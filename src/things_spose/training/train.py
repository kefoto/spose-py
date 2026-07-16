"""Train a SPoSE embedding from odd-one-out triplet judgments (PyTorch).

This is a faithful reimplementation of the model described in the Methods of

    Hebart, Zheng, Pereira & Baker (2020). Revealing the multidimensional
    mental representations of natural objects underlying human similarity
    judgments. *Nature Human Behaviour*, 4, 1173-1185.

    "Details of computational modeling procedure" (paper p. 13)

The model itself lives in :mod:`things_spose.training.model` (``SPoSE``,
``l1_regularization``); this module supplies the objective, the training loop,
and the pruning/sorting step. Both follow the conventions of the reference
implementation (ViCCo-Group/SPoSE, Muttenthaler & Hebart).

Dataset
-------
The raw human triplets ship with this repository under ``data/triplet_dataset/``
(see :data:`things_spose.core.paths.TRIPLET_DIR`): ``trainset.txt``,
``validationset.txt``, and several test splits. The pre-trained 49-d embedding
that reproduces the paper's figures is separate, in ``data/spose/data/``. The
full dataset is also distributed by the THINGS initiative:
https://things-initiative.org/ (mirrored on OSF).

Assumptions this code makes about the data / weights
----------------------------------------------------
1. **Triplet layout** — each row is ``(i, j, k)`` with the participant's
   *chosen (kept-together) pair in columns 0 and 1* and the odd-one-out in
   column 2. The official THINGS ``trainset.txt``/``validationset.txt`` are
   already stored this way ("resorted to have the chosen pair first, followed
   by the odd one out at the end"), so the softmax target is always class 0.
2. **0-based, contiguous object IDs** — indices are ``0 .. n_objects-1`` and
   select rows of the identity matrix used for one-hot encoding. The raw files
   are already 0-based; ``n_objects`` defaults to ``max(index) + 1`` (1854 for
   THINGS).
3. **No ``sortind`` remap is applied or needed for training.** Object IDs are
   arbitrary labels; the fitted dimensions are ordered post-hoc by descending
   column sum, matching the paper. A freshly trained embedding therefore aligns
   *row-for-row* with the shipped ``spose_embedding_49d_sorted.txt`` only if you
   trained on the same objects AND apply the same ``sortind`` object-order map
   (see ``dataio.load_sortind``); the *set* of dimensions is reproducible, but
   their exact identities/order depend on the random init and the data.
4. **Weights are non-negative and sparse** by construction (an explicit
   positivity penalty; L1-penalised). A dimension is considered "dead" and
   pruned when its largest weight across all objects falls below
   ``prune_threshold`` (0.1).

Model
-----
:class:`things_spose.training.model.SPoSE` stores the embedding as an
``nn.Linear(n_objects, n_dim, bias=False)``, so ``model.fc.weight`` is ``D x M``
(dimensions by objects) -- the *transpose* of the ``M x D`` layout used
everywhere else in this package and by the shipped embedding file. A triplet is
fed in as three one-hot rows, and the linear layer acts as an embedding lookup.
:func:`train_spose` transposes back to ``M x D`` on the way out, so
``TrainResult.embedding`` matches the shipped orientation.

For a triplet ``(i, j, k)`` in which a participant judged the pair ``(i, j)`` to
belong together (so ``k`` is the odd-one-out), proximity is the dot product of
the two object vectors, and the choice probability is a softmax over the three
possible pairs::

    p(i, j | i, j, k) = exp(x_i . x_j)
                        ---------------------------------------------
                        exp(x_i . x_j) + exp(x_i . x_k) + exp(x_j . x_k)

Objective (paper eq., p. 13): cross-entropy (negative log of the softmax
above) summed over triplets, plus an L1 penalty ``lambda * sum |X|`` that
encourages sparsity, plus a penalty pushing weights non-negative.

Optimisation: Adam with default parameters, minibatches of 100 triplets. After
training, dimensions whose weights are below ``prune_threshold`` (0.1) for
*every* object are dropped, and the survivors are sorted by descending column
sum -- exactly as in the paper, which is left with 49 dimensions.

Triplet convention
-------------------
Each triplet row is ``(i, j, k)`` where **columns 0 and 1 are the pair the
participant kept together** and column 2 is the odd-one-out. This was verified
against the shipped embedding: scoring ``data1854_batch5_test10.txt`` under this
convention reproduces the paper's 64.60% test accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import checkpoint, run_log
from .model import SPoSE, l1_regularization


# --------------------------------------------------------------------------- #
# Config + history containers
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    n_dim: int = 90            # initial dimensionality (paper: 90)
    lambda_l1: float = 0.008   # sparsity weight (paper: cross-validated to 0.008)
    lr: float = 1e-3           # Adam default
    batch_size: int = 100      # paper minibatch size
    epochs: int = 200
    seed: int = 0
    prune_threshold: float = 0.1   # drop dims whose max weight < this (paper: 0.1)
    device: str = "cpu"
    temperature: float = 1.0   # softmax temperature (beta) for choice randomness
    pos_penalty: float = 0.01  # weight of the non-negativity penalty
    # Report validation accuracy every ``eval_every`` epochs (0 = only at end).
    eval_every: int = 5
    # Stop if val accuracy has not improved for this many evaluations (0 = off).
    patience: int = 0

    # --- run logging / checkpointing (all optional; None = off) ------------ #
    # Directory for ``model_epoch*.tar`` checkpoints. None disables checkpointing.
    checkpoint_dir: str | Path | None = None
    # Write a checkpoint every ``checkpoint_every`` epochs (0 = only at the end).
    checkpoint_every: int = 0
    # Resume from the newest checkpoint in ``checkpoint_dir`` if one exists.
    resume: bool = False
    # File to log per-epoch progress to. None = console only.
    log_file: str | Path | None = None


@dataclass
class TrainResult:
    embedding: np.ndarray                 # (M, D_pruned) sorted, non-negative
    keep_dims: np.ndarray                 # indices (into the D=n_dim model) kept
    val_accuracy: float | None            # odd-one-out accuracy on val set
    history: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# One-hot encoding + similarities
# --------------------------------------------------------------------------- #
def encode_as_onehot(I: torch.Tensor, triplets: torch.Tensor) -> torch.Tensor:
    """Encode a ``(B, 3)`` batch of object indices as ``(B*3, M)`` one-hot rows.

    ``I`` is an ``M x M`` identity matrix. Rows come out in triplet-major order
    (``i0, j0, k0, i1, j1, k1, ...``), which is what :func:`unbind_embeddings`
    relies on.
    """
    return I[triplets.flatten(), :]


def unbind_embeddings(logits: torch.Tensor, n_dim: int
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a ``(B*3, D)`` stack back into ``(anchor, positive, negative)``.

    Inverse of :func:`encode_as_onehot`'s flattening: reshaping to ``(B, 3, D)``
    recovers one row per triplet, and unbinding dim 1 yields the three object
    embeddings. ``anchor`` and ``positive`` are the kept-together pair (input
    columns 0 and 1); ``negative`` is the odd-one-out (column 2).
    """
    return torch.unbind(torch.reshape(logits, (-1, 3, n_dim)), dim=1)


def compute_similarities(anchor: torch.Tensor, positive: torch.Tensor,
                         negative: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Dot-product proximity of each of the three pairs in a triplet.

    Returns ``(anchor.positive, anchor.negative, positive.negative)``. The first
    element is the pair the participant kept together, so the softmax target is
    always class 0.
    """
    pos_sim = torch.sum(anchor * positive, dim=1)
    neg_sim = torch.sum(anchor * negative, dim=1)
    neg_sim_2 = torch.sum(positive * negative, dim=1)
    return pos_sim, neg_sim, neg_sim_2


def _softmax(sims: tuple[torch.Tensor, ...], t: torch.Tensor) -> torch.Tensor:
    return torch.exp(sims[0] / t) / torch.sum(
        torch.stack([torch.exp(sim / t) for sim in sims]), dim=0)


def trinomial_loss(anchor: torch.Tensor, positive: torch.Tensor,
                   negative: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy of the three-way choice (target = the kept pair)."""
    sims = compute_similarities(anchor, positive, negative)
    return torch.mean(-torch.log(_softmax(sims, t)))


def spose_loss(anchor: torch.Tensor, positive: torch.Tensor,
               negative: torch.Tensor, model: SPoSE, t: torch.Tensor,
               lambda_l1: float, n_objects: int,
               pos_penalty: float = 0.01) -> torch.Tensor:
    """SPoSE objective on one minibatch.

    ``mean`` cross-entropy over the batch plus an L1 penalty scaled by
    ``lambda_l1 / n_objects`` and a penalty pushing weights non-negative. This
    matches the reference SPoSE implementation (ViCCo-Group/SPoSE, Muttenthaler
    & Hebart), where the per-batch objective is::

        loss = mean_CE  +  0.01 * sum(relu(-W))  +  (lmbda / n_items) * ||W||_1

    with ``n_items`` the number of objects and ``lmbda = 0.008``. Calibrating
    the L1 weight *per object* (not per triplet) is what makes ``0.008`` exert
    enough sparsity pressure to drive unused dimensions to zero while leaving
    informative ones intact -- so :func:`prune_and_sort` can drop them.

    Scaling pitfalls found empirically on the full THINGS data:
      * ``lambda / n_train`` (per *triplet*, ~2e3x weaker): accuracy is fine but
        nothing prunes -- all 90 dimensions stay alive.
      * ``lambda`` unscaled (~n_objects x stronger): the whole embedding
        collapses to zero within one epoch.
    """
    c_entropy = trinomial_loss(anchor, positive, negative, t)
    pos_pen = torch.sum(F.relu(-model.fc.weight))
    complexity = (lambda_l1 / n_objects) * l1_regularization(model)
    return c_entropy + pos_penalty * pos_pen + complexity


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #
@torch.no_grad()
def odd_one_out_accuracy(model: SPoSE, triplets: torch.Tensor,
                         batch_size: int = 1024) -> float:
    """Fraction of triplets whose kept-together pair (col 0) is the argmax.

    Degenerate ties (all three proximities equal, e.g. a collapsed all-zero
    embedding) are counted as *incorrect*, matching the reference ``accuracy_``.
    Without this, a collapsed model would report a spurious 100%.
    """
    model.eval()
    device = model.fc.weight.device
    I = torch.eye(model.in_size, device=device)
    correct = 0
    for start in range(0, triplets.shape[0], batch_size):
        batch = triplets[start:start + batch_size]
        out = model(encode_as_onehot(I, batch))
        anchor, positive, negative = unbind_embeddings(out, model.out_size)
        sims = torch.stack(compute_similarities(anchor, positive, negative), dim=-1)
        pred = sims.argmax(dim=1)
        tie = (sims == sims.amax(dim=1, keepdim=True)).all(dim=1)  # all equal
        correct += int(((pred == 0) & ~tie).sum())
    return correct / triplets.shape[0]


# --------------------------------------------------------------------------- #
# Pruning + sorting
# --------------------------------------------------------------------------- #
def prune_and_sort(weights: np.ndarray, threshold: float = 0.1
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Drop near-zero dimensions and sort by descending column sum.

    ``weights`` is ``M x D``. Returns ``(pruned_embedding, keep_dims)`` where
    ``keep_dims`` indexes the original columns in their new (sorted) order. For
    non-negative weights, sorting by column sum is equivalent to the reference's
    sort by descending L1 norm.
    """
    keep = np.where(weights.max(axis=0) >= threshold)[0]
    order = keep[np.argsort(-weights[:, keep].sum(axis=0))]
    return weights[:, order], order


def embedding_from(model: SPoSE) -> np.ndarray:
    """Extract the ``M x D`` non-negative embedding from a trained model.

    ``model.fc.weight`` is ``D x M``, so it is transposed here. Non-negativity is
    enforced by a penalty rather than a hard projection, so small negative values
    can survive; ``abs`` cleans them up, as the reference ``save_weights_`` does.
    """
    return np.abs(model.fc.weight.detach().cpu().numpy().T)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_spose(train_triplets: np.ndarray,
                n_objects: int,
                cfg: TrainConfig = TrainConfig(),
                val_triplets: np.ndarray | None = None,
                verbose: bool = True) -> TrainResult:
    """Fit a SPoSE embedding from odd-one-out triplets.

    Parameters
    ----------
    train_triplets : (N, 3) int array
        Object indices ``(i, j, k)``; columns 0,1 = kept-together pair.
    n_objects : int
        Total number of objects (defines the embedding's row count).
    cfg : TrainConfig
    val_triplets : (V, 3) int array, optional
        Held-out triplets for odd-one-out accuracy reporting.
    verbose : bool
        Echo per-epoch progress to the console. Progress always goes to
        ``cfg.log_file`` if one is set, independently of this flag.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    logger = run_log.setup(cfg.log_file, console=verbose, append=cfg.resume)

    model = SPoSE(in_size=n_objects, out_size=cfg.n_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    temperature = torch.tensor(cfg.temperature, device=device)
    I = torch.eye(n_objects, device=device)

    train = torch.as_tensor(np.asarray(train_triplets), dtype=torch.long, device=device)
    val = (torch.as_tensor(np.asarray(val_triplets), dtype=torch.long, device=device)
           if val_triplets is not None else None)
    n_train = train.shape[0]

    history: list[dict] = []
    best_acc, best_state, stale = -1.0, None, 0
    start_epoch = 1

    if cfg.resume and cfg.checkpoint_dir is not None:
        prev = checkpoint.latest(cfg.checkpoint_dir)
        if prev is None:
            logger.info(f"no checkpoint in {cfg.checkpoint_dir}; starting from scratch")
        else:
            ckpt = checkpoint.load(prev, model, opt, device=device)
            start_epoch = ckpt["epoch"] + 1
            history = list(ckpt.get("history", []))
            logger.info(f"resumed from {prev} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        for start in range(0, n_train, cfg.batch_size):
            batch = train[perm[start:start + cfg.batch_size]]
            opt.zero_grad()
            out = model(encode_as_onehot(I, batch))
            anchor, positive, negative = unbind_embeddings(out, cfg.n_dim)
            loss = spose_loss(anchor, positive, negative, model, temperature,
                              cfg.lambda_l1, n_objects, cfg.pos_penalty)
            loss.backward()
            opt.step()
            epoch_loss += float(loss)

        do_eval = cfg.eval_every and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs)
        if do_eval:
            n_active = int((embedding_from(model).max(0) >= cfg.prune_threshold).sum())
            rec = {"epoch": epoch, "loss": epoch_loss, "active_dims": n_active}
            if val is not None:
                rec["val_acc"] = odd_one_out_accuracy(model, val)
            history.append(rec)
            run_log.log_epoch(logger, rec)

            if val is not None and cfg.patience:
                if rec["val_acc"] > best_acc + 1e-5:
                    best_acc = rec["val_acc"]
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= cfg.patience:
                        logger.info(f"early stop at epoch {epoch} "
                                    f"(best val_acc {best_acc*100:.2f}%)")
                        break

        if (cfg.checkpoint_dir is not None and cfg.checkpoint_every
                and epoch % cfg.checkpoint_every == 0):
            p = checkpoint.save(checkpoint.epoch_path(cfg.checkpoint_dir, epoch),
                                epoch, model, opt, history)
            logger.info(f"wrote checkpoint {p}")

    if best_state is not None:
        model.load_state_dict(best_state)

    weights = embedding_from(model)
    pruned, keep_dims = prune_and_sort(weights, cfg.prune_threshold)
    val_acc = odd_one_out_accuracy(model, val) if val is not None else None
    logger.info(f"done: {weights.shape[1]} initial dims -> {pruned.shape[1]} after pruning"
                + (f"  |  final val_acc {val_acc*100:.2f}%" if val_acc is not None else ""))

    if cfg.checkpoint_dir is not None:
        p = checkpoint.save(checkpoint.epoch_path(cfg.checkpoint_dir, cfg.epochs),
                            cfg.epochs, model, opt, history, val_accuracy=val_acc)
        logger.info(f"wrote final checkpoint {p}")

    return TrainResult(embedding=pruned, keep_dims=keep_dims,
                       val_accuracy=val_acc, history=history)
