"""Train a SPoSE embedding from odd-one-out triplet judgments (PyTorch).

This is a faithful reimplementation of the model described in the Methods of

    Hebart, Zheng, Pereira & Baker (2020). Revealing the multidimensional
    mental representations of natural objects underlying human similarity
    judgments. *Nature Human Behaviour*, 4, 1173-1185.

    "Details of computational modeling procedure" (paper p. 13)

Only the *pre-trained* 49-d embedding ships with this repository; the code
that produced it does not. This module reconstructs that training procedure so
you can fit a fresh embedding from raw triplet data.

Dataset
-------
The full human triplet dataset (the ~4.1M/453K train/validation split, the
1,854 THINGS object images, and the pre-trained SPoSE weights) is distributed
by the THINGS initiative: https://things-initiative.org/ (data also mirrored on
OSF). This repository only ships the 10% odd-one-out *test* set
(``data1854_batch5_test10.txt``) and the pre-trained ``spose_embedding_49d``,
so the shipped files alone reproduce the paper's *figures* but not its
*training*. Point ``train_spose`` at the full ``trainset.txt`` /
``validationset.txt`` to fit a fresh embedding.

Assumptions this code makes about the data / weights
----------------------------------------------------
1. **Triplet layout** — each row is ``(i, j, k)`` with the participant's
   *chosen (kept-together) pair in columns 0 and 1* and the odd-one-out in
   column 2. The official THINGS ``trainset.txt``/``validationset.txt`` are
   already stored this way ("resorted to have the chosen pair first, followed
   by the odd one out at the end"), so the softmax target is always class 0.
2. **0-based, contiguous object IDs** — indices are ``0 .. n_objects-1`` and
   index directly into the embedding rows. The raw files are already 0-based;
   ``n_objects`` defaults to ``max(index) + 1`` (1854 for THINGS).
3. **No ``sortind`` remap is applied or needed for training.** Object IDs are
   arbitrary labels; the fitted dimensions are ordered post-hoc by descending
   column sum, matching the paper. A freshly trained embedding therefore aligns
   *row-for-row* with the shipped ``spose_embedding_49d_sorted.txt`` only if you
   trained on the same objects AND apply the same ``sortind`` object-order map
   (see ``dataio.load_sortind``); the *set* of dimensions is reproducible, but
   their exact identities/order depend on the random init and the data.
4. **Weights are non-negative and sparse** by construction (clamped ``>= 0``
   every step; L1-penalised). A dimension is considered "dead" and pruned when
   its largest weight across all objects falls below ``prune_threshold`` (0.1).

Model
-----
The embedding ``X`` is an ``M x D`` matrix (``M`` objects, ``D`` latent
dimensions, initialised ``D = 90``). Every weight is constrained to be
**non-negative**. For a triplet ``(i, j, k)`` in which a participant judged the
pair ``(i, j)`` to belong together (so ``k`` is the odd-one-out), proximity is
the dot product of the two object vectors, and the choice probability is a
softmax over the three possible pairs::

    p(i, j | i, j, k) = exp(x_i . x_j)
                        ---------------------------------------------
                        exp(x_i . x_j) + exp(x_i . x_k) + exp(x_j . x_k)

Objective (paper eq., p. 13): cross-entropy (negative log of the softmax
above) summed over triplets, plus an L1 penalty ``lambda * sum |X|`` that
encourages sparsity. Because ``X >= 0`` the L1 term is just ``lambda * sum X``.

Optimisation: Adam with default parameters, minibatches of 100 triplets, with
non-negativity enforced by projecting the weights back to ``>= 0`` after every
optimiser step (projected gradient descent). After training, dimensions whose
weights are below ``prune_threshold`` (0.1) for *every* object are dropped, and
the survivors are sorted by descending column sum -- exactly as in the paper,
which is left with 49 dimensions.

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
from torch import nn


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class SPoSE(nn.Module):
    """Sparse Positive Similarity Embedding.

    Parameters
    ----------
    n_objects : int
        Number of objects ``M`` (1854 for the THINGS dataset).
    n_dim : int
        Initial dimensionality ``D`` (90 in the paper).
    init_range : tuple[float, float]
        Uniform init range for the weights (paper: ``(0, 1)``).
    """

    def __init__(self, n_objects: int, n_dim: int = 90,
                 init_range: tuple[float, float] = (0.0, 1.0),
                 seed: int | None = None):
        super().__init__()
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        lo, hi = init_range
        w = torch.rand(n_objects, n_dim, generator=g) * (hi - lo) + lo
        self.weights = nn.Parameter(w)

    @property
    def embedding(self) -> torch.Tensor:
        return self.weights

    def forward(self, triplets: torch.Tensor) -> torch.Tensor:
        """Return the 3 pairwise-proximity logits for a batch of triplets.

        ``triplets`` is a ``(B, 3)`` long tensor of object indices
        ``(i, j, k)``. Output is ``(B, 3)`` = ``[x_i.x_j, x_i.x_k, x_j.x_k]``.
        The correct class (the kept-together pair) is always column 0.
        """
        xi = self.weights[triplets[:, 0]]
        xj = self.weights[triplets[:, 1]]
        xk = self.weights[triplets[:, 2]]
        sim_ij = (xi * xj).sum(dim=1)
        sim_ik = (xi * xk).sum(dim=1)
        sim_jk = (xj * xk).sum(dim=1)
        return torch.stack([sim_ij, sim_ik, sim_jk], dim=1)

    @torch.no_grad()
    def clamp_nonnegative(self) -> None:
        """Project weights back onto the non-negative orthant."""
        self.weights.clamp_(min=0.0)


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
    # Report validation accuracy every ``eval_every`` epochs (0 = only at end).
    eval_every: int = 5
    # Stop if val accuracy has not improved for this many evaluations (0 = off).
    patience: int = 0


@dataclass
class TrainResult:
    embedding: np.ndarray                 # (M, D_pruned) sorted, non-negative
    keep_dims: np.ndarray                 # indices (into the D=n_dim model) kept
    val_accuracy: float | None            # odd-one-out accuracy on val set
    history: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Loss + accuracy
# --------------------------------------------------------------------------- #
def triplet_loss(logits: torch.Tensor, weights: torch.Tensor,
                 lambda_l1: float, n_objects: int) -> torch.Tensor:
    """SPoSE objective on one minibatch.

    ``mean`` cross-entropy over the batch (target class 0) plus an L1 penalty
    scaled by ``lambda_l1 / n_objects``. This matches the reference SPoSE
    implementation (ViCCo-Group/SPoSE, Muttenthaler & Hebart), where the
    per-batch objective is::

        loss = mean_CE  +  (lmbda / n_items) * ||W||_1      # + soft positivity

    with ``n_items`` the number of objects and ``lmbda = 0.008``. Calibrating
    the L1 weight *per object* (not per triplet) is what makes ``0.008`` exert
    enough sparsity pressure to drive unused dimensions to zero while leaving
    informative ones intact -- so :func:`prune_and_sort` can drop them.

    Scaling pitfalls found empirically on the full THINGS data:
      * ``lambda / n_train`` (per *triplet*, ~2e3x weaker): accuracy is fine but
        nothing prunes -- all 90 dimensions stay alive.
      * ``lambda`` unscaled (~n_objects x stronger): the whole embedding
        collapses to zero within one epoch.

    Note the reference enforces positivity with a *soft* penalty
    ``0.01 * sum(relu(-W))``; here we instead use a hard projection to ``>= 0``
    after each optimiser step (see :meth:`SPoSE.clamp_nonnegative`), which is a
    stricter reading of the paper's "strictly enforcing weights ... positive".
    """
    batch = logits.shape[0]
    target = logits.new_zeros(batch, dtype=torch.long)  # column 0 is correct
    ce = torch.nn.functional.cross_entropy(logits, target, reduction="mean")
    l1 = weights.abs().sum()  # == weights.sum() since X >= 0
    return ce + (lambda_l1 / n_objects) * l1


@torch.no_grad()
def odd_one_out_accuracy(model: SPoSE, triplets: torch.Tensor,
                         batch_size: int = 4096) -> float:
    """Fraction of triplets whose kept-together pair (col 0) is the argmax.

    Degenerate ties (all three proximities equal, e.g. a collapsed all-zero
    embedding) are counted as *incorrect*, matching the reference ``accuracy_``.
    Without this, a collapsed model would report a spurious 100%.
    """
    model.eval()
    correct = 0
    for start in range(0, triplets.shape[0], batch_size):
        batch = triplets[start:start + batch_size]
        logits = model(batch)
        pred = logits.argmax(dim=1)
        tie = (logits == logits.amax(dim=1, keepdim=True)).all(dim=1)  # all equal
        correct += int(((pred == 0) & ~tie).sum())
    return correct / triplets.shape[0]


# --------------------------------------------------------------------------- #
# Pruning + sorting
# --------------------------------------------------------------------------- #
def prune_and_sort(weights: np.ndarray, threshold: float = 0.1
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Drop near-zero dimensions and sort by descending column sum.

    Returns ``(pruned_embedding, keep_dims)`` where ``keep_dims`` indexes the
    original columns in their new (sorted) order.
    """
    keep = np.where(weights.max(axis=0) >= threshold)[0]
    order = keep[np.argsort(-weights[:, keep].sum(axis=0))]
    return weights[:, order], order


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
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    model = SPoSE(n_objects, n_dim=cfg.n_dim, seed=cfg.seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train = torch.as_tensor(np.asarray(train_triplets), dtype=torch.long, device=device)
    val = (torch.as_tensor(np.asarray(val_triplets), dtype=torch.long, device=device)
           if val_triplets is not None else None)
    n_train = train.shape[0]

    history: list[dict] = []
    best_acc, best_state, stale = -1.0, None, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        for start in range(0, n_train, cfg.batch_size):
            batch = train[perm[start:start + cfg.batch_size]]
            opt.zero_grad()
            loss = triplet_loss(model(batch), model.weights, cfg.lambda_l1, n_objects)
            loss.backward()
            opt.step()
            model.clamp_nonnegative()  # projected gradient: enforce X >= 0
            epoch_loss += float(loss)

        do_eval = cfg.eval_every and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs)
        if do_eval:
            n_active = int((model.weights.detach().cpu().numpy().max(0) >= cfg.prune_threshold).sum())
            rec = {"epoch": epoch, "loss": epoch_loss, "active_dims": n_active}
            if val is not None:
                rec["val_acc"] = odd_one_out_accuracy(model, val)
            history.append(rec)
            if verbose:
                msg = f"epoch {epoch:4d}  loss {epoch_loss:12.1f}  active_dims {n_active:3d}"
                if val is not None:
                    msg += f"  val_acc {rec['val_acc']*100:5.2f}%"
                print(msg)

            if val is not None and cfg.patience:
                if rec["val_acc"] > best_acc + 1e-5:
                    best_acc = rec["val_acc"]
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= cfg.patience:
                        if verbose:
                            print(f"early stop at epoch {epoch} (best val_acc {best_acc*100:.2f}%)")
                        break

    if best_state is not None:
        model.load_state_dict(best_state)

    weights = model.weights.detach().cpu().numpy()
    pruned, keep_dims = prune_and_sort(weights, cfg.prune_threshold)
    val_acc = odd_one_out_accuracy(model, val) if val is not None else None
    if verbose:
        print(f"done: {weights.shape[1]} initial dims -> {pruned.shape[1]} after pruning"
              + (f"  |  final val_acc {val_acc*100:.2f}%" if val_acc is not None else ""))

    return TrainResult(embedding=pruned, keep_dims=keep_dims,
                       val_accuracy=val_acc, history=history)


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def load_triplets(path: str | Path) -> np.ndarray:
    """Load a whitespace-delimited triplet file (0-based ``i j k`` per row)."""
    return np.loadtxt(path, dtype=np.int64)


def train_val_split(triplets: np.ndarray, val_frac: float = 0.1,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random split into (train, val). The paper trains on 90%, tests on 10%."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(triplets))
    n_val = int(round(len(triplets) * val_frac))
    return triplets[idx[n_val:]], triplets[idx[:n_val]]


def save_embedding(embedding: np.ndarray, path: str | Path) -> None:
    """Write the embedding as a space-delimited txt (like the shipped file)."""
    np.savetxt(path, embedding, fmt="%.8f")
