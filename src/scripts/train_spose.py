#!/usr/bin/env python3
"""Train a SPoSE embedding from odd-one-out triplet data.

Reimplements the model from Hebart et al. (2020), Nature Human Behaviour.
See ``things_spose/training/model.py`` for the model and
``things_spose/training/train.py`` for the objective and training loop.

Examples
--------
Train on the shipped THINGS train/validation splits (``data/triplet_dataset/``)::

    python scripts/train_spose.py --out spose_embedding_trained.txt

Train on your own splits::

    python scripts/train_spose.py \
        --triplets path/to/train_triplets.txt \
        --val-triplets path/to/test_triplets.txt \
        --out my_embedding.txt

Quick smoke test on the small 10% test set, auto-split 90/10::

    python scripts/train_spose.py --demo --epochs 30

``--demo`` is for wiring checks only: 131k triplets is too little data to prune
dimensions or to approach the published accuracy, and it overfits within ~20
epochs. Use the full training split for a real fit.

Triplet file format: one triplet per line, three 0-based object indices
``i j k`` separated by whitespace, where columns 0 and 1 are the pair the
participant kept together and column 2 is the odd-one-out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow running as a plain script (``python scripts/train_spose.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from things_spose.core import paths
from things_spose.data.sampling import load_triplets, train_val_split
from things_spose.training.checkpoint import save_embedding
from things_spose.training.train import TrainConfig, train_spose


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a SPoSE embedding from odd-one-out triplets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--triplets", type=Path,
                   help="Training triplet file (0-based 'i j k' per row). "
                        "Default: trainset.txt from data/triplet_dataset/.")
    p.add_argument("--val-triplets", type=Path, default=None,
                   help="Optional held-out triplet file for accuracy reporting. "
                        "Default: validationset.txt from data/triplet_dataset/ "
                        "when --triplets is also defaulted.")
    p.add_argument("--val-split", type=float, default=0.0,
                   help="If >0 and --val-triplets is unset, hold out this "
                        "fraction of --triplets for validation.")
    p.add_argument("--demo", action="store_true",
                   help="Use the small 10%% test set (data1854_batch5_test10.txt) "
                        "with a 90/10 split. For smoke-testing only.")
    p.add_argument("--n-objects", type=int, default=None,
                   help="Number of objects. Default: max index in data + 1.")

    p.add_argument("--dim", type=int, default=90, help="Initial dimensionality.")
    p.add_argument("--lambda", dest="lambda_l1", type=float, default=0.008,
                   help="L1 sparsity weight (paper value: 0.008).")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Softmax temperature (beta) for choice randomness.")
    p.add_argument("--prune-threshold", type=float, default=0.1,
                   help="Drop dims whose max weight across objects is below this.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=0,
                   help="Early-stop after this many non-improving evals (0=off).")
    p.add_argument("--device", default=None,
                   help="cpu / cuda / mps. Default: auto-detect.")
    p.add_argument("--out", type=Path, default=Path("spose_embedding_trained.txt"),
                   help="Where to write the fitted embedding.")

    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Directory for model_epoch*.tar checkpoints. "
                        "Default: none (no checkpointing).")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="Checkpoint every N epochs (0 = only at the end). "
                        "Requires --checkpoint-dir.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the newest checkpoint in --checkpoint-dir.")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Also write per-epoch progress to this file.")
    return p.parse_args(argv)


def auto_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv=None) -> int:
    args = parse_args(argv)

    # ---- Resolve the triplet source ------------------------------------- #
    if args.demo:
        src = paths.data("data1854_batch5_test10.txt")
        print(f"[demo] loading 10% test set: {src}")
        triplets = load_triplets(src)
        train_tri, val_tri = train_val_split(triplets, val_frac=0.1, seed=args.seed)
        print("[demo] NOTE: this is only the 10% test set (146k triplets). For a "
              "real fit, drop --demo to use the full training split.")
    else:
        # Default to the shipped THINGS splits in data/triplet_dataset/.
        train_path = args.triplets or paths.triplets("trainset.txt")
        val_path = args.val_triplets
        if val_path is None and args.triplets is None:
            val_path = paths.triplets("validationset.txt")
        if not train_path.exists():
            print(f"error: triplet file not found: {train_path}", file=sys.stderr)
            return 2

        print(f"loading triplets: {train_path}")
        triplets = load_triplets(train_path)
        if val_path:
            train_tri = triplets
            print(f"loading validation triplets: {val_path}")
            val_tri = load_triplets(val_path)
        elif args.val_split > 0:
            train_tri, val_tri = train_val_split(triplets, args.val_split, args.seed)
        else:
            train_tri, val_tri = triplets, None

    n_objects = args.n_objects or int(triplets.max()) + 1
    print(f"objects: {n_objects}  |  train triplets: {len(train_tri):,}"
          + (f"  |  val triplets: {len(val_tri):,}" if val_tri is not None else ""))

    device = auto_device(args.device)
    print(f"device: {device}")

    cfg = TrainConfig(
        n_dim=args.dim,
        lambda_l1=args.lambda_l1,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        prune_threshold=args.prune_threshold,
        device=device,
        temperature=args.temperature,
        eval_every=args.eval_every,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        log_file=args.log_file,
    )

    result = train_spose(train_tri, n_objects, cfg, val_triplets=val_tri)

    save_embedding(result.embedding, args.out)
    print(f"saved embedding {result.embedding.shape} -> {args.out}")
    if result.val_accuracy is not None:
        print(f"final odd-one-out val accuracy: {result.val_accuracy*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
