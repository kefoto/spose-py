#!/usr/bin/env python3
"""Train a SPoSE embedding from odd-one-out triplet data.

Reimplements the model from Hebart et al. (2020), Nature Human Behaviour.
See ``things_spose/train.py`` for the model and objective.

Examples
--------
Train on your own 90% training split and validate on a held-out file::

    python scripts/train_spose.py \
        --triplets path/to/train_triplets.txt \
        --val-triplets path/to/test_triplets.txt \
        --out my_embedding.txt

Quick smoke test using only the shipped 10% test set, auto-split 90/10
(this will *not* reproduce the paper -- the real training set is not shipped)::

    python scripts/train_spose.py --demo --epochs 30

Triplet file format: one triplet per line, three 0-based object indices
``i j k`` separated by whitespace, where columns 0 and 1 are the pair the
participant kept together and column 2 is the odd-one-out.

Full dataset (real train/validation splits, images, pre-trained weights):
    https://things-initiative.org/   (also mirrored on OSF)

    python scripts/train_spose.py \
        --triplets   /path/to/trainset.txt \
        --val-triplets /path/to/validationset.txt \
        --out spose_embedding_trained.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as a plain script (``python scripts/train_spose.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from things_spose import paths
from things_spose.train import (
    TrainConfig,
    load_triplets,
    save_embedding,
    train_spose,
    train_val_split,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a SPoSE embedding from odd-one-out triplets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--triplets", type=Path,
                   help="Training triplet file (0-based 'i j k' per row).")
    p.add_argument("--val-triplets", type=Path, default=None,
                   help="Optional held-out triplet file for accuracy reporting.")
    p.add_argument("--val-split", type=float, default=0.0,
                   help="If >0 and --val-triplets is unset, hold out this "
                        "fraction of --triplets for validation.")
    p.add_argument("--demo", action="store_true",
                   help="Use the shipped test set (data1854_batch5_test10.txt) "
                        "with a 90/10 split. For smoke-testing only.")
    p.add_argument("--n-objects", type=int, default=None,
                   help="Number of objects. Default: max index in data + 1.")

    p.add_argument("--dim", type=int, default=90, help="Initial dimensionality.")
    p.add_argument("--lambda", dest="lambda_l1", type=float, default=0.008,
                   help="L1 sparsity weight (paper value: 0.008).")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--epochs", type=int, default=200)
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
        print(f"[demo] loading shipped test set: {src}")
        triplets = load_triplets(src)
        train_tri, val_tri = train_val_split(triplets, val_frac=0.1, seed=args.seed)
        print("[demo] NOTE: this is only the 10% test set (146k triplets); the "
              "paper's 90% training set is not distributed, so results will not "
              "match the published embedding.")
    else:
        if not args.triplets:
            print("error: provide --triplets (or use --demo).", file=sys.stderr)
            return 2
        triplets = load_triplets(args.triplets)
        if args.val_triplets:
            train_tri = triplets
            val_tri = load_triplets(args.val_triplets)
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
        eval_every=args.eval_every,
        patience=args.patience,
    )

    result = train_spose(train_tri, n_objects, cfg, val_triplets=val_tri)

    save_embedding(result.embedding, args.out)
    print(f"saved embedding {result.embedding.shape} -> {args.out}")
    if result.val_accuracy is not None:
        print(f"final odd-one-out val accuracy: {result.val_accuracy*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
