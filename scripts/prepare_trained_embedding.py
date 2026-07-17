#!/usr/bin/env python3
"""Remap a freshly trained embedding from raw object-ID order into sorted order.

``scripts/train_spose.py`` writes an embedding whose rows are indexed by the
raw 0-based object IDs used in ``data/triplet_dataset/``. Every downstream
analysis (``things_spose.data.dataio.load_dataset``, the notebooks, figure
scripts) expects rows in the same sorted object order as the shipped
``spose_embedding_49d_sorted.txt``. This script applies that one-time reorder
(``sorted_embedding = raw_embedding[sortind0]``, the same convention used for
the Extended Data Fig. 1 reference models in
``things_spose.analysis.reproducibility.load_reference_models``) and writes
the result next to the input.

Usage
-----
    python scripts/prepare_trained_embedding.py \
        --in cache/spose_embedding_trained.txt \
        --out cache/spose_embedding_trained_sorted.txt

Then point analyses at it, e.g.::

    THINGS_EMBEDDING_PATH=cache/spose_embedding_trained_sorted.txt \
        python scripts/run_analyses.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from things_spose.data import dataio


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--in", dest="in_path", type=Path,
                   default=REPO_ROOT / "cache" / "spose_embedding_trained.txt",
                   help="Raw trained embedding (object-ID order).")
    p.add_argument("--out", dest="out_path", type=Path, default=None,
                   help="Where to write the sorted embedding. Default: "
                        "<in>_sorted.txt next to the input.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite --out if it already exists.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_path = args.out_path or args.in_path.with_name(
        args.in_path.stem + "_sorted" + args.in_path.suffix
    )

    if out_path.exists() and not args.force:
        print(f"already exists, skipping (use --force to redo): {out_path}")
        return 0

    if not args.in_path.exists():
        print(f"error: input embedding not found: {args.in_path}", file=sys.stderr)
        return 2

    raw = np.loadtxt(args.in_path, dtype=np.float64)
    sortind0 = dataio.load_sortind()
    if raw.shape[0] != sortind0.shape[0]:
        print(f"error: embedding has {raw.shape[0]} rows, expected "
              f"{sortind0.shape[0]} (one per object) -- was this trained on "
              "the full 1854-object THINGS set?", file=sys.stderr)
        return 2

    sorted_embedding = raw[sortind0]
    np.savetxt(out_path, sorted_embedding, fmt="%.8f")
    print(f"wrote {sorted_embedding.shape} sorted embedding -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
