"""Loading and sampling raw odd-one-out triplets for training.

Triplet files are whitespace-delimited, one triplet per line, three **0-based**
object indices ``i j k`` where columns 0 and 1 are the pair the participant kept
together and column 2 is the odd-one-out. The THINGS ``trainset.txt`` /
``validationset.txt`` in ``data/triplet_dataset/`` are already stored this way
("resorted to have the chosen pair first, followed by the odd one out at the
end"), so the training softmax target is always class 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core import paths


def load_triplets(path: str | Path) -> np.ndarray:
    """Load a whitespace-delimited triplet file (0-based ``i j k`` per row)."""
    return np.loadtxt(path, dtype=np.int64)


def load_split(name: str) -> np.ndarray:
    """Load a named split from ``data/triplet_dataset`` (e.g. ``trainset.txt``)."""
    return load_triplets(paths.triplets(name))


def n_objects(triplets: np.ndarray) -> int:
    """Number of objects implied by a triplet set (indices are 0-based)."""
    return int(triplets.max()) + 1


def train_val_split(triplets: np.ndarray, val_frac: float = 0.1,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random split into (train, val). The paper trains on 90%, tests on 10%."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(triplets))
    n_val = int(round(len(triplets) * val_frac))
    return triplets[idx[n_val:]], triplets[idx[:n_val]]
