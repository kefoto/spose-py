"""Saving and restoring training state.

Two different things get written, for two different purposes:

* **The embedding** (:func:`save_embedding`) — the *result*. A plain
  ``M x D`` text file matching the shipped ``spose_embedding_49d_sorted.txt``
  format, which every downstream analysis reads.
* **A checkpoint** (:func:`save`) — the *run*. Model and optimizer state plus
  enough bookkeeping to resume mid-training, as a ``.tar`` (PyTorch convention).

Checkpoints are keyed by epoch (``model_epoch0042.tar``) so :func:`latest` can
find the newest one to resume from.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

_CKPT_RE = re.compile(r"model_epoch(\d+)\.tar$")


def save_embedding(embedding: np.ndarray, path: str | Path) -> None:
    """Write the embedding as a space-delimited txt (like the shipped file)."""
    np.savetxt(path, embedding, fmt="%.8f")


def save(path: str | Path, epoch: int, model: torch.nn.Module,
         optimizer: torch.optim.Optimizer, history: list[dict],
         **extra: Any) -> Path:
    """Write a resumable checkpoint for ``epoch``; returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "history": history,
        **extra,
    }, path)
    return path


def load(path: str | Path, model: torch.nn.Module,
         optimizer: torch.optim.Optimizer | None = None,
         device: str | torch.device = "cpu") -> dict:
    """Restore ``model`` (and optionally ``optimizer``) from a checkpoint.

    Returns the raw checkpoint dict, so callers can pick up ``epoch`` and
    ``history`` to resume from where the run left off.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optim_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state_dict"])
    return ckpt


def epoch_path(directory: str | Path, epoch: int) -> Path:
    """Canonical checkpoint path for ``epoch`` inside ``directory``."""
    return Path(directory) / f"model_epoch{epoch:04d}.tar"


def latest(directory: str | Path) -> Path | None:
    """Newest checkpoint in ``directory`` by epoch number, or None if there are none."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    found = [(int(m.group(1)), p) for p in directory.glob("model_epoch*.tar")
             if (m := _CKPT_RE.search(p.name))]
    return max(found)[1] if found else None
