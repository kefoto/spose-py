"""Python reimplementation of the SPoSE analyses from Hebart et al. (2020),
*Nature Human Behaviour*.

The package is organised by pipeline stage::

    core/       shared infra — paths, compute backend, cache, stats
    data/       stage 1 — load / assemble / sample the raw data
    training/   stage 2 — model, training loop, run logging, checkpoints
    analysis/   stage 3 — numerics and figures from a trained embedding

Import submodules directly; the public API is intentionally small::

    from things_spose.core import paths, backend
    from things_spose.data import dataio, sampling
    from things_spose.training import train, checkpoint
    from things_spose.analysis import analyses, viz

    ds = dataio.load_dataset()

``training`` pulls in PyTorch and ``analysis`` pulls in matplotlib/scikit-learn,
so the four subpackages are exposed lazily here — a bare ``import things_spose``
stays light.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = ["core", "data", "training", "analysis"]

# Lazily expose the stage subpackages (PEP 562) so that ``things_spose.training``
# works without importing torch at package-import time.
_LAZY_SUBMODULES = set(__all__)


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module  # cache so subsequent access is a plain lookup
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_SUBMODULES))


if TYPE_CHECKING:  # help static analysers see the lazy subpackages
    from . import analysis, core, data, training
