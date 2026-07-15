"""Python reimplementation of the SPoSE analyses from Hebart et al. (2020),
*Nature Human Behaviour*.

Public API is intentionally small; import submodules directly for details::

    from things_spose import dataio, similarity, analyses
    ds = dataio.load_dataset()

To *fit* a fresh embedding from odd-one-out triplets (rather than load the
shipped one), use the ``train`` submodule::

    from things_spose import train
    res = train.train_spose(train_triplets, n_objects=1854)

``train`` pulls in PyTorch, so it is loaded lazily on first access to keep a
bare ``import things_spose`` lightweight.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from . import backend, dataio, paths

__all__ = ["backend", "dataio", "paths", "train"]

# Optional submodules (imported lazily by callers to keep import light — e.g.
# ``viz`` pulls in matplotlib, ``tsne`` pulls in scikit-learn, the GPU
# similarity backend and ``train`` pull in torch, so none are imported at top
# level):
#   similarity, analyses, reproducibility, classify, clustering
#   dimlabels, tsne, viz, artifacts, train

# Lazily expose heavy submodules named in ``__all__`` (PEP 562) so that
# ``things_spose.train`` / ``from things_spose import train`` work without
# importing torch at package-import time.
_LAZY_SUBMODULES = {"train"}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module  # cache so subsequent access is a plain lookup
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_SUBMODULES))


if TYPE_CHECKING:  # help static analysers see the lazy submodule
    from . import train
