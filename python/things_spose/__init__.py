"""Python reimplementation of the SPoSE analyses from Hebart et al. (2020),
*Nature Human Behaviour*.

Public API is intentionally small; import submodules directly for details::

    from things_spose import dataio, similarity, analyses
    ds = dataio.load_dataset()
"""
from __future__ import annotations

from . import backend, dataio, paths

__all__ = ["backend", "dataio", "paths"]

# Optional submodules (imported lazily by callers to keep import light — e.g.
# ``viz`` pulls in matplotlib, ``tsne`` pulls in scikit-learn, the GPU
# similarity backend pulls in torch, so none are imported at top level):
#   similarity, analyses, reproducibility, classify, clustering
#   dimlabels, tsne, viz, artifacts
