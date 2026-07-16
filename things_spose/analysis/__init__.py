"""Stage 3 — numbers and figures derived from a trained embedding.

``similarity`` turns an embedding into pairwise similarity; ``analyses``,
``classify``, ``clustering``, ``reproducibility``, ``dimlabels``, and ``tsne``
compute the paper's numerics; ``viz`` draws the figures. Import submodules
directly (``viz`` pulls in matplotlib, ``tsne`` scikit-learn)::

    from things_spose.analysis import analyses, viz
"""
