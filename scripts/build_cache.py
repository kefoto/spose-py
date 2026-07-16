#!/usr/bin/env python3
"""Build the rebuildable cache artifacts for the SPoSE analyses.

The only genuinely expensive precompute in the paper. Meant to run **once** on a
GPU node (fast ``embedding2sim``) or a fat CPU node (parallel Numba kernel), after
which every analysis/figure loads the artifacts instantly.

Artifacts written to ``THINGS_CACHE_DIR`` (default ``cache/``):

* ``reduced_embeddings.npy``   (49, N, 49) float32 — Fig-6 ablation embeddings.
* ``reduced_sim_pairvecs.npy`` (49, n_pairs) float32 — below-diagonal vectors of
  each reduced similarity matrix (all ``dimension_ablation`` needs). ~168 MB.
* ``tsne_layout.npy``          (N, 2) — Fig-4/5 crystal-map layout (with --tsne).
* ``spose_sim.npy``            (N, N) — similarity recomputed from the embedding,
  for a self-contained pipeline (with --recompute-sim).

Examples
--------
    python scripts/build_cache.py                 # ablation cache (all 49 dims)
    python scripts/build_cache.py --tsne          # + t-SNE layout
    THINGS_DEVICE=cuda python scripts/build_cache.py --backend gpu
    python scripts/build_cache.py --quick 3       # only first 3 dims (smoke test)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from things_spose.analysis import analyses, similarity
from things_spose.analysis.similarity import squareformq
from things_spose.core import artifacts, backend, paths
from things_spose.data import dataio


def build_ablation(ds, backend_name: str, quick: int | None, save_full: bool):
    reduced = analyses.build_reduced_embeddings(ds.embedding).astype(np.float32)
    n_reduc = reduced.shape[0] if quick is None else min(quick, reduced.shape[0])

    n_pairs = ds.n_objects * (ds.n_objects - 1) // 2
    pairvecs = np.zeros((reduced.shape[0], n_pairs), dtype=np.float32)
    full_mm = None
    if save_full:
        full_mm = np.lib.format.open_memmap(
            paths.CACHE_DIR / "reduced_sims.npy", mode="w+", dtype=np.float32,
            shape=(reduced.shape[0], ds.n_objects, ds.n_objects))

    print(f"Computing {n_reduc} reduced similarity matrices "
          f"(backend={backend.resolve_backend(backend_name)}) ...", flush=True)
    for d in range(n_reduc):
        t = time.time()
        sim = similarity.embedding2sim(
            reduced[d], backend_name=backend_name, dtype=np.float32)
        pairvecs[d] = squareformq(sim).astype(np.float32)
        if full_mm is not None:
            full_mm[d] = sim
        print(f"  dim {d + 1:2d}/{n_reduc}  {time.time() - t:5.1f}s", flush=True)

    np.save(paths.CACHE_DIR / artifacts.REDUCED_EMB_FILE, reduced)
    np.save(paths.CACHE_DIR / artifacts.REDUCED_PAIRVEC_FILE, pairvecs)
    if full_mm is not None:
        full_mm.flush()
    print(f"Saved reduced embeddings + pairvecs to {paths.CACHE_DIR}", flush=True)


def build_tsne(ds):
    from things_spose.analysis import tsne

    print("Computing t-SNE crystal layout (MDS init -> multiscale t-SNE) ...",
          flush=True)
    t = time.time()
    layout = tsne.crystal_layout(ds.dissim)
    np.save(paths.CACHE_DIR / artifacts.TSNE_LAYOUT_FILE, layout)
    print(f"Saved t-SNE layout ({time.time() - t:.1f}s) to "
          f"{paths.CACHE_DIR / artifacts.TSNE_LAYOUT_FILE}", flush=True)


def recompute_sim(ds, backend_name: str):
    print("Recomputing full spose_sim from the embedding ...", flush=True)
    t = time.time()
    sim = similarity.embedding2sim(ds.embedding, backend_name=backend_name)
    np.save(paths.CACHE_DIR / artifacts.SPOSE_SIM_FILE, sim)
    r = np.corrcoef(squareformq(sim), squareformq(ds.spose_sim))[0, 1]
    print(f"Saved spose_sim ({time.time() - t:.1f}s); corr vs shipped = {r:.6f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "gpu", "numba", "numpy"],
                    help="similarity backend (default: auto)")
    ap.add_argument("--tsne", action="store_true", help="also build the t-SNE layout")
    ap.add_argument("--recompute-sim", action="store_true",
                    help="also recompute full spose_sim from the embedding")
    ap.add_argument("--full", action="store_true",
                    help="also store the full (49, N, N) reduced_sims memmap")
    ap.add_argument("--quick", type=int, default=None,
                    help="only build the first K reduced dims (smoke test)")
    ap.add_argument("--skip-ablation", action="store_true",
                    help="skip the ablation cache (e.g. only build --tsne)")
    args = ap.parse_args()

    backend.configure_threads()
    print(backend.describe(), flush=True)
    paths.ensure_cache_dir()
    ds = dataio.load_dataset()

    if not args.skip_ablation:
        build_ablation(ds, args.backend, args.quick, args.full)
    if args.recompute_sim:
        recompute_sim(ds, args.backend)
    if args.tsne:
        build_tsne(ds)
    print("Cache build complete.", flush=True)


if __name__ == "__main__":
    main()
