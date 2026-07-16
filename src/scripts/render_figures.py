#!/usr/bin/env python3
"""Render every figure to SVG/PNG headlessly (Agg backend).

    python scripts/render_figures.py --out figs/ --ext svg

Figures that need the cache (Fig 4/5 t-SNE layout, Fig 6 ablation) are rendered
only when the artifacts from ``scripts/build_cache.py`` are present; otherwise
they are skipped with a message.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — must precede pyplot import inside viz
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from things_spose.analysis import viz
from things_spose.core import artifacts
from things_spose.data import dataio


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figs", help="output directory")
    ap.add_argument("--ext", default="svg", choices=["svg", "png", "pdf"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = dataio.load_dataset()

    def save(name, fig):
        p = out / f"{name}.{args.ext}"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p}")

    print("Rendering figures ...")
    save("fig1_similarity_overview", viz.fig1_similarity_overview(ds))
    save("fig2_prediction", viz.fig2_prediction(ds))
    save("fig3_dimension_examples", viz.fig3_dimension_examples(ds))
    save("fig7_typicality", viz.fig7_typicality(ds))
    save("fig8_human_ratings", viz.fig8_human_ratings(ds))
    save("extfig2_all_dimension_clouds", viz.extfig2_all_dimension_clouds(ds))

    sortind0 = dataio.load_sortind()
    save("extfig1_reproducibility", viz.extfig1_reproducibility(ds, sortind0))

    # Cache-dependent figures.
    if artifacts.exists(artifacts.TSNE_LAYOUT_FILE):
        layout = artifacts.load_tsne_layout()
        save("fig4_crystal_map", viz.fig4_crystal_map(ds, layout))
        for obj in viz._FIG5_OBJECTS[:4]:
            save(f"fig5_crystal_{ds.words[obj]}", viz.fig5_crystal_closeup(ds, int(obj)))
    else:
        print("  [skip] fig4/fig5 — no t-SNE layout (run build_cache.py --tsne)")

    if artifacts.exists(artifacts.REDUCED_PAIRVEC_FILE):
        from things_spose.analysis import analyses
        cache = artifacts.load_ablation_cache()
        pred = analyses.predict_triplets(ds.dot_product, ds.triplets_test, ds.n_objects)
        abl = analyses.dimension_ablation(
            cache.reduced_embeddings, None, ds.spose_sim, ds.triplets_test,
            pred.accuracy, reduced_sim_pairvecs=cache.reduced_sim_pairvecs)
        save("fig6_ablation", viz.fig6_ablation(abl))
    else:
        print("  [skip] fig6 — no ablation cache (run build_cache.py)")

    print("Done.")


if __name__ == "__main__":
    main()
