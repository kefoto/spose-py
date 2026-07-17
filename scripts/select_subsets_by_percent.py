#!/usr/bin/env python3
"""Generate category-stratified THINGS subsets at increasing size (5%, 10%,
..., 100% of the 1854 objects) for later use as object filters.

Each step selects `n_total = round(1854 * pct / 100)` objects via
`things_spose.analysis.subset_selection.select_subset` (MDS + K-means per
category, snapped to nearest real object) and writes the selected object
indices/names as `data/subsets/subset_<pct>pct.csv`, plus a
`subsets_summary.csv` with per-step diagnostics (columns: `mean/median/
min_dimension_coverage`, `rdm_fidelity_spearman`, `weighted_centroid_shift_49d`,
`image_diversity_ratio`, `image_min/median_nn_dist`, `*_image_pca_coverage`,
`triplet_accuracy`). Percent is only the sweep variable -- see the module
docstring of `things_spose.analysis.subset_selection` for what each metric
means and how it connects to representativeness, and which two are the actual
sizing criterion vs. reference-only.

The script also prints the smallest generated percentage that clears
`--coverage_threshold` and `--rdm_threshold` simultaneously (default 0.95
each) -- treat that as the recommendation, not a fixed percentage.

Examples
--------
    python scripts/select_subsets_by_percent.py
    python scripts/select_subsets_by_percent.py --step 10 --max_percent 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from things_spose.analysis.subset_selection import (
    build_category_labels,
    category_centroid_shift,
    dimension_coverage,
    evaluate_on_subset,
    image_diversity,
    rdm_fidelity,
    rsa_check,
    select_subset,
    weighted_mean_centroid_shift,
)
from things_spose.core import paths
from things_spose.data import dataio

DEFAULT_OUT_DIR = paths._REPO_ROOT / "data" / "subsets"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR,
                     help="where to write one subset_<pct>pct.csv per percentage step")
    ap.add_argument("--step", type=int, default=5, help="percentage increment (default 5)")
    ap.add_argument("--max_percent", type=int, default=100, help="largest percentage to generate")
    ap.add_argument("--min_per_category", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="regenerate even if a subset CSV already exists")
    ap.add_argument("--coverage_threshold", type=float, default=0.95,
                     help="min mean_dimension_coverage required to recommend a size")
    ap.add_argument("--rdm_threshold", type=float, default=0.95,
                     help="min rdm_fidelity_spearman required to recommend a size")
    args = ap.parse_args()

    ds = dataio.load_dataset()
    X = ds.embedding
    categories = build_category_labels(ds.category_mat_manual)
    unique_id = ds.unique_id
    triplets = ds.triplets_test

    print("Fitting image PCA (once, reused across all subset sizes) ...")
    from sklearn.decomposition import PCA
    flat_images = np.stack([im.astype(np.float32).ravel() for im in ds.images])
    image_pca_coords = PCA(n_components=20, random_state=0).fit_transform(flat_images)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    percents = list(range(args.step, args.max_percent + 1, args.step))
    summary_rows = []

    for pct in percents:
        out_csv = args.out_dir / f"subset_{pct:03d}pct.csv"
        n_total = round(ds.n_objects * pct / 100)
        print(f"\n=== {pct}% ({n_total}/{ds.n_objects} objects) ===")

        if out_csv.exists() and not args.force:
            print(f"  [cached] {out_csv}")
            subset_df = pd.read_csv(out_csv)
            selected = subset_df["embedding_row"].values
        else:
            selected = select_subset(X, categories, n_total=n_total,
                                      min_per_category=args.min_per_category)
            subset_df = pd.DataFrame({
                "embedding_row": selected,
                "unique_id": [unique_id[i] for i in selected],
                "category": categories[selected],
            })
            subset_df.to_csv(out_csv, index=False)
            print(f"  wrote {out_csv}")

        print(f"  selected {len(selected)} objects")

        rho = rsa_check(X, selected)
        rdm_rho = rdm_fidelity(ds.dissim, X, selected)
        result = evaluate_on_subset(X, triplets, selected)

        cov = dimension_coverage(X, selected)["coverage_ratio"]
        img_cov = dimension_coverage(image_pca_coords, selected)["coverage_ratio"]
        div = image_diversity(image_pca_coords, selected)

        centroid_df = category_centroid_shift(X, categories, selected)
        centroid_shift = weighted_mean_centroid_shift(centroid_df)

        print(f"  dimension coverage (mean/median/min): {cov.mean():.4f} / {cov.median():.4f} / {cov.min():.4f}")
        print(f"  RDM fidelity vs. shipped model (Spearman): {rdm_rho:.4f}")
        print(f"  weighted category centroid shift (49D): {centroid_shift:.4f}")
        print(f"  image diversity ratio: {div['diversity_ratio']:.4f}, "
              f"min/median NN dist: {div['min_nn_dist']:.4f} / {div['median_nn_dist']:.4f}")
        print(f"  image PCA coverage (mean/min): {img_cov.mean():.4f} / {img_cov.min():.4f}")
        print(f"  [reference only] RSA indexing sanity check: {rho:.4f}; "
              f"triplets fully inside subset: {result['n_triplets']}, accuracy: {result['accuracy']}")

        summary_rows.append({
            "percent": pct,
            "n_selected": len(selected),
            "mean_dimension_coverage": cov.mean(),
            "median_dimension_coverage": cov.median(),
            "min_dimension_coverage": cov.min(),
            "rdm_fidelity_spearman": rdm_rho,
            "weighted_centroid_shift_49d": centroid_shift,
            "image_diversity_ratio": div["diversity_ratio"],
            "image_min_nn_dist": div["min_nn_dist"],
            "image_median_nn_dist": div["median_nn_dist"],
            "mean_image_pca_coverage": img_cov.mean(),
            "min_image_pca_coverage": img_cov.min(),
            "rsa_sanity_check": rho,
            "n_triplets_in_subset": result["n_triplets"],
            "triplet_accuracy": result["accuracy"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = args.out_dir / "subsets_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nWrote {summary_csv}")
    print(f"\nGenerated {len(percents)} subsets ({percents[0]}%-{percents[-1]}% "
          f"in steps of {args.step}) in {args.out_dir}")

    meets = summary_df[
        (summary_df["mean_dimension_coverage"] >= args.coverage_threshold)
        & (summary_df["rdm_fidelity_spearman"] >= args.rdm_threshold)
    ]
    print(f"\nSizing recommendation (mean_dimension_coverage >= {args.coverage_threshold} "
          f"and rdm_fidelity_spearman >= {args.rdm_threshold}; percent is only the sweep "
          f"variable, these thresholds are the actual criterion):")
    if len(meets):
        rec = meets.sort_values("percent").iloc[0]
        print(f"  -> {int(rec['percent'])}% ({int(rec['n_selected'])} objects): "
              f"mean_coverage={rec['mean_dimension_coverage']:.4f}, "
              f"rdm_fidelity={rec['rdm_fidelity_spearman']:.4f}, "
              f"image_diversity_ratio={rec['image_diversity_ratio']:.4f}")
    else:
        print("  no generated size clears both thresholds; lower --coverage_threshold/"
              "--rdm_threshold or extend --max_percent.")


if __name__ == "__main__":
    main()
