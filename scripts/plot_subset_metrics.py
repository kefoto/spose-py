#!/usr/bin/env python3
"""Plot subset size (%) vs. the diagnostics in `data/subsets/subsets_summary.csv`
(produced by `scripts/select_subsets_by_percent.py`).

Three figures:
  subset_accuracy.png       -- percent vs. triplet_accuracy alone (flat, and
                               thus not a useful downsizing signal on its own).
  subset_metrics_v1_previous.png -- the original 3-panel version: mean/min
                               dimension coverage, weighted category centroid
                               shift, and image-PCA coverage. Uses
                               `rsa_sanity_check` in place of true RDM fidelity
                               (that check is ~1.0 by construction and cannot
                               detect representativeness loss -- kept only for
                               comparison against v2).
  subset_metrics_v2_current.png -- current version: mean/median/min dimension
                               coverage, true RDM fidelity vs. the shipped
                               model, centroid shift, and image diversity
                               ratio, with the recommended size (smallest
                               percent clearing both thresholds) marked.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SUMMARY = REPO_ROOT / "data" / "subsets" / "subsets_summary.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "figs"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--ext", default="png", choices=["png", "svg", "pdf"])
    ap.add_argument("--coverage_threshold", type=float, default=0.95)
    ap.add_argument("--rdm_threshold", type=float, default=0.95)
    args = ap.parse_args()

    if not args.summary.exists():
        raise FileNotFoundError(
            f"{args.summary} not found -- run scripts/select_subsets_by_percent.py first."
        )
    df = pd.read_csv(args.summary).sort_values("percent")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meets = df[(df["mean_dimension_coverage"] >= args.coverage_threshold)
               & (df["rdm_fidelity_spearman"] >= args.rdm_threshold)]
    rec_pct = meets["percent"].min() if len(meets) else None

    # --- Figure 1: percent vs. triplet accuracy -----------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    ax.plot(df["percent"], df["triplet_accuracy"], marker="o", color="#0072B2")
    ax.axhline(1 / 3, color="gray", linestyle="--", linewidth=1, label="chance (1/3)")
    ax.set_xlabel("Subset size (%)")
    ax.set_ylabel("Held-in triplet accuracy")
    ax.set_title("Subset size vs. triplet odd-one-out accuracy\n"
                  "(flat across sizes -- not informative for downsizing)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    out1 = args.out_dir / f"subset_accuracy.{args.ext}"
    fig.savefig(out1)
    plt.close(fig)
    print(f"Wrote {out1}")

    # --- Figure 2 (v1): the previous version, kept for comparison -----------
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 11), dpi=150, sharex=True)

    ax = axes[0]
    ax.plot(df["percent"], df["mean_dimension_coverage"], marker="o",
            color="#009E73", label="mean")
    ax.plot(df["percent"], df["min_dimension_coverage"], marker="o",
            color="#009E73", linestyle="--", alpha=0.6, label="min")
    ax.set_ylabel("SPoSE dimension\ncoverage ratio")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Subset size vs. embedding + image representativeness (v1, previous)")

    ax = axes[1]
    ax.plot(df["percent"], df["weighted_centroid_shift_49d"], marker="o", color="#D55E00")
    ax.set_ylabel("Weighted category\ncentroid shift (49D)")

    ax = axes[2]
    ax.plot(df["percent"], df["mean_image_pca_coverage"], marker="o",
            color="#CC79A7", label="mean")
    ax.plot(df["percent"], df["min_image_pca_coverage"], marker="o",
            color="#CC79A7", linestyle="--", alpha=0.6, label="min")
    ax.set_xlabel("Subset size (%)")
    ax.set_ylabel("Image PCA\ncoverage ratio")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    out_v1 = args.out_dir / f"subset_metrics_v1_previous.{args.ext}"
    fig.savefig(out_v1)
    plt.close(fig)
    print(f"Wrote {out_v1}")

    # --- Figure 3 (v2): current version --------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(6.5, 14), dpi=150, sharex=True)

    def mark_rec(ax):
        if rec_pct is not None:
            ax.axvline(rec_pct, color="black", linestyle=":", linewidth=1.2,
                       label=f"recommended: {int(rec_pct)}%")

    ax = axes[0]
    ax.plot(df["percent"], df["mean_dimension_coverage"], marker="o",
            color="#009E73", label="mean")
    ax.plot(df["percent"], df["median_dimension_coverage"], marker="o",
            color="#009E73", linestyle="-.", alpha=0.7, label="median")
    ax.plot(df["percent"], df["min_dimension_coverage"], marker="o",
            color="#009E73", linestyle="--", alpha=0.6, label="min")
    ax.axhline(args.coverage_threshold, color="gray", linestyle="--", linewidth=1)
    mark_rec(ax)
    ax.set_ylabel("SPoSE dimension\ncoverage ratio")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title("Subset size vs. embedding + image representativeness (v2, current)\n"
                  "(sizing criterion: coverage + RDM fidelity thresholds, not percent itself)")

    ax = axes[1]
    ax.plot(df["percent"], df["rdm_fidelity_spearman"], marker="o", color="#0072B2",
            label="RDM fidelity vs. shipped model")
    ax.axhline(args.rdm_threshold, color="gray", linestyle="--", linewidth=1)
    mark_rec(ax)
    ax.set_ylabel("RDM fidelity\n(Spearman rho)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(df["percent"], df["weighted_centroid_shift_49d"], marker="o", color="#D55E00")
    mark_rec(ax)
    ax.set_ylabel("Weighted category\ncentroid shift (49D)")

    ax = axes[3]
    ax.plot(df["percent"], df["image_diversity_ratio"], marker="o",
            color="#CC79A7", label="diversity ratio (subset/full)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    mark_rec(ax)
    ax.set_xlabel("Subset size (%)")
    ax.set_ylabel("Image diversity ratio\n(pixel-PCA space)")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out_v2 = args.out_dir / f"subset_metrics_v2_current.{args.ext}"
    fig.savefig(out_v2)
    plt.close(fig)
    print(f"Wrote {out_v2}")


if __name__ == "__main__":
    main()
