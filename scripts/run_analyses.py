#!/usr/bin/env python3
"""Run every numeric analysis and emit a JSON + printed summary.

This is the headless spine of ``make_figures_behavsim.m`` — no plotting. It is
the script a Slurm job runs. Figure 6 (dimension ablation) is included only when
the cache built by ``scripts/build_cache.py`` is present; pass ``--skip-ablation``
to omit it, or build the cache first.

    python scripts/run_analyses.py --out results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from things_spose.analysis import (analyses, classify, reproducibility,
                                   similarity)
from things_spose.core import artifacts, backend
from things_spose.data import dataio


def _round(obj):
    """Recursively make numpy values JSON-serialisable and lightly rounded."""
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 4)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _round(obj.tolist())
    return obj


def run(skip_ablation: bool, backend_name: str) -> dict:
    ds = dataio.load_dataset()
    results: dict = {"environment": backend.describe()}

    # Fig 2a — triplet prediction + noise ceiling
    pred = analyses.predict_triplets(ds.dot_product, ds.triplets_test, ds.n_objects)
    ceil = analyses.noise_ceiling()
    pct, pct_se = analyses.percent_performance(pred.per_object_acc, ceil.consistency)
    results["fig2a_prediction"] = {
        "accuracy": pred.accuracy, "accuracy_ci95": pred.accuracy_ci95,
        "noise_ceiling": ceil.ceiling, "noise_ceiling_ci95": ceil.ci95,
        "percent_of_ceiling": pct, "percent_of_ceiling_se": pct_se,
    }

    # Fig 2b — 48-object similarity correlation
    spose_sim48 = similarity.embedding2sim48(ds.embedding, ds.wordposition48)
    sim48 = analyses.sim48_correlation(
        spose_sim48, ds.rdm48, ds.rdm48_split1, ds.rdm48_split2)
    results["fig2b_similarity48"] = {
        "r": sim48.r, "r_ci95": list(sim48.r_ci95),
        "reliability": sim48.reliability, "splithalf": sim48.splithalf,
        "variance_explained": sim48.variance_explained,
    }

    # Classification (SPoSE vs word vectors)
    cls = classify.predict_category(ds)
    results["classification"] = {
        "accuracy_spose": cls.accuracy_spose,
        "accuracy_wordvec": cls.accuracy_wordvec,
        "n_categories": len(cls.categories), "n_objects": int(cls.labels.size),
    }

    # Fig 7 — typicality
    typ = analyses.typicality_correlations(ds)
    results["fig7_typicality"] = {
        "rho": typ.rho, "p_adjusted": typ.p_adjusted,
        "n_significant_fdr": int((typ.p_adjusted < 0.05).sum()),
    }

    # Fig 8 — human dimension ratings -> similarity. Substitutes raw human
    # ratings for the paper's 49 named dimensions into the embedding, so it
    # only makes sense for a 49-dim embedding aligned to those dimensions.
    if ds.n_dims == 49:
        human = analyses.human_rating_similarity(ds, backend_name=backend_name)
        results["fig8_human_ratings"] = {
            "r": human.r, "r_ci95": list(human.r_ci95),
            "p_randomization": human.p_randomization,
        }
    else:
        results["fig8_human_ratings"] = (
            f"skipped (embedding has {ds.n_dims} dims, needs 49 to align with "
            "the paper's human dimension ratings)"
        )

    # Ext Data Fig 1 — reproducibility against the 20 reference model fits,
    # which are themselves 49-dim; only meaningful for a 49-dim embedding.
    if ds.n_dims == 49:
        sortind0 = dataio.load_sortind()
        repro = reproducibility.dimension_reproducibility(ds.embedding, sortind0)
        results["extfig1_reproducibility"] = {
            "mean_reproducibility_first": repro.mean[:5], "rank_corr": repro.rank_corr,
            "rank_p": repro.rank_p, "rank_ci95": list(repro.rank_ci95),
        }
    else:
        results["extfig1_reproducibility"] = (
            f"skipped (embedding has {ds.n_dims} dims, reference models are 49-dim)"
        )

    # Fig 6 — dimension ablation (needs cache; the cache is built from the
    # shipped 49-dim embedding, so it's only valid to use when ds also has 49
    # dims -- otherwise it's silently comparing a different embedding).
    if not skip_ablation and ds.n_dims == 49 and artifacts.exists(artifacts.REDUCED_PAIRVEC_FILE):
        cache = artifacts.load_ablation_cache()
        abl = analyses.dimension_ablation(
            cache.reduced_embeddings, None, ds.spose_sim, ds.triplets_test,
            pred.accuracy, reduced_sim_pairvecs=cache.reduced_sim_pairvecs)
        results["fig6_ablation"] = {
            "mindim_acc": abl.mindim_acc, "maxdim_acc": abl.maxdim_acc,
            "mindim_var": abl.mindim_var, "maxdim_var": abl.maxdim_var,
        }
    elif ds.n_dims != 49:
        results["fig6_ablation"] = (
            f"skipped (embedding has {ds.n_dims} dims, cache is 49-dim)"
        )
    else:
        results["fig6_ablation"] = "skipped (no cache; run scripts/build_cache.py)"

    return results


def print_table(results: dict):
    print("\n===== SPoSE analysis summary =====")
    f2a = results["fig2a_prediction"]
    print(f"Triplet accuracy      : {f2a['accuracy']:.2f}%  (CI {f2a['accuracy_ci95']:.2f})")
    print(f"Noise ceiling         : {f2a['noise_ceiling']:.2f}%  (CI {f2a['noise_ceiling_ci95']:.2f})")
    print(f"Percent of ceiling    : {f2a['percent_of_ceiling']:.1f}%  (SE {f2a['percent_of_ceiling_se']:.1f})")
    print(f"48-object sim r        : {results['fig2b_similarity48']['r']:.3f}")
    c = results["classification"]
    print(f"Classification (SPoSE) : {c['accuracy_spose']:.1f}%   (word-vec {c['accuracy_wordvec']:.1f}%)")
    print(f"Typicality FDR-signif  : {results['fig7_typicality']['n_significant_fdr']} categories")
    f8 = results["fig8_human_ratings"]
    if isinstance(f8, dict):
        print(f"Human-rating sim r     : {f8['r']:.3f}  (p_rand {f8['p_randomization']:.4f})")
    else:
        print(f"Human-rating sim r     : {f8}")
    repro = results["extfig1_reproducibility"]
    if isinstance(repro, dict):
        print(f"Reproducibility rank r : {repro['rank_corr']:.3f}")
    else:
        print(f"Reproducibility rank r : {repro}")
    print(f"Fig6 ablation          : {results['fig6_ablation']}")
    print("==================================\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="write JSON summary to this path")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "gpu", "numba", "numpy"])
    args = ap.parse_args()

    backend.configure_threads()
    t = time.time()
    results = run(args.skip_ablation, args.backend)
    results["runtime_seconds"] = round(time.time() - t, 1)

    rounded = _round(results)
    print_table(rounded)
    if args.out:
        Path(args.out).write_text(json.dumps(rounded, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
