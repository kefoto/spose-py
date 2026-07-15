#!/usr/bin/env python3
"""Assert the Python port matches the MATLAB / paper values within tolerance.

Prints PASS/FAIL per target with the observed value. Exits non-zero if any check
fails, so it can gate CI or a cluster smoke test.

    python scripts/verify_parity.py
    python scripts/verify_parity.py --backends   # also cross-check gpu/numba/numpy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from things_spose import analyses, backend, classify, dataio, similarity
from things_spose.similarity import squareformq

_FAILS = 0


def check(name, ok, detail=""):
    global _FAILS
    tag = "PASS" if ok else "FAIL"
    if not ok:
        _FAILS += 1
    print(f"  [{tag}] {name}{('  — ' + detail) if detail else ''}")


def approx(name, value, target, tol):
    check(f"{name} = {value:.4f} (target {target}±{tol})",
          abs(value - target) <= tol, f"|Δ|={abs(value - target):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", action="store_true",
                    help="also verify gpu/numba/numpy backends agree")
    args = ap.parse_args()

    ds = dataio.load_dataset()
    print(backend.describe())
    print("\n=== Parity checks ===")

    # 1. embedding2sim reproduces the shipped similarity matrix.
    sim = similarity.embedding2sim(ds.embedding, dtype=np.float64)
    r_ship = np.corrcoef(squareformq(sim), squareformq(ds.spose_sim))[0, 1]
    check(f"embedding2sim vs shipped spose_sim (r={r_ship:.6f})", r_ship > 0.999)

    # 2. squareformq is MATLAB column-major (round-trips and is symmetric).
    m = ds.spose_sim[:20, :20].copy()
    np.fill_diagonal(m, 0.0)
    rt = squareformq(squareformq(m))
    check("squareformq round-trip", np.allclose(rt, m))
    # column-major order: first element is (row 1, col 0)
    check("squareformq column-major order", squareformq(m)[0] == m[1, 0])

    # 3. Fig 2b similarity correlation ~ 0.87.
    spose_sim48 = similarity.embedding2sim48(ds.embedding, ds.wordposition48)
    sim48 = analyses.sim48_correlation(
        spose_sim48, ds.rdm48, ds.rdm48_split1, ds.rdm48_split2)
    approx("Fig2b r", sim48.r, 0.87, 0.02)

    # 4. Fig 2a triplet accuracy and noise ceiling (sane ranges vs paper ~64/67%).
    pred = analyses.predict_triplets(ds.dot_product, ds.triplets_test, ds.n_objects)
    ceil = analyses.noise_ceiling()
    approx("Fig2a triplet accuracy", pred.accuracy, 64.0, 2.0)
    approx("Fig2a noise ceiling", ceil.ceiling, 67.0, 2.0)
    check("accuracy below noise ceiling", pred.accuracy < ceil.ceiling)

    # 5. Classification: SPoSE beats the word-vector baseline.
    cls = classify.predict_category(ds)
    check(f"classification SPoSE ({cls.accuracy_spose:.1f}%) > "
          f"wordvec ({cls.accuracy_wordvec:.1f}%)",
          cls.accuracy_spose > cls.accuracy_wordvec)
    check(f"classification SPoSE plausible (>50%)", cls.accuracy_spose > 50.0)

    # 6. Typicality: several categories survive FDR.
    typ = analyses.typicality_correlations(ds)
    n_sig = int((typ.p_adjusted < 0.05).sum())
    check(f"Typicality FDR-significant categories ({n_sig})", n_sig >= 8)

    # 7. Backends agree (optional; the expensive one).
    if args.backends:
        ref = similarity.embedding2sim(ds.embedding[:300], backend_name="numpy")
        for bk in ("numba", "gpu"):
            resolved = backend.resolve_backend(bk)
            got = similarity.embedding2sim(ds.embedding[:300], backend_name=bk)
            check(f"backend {bk} (->{resolved}) matches numpy",
                  np.allclose(got, ref, atol=1e-3))

    print(f"\n{_FAILS} failure(s)." if _FAILS else "\nAll checks passed.")
    sys.exit(1 if _FAILS else 0)


if __name__ == "__main__":
    main()
