#!/usr/bin/env python3
"""Execute the figure notebooks headlessly with papermill.

Runs each ``notebooks/*.ipynb`` end-to-end (no UI) and writes the executed copy
to an output directory — how the cluster produces rendered notebooks in a batch
job. Requires ``papermill`` (in requirements.txt).

    python scripts/run_notebooks.py --out executed/
    python scripts/run_notebooks.py --only fig2 fig6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_NB_DIR = _ROOT / "notebooks"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="notebooks/executed", help="output directory")
    ap.add_argument("--only", nargs="*", default=None,
                    help="stems to run (default: all notebooks, sorted)")
    args = ap.parse_args()

    try:
        import papermill as pm
    except ImportError:
        sys.exit("papermill not installed. `pip install papermill` "
                 "(it is in requirements.txt).")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    notebooks = sorted(_NB_DIR.glob("*.ipynb"))
    if args.only:
        wanted = set(args.only)
        notebooks = [nb for nb in notebooks if nb.stem in wanted]
    if not notebooks:
        sys.exit(f"No notebooks found in {_NB_DIR}.")

    failures = []
    for nb in notebooks:
        dst = out / nb.name
        print(f"Executing {nb.name} ...", flush=True)
        try:
            pm.execute_notebook(str(nb), str(dst), kernel_name="python3")
            print(f"  -> {dst}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((nb.name, str(exc)))
            print(f"  FAILED: {exc}")

    if failures:
        print(f"\n{len(failures)} notebook(s) failed:")
        for name, err in failures:
            print(f"  {name}: {err.splitlines()[-1] if err else ''}")
        sys.exit(1)
    print("\nAll notebooks executed successfully.")


if __name__ == "__main__":
    main()
