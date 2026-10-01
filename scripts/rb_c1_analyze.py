#!/usr/bin/env python3
"""CLI script for C1 analysis: RQ4 / H3, near-RT feasibility, C1 latency, quality and cost.

Usage:
    python scripts/rb_c1_analyze.py --runs <runs_dir> ... --corpus <corpus_dir> --out <out_dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.ranbench.c1_analysis import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    run_c1_analysis,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXP-2026-003 C1 pre-registered analysis pipeline (RQ4 / H3 / near-RT / cost)"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more directories containing C1 run ledgers, probes, and traces",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to the RANIntent v1 corpus directory (e.g. data/ranbench/ranintent-v1)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for generated CSV tables and results.md",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for bootstrap resampling (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=DEFAULT_RESAMPLES,
        help=f"Number of bootstrap resamples (default: {DEFAULT_RESAMPLES})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Running C1 Analysis on {len(args.runs)} run directory path(s)...")
    print(f"Corpus: {args.corpus}")
    print(f"Output: {args.out}")
    pre_registered = args.seed == DEFAULT_SEED and args.n_resamples == DEFAULT_RESAMPLES
    print(
        f"Bootstrap Seed: {args.seed} ({args.n_resamples} resamples)"
        + ("" if pre_registered else " -- SENSITIVITY (differs from the pre-registered values)")
    )

    try:
        res = run_c1_analysis(
            run_dirs=args.runs,
            corpus_dir=args.corpus,
            out_dir=args.out,
            seed=args.seed,
            n_resamples=args.n_resamples,
        )
    except Exception as exc:
        print(f"[ERROR] C1 analysis failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    print("\nAnalysis completed successfully!")
    print(f"Output files generated in {args.out}:")
    for f in res["files_written"]:
        print(f"  - {f}")

    print("\n--- H3 Confirmatory Hypothesis Excerpt ---")
    for r in res["h3_rows"]:
        if r["status"] in ("tested", "reference"):
            print(
                f"  {r['model']:20}: acc(c3)={r['acc_c3_fresh']:.4f}  "
                f"acc(c57)={r['acc_c57_fresh']:.4f}  delta={r['delta_c3_minus_c57']:+.4f}  "
                f"95%CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
                f"p_holm={r['p_holm']:.4e}  resolved={r['resolved']}  [{r['status']}]"
            )
        else:
            print(f"  {r['model']:20}: {r['status']}")

    print("\n--- Near-RT Feasibility (phi) Excerpt (c3_fresh / c57_fresh) ---")
    for r in res["near_rt_rows"]:
        if r["condition"] in ("c3_fresh", "c57_fresh") and r["n"] > 0:
            print(
                f"  {r['model']:20} ({r['condition']:10}): phi={r['phi']:.4f}  "
                f"95%CI=[{r['phi_ci_low']:.4f}, {r['phi_ci_high']:.4f}] (n={r['n']})"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
