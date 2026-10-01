"""RANIntent v1 C1 runner CLI (EXP-2026-003 protocol step 1).

  python -m src.ranbench.interpreters.cli run --condition c21_fresh --split test \
      --interpreters Jev-1.13.0,DeepSeek-V4.1-Flash,GLM-5.3-Flash,Qwen3.8-Flash \
      --out runs/EXP-2026-003/C1/c21_fresh/test --client-location "<city, network>" [--probe-every 10]

Rerunning the same command resumes. A D-3 gap-fill block adds --only-cases <list> and a new --out. Hosted
interpreters read OPENROUTER_API_KEY through src.credentials (environment first, then the saved key file); the key is
never printed or written.
"""
from __future__ import annotations

import argparse
import json
import sys

from src.credentials import load_openrouter_key
from src.ranbench.interpreters.manifest import load_manifest
from src.ranbench.interpreters.runner import DEFAULT_SEED, SELF_HOSTED_WARMUP, run_c1

DEFAULT_CORPUS_DIR = "data/ranbench/ranintent-v1"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RANIntent v1 C1 interpreter runs")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    r.add_argument("--condition", required=True, help="e.g. c21_fresh")
    r.add_argument("--split", required=True, choices=["test", "dev"])
    r.add_argument("--interpreters", required=True, help="Comma-separated manifest names")
    r.add_argument("--out", required=True)
    r.add_argument("--manifest", default=None)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--seed", type=int, default=DEFAULT_SEED)
    r.add_argument("--spend-cap", type=float, default=20.0, help="USD, counting ledger and probe costs")
    r.add_argument("--probe-every", type=int, default=10, help="RTT probe per hosted interpreter every N cases; 0=off")
    r.add_argument("--limit", type=int, default=None, help="First N cases of the shuffled order (smoke tests)")
    r.add_argument("--client-location", default=None, help="Required for hosted runs")
    r.add_argument("--base-url", default=None, help="Self-hosted server URL override")
    r.add_argument("--warmup", type=int, default=SELF_HOSTED_WARMUP,
                   help="Self-hosted only: untimed, unrecorded decide() on the first N cases of the seeded order")
    r.add_argument("--fresh", action="store_true",
                   help="Order-dependent interpreters (AnyJev): delete their rows for this condition and restart")
    r.add_argument("--only-cases", default=None,
                   help="Gap-fill list (D-3): one case_id per line; keeps the seeded order, issues only these cases")
    r.add_argument("--allow-dirty", action="store_true")
    r.add_argument("--allow-stub", action="store_true", help="Accept a stub (dry-run) corpus")
    args = p.parse_args(argv)

    # D-3 gap-fill requests must preserve the seeded order on the wire as well
    # as in the case list.  A multi-worker executor can overlap later cases.
    if args.only_cases is not None:
        args.workers = 1

    names = [n.strip() for n in args.interpreters.split(",") if n.strip()]
    manifest = load_manifest(args.manifest)
    api_key = None
    if any(manifest[n]["deployment"] == "hosted" for n in names if n in manifest):
        api_key = load_openrouter_key()
        if not api_key:
            print("OPENROUTER_API_KEY not found (environment or key file).", file=sys.stderr)
            return 2
    report = run_c1(
        corpus_dir=args.corpus_dir, condition=args.condition, split=args.split, interpreter_names=names,
        out_dir=args.out, workers=args.workers, seed=args.seed, spend_cap_usd=args.spend_cap,
        allow_dirty=args.allow_dirty, allow_stub=args.allow_stub, manifest_path=args.manifest, api_key=api_key,
        base_url=args.base_url, probe_every=args.probe_every, limit=args.limit, client_location=args.client_location,
        warmup=args.warmup, fresh=args.fresh, only_cases=args.only_cases,
    )
    summary = {k: report[k] for k in ("run_id", "n_cases", "stop_reason", "total_spend_usd", "duplicates", "probes_written")}
    summary["rows"] = {m: v["actual_rows"] for m, v in report["per_model"].items()}
    print(json.dumps(summary, indent=2))
    return 0 if report["stop_reason"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
