#!/usr/bin/env python3
"""D-3 gap-fill lists for Qwen3.8-Flash (EXP-2026-003 deviations.md, D-3).

Reads every Qwen3.8-Flash attempt in its block directories under --runs-root (c1_hosted, c1_hosted_qwen_block2,
c1_hosted_qwen_block3 and every c1_hosted_qwen_gapfill*; ledgers found recursively) and writes, per condition,
<out>/<condition>.txt: the test case_ids with no transport-error-free attempt (HTTP 200) yet, one per line.
Schema-invalid answers are outcomes, so their cases are done. Conditions with nothing left get no file.

    python scripts/rb_c1_gapfill.py --runs-root runs/EXP-2026-003 --out runs/EXP-2026-003/gapfill_lists/round1
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.ranbench.c1_analysis import (  # noqa: E402
    EXPECTED_CONDITIONS,
    _read_jsonl_strict,
    canonical_model_name,
    is_transport_ok,
    load_corpus_cases,
)

MODEL = "Qwen3.8-Flash"
FIXED_BLOCKS = ("c1_hosted", "c1_hosted_qwen_block2", "c1_hosted_qwen_block3")
GAPFILL_GLOB = "c1_hosted_qwen_gapfill*"


def block_dirs(runs_root: Path) -> list[Path]:
    missing = [b for b in FIXED_BLOCKS if not (runs_root / b).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing block directories under {runs_root}: {', '.join(missing)}")
    return [runs_root / b for b in FIXED_BLOCKS] + sorted(p for p in runs_root.glob(GAPFILL_GLOB) if p.is_dir())


def gap_lists(runs_root: Path, corpus_dir: Path) -> tuple[dict[str, list[str]], dict[str, dict[str, int]], list[Path]]:
    """(condition -> sorted case ids still without an HTTP 200 attempt, condition -> counts, ledgers read)."""
    cases = load_corpus_cases(corpus_dir)
    absent = [c for c in EXPECTED_CONDITIONS if not cases.get(c)]
    if absent:
        raise ValueError(f"Corpus has no test cases for {', '.join(absent)}")
    dirs = block_dirs(runs_root)
    ledgers = sorted(f for d in dirs for f in d.rglob("ledger.jsonl"))
    fixed_ledgers = sorted(f for d in dirs[:len(FIXED_BLOCKS)] for f in d.rglob("ledger.jsonl"))
    attempted: dict[str, set[str]] = {c: set() for c in EXPECTED_CONDITIONS}
    initially_attempted: dict[str, set[str]] = {c: set() for c in EXPECTED_CONDITIONS}
    done: dict[str, set[str]] = {c: set() for c in EXPECTED_CONDITIONS}
    for lf in ledgers:
        for r in _read_jsonl_strict(lf):
            cond, cid = str(r.get("condition", "")), str(r.get("case_id"))
            if r.get("rq") != "C1" or canonical_model_name(str(r.get("model", ""))) != MODEL:
                continue
            if cond not in cases or cid not in cases[cond]:  # not a test case of this condition
                continue
            attempted[cond].add(cid)
            if is_transport_ok(r):
                done[cond].add(cid)
    for lf in fixed_ledgers:
        for r in _read_jsonl_strict(lf):
            cond, cid = str(r.get("condition", "")), str(r.get("case_id"))
            if (r.get("rq") == "C1" and canonical_model_name(str(r.get("model", ""))) == MODEL
                    and cond in cases and cid in cases[cond]):
                initially_attempted[cond].add(cid)
    incomplete = {
        cond: sorted(set(cases[cond]) - initially_attempted[cond])
        for cond in EXPECTED_CONDITIONS
        if set(cases[cond]) - initially_attempted[cond]
    }
    if incomplete:
        detail = "; ".join(f"{cond}: {len(ids)} unattempted" for cond, ids in incomplete.items())
        raise ValueError(
            "Refusing D-3 gap-list generation before the three fixed blocks cover every test case "
            f"({detail})"
        )
    lists = {c: sorted(set(cases[c]) - done[c]) for c in EXPECTED_CONDITIONS}
    counts = {c: {"cases": len(cases[c]), "attempted": len(attempted[c]), "done": len(done[c]),
                  "to_fill": len(lists[c])} for c in EXPECTED_CONDITIONS}
    return lists, counts, ledgers


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="D-3 Qwen3.8-Flash gap-fill lists")
    p.add_argument("--runs-root", type=Path, default=Path("runs/EXP-2026-003"))
    p.add_argument("--corpus-dir", type=Path, default=Path("data/ranbench/ranintent-v1"))
    p.add_argument("--out", type=Path, required=True, help="New directory for <condition>.txt lists")
    args = p.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        print(f"Refusing to write into non-empty {args.out} (stale lists); use a new directory.", file=sys.stderr)
        return 2
    lists, counts, ledgers = gap_lists(args.runs_root, args.corpus_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    for cond, ids in lists.items():
        if ids:
            (args.out / f"{cond}.txt").write_text("".join(i + "\n" for i in ids), encoding="utf-8")
    print("ledgers read:")
    for lf in ledgers:
        print(f"  {lf}")
    print(f"{'condition':<22}{'cases':>6}{'attempted':>10}{'done':>6}{'to_fill':>8}")
    for cond, c in counts.items():
        print(f"{cond:<22}{c['cases']:>6}{c['attempted']:>10}{c['done']:>6}{c['to_fill']:>8}")
    total = sum(c["to_fill"] for c in counts.values())
    print(f"{'total':<22}{sum(c['cases'] for c in counts.values()):>6}{'':>10}"
          f"{sum(c['done'] for c in counts.values()):>6}{total:>8}")
    print(f"{sum(1 for ids in lists.values() if ids)} list(s) written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
