#!/usr/bin/env python3
"""Self-check for finished EXP-2026-003 C2 run directories.

For every run_id in the run list, checks that <run-root>/<run_id> holds a complete
manifest, every scenario output with its exact CSV header, the expected number of
100 ms report rows, and input/binary hashes that match the frozen input tree.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ranbench.c2.verify import (
    print_table,
    read_matrix,
    read_run_list,
    sha256_file,
    verify_many,
    verify_run,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--run-list", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--binary-sha256", help="expected ns-3 binary sha256 of this platform")
    parser.add_argument("--allow-missing", action="store_true",
                        help="MISSING/INCOMPLETE runs do not fail the check (progress view)")
    parser.add_argument("--quarantine", action="store_true",
                        help="move INCOMPLETE/FAIL run dirs to <run-root>/_incomplete/<run_id>.<UTC> "
                             "(only while no queue is running)")
    args = parser.parse_args(argv)
    results = verify_many(args.run_root, read_run_list(args.run_list), read_matrix(args.matrix),
                          args.input_root, args.binary_sha256)
    print_table(results)
    if args.quarantine:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for run_id, status, _ in results:
            if status in ("INCOMPLETE", "FAIL") and (args.run_root / run_id).is_dir():
                target = args.run_root / "_incomplete" / f"{run_id}.{stamp}"
                target.parent.mkdir(exist_ok=True)
                (args.run_root / run_id).rename(target)
                print(f"QUARANTINED {run_id} -> {target}")
        return 0
    ok = {"PASS", "MISSING", "INCOMPLETE"} if args.allow_missing else {"PASS"}
    passed = all(status in ok for _, status, _ in results)
    print("RESULT " + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
