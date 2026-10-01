"""Resumable local/remote single-thread C2 queue runner.

Strictly verified run directories are skipped. Any incomplete or invalid run
directory is moved aside before its row is re-queued.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import errno
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from src.ranbench.c2.verify import verify_run

BLOCKING_STATUSES = {"refuse_existing", "missing_input", "bad_rng_run"}
INPUT_DIRS = {"config_file": "configs", "schedule_file": "schedules", "stream_file": "streams"}
_PROCESS_LOCKS: list[BinaryIO] = []


@dataclass(frozen=True)
class QueueResult:
    run_id: str
    status: str
    detail: str = ""


def manifest_complete(run_dir: Path) -> bool:
    path = run_dir / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return data.get("complete") is True and data.get("status") == "complete"


def load_rows(matrix_path: str | Path, run_ids: list[str] | set[str] | None = None,
              tiers: set[int] | None = None) -> list[dict[str, str]]:
    with Path(matrix_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row.get("run_id", "") for row in rows]
    if not ids or any(not run_id for run_id in ids):
        raise ValueError("matrix has no rows or a row without run_id")
    if len(set(ids)) != len(ids):
        raise ValueError("matrix contains duplicate run_id values")
    if run_ids is not None:
        requested = list(run_ids)
        missing = sorted(set(requested) - set(ids))
        if missing:
            raise ValueError(f"run list contains {len(missing)} unknown run_id(s): {missing[:3]}")
        by_id = {row["run_id"]: row for row in rows}
        rows = [by_id[run_id] for run_id in requested]
    if tiers is not None:
        rows = [row for row in rows if int(row["priority_tier"]) in tiers]
    return rows


def read_run_list(path: str | Path) -> list[str]:
    values = [
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values:
        raise ValueError(f"empty run list: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate run ID in run list: {path}")
    return values


def relocate_inputs(rows: list[dict[str, str]], input_root: str | Path) -> None:
    """Relocate the matrix's immutable input tree to a platform-local root."""
    root = Path(input_root)
    for row in rows:
        for field, directory in INPUT_DIRS.items():
            original = Path(row[field])
            if original.parent.name != directory:
                raise ValueError(f"{row['run_id']}: malformed {field} path {original}")
            row[field] = str(root / directory / original.name)


def command_for(row: dict[str, str], run_dir: Path, runner: Path, binary: Path,
                python: str) -> list[str]:
    return [
        python,
        str(runner),
        "--config", row["config_file"],
        "--schedule", row["schedule_file"],
        "--stream", row["stream_file"],
        "--run-dir", str(run_dir),
        "--binary", str(binary),
        "--rng-run", row["rng_run"],
        "--sim-time", row["simulated_seconds"],
    ]


def check_binary(binary: Path, expect_sha256: str | None) -> None:
    """Fail before any run if the binary is not the platform's pinned build.

    run_c2.py reads build facts from <ns-3 root>/cmake-cache only after the
    simulation, so a binary outside its build tree without C2_NS3_ROOT would
    lose a finished run; that is refused here instead.
    """
    if not binary.is_file():
        raise FileNotFoundError(f"binary not found: {binary}")
    if expect_sha256 is not None:
        got = hashlib.sha256(binary.read_bytes()).hexdigest()
        if got != expect_sha256.lower():
            raise ValueError(f"binary sha256 {got} != expected {expect_sha256}")
    ns3_root = Path(os.environ.get("C2_NS3_ROOT") or binary.resolve().parents[2])
    if not (ns3_root / "cmake-cache" / "CMakeCache.txt").is_file():
        raise FileNotFoundError(
            f"no {ns3_root}/cmake-cache/CMakeCache.txt: set C2_NS3_ROOT to the binary's ns-3 build tree")


def _lock_run(output_root: Path, run_id: str, dry_run: bool) -> tuple[BinaryIO | None, str]:
    lock_path = output_root / ".locks" / f"{run_id}.lock"
    if dry_run and not lock_path.exists():
        return None, "absent"
    if not dry_run:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("rb" if dry_run else "a+b")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            return None, "locked"
        return None, "unsupported"
    return handle, "acquired"


def preflight(rows: Iterable[dict[str, str]], output_root: Path, runner: Path,
              binary: Path, expect_sha256: str | None = None, dry_run: bool = False,
              _locks: dict[str, BinaryIO] | None = None,
              _binary_sha256: str | None = None) -> list[QueueResult]:
    results: list[QueueResult] = []
    if not runner.is_file():
        raise FileNotFoundError(f"runner not found: {runner}")
    check_binary(binary, expect_sha256)
    binary_sha256 = _binary_sha256 or hashlib.sha256(binary.read_bytes()).hexdigest()
    for row in rows:
        run_id = row["run_id"]
        run_dir = output_root / run_id
        lock, lock_status = _lock_run(output_root, run_id, dry_run)
        if lock_status == "locked":
            results.append(QueueResult(run_id, "locked"))
            continue
        if lock is not None:
            if _locks is None:
                _PROCESS_LOCKS.append(lock)
            else:
                _locks[run_id] = lock
        if run_dir.exists():
            if not run_dir.is_dir():
                results.append(QueueResult(run_id, "refuse_existing", str(run_dir)))
                continue
            status, problems = verify_run(run_dir, row, None, binary_sha256)
            if status == "PASS":
                results.append(QueueResult(run_id, "skip_complete"))
                continue
            if lock_status == "unsupported":
                results.append(QueueResult(
                    run_id, "refuse_existing", f"{run_dir}: {status}: {'; '.join(problems)}"))
                continue
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            target = run_dir.with_name(f"{run_dir.name}.incomplete-{stamp}")
            action = "would move" if dry_run else "moved"
            if not dry_run:
                run_dir.rename(target)
            detail = f"{action} {status} run to {target}: {'; '.join(problems)}"
        else:
            detail = ""
        missing = [name for name in ("config_file", "schedule_file", "stream_file")
                   if not Path(row[name]).is_file()]
        if missing:
            results.append(QueueResult(run_id, "missing_input", ",".join(missing)))
            continue
        if row.get("rng_run") != "1":
            results.append(QueueResult(run_id, "bad_rng_run", row.get("rng_run", "")))
            continue
        results.append(QueueResult(run_id, "ready", detail))
    return results


def run_queue(rows: list[dict[str, str]], output_root: Path, runner: Path,
              binary: Path, jobs: int, python: str = sys.executable,
              dry_run: bool = False, expect_sha256: str | None = None) -> list[QueueResult]:
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
    locks: dict[str, BinaryIO] = {}
    binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    checked = preflight(
        rows, output_root, runner, binary, expect_sha256, dry_run, locks, binary_sha256)
    by_id = {row["run_id"]: row for row in rows}
    ready = [result.run_id for result in checked if result.status == "ready"]
    if dry_run or any(result.status in BLOCKING_STATUSES for result in checked):
        _PROCESS_LOCKS.extend(locks.values())
        return checked

    for result in checked:
        if result.status != "ready" and result.run_id in locks:
            _PROCESS_LOCKS.append(locks.pop(result.run_id))

    def execute(run_id: str) -> QueueResult:
        try:
            row = by_id[run_id]
            cmd = command_for(row, output_root / run_id, runner, binary, python)
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout)[-2000:]
                return QueueResult(run_id, "failed", detail)
            status, problems = verify_run(output_root / run_id, row, None, binary_sha256)
            if status != "PASS":
                return QueueResult(run_id, "failed", "; ".join(problems))
            return QueueResult(run_id, "complete")
        finally:
            locks[run_id].close()

    finished: dict[str, QueueResult] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(execute, run_id): run_id for run_id in ready}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            finished[result.run_id] = result
            print(json.dumps(result.__dict__), flush=True)
    return [finished.get(result.run_id, result) for result in checked]


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Run a resumable C2 matrix queue")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--expect-binary-sha256",
                        help="refuse to start unless the binary is this platform's pinned build")
    parser.add_argument("--input-root", help="platform-local replacement for the matrix input-tree root")
    parser.add_argument("--runner", default=str(repo_root / "src/ranbench/ns3/run_c2.py"))
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("C2_JOBS", "32")))
    parser.add_argument("--run-list")
    parser.add_argument("--only-run-id", action="append", default=[])
    parser.add_argument("--tiers", help="comma-separated tiers, e.g. 1,2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    selected: list[str] | None = None
    if args.run_list:
        selected = read_run_list(args.run_list)
    if args.only_run_id:
        selected = list(dict.fromkeys((selected or []) + args.only_run_id))
    tiers = {int(value) for value in args.tiers.split(",")} if args.tiers else None
    rows = load_rows(args.matrix, selected, tiers)
    if args.input_root:
        relocate_inputs(rows, args.input_root)
    results = run_queue(rows, Path(args.output_root), Path(args.runner), Path(args.binary),
                        args.jobs, dry_run=args.dry_run, expect_sha256=args.expect_binary_sha256)
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selected": len(rows),
        "jobs": args.jobs,
        "status_counts": counts,
    }
    print(json.dumps(summary, indent=2))
    bad = BLOCKING_STATUSES | {"failed"}
    return 1 if any(result.status in bad for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
