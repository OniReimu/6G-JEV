"""Shared strict validation for completed EXP-2026-003 C2 run directories."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

NS3_VERSION = "ns-3.48"
# D-7/D-7b/D-11 rerun builds: the same LENA commit plus the abort guards (cluster-B tree ns-3.48-d7, M4 tree ns-3.48-d7b).
NS3_PATCHED_VERSIONS = ("ns-3.48-d7", "ns-3.48-d7b")
LENA_COMMIT = "cedceadda17392c90587fb9400eb9b1f8c236713"
SLOT_US = 100_000

CSV_HEADERS = {
    "ue_slots.csv": "t_slot,imsi,serving_cell,class,rx_bytes,n_rx_pkts,mean_delay_ms,p95_delay_ms",
    "tx_trace.csv": "t_slot,imsi,tx_bytes,n_tx_pkts",
    "positions.csv": "t,imsi,x,y,serving_cell",
    "cell_prb.csv": "t_slot,cell,prb_share",
    "handover.csv": "t_start,t_end_ok,imsi,source_cell,target_cell,handover_total_time_ms,outcome",
    "rlf.csv": "t,imsi,cell,cause",
    "enforcement.csv": "t,cell,class,old,new,source",
    "lc_priority.csv": "t,cell,rnti,imsi,class,lc_m_priority,cell_class_level",
    "cells.csv": "cell,site,sector,x,y,bearing_deg",
    "ue_map.csv": "ue_index,imsi,drop_cell,class,port",
    "schedule.csv": "t_s,scope,class,priority",
}
NONEMPTY_FILES = ("manifest.json", "config.json", "stream.json", "streams.json", "stderr.log")
PRESENT_FILES = ("stdout.log",)
MAY_BE_HEADER_ONLY = {"handover.csv", "rlf.csv", "enforcement.csv", "schedule.csv"}
INPUT_DIRS = {"config_file": "configs", "schedule_file": "schedules", "stream_file": "streams"}
MANIFEST_SHA = {"config_file": "config_sha256", "schedule_file": "schedule_sha256",
                "stream_file": "stream_sha256"}
RUN_COPY = {"config_file": "config.json", "schedule_file": "schedule.csv", "stream_file": "stream.json"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_run_list(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    if not ids or len(ids) != len(set(ids)):
        raise SystemExit(f"run list is empty or has duplicates: {path}")
    return ids


def read_matrix(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["run_id"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise SystemExit(f"empty matrix: {path}")
    return rows


def input_path(row: dict[str, str], field: str, input_root: Path | None) -> Path:
    """Resolve an input directly or under a platform-local relocated input root."""
    original = Path(row[field])
    if original.parent.name != INPUT_DIRS[field]:
        raise ValueError(f"malformed {field} path {original}")
    return original if input_root is None else input_root / INPUT_DIRS[field] / original.name


def n_report_slots(sim_time_s: float) -> int:
    return (round(sim_time_s * 1_000_000) + 1) // SLOT_US


def _csv_rows(path: Path) -> tuple[str, list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        lines = handle.read().splitlines()
    return (lines[0] if lines else ""), lines[1:]


def verify_run(run_dir: Path, row: dict[str, str], input_root: Path | None,
               binary_sha256: str | None = None) -> tuple[str, list[str]]:
    """Return (status, problems); status is PASS, FAIL, MISSING or INCOMPLETE."""
    if not run_dir.is_dir():
        return "MISSING", ["run directory absent"]
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return "INCOMPLETE", ["no manifest.json (still running, or crashed)"]
    problems: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "FAIL", [f"manifest.json unreadable: {exc}"]
    if manifest.get("complete") is not True or manifest.get("status") != "complete":
        problems.append("manifest not complete")
    # D-2: RQ5 matrix rows are run by the RQ5 live controller. Its schedule is produced during the run (bound to the
    # manifest hash below) and it runs ns-3 without /usr/bin/time, so a clean run leaves stderr.log empty. Scoped by
    # the matrix row, not only the manifest, so no other run can claim these exceptions.
    live = row.get("rqs") == "RQ5" and manifest.get("mode") == "live"

    for name in NONEMPTY_FILES:
        path = run_dir / name
        if live and name == "stderr.log":
            if not path.is_file():
                problems.append(f"{name} missing")
            continue
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"{name} missing or empty")
    for name in PRESENT_FILES:
        if not (run_dir / name).is_file():
            problems.append(f"{name} missing")

    data: dict[str, list[str]] = {}
    for name, header in CSV_HEADERS.items():
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"{name} missing or empty")
            continue
        got, rows = _csv_rows(path)
        if got != header:
            problems.append(f"{name} header {got!r} != {header!r}")
        if not rows and name not in MAY_BE_HEADER_ONLY:
            problems.append(f"{name} has no data rows")
        data[name] = rows

    if manifest.get("ns3_version") not in (NS3_VERSION, *NS3_PATCHED_VERSIONS):
        problems.append(f"manifest ns3_version={manifest.get('ns3_version')!r}, expected {NS3_VERSION!r} "
                        f"or one of {NS3_PATCHED_VERSIONS}")
    expect = {"rng_run": 1, "lena_commit": LENA_COMMIT, "build_profile": "optimized"}
    for key, value in expect.items():
        if manifest.get(key) != value:
            problems.append(f"manifest {key}={manifest.get(key)!r}, expected {value!r}")
    sim_time = manifest.get("sim_time_s")
    try:
        want_sim = float(row["simulated_seconds"])
        if sim_time is None or not math.isclose(float(sim_time), want_sim, abs_tol=1e-9):
            problems.append(f"manifest sim_time_s={sim_time!r}, matrix {want_sim}")
    except (TypeError, ValueError):
        problems.append(f"bad sim_time_s {sim_time!r}")
    if binary_sha256 and manifest.get("binary_sha256") != binary_sha256:
        problems.append(f"binary_sha256 {manifest.get('binary_sha256')} != expected {binary_sha256}")

    for field, key in MANIFEST_SHA.items():
        if live and field == "schedule_file":
            copy = run_dir / RUN_COPY[field]
            if not copy.is_file() or sha256_file(copy) != manifest.get(key):
                problems.append(f"live schedule {copy.name} does not match manifest {key}")
            continue
        try:
            source = input_path(row, field, input_root)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        copy = run_dir / RUN_COPY[field]
        if not source.is_file():
            problems.append(f"input {source} not found")
            continue
        want = sha256_file(source)
        if manifest.get(key) != want:
            problems.append(f"manifest {key} != sha256 of {source.name}")
        if copy.is_file() and sha256_file(copy) != want:
            problems.append(f"{copy.name} differs from input {source.name}")

    try:
        streams = json.loads((run_dir / "streams.json").read_text(encoding="utf-8"))
        if streams.get("rng_run") != 1:
            problems.append(f"streams.json rng_run={streams.get('rng_run')!r}")
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        n_cells = int(config["scenario"]["num_cells"])
        ues = int(row["ues"])
        slots = n_report_slots(float(sim_time))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append(f"cannot derive expected row counts: {exc}")
    else:
        counts = {"ue_slots.csv": ues * slots, "tx_trace.csv": ues * slots,
                  "positions.csv": ues * slots, "cell_prb.csv": n_cells * slots,
                  "ue_map.csv": ues, "cells.csv": n_cells}
        for name, want in counts.items():
            if name in data and len(data[name]) != want:
                problems.append(f"{name} has {len(data[name])} data rows, expected {want}")
        last = f"{slots * SLOT_US / 1_000_000:.3f}"
        if data.get("ue_slots.csv") and data["ue_slots.csv"][-1].split(",")[0] != last:
            problems.append(f"ue_slots.csv last t_slot != {last}")
    return ("FAIL" if problems else "PASS"), problems


def verify_many(run_root: Path, run_ids: list[str], matrix: dict[str, dict[str, str]],
                input_root: Path, binary_sha256: str | None = None
                ) -> list[tuple[str, str, list[str]]]:
    results = []
    for run_id in run_ids:
        if run_id not in matrix:
            results.append((run_id, "FAIL", ["run_id not in matrix"]))
            continue
        status, problems = verify_run(run_root / run_id, matrix[run_id], input_root, binary_sha256)
        results.append((run_id, status, problems))
    shas = {}
    for run_id, status, _ in results:
        if status in ("PASS", "FAIL"):
            try:
                manifest = json.loads((run_root / run_id / "manifest.json").read_text(encoding="utf-8"))
                shas.setdefault(manifest.get("binary_sha256"), []).append(run_id)
            except (OSError, json.JSONDecodeError):
                pass
    if len(shas) > 1:
        results = [(run_id, "FAIL" if status == "PASS" else status,
                    problems + (["runs used different binaries"] if status in ("PASS", "FAIL") else []))
                   for run_id, status, problems in results]
    return results


def print_table(results: list[tuple[str, str, list[str]]]) -> None:
    width = max([len(result[0]) for result in results] + [6])
    print(f"{'run_id'.ljust(width)}  {'status'.ljust(15)}  detail")
    for run_id, status, problems in results:
        print(f"{run_id.ljust(width)}  {status.ljust(15)}  {'; '.join(problems)}")
    counts: dict[str, int] = {}
    for _, status, _ in results:
        counts[status] = counts.get(status, 0) + 1
    print("SUMMARY " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())))
