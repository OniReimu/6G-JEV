"""Materialise immutable production inputs named by a C2 matrix."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.ranbench.c2 import schedule as sch
from src.ranbench.c2.streams import (
    load_completed_rq3_trace,
    load_pool,
    make_stream,
    stream_from_trace,
    write_stream,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"matrix input path escapes input root: {path}") from exc
    return resolved


def read_matrix(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty production matrix")
    return rows


def materialize(
    rows: list[dict[str, str]],
    input_root: str | Path,
    base_config: str | Path,
    pool_path: str | Path,
    block_s: float,
    put_ack_median_s: float,
    record_paths: dict[str, str | Path] | None = None,
    trace_paths: dict[str, str | Path] | None = None,
    allow_a1_put_ack_override: bool = False,
) -> dict[str, Any]:
    """Create configs, ordinary streams, and schedules whose prerequisites exist.

    RQ3 stays deferred until its per-model/rate load trace exists, and RQ5 stays
    deferred until its exploratory live-coupling implementation exists.
    Interpreter schedules stay deferred unless a selected C1 ledger is supplied.
    """
    root = Path(input_root).resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite input root: {root}")
    if block_s <= 0 or put_ack_median_s < 0:
        raise ValueError("block length must be positive and A1 PUT-ack median non-negative")
    d_a1 = sch.delta_a1(put_ack_median_s, allow_override=allow_a1_put_ack_override)
    for row in rows:
        for field in ("config_file", "stream_file", "schedule_file"):
            _inside(Path(row[field]), root)

    base = json.loads(Path(base_config).read_text(encoding="utf-8"))
    pool, pool_sha = load_pool(pool_path)
    records = {
        model: sch.load_records(path, model)
        for model, path in (record_paths or {}).items()
    }
    traces = dict(trace_paths or {})
    root.mkdir(parents=True)

    created: set[Path] = set()
    config_rows: dict[Path, list[dict[str, str]]] = {}
    for row in rows:
        path = _inside(Path(row["config_file"]), root)
        config_rows.setdefault(path, []).append(row)
    for path, refs in config_rows.items():
        config = json.loads(json.dumps(base))
        config["ues"]["speed_kmh"] = float(refs[0]["speed_kmh"])
        config["ues"]["ues_per_cell"] = int(refs[0]["ues_per_cell"])
        # The launcher always passes the per-row --sim-time. This fallback makes
        # a direct invocation long enough for every row sharing this config.
        config["simulation"]["sim_time_s"] = max(float(row["simulated_seconds"]) for row in refs)
        _write_json(path, config)
        created.add(path)

    streams: dict[Path, dict[str, Any]] = {}
    rq3_records: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["rqs"] == "RQ5":
            continue
        path = _inside(Path(row["stream_file"]), root)
        if path not in streams:
            if row["rqs"] == "RQ3":
                trace_key = f"{row['interpreter']}@{float(row['rate_per_s']):g}"
                if trace_key not in traces:
                    continue
                trace_rows = load_completed_rq3_trace(
                    traces[trace_key], row["interpreter"], float(row["rate_per_s"])
                )
                trace_records = {str(item["intent_id"]): item for item in trace_rows}
                stream = stream_from_trace(trace_rows, pool, block_s, trace_key, pool_sha256=pool_sha)
                if abs(float(row["simulated_seconds"]) - stream["meta"]["sim_time_s"]) > 1e-6:
                    raise ValueError(
                        f"{row['run_id']}: matrix simulated_seconds does not match RQ3 trace; regenerate matrix"
                    )
                rq3_records[trace_key] = trace_records
            else:
                key = f"rate={float(row['rate_per_s']):g},speed={float(row['speed_kmh']):g}"
                stream = make_stream(pool, float(row["rate_per_s"]), block_s, key, pool_sha256=pool_sha)
            write_stream(stream, path)
            streams[path] = stream
            created.update((path, Path(str(path) + ".sha256")))

    row_status: dict[str, str] = {}
    for row in rows:
        run_id = row["run_id"]
        if row["rqs"] == "RQ3":
            trace_key = f"{row['interpreter']}@{float(row['rate_per_s']):g}"
            if trace_key not in traces:
                row_status[run_id] = "waiting_for_load_trace"
                continue
            stream = streams[_inside(Path(row["stream_file"]), root)]
            arm = sch.build_arm(stream, "N", row["mode"], records=rq3_records[trace_key], d_a1=d_a1)
        if row["rqs"] == "RQ5":
            row_status[run_id] = "exploratory_not_materialised"
            continue
        if row["rqs"] != "RQ3":
            model = row["interpreter"]
            if model != "control" and model not in records:
                row_status[run_id] = "waiting_for_c1_ledger"
                continue
            stream = streams[_inside(Path(row["stream_file"]), root)]
            arm_name = row["arm"]
            if arm_name == "oracle":
                arm = sch.build_arm(stream, "oracle", d_a1=d_a1)
            elif arm_name == "no-update":
                arm = sch.build_arm(stream, "no_update", d_a1=d_a1)
            elif arm_name.startswith("fixed-"):
                delay = float(arm_name.removeprefix("fixed-"))
                arm = sch.build_arm(stream, "fixed", row["mode"], fixed_d=delay, d_a1=d_a1)
            else:
                arm = sch.build_arm(stream, arm_name, row["mode"], records=records[model], d_a1=d_a1)
        schedule_path = _inside(Path(row["schedule_file"]), root)
        schedule_path.parent.mkdir(parents=True, exist_ok=True)
        sch.write_schedule(arm["schedule"], schedule_path)
        events_path = schedule_path.with_suffix(".events.json")
        _write_json(events_path, {"meta": arm["meta"], "events": arm["events"]})
        created.update((schedule_path, events_path))
        row_status[run_id] = "ready"

    files = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(created)
    }
    summary = {
        "input_root": str(root),
        "block_s": block_s,
        "a1_put_ack_median_s": put_ack_median_s,
        "a1_put_ack_override": (
            allow_a1_put_ack_override
            and abs(put_ack_median_s - sch.PUT_ACK_MEDIAN_S) > 1e-12
        ),
        "delta_a1_s": d_a1,
        "delta_e2_s": sch.DELTA_E2_S,
        "base_config": str(Path(base_config).resolve()),
        "base_config_sha256": _sha256(Path(base_config)),
        "pool": str(Path(pool_path).resolve()),
        "pool_sha256": pool_sha,
        "records": {model: {"path": str(Path(path).resolve()), "sha256": _sha256(Path(path))}
                    for model, path in (record_paths or {}).items()},
        "load_traces": {
            key: {
                "path": str(Path(path).resolve()),
                "sha256": _sha256(Path(path)),
                "integrity_path": str((Path(path).parent / "integrity.json").resolve()),
                "integrity_sha256": _sha256(Path(path).parent / "integrity.json"),
            }
            for key, path in traces.items()
        },
        "rows": len(rows),
        "row_status_counts": dict(sorted(Counter(row_status.values()).items())),
        "row_status": row_status,
        "files": files,
    }
    _write_json(root / "input-manifest.json", summary)
    return summary
