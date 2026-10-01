#!/usr/bin/env python3
"""Check EXP-2026-003 C3 records against their raw A1, xApp, gNB, and iperf3 evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any


GNB_REQUEST = re.compile(
    r"^(?P<t>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)\s+\[E2-DU\s*\]\s+\[I\]\s+"
    r"Received RIC Control Request\s*$"
)
GNB_ACK = re.compile(
    r"^(?P<t>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)\s+\[E2-DU\s*\]\s+\[I\]\s+"
    r"Sending E2 RIC Control Acknowledge\s*$"
)
DECISION_NONNEGATIVE_FIELDS = ("ell_s", "adapter_latency_s")
CONTROL_NONNEGATIVE_FIELDS = (
    "a1_put_ack_latency_s",
    "delta_a1_s",
    "a1_poll_wait_s",
    "delta_e2_s",
)
CONTROL_PATH_FIELDS = (
    "policy_id",
    "control",
    "a1_http_status",
    "a1_put_ack_latency_s",
    "delta_a1_s",
    "a1_poll_wait_s",
    "delta_e2_s",
    "kpm_change_upper_bound_s",
    "kpm_change",
    "t_a1_put_requested",
    "t_a1_put_acknowledged",
    "t_seen",
    "t_control_sent",
    "t_gnb_acknowledged",
    "t_first_kpm_change",
    "remote_timestamps",
    "clock_probes",
)
SUMMARY_FIELDS = ("ell_s", "delta_a1_s", "delta_e2_s", "kpm_change_upper_bound_s")


def parse_timestamp(value: str) -> float:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def json_events(path: Path, event_name: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("event") == event_name:
                events.append(value)
    return events


def gnb_exchanges(path: Path) -> tuple[list[dict[str, float]], list[str]]:
    exchanges: list[dict[str, float]] = []
    errors: list[str] = []
    pending: float | None = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            request = GNB_REQUEST.match(stripped)
            if request:
                if pending is not None:
                    errors.append(f"gNB line {line_number}: nested control request")
                pending = parse_timestamp(request.group("t"))
                continue
            acknowledge = GNB_ACK.match(stripped)
            if acknowledge:
                if pending is None:
                    errors.append(f"gNB line {line_number}: acknowledge without a request")
                    continue
                ack = parse_timestamp(acknowledge.group("t"))
                if ack < pending:
                    errors.append(f"gNB line {line_number}: acknowledge predates its request")
                exchanges.append({"t_request_s": pending, "t_ack_s": ack})
                pending = None
    if pending is not None:
        errors.append("gNB log ends with an unacknowledged control request")
    return exchanges, errors


def duplicate_values(values: list[str]) -> list[str]:
    return sorted({value for value in values if values.count(value) > 1})


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    return {
        "n": len(values),
        "null_count": len(rows) - len(values),
        "median_s": statistics.median(values) if values else None,
        "min_s": min(values) if values else None,
        "max_s": max(values) if values else None,
    }


def iperf3_interval_end_s(intervals: Any) -> float | None:
    if not isinstance(intervals, list) or not intervals:
        return None
    ends: list[float] = []
    elapsed = 0.0
    for interval in intervals:
        if not isinstance(interval, dict) or not isinstance(interval.get("sum"), dict):
            return None
        summary = interval["sum"]
        if isinstance(summary.get("end"), (int, float)):
            ends.append(float(summary["end"]))
        elif isinstance(summary.get("seconds"), (int, float)):
            elapsed += float(summary["seconds"])
            ends.append(elapsed)
        else:
            return None
    return max(ends)


def iperf3_start_s(start: Any) -> float | None:
    if not isinstance(start, dict) or not isinstance(start.get("timestamp"), dict):
        return None
    timestamp = start["timestamp"]
    if isinstance(timestamp.get("timesecs"), (int, float)):
        return float(timestamp["timesecs"])
    value = timestamp.get("time")
    if not isinstance(value, str):
        return None
    try:
        return parse_timestamp(value)
    except ValueError:
        try:
            return parsedate_to_datetime(value).timestamp()
        except (TypeError, ValueError):
            return None


def record_time_bounds(records: list[dict[str, Any]]) -> tuple[float, float] | None:
    starts: list[float] = []
    ends: list[float] = []
    for row in records:
        start_value = row.get("t_intent_issued") or row.get("t_a1_put_requested")
        end_value = (
            row.get("t_gnb_acknowledged")
            or row.get("t_decision_returned")
            or row.get("t_a1_put_acknowledged")
        )
        try:
            if isinstance(start_value, str):
                starts.append(parse_timestamp(start_value))
            if isinstance(end_value, str):
                ends.append(parse_timestamp(end_value))
        except ValueError:
            continue
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def check(args: argparse.Namespace) -> dict[str, Any]:
    sources = {
        "records": str(args.records.resolve()),
        "a1": str(args.a1.resolve()),
        "xapp": str(args.xapp.resolve()),
        "gnb": str(args.gnb.resolve()),
        "iperf3": str(args.iperf3.resolve()),
    }
    errors: list[str] = []
    try:
        records = read_jsonl(args.records)
        a1_events = json_events(args.a1, "a1_put_ack")
        xapp_events_all = json_events(args.xapp, "control_sent")
        exchanges, gnb_errors = gnb_exchanges(args.gnb)
        errors.extend(gnb_errors)
        with args.iperf3.open(encoding="utf-8") as handle:
            iperf3 = json.load(handle)
        if not isinstance(iperf3, dict):
            errors.append("iperf3 JSON root is not an object")
            iperf3 = {}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "sources": sources, "errors": [str(exc)]}

    run_ids = {str(row.get("run_id")) for row in records}
    interpreters = {str(row.get("interpreter")) for row in records}
    run_id = next(iter(run_ids)) if len(run_ids) == 1 else None
    interpreter = next(iter(interpreters)) if len(interpreters) == 1 else None
    if len(records) != args.expected:
        errors.append(f"decision count {len(records)} != expected {args.expected}")
    if len(run_ids) != 1:
        errors.append(f"records contain {len(run_ids)} run IDs")
    if len(interpreters) != 1:
        errors.append(f"records contain {len(interpreters)} interpreters")
    sequences = [row.get("sequence") for row in records]
    if sequences != list(range(len(records))):
        errors.append("record sequences are not contiguous from zero in file order")
    intent_ids = [str(row.get("intent_id")) for row in records]
    if duplicate_values(intent_ids):
        errors.append(f"duplicate intent IDs: {duplicate_values(intent_ids)}")

    actuated: list[tuple[int, dict[str, Any], str]] = []
    invalid_decisions = 0
    unactuatable_decisions = 0
    for index, row in enumerate(records):
        prefix = f"record {index}"
        decision_invalid = row.get("decision_valid") is not True or row.get("decision_error") is not None
        unactuatable = row.get("actuation_error") is not None
        if decision_invalid:
            invalid_decisions += 1
        if unactuatable:
            unactuatable_decisions += 1
        for field in DECISION_NONNEGATIVE_FIELDS:
            value = row.get(field)
            if not isinstance(value, (int, float)) or value < 0:
                errors.append(f"{prefix}: {field} is missing or negative")
        policy_id = row.get("policy_id")
        if decision_invalid or unactuatable:
            present = [field for field in CONTROL_PATH_FIELDS if row.get(field) is not None]
            if present:
                errors.append(f"{prefix}: non-actuated decision has control-path fields: {present}")
            continue
        if not isinstance(policy_id, str) or not policy_id:
            errors.append(f"{prefix}: missing policy_id/A1 actuation")
            continue
        actuated.append((index, row, policy_id))
        if not isinstance(row.get("control"), dict):
            errors.append(f"{prefix}: missing control")
        if row.get("a1_http_status") not in (200, 201, 202, 204):
            errors.append(f"{prefix}: A1 PUT was not acknowledged")
        for field in CONTROL_NONNEGATIVE_FIELDS:
            value = row.get(field)
            if not isinstance(value, (int, float)) or value < 0:
                errors.append(f"{prefix}: {field} is missing or negative")
        kpm_value = row.get("kpm_change_upper_bound_s")
        if kpm_value is not None and (not isinstance(kpm_value, (int, float)) or kpm_value < 0):
            errors.append(f"{prefix}: kpm_change_upper_bound_s is negative or non-numeric")
        for field in (
            "t_a1_put_requested", "t_a1_put_acknowledged", "t_control_sent", "t_gnb_acknowledged",
        ):
            if not isinstance(row.get(field), str):
                errors.append(f"{prefix}: missing {field}")
        timestamps = (
            row.get("t_a1_put_acknowledged"), row.get("t_a1_put_requested"),
            row.get("t_gnb_acknowledged"), row.get("t_control_sent"),
        )
        try:
            if not all(isinstance(value, str) for value in timestamps):
                raise ValueError("missing timestamp")
            if parse_timestamp(row["t_a1_put_acknowledged"]) < parse_timestamp(row["t_a1_put_requested"]):
                errors.append(f"{prefix}: A1 acknowledgement predates request")
            if parse_timestamp(row["t_gnb_acknowledged"]) < parse_timestamp(row["t_control_sent"]):
                errors.append(f"{prefix}: normalized gNB acknowledgement predates control")
            if row.get("t_first_kpm_change") is not None and (
                parse_timestamp(row["t_first_kpm_change"]) < parse_timestamp(row["t_control_sent"])
            ):
                errors.append(f"{prefix}: normalized KPM change predates control")
        except (AttributeError, KeyError, TypeError, ValueError):
            errors.append(f"{prefix}: invalid timestamp field")

    policy_ids = [policy_id for _, _, policy_id in actuated]
    if duplicate_values(policy_ids):
        errors.append(f"duplicate policy IDs: {duplicate_values(policy_ids)}")
    a1_by_policy = {str(event.get("policy_id")): event for event in a1_events}
    if len(a1_by_policy) != len(a1_events):
        errors.append("A1 log contains duplicate policy acknowledgements")
    if set(a1_by_policy) != set(policy_ids):
        errors.append("A1 acknowledged policy IDs do not exactly match records")
    for record_index, row, policy_id in actuated:
        event = a1_by_policy.get(policy_id)
        if event is None:
            continue
        if event.get("status") not in (200, 201, 202, 204):
            errors.append(f"record {record_index}: A1 event has non-success status")
        if row.get("t_a1_put_acknowledged") != event.get("t_utc"):
            errors.append(f"record {record_index}: A1 acknowledgement timestamp mismatch")

    xapp_events = [event for event in xapp_events_all if event.get("run_id") == run_id]
    xapp_policy_ids = [str(event.get("policy_id")) for event in xapp_events]
    if xapp_policy_ids != policy_ids:
        errors.append("run-scoped xApp controls do not exactly match record order")
    if len(exchanges) != len(policy_ids):
        errors.append(f"strict gNB exchange count {len(exchanges)} != control count {len(policy_ids)}")
    for (record_index, row, _), event in zip(actuated, xapp_events):
        remote = row.get("remote_timestamps")
        if not isinstance(remote, dict):
            errors.append(f"record {record_index}: missing remote_timestamps")
            continue
        if remote.get("xapp_t_control_sent") != event.get("t_control_sent"):
            errors.append(f"record {record_index}: xApp control timestamp mismatch")
        if remote.get("xapp_t_seen") != event.get("t_seen"):
            errors.append(f"record {record_index}: xApp seen timestamp mismatch")
    for (record_index, row, _), exchange in zip(actuated, exchanges):
        remote = row.get("remote_timestamps")
        try:
            recorded_ack = parse_timestamp(str(remote["gnb_t_acknowledged"]))
        except (KeyError, TypeError, ValueError):
            errors.append(f"record {record_index}: invalid raw gNB acknowledgement timestamp")
            continue
        if abs(recorded_ack - exchange["t_ack_s"]) > 1e-6:
            errors.append(f"record {record_index}: raw gNB acknowledgement timestamp mismatch")

    has_end = isinstance(iperf3.get("end"), dict)
    interval_end_s = iperf3_interval_end_s(iperf3.get("intervals"))
    iperf_start_s = iperf3_start_s(iperf3.get("start"))
    if not isinstance(iperf3.get("start"), dict) or not has_end:
        errors.append("iperf3 JSON lacks start/end objects")
    if interval_end_s is None:
        errors.append("iperf3 JSON lacks non-empty intervals")
    coverage_ok = False
    bounds = record_time_bounds(records)
    if bounds is not None and iperf_start_s is not None and interval_end_s is not None:
        run_start_s, run_end_s = bounds
        coverage_ok = iperf_start_s <= run_start_s and iperf_start_s + interval_end_s >= run_end_s
        if not coverage_ok:
            errors.append("iperf3 intervals do not cover the recorded run")
    elif records:
        errors.append("could not verify iperf3 coverage of the recorded run")
    iperf_error = iperf3.get("error")
    expected_interrupt = iperf_error == "interrupt - the client has terminated"
    if iperf_error and not (expected_interrupt and has_end and interval_end_s is not None and coverage_ok):
        errors.append(f"iperf3 reported an error: {iperf_error}")

    report: dict[str, Any] = {
        "status": "PASS" if not errors else "FAIL",
        "sources": sources,
        "expected_decisions": args.expected,
        "actual_decisions": len(records),
        "run_id": run_id,
        "interpreter": interpreter,
        "invalid_decisions": invalid_decisions,
        "unactuatable_decisions": unactuatable_decisions,
        "actuated_decisions": len(actuated),
        "a1_acknowledgements": len(a1_events),
        "xapp_controls": len(xapp_events),
        "strict_gnb_exchanges": len(exchanges),
        "iperf3_intervals": len(iperf3.get("intervals", [])) if isinstance(iperf3.get("intervals"), list) else 0,
        "summary": {field: distribution(records, field) for field in SUMMARY_FIELDS},
        "errors": errors,
    }
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--records", type=Path, required=True)
    result.add_argument("--a1", type=Path, required=True)
    result.add_argument("--xapp", type=Path, required=True)
    result.add_argument("--gnb", type=Path, required=True)
    result.add_argument("--iperf3", type=Path, required=True)
    result.add_argument("--expected", type=int, default=30)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.expected < 1:
        parser().error("--expected must be positive")
    report = check(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"C3 integrity {report['status']}: {args.output}")
    for error in report.get("errors", []):
        print(f"ERROR: {error}", file=sys.stderr)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
