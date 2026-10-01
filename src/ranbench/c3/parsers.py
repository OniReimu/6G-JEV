"""Parsers for the three timestamp-bearing C3 evidence logs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Any, Iterable

_GNB_ACK = re.compile(
    r"^(?P<t>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)\s+\[E2-DU\s*\]\s+\[I\]\s+"
    r"Sending E2 RIC Control Acknowledge\s*$"
)
_GNB_REQUEST = re.compile(
    r"^(?P<t>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)\s+\[E2-DU\s*\]\s+\[I\]\s+"
    r"Received RIC Control Request\s*$"
)


def parse_timestamp(value: str) -> float:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _lines(text_or_lines: str | Iterable[str]) -> Iterable[str]:
    return text_or_lines.splitlines() if isinstance(text_or_lines, str) else text_or_lines


def _json_events(text_or_lines: str | Iterable[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in _lines(text_or_lines):
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def parse_a1_put_acknowledgements(text_or_lines: str | Iterable[str]) -> list[dict[str, Any]]:
    """Parse driver JSON lines written immediately after an A1 simulator PUT response."""
    return [event for event in _json_events(text_or_lines) if event.get("event") == "a1_put_ack"]


def parse_xapp_controls(text_or_lines: str | Iterable[str]) -> list[dict[str, Any]]:
    """Parse xApp JSON lines carrying paired t_seen/t_control_sent timestamps."""
    return [event for event in _json_events(text_or_lines) if event.get("event") == "control_sent"]


def first_gnb_acknowledge(text_or_lines: str | Iterable[str], after_s: float = float("-inf")) -> float | None:
    """Return the first B2-format gNB E2 control-acknowledge timestamp after ``after_s``."""
    for line in _lines(text_or_lines):
        match = _GNB_ACK.match(line.strip())
        if match:
            timestamp = parse_timestamp(match.group("t"))
            if timestamp >= after_s:
                return timestamp
    return None


def parse_gnb_control_exchanges(text_or_lines: str | Iterable[str]) -> list[dict[str, float]]:
    """Pair strictly serial gNB RIC-control requests and acknowledgements.

    The deployed srsRAN log does not print the E2 requestor identity on these
    lines.  C3 therefore enforces a single in-flight control and rejects a
    nested request or an acknowledgement without a request instead of making
    a timestamp-only guess.
    """
    exchanges: list[dict[str, float]] = []
    pending: float | None = None
    for line in _lines(text_or_lines):
        stripped = line.strip()
        request = _GNB_REQUEST.match(stripped)
        if request:
            if pending is not None:
                raise ValueError("gNB logged a second RIC Control Request before acknowledging the first")
            pending = parse_timestamp(request.group("t"))
            continue
        acknowledge = _GNB_ACK.match(stripped)
        if acknowledge:
            if pending is None:
                raise ValueError("gNB logged a RIC Control Acknowledge without a preceding request")
            ack_s = parse_timestamp(acknowledge.group("t"))
            exchanges.append({"t_request_s": pending, "t_ack_s": ack_s})
            pending = None
    return exchanges


def first_kpm_change(
    text_or_lines: str | Iterable[str],
    after_s: float,
    *,
    relative_threshold: float = 0.10,
    prb_threshold: int = 2,
) -> dict[str, Any] | None:
    """Find the first material KPM change after a control.

    The baseline is the final KPM report before ``after_s``.  A report changes
    when DRB.UEThpDl moves by at least 10% of a non-zero baseline or
    RRU.PrbTotDl moves by at least two PRBs.  This suppresses one-unit and tiny
    throughput jitter while retaining the first changes seen in spike B2.
    """
    reports = [event for event in _json_events(text_or_lines) if event.get("event") == "kpm"]
    timed = [(parse_timestamp(str(event["t_utc"])), event) for event in reports if event.get("t_utc")]
    before = [(timestamp, event) for timestamp, event in timed if timestamp < after_s]
    if not before:
        return None
    _, baseline = before[-1]
    baseline_thp = float(baseline.get("DRB.UEThpDl", 0.0))
    baseline_prb = int(baseline.get("RRU.PrbTotDl", 0))
    for timestamp, event in timed:
        if timestamp < after_s:
            continue
        throughput = float(event.get("DRB.UEThpDl", 0.0))
        prbs = int(event.get("RRU.PrbTotDl", 0))
        thp_changed = (
            abs(throughput - baseline_thp) / abs(baseline_thp) >= relative_threshold
            if baseline_thp
            else throughput != 0.0
        )
        if thp_changed or abs(prbs - baseline_prb) >= prb_threshold:
            return dict(event, timestamp_s=timestamp)
    return None
