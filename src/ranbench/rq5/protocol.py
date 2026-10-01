"""Shared RQ5 per-cell question, rendering, schema, and fail-closed normalisation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

RQ5_ACTIONS = ("raise", "hold", "lower")
RQ5_CLASSES = ("video", "xr", "iot", "be")
RQ5_DEADLINE_S = 30.0
RQ5_CRITERIA = {
    "raise": "increase this class's scheduler weight by one configured step",
    "hold": "leave this class's scheduler weight unchanged",
    "lower": "decrease this class's scheduler weight by one configured step",
}
RQ5_RULES = (
    "Choose one action for every cell/class row. Use only raise, hold, or lower. "
    "Raise means more scheduler weight; lower means less. Priorities are scheduler levels where a "
    "lower number means more weight: raise subtracts step, lower adds step, kept within "
    "base_priority +/- band. Use the active policy (base_priority), the class target, and the last "
    "one-second throughput, p95 delay, and PRB share. Return every row exactly once."
)
RQ5_DELTA_E2_S = 0.005


def question_id(cell: int, cls: str) -> str:
    return f"cell_{cell}__{cls}"


def application_time(tick_s: float, latency_s: float, delta_e2_s: float = RQ5_DELTA_E2_S) -> float:
    if latency_s < 0 or delta_e2_s < 0:
        raise ValueError("latency and E2 delay must be non-negative")
    return tick_s + latency_s + delta_e2_s


def tick_is_skipped(tick_s: float, pending_until_s: float) -> bool:
    return pending_until_s > tick_s + 1e-12


def apply_action(current: int, base: int, action: str, step: int, band: int) -> int:
    """The numerical-xApp direction convention and active-policy clamp used by ns-3."""
    if action not in RQ5_ACTIONS:
        action = "hold"
    desired = current - step if action == "raise" else current + step if action == "lower" else current
    return min(max(desired, max(1, base - band)), min(99, base + band))


def expected_pairs(snapshot: dict[str, Any]) -> list[tuple[int, str]]:
    rows = snapshot.get("rows")
    if not isinstance(rows, list):
        raise TypeError("snapshot rows must be a list")
    pairs: list[tuple[int, str]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("cell"), int) or row.get("class") not in RQ5_CLASSES:
            raise ValueError("snapshot row has an invalid cell/class")
        pairs.append((row["cell"], row["class"]))
    if len(pairs) != len(set(pairs)):
        raise ValueError("snapshot contains a duplicate cell/class row")
    return pairs


def rq5_context(snapshot: dict[str, Any]) -> dict[str, str]:
    """C1-style rules/case split; all adapters receive the same content."""
    expected_pairs(snapshot)
    columns = (
        "cell,class,throughput_kbps,p95_delay_ms,prb_share,base_priority,current_priority,"
        "target_type,target_value"
    )
    lines = [
        f"family=rq5_percell tick_s={float(snapshot['tick_s']):.6f} window_s={float(snapshot['window_s']):.6f}",
        f"step={int(snapshot['step'])} band={int(snapshot['band'])}",
        columns,
    ]
    def num(value: Any, digits: int) -> str:
        return "null" if value is None else f"{float(value):.{digits}f}"

    for row in snapshot["rows"]:
        # Measured values rounded for a short prompt; the logged snapshot keeps full precision.
        values = [
            str(row["cell"]), row["class"], num(row.get("throughput_kbps"), 1),
            num(row.get("p95_delay_ms"), 1), num(row.get("prb_share"), 3), str(row["base_priority"]),
            str(row["current_priority"]), row["target_type"], f"{float(row['target_value']):g}",
        ]
        lines.append(",".join(values))
    return {"rules": RQ5_RULES, "case": "\n".join(lines)}


def decision_state(snapshot: dict[str, Any]) -> str:
    ctx = rq5_context(snapshot)
    return ctx["rules"] + "\n\n" + ctx["case"]


def rq5_questions(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        question_id(cell, cls): {
            "type": "choice",
            "instructions": f"Choose the class-priority action for cell {cell}, class {cls}.",
            "criteria": dict(RQ5_CRITERIA),
        }
        for cell, cls in expected_pairs(snapshot)
    }


def response_schema(snapshot: dict[str, Any]) -> dict[str, Any]:
    n = len(expected_pairs(snapshot))
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {
                    "type": "object",
                    "properties": {
                        "cell": {"type": "integer"},
                        "class": {"type": "string", "enum": list(RQ5_CLASSES)},
                        "action": {"type": "string", "enum": list(RQ5_ACTIONS)},
                    },
                    "required": ["cell", "class", "action"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["actions"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class Rq5Result:
    actions: list[dict[str, Any]]
    invalid_fields: int
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalise_actions(snapshot: dict[str, Any], returned: Any, error_type: str | None = None) -> Rq5Result:
    """Map every invalid/missing expected field to hold and count it."""
    pairs = expected_pairs(snapshot)
    raw = returned.get("actions") if isinstance(returned, dict) else returned
    raw = raw if isinstance(raw, list) else []
    by_pair: dict[tuple[int, str], str] = {}
    invalid_pairs: set[tuple[int, str]] = set()
    extra_errors = 0
    for item in raw:
        if not isinstance(item, dict):
            extra_errors += 1
            continue
        pair = (item.get("cell"), item.get("class"))
        action = item.get("action")
        if pair not in pairs:
            extra_errors += 1
            continue
        if pair in by_pair or action not in RQ5_ACTIONS:
            invalid_pairs.add(pair)
            by_pair.pop(pair, None)
            continue
        by_pair[pair] = action
    actions = []
    invalid = extra_errors
    for cell, cls in pairs:
        pair = (cell, cls)
        action = None if pair in invalid_pairs else by_pair.get(pair)
        if action is None:
            action = "hold"
            invalid += 1
        actions.append({"cell": cell, "class": cls, "action": action})
    return Rq5Result(actions=actions, invalid_fields=invalid, error_type=error_type)


def canonical_snapshot(snapshot: dict[str, Any]) -> str:
    """Stable replay comparison representation."""
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
