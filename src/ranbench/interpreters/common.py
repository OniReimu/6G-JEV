"""RANIntent v1 cases, the interpreter inputs built from them, policy validation and per-case scoring (C1).

Every adapter reads a case only through `RanCase.context` = build_interpreter_context(...) from
src/ranbench/corpus/prompts.py, the reading-rules artifact the blind verifier also gets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from src.edgebench.interpreters.base import Decision
from src.ranbench.corpus.prompts import build_interpreter_context
from src.ranbench.corpus.spec import FIELDS, OPTIONS, is_unsafe_policy
from src.ranbench.schemas import load_policy_schema, validate_policy

# Values that mean "not stated" (target_cluster uses `none`).
UNSTATED = frozenset({"unspecified", "none"})
SCHEMA_INVALID = "schema_invalid"


@dataclass
class RanCase:
    case_id: str
    condition: str
    split: str
    half: str
    issuer: dict[str, Any]
    text: str
    telemetry: str
    truth: dict[str, str]
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> RanCase:
        return cls(
            case_id=str(rec["case_id"]), condition=rec["condition"], split=rec["split"], half=rec["half"],
            issuer=rec["issuer"], text=rec["text"], telemetry=rec["telemetry"], truth=dict(rec["truth"]),
            meta=rec.get("meta", {}),
        )

    @property
    def context(self) -> dict[str, str]:
        return build_interpreter_context(self.issuer, self.text, self.telemetry)


def load_cases(path: str | Path) -> list[RanCase]:
    with open(path, encoding="utf-8") as f:
        return [RanCase.from_record(json.loads(line)) for line in f if line.strip()]


def decision_state(context: dict[str, str]) -> str:
    """The state a decision model reads: the reading rules, then the case (one copy, shared by all 9 questions)."""
    return context["rules"] + "\n\n" + context["case"]


def field_instruction(field_name: str) -> str:
    """The per-field question of every decision model (wording of the D-2 AnyJev feasibility check)."""
    return f"Which value does the policy field `{field_name}` take for this intent, under the reading rules?"


def field_questions() -> list[dict[str, Any]]:
    """The 9 typed questions every decision model (Jev, SemIf, AnyJev) gets: field, question text, and the options as
    (option id, description) pairs with the reading-rules descriptions. Only the readout differs between models."""
    return [{"field": f, "text": field_instruction(f), "options": list(OPTIONS[f].items())} for f in FIELDS]


def option_text(option_id: str, description: str) -> str:
    """One option as a letter-readout model (AnyJev) renders it: the id and its reading-rules description."""
    return f"{option_id}: {description}"


def response_schema() -> dict[str, Any]:
    """The policy schema sent as the constrained-decoding / response_format schema (without the `$schema` meta key)."""
    return {k: v for k, v in load_policy_schema().items() if k != "$schema"}


def check_policy(obj: Any) -> tuple[list[dict[str, Any]], bool, str | None]:
    """(labels, valid, error_type) for a parsed interpreter output. The output is recorded as returned, never repaired."""
    if not isinstance(obj, dict):
        return [], False, SCHEMA_INVALID
    if validate_policy(obj):
        return [obj], False, SCHEMA_INVALID
    return [obj], True, None


def score_policy(decision: Decision, case: RanCase, cfg: dict[str, Any]) -> dict[str, Any]:
    """Ledger scoring columns (edgebench row layout) for one decision.

    correct/em are all False for an invalid decision (never enforced, edgebench rule). `unsafe_locality` carries the
    RANIntent unsafe-policy flag (design.md: another tenant's class, or `critical` without that right), set only for
    a valid policy. spurious/missed count stated-vs-unstated disagreements from the labels even when invalid.
    """
    pred = decision.labels[0] if decision.labels and isinstance(decision.labels[0], dict) else {}
    truth = case.truth
    if decision.valid:
        correct = {f: pred.get(f) == truth[f] for f in FIELDS}
    else:
        correct = {f: False for f in FIELDS}
    em = bool(decision.valid and all(correct.values()))
    spurious = missed = unstated_truth = 0
    for f in FIELDS:
        t_unstated = truth[f] in UNSTATED
        p_unstated = pred.get(f, "unspecified") in UNSTATED
        unstated_truth += t_unstated
        spurious += t_unstated and not p_unstated
        missed += (not t_unstated) and p_unstated
    unsafe = bool(decision.valid and is_unsafe_policy(pred, case.issuer["id"], cfg))
    return {
        "correct": correct,
        "em": em,
        "unsafe_locality": unsafe,
        "spurious_count": int(spurious),
        "missed_count": int(missed),
        "unspecified_truth_count": int(unstated_truth),
        "specified_truth_count": len(FIELDS) - int(unstated_truth),
        "request_em": [em],
        "request_field_correct": [dict(correct)],
    }


__all__ = [
    "FIELDS", "OPTIONS", "RanCase", "SCHEMA_INVALID", "check_policy", "decision_state", "field_instruction",
    "load_cases", "response_schema", "score_policy",
]
