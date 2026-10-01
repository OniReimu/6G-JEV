"""Edgebench contract v2: ordered fields, options, descriptions, and Case dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

# Core 4 fields and criteria descriptions taken verbatim from src/openrouter_decisions.CRITERIA and src/schema.FIELDS
CORE_FIELDS: list[str] = ["service_type", "locality", "quality_floor", "urgency"]

DEFAULT_CRITERIA: dict[str, dict[str, str]] = {
    "service_type": {
        "count": "count objects: return only a number",
        "detection": "detect and localize objects with bounding boxes",
        "ocr": "OCR: read text from an image",
        "unsupported": "unsupported service",
    },
    "locality": {
        "site_only": "data must stay at the originating site",
        "remote_allowed": "remote processing explicitly allowed",
        "unspecified": "locality is not specified",
    },
    "quality_floor": {
        "standard": "standard quality explicitly sufficient",
        "high": "high quality explicitly required",
        "unspecified": "quality is not specified",
    },
    "urgency": {
        "normal": "normal priority explicitly requested, including not urgent",
        "urgent": "urgent or priority processing explicitly requested",
        "unspecified": "urgency is not specified",
    },
    "retention": {
        "discard_after_use": "data must be discarded immediately after use",
        "retain_allowed": "data retention or storage explicitly permitted",
        "unspecified": "retention is not specified",
    },
    "energy": {
        "eco": "energy saving or eco-mode explicitly preferred",
        "performance": "maximum performance or low-latency mode explicitly required",
        "unspecified": "energy profile is not specified",
    },
    "redundancy": {
        "single": "single instance or non-redundant execution sufficient",
        "replicated": "replicated execution or high availability explicitly required",
        "unspecified": "redundancy is not specified",
    },
    "latency_class": {
        "realtime": "hard real-time processing deadline explicitly required",
        "interactive": "interactive processing latency explicitly requested",
        "batch": "background or batch processing acceptable",
        "unspecified": "latency class is not specified",
    },
}

FIELD_SETS: dict[int, list[str]] = {
    4: list(CORE_FIELDS),
    6: CORE_FIELDS + ["retention", "energy"],
    8: CORE_FIELDS + ["retention", "energy", "redundancy", "latency_class"],
}

DEFAULT_FIELD_OPTIONS: dict[str, list[str]] = {
    f: list(DEFAULT_CRITERIA[f].keys()) for f in DEFAULT_CRITERIA
}

MAX_SERVICE_OPTIONS = 255


def validate_service_options(service_options: dict[str, str] | None) -> None:
    """Validate catalog service options."""
    if service_options is None:
        return
    if "unsupported" not in service_options:
        raise ValueError("service_options must contain 'unsupported' option")
    if len(service_options) > MAX_SERVICE_OPTIONS:
        raise ValueError(
            f"service_options cannot exceed {MAX_SERVICE_OPTIONS} options (got {len(service_options)})"
        )


SERVICE_PRECEDENCE_RULE = (
    "Choose the most specific catalog service that matches the request. "
    "Choose a general-purpose service (count, detection, ocr) only if no more specific service matches. "
    "Choose unsupported only if no catalog service, including the general-purpose ones, matches."
)


READING_RULES = (
    "Count a requirement if it is stated directly, indirectly or through negation; "
    "if it is revised, use the final value. "
    "Ignore requirements for something else or explicitly not wanted. "
    "Never infer from the setting or fill in defaults."
)


def get_field_instruction(
    field: str,
    request_index: int | None = None,
    bundle_size: int = 1,
    is_catalog: bool = False,
) -> str:
    """Shared per-field instruction string for decision queries."""
    if bundle_size > 1 and request_index is not None:
        base = (
            f"Extract {field} from the requirements stated for request number {request_index} in the text. "
            f"{READING_RULES}"
        )
    else:
        base = f"Extract {field} from the requirements stated in the text. {READING_RULES}"
    if field == "service_type" and is_catalog:
        return f"{base} {SERVICE_PRECEDENCE_RULE}"
    return base


@dataclass
class Case:
    case_id: str
    text: str
    fields: list[str] = field(default_factory=lambda: list(FIELD_SETS[4]))
    service_options: dict[str, str] | None = None
    bundle_size: int = 1
    truth: list[dict[str, str]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bundle_size < 1:
            raise ValueError(f"bundle_size must be >= 1, got {self.bundle_size}")
        if self.service_options is not None:
            validate_service_options(self.service_options)
        if len(self.truth) != self.bundle_size:
            raise ValueError(
                f"truth length ({len(self.truth)}) must equal bundle_size ({self.bundle_size})"
            )
        if isinstance(self.fields, tuple):
            self.fields = list(self.fields)

    def to_dict(self) -> dict[str, Any]:
        return case_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Case:
        return case_from_dict(data)


def case_to_dict(case: Case) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "text": case.text,
        "fields": list(case.fields),
        "service_options": case.service_options,
        "bundle_size": case.bundle_size,
        "truth": case.truth,
        "meta": case.meta,
    }


def case_from_dict(d: dict[str, Any]) -> Case:
    bundle_size = d.get("bundle_size", 1)
    truth = d.get("truth", [])
    if isinstance(truth, dict):
        truth = [truth]
        bundle_size = 1
    elif not truth and bundle_size == 1:
        # If truth wasn't provided, initialize empty dict for k=1
        truth = [{}]

    fields = d.get("fields", FIELD_SETS[4])
    if isinstance(fields, int):
        fields = FIELD_SETS[fields]

    service_options = d.get("service_options")
    if service_options is not None:
        validate_service_options(service_options)

    return Case(
        case_id=str(d.get("case_id", d.get("id", ""))),
        text=str(d.get("text", "")),
        fields=list(fields),
        service_options=service_options,
        bundle_size=bundle_size,
        truth=truth,
        meta=dict(d.get("meta", {})),
    )


def load_cases_jsonl(path: str | Path) -> list[Case]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                cases.append(case_from_dict(data))
            except Exception as exc:
                raise ValueError(f"Error parsing case at line {line_no} in {path}: {exc}") from exc
    return cases


def dump_cases_jsonl(cases: list[Case], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case.to_dict()) + "\n")
