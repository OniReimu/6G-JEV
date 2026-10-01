"""Base classes and Decision dataclass for Edgebench interpreters."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any

from src.edgebench.contract import Case


@dataclass
class Decision:
    labels: list[dict[str, str]] = field(default_factory=list)
    probabilities: list[dict[str, dict[str, float]]] | None = None
    confidence: float | None = None
    valid: bool = True
    error_type: str | None = None
    http_status: int | None = None
    latency_s: float | None = 0.0
    t_send_wall: float = 0.0
    t_recv_wall: float = 0.0
    # Usage / cost: None means "not reported" (unknown), never a silent 0.
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    completion_tokens_raw: int | None = None
    cost_usd: float | None = None
    resolved_model: str = ""
    provider: str = ""
    raw_response: str = ""
    raw_response_sha256: str = ""
    # reasoning disabled but the provider did not report reasoning_tokens: validity not verifiable
    reasoning_tokens_missing: bool = False
    # untimed wait for the self-hosted server to drain an abandoned (timed-out) request
    post_timeout_wait_s: float | None = None

    def __post_init__(self) -> None:
        if self.raw_response and not self.raw_response_sha256:
            self.raw_response_sha256 = hashlib.sha256(
                self.raw_response.encode("utf-8")
            ).hexdigest()
        if self.completion_tokens_raw is None and self.output_tokens is not None:
            self.completion_tokens_raw = self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def usage_int(usage: dict[str, Any], *keys: str) -> int | None:
    """First of `keys` present in `usage` with a non-null value, as int; None when none is reported."""
    for key in keys:
        if usage.get(key) is not None:
            return int(usage[key])
    return None


def usage_float(usage: dict[str, Any], key: str) -> float | None:
    return float(usage[key]) if usage.get(key) is not None else None


class Interpreter:
    platform: str | None = None  # set from the manifest: "M4-Max" | "H100-NVL" | "hosted"

    def __init__(self, name: str, deployment: str) -> None:
        if deployment not in ("hosted", "self-hosted", "reference"):
            raise ValueError(f"Unknown deployment mode: {deployment}")
        self.name = name
        self.deployment = deployment

    def decide(self, case: Case) -> Decision:
        raise NotImplementedError
