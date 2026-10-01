"""Reference interpreter wrapping src/rule_parser.py."""
from __future__ import annotations

import hashlib
import re
import time

from src.edgebench.contract import Case
from src.edgebench.interpreters.base import Decision, Interpreter
from src.rule_parser import parse as parse_rule


class RuleInterpreter(Interpreter):
    def __init__(self, name: str = "Rule") -> None:
        super().__init__(name=name, deployment="reference")

    def _split_text(self, text: str, k: int) -> list[str]:
        if k == 1:
            return [text]
        # Match "1. ... 2. ..." or "1) ... 2) ..."
        parts = re.split(r"(?:^|\n)\s*\d+[\.\)]\s*", text.strip())
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) == k:
            return parts
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) == k:
            return lines
        return [text] * k

    def decide(self, case: Case) -> Decision:
        t_send_wall = time.time()
        t0 = time.perf_counter()

        text_segments = self._split_text(case.text, case.bundle_size)
        labels: list[dict[str, str]] = []

        for seg in text_segments:
            parsed = parse_rule(seg)
            req_labels: dict[str, str] = {}
            for field in case.fields:
                if field == "service_type":
                    raw_service = parsed.get("service_type", "unsupported")
                    if case.service_options is not None:
                        if raw_service in case.service_options:
                            req_labels[field] = raw_service
                        else:
                            req_labels[field] = "unsupported"
                    else:
                        req_labels[field] = raw_service
                elif field in ("locality", "quality_floor", "urgency"):
                    req_labels[field] = parsed.get(field, "unspecified")
                else:
                    req_labels[field] = "unspecified"
            labels.append(req_labels)

        latency_s = time.perf_counter() - t0
        t_recv_wall = time.time()

        return Decision(
            labels=labels,
            valid=True,
            latency_s=latency_s,
            t_send_wall=t_send_wall,
            t_recv_wall=t_recv_wall,
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            cost_usd=0.0,
            resolved_model="reference:rule",
            provider="local",
            raw_response="rule_parser",
            raw_response_sha256=hashlib.sha256(b"rule_parser").hexdigest(),
        )
