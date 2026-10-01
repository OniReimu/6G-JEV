"""Positive-control Oracle interpreter returning exact case truth."""
from __future__ import annotations

import hashlib
import time

from src.edgebench.contract import Case
from src.edgebench.interpreters.base import Decision, Interpreter


class OracleInterpreter(Interpreter):
    def __init__(self, name: str = "Oracle") -> None:
        super().__init__(name=name, deployment="reference")

    def decide(self, case: Case) -> Decision:
        now = time.time()
        return Decision(
            labels=[dict(row) for row in case.truth],
            valid=True,
            latency_s=0.0,
            t_send_wall=now,
            t_recv_wall=now,
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            cost_usd=0.0,
            resolved_model="reference:oracle",
            provider="local",
            raw_response="oracle",
            raw_response_sha256=hashlib.sha256(b"oracle").hexdigest(),
        )
