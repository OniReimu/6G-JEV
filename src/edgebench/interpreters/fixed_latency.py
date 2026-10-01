"""Fixed-latency reference interpreter returning exact truth after a specified delay."""
from __future__ import annotations

import hashlib
import time

from src.edgebench.contract import Case
from src.edgebench.interpreters.base import Decision, Interpreter


class FixedLatencyInterpreter(Interpreter):
    def __init__(self, name: str = "fixed-latency", delay_s: float = 0.6) -> None:
        super().__init__(name=name, deployment="reference")
        self.delay_s = float(delay_s)

    def decide(self, case: Case) -> Decision:
        t_send = time.time()
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        t_recv = time.time()
        raw = f"fixed-latency-{self.delay_s}"
        return Decision(
            labels=[dict(row) for row in case.truth],
            valid=True,
            latency_s=self.delay_s,
            t_send_wall=t_send,
            t_recv_wall=t_recv,
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            cost_usd=0.0,
            resolved_model=f"reference:{raw}",
            provider="local",
            raw_response=raw,
            raw_response_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )
