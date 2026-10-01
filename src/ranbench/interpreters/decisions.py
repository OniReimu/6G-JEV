"""Decision-API client for RANIntent v1 (Jev-1.13.0 via OpenRouter/TypeSafe; SemIf on its reference server).

One typed choice question per policy field over that field's option list; the state is the reading rules plus
the case. Same request/response shape and checks as src/edgebench/interpreters/decisions.py.
"""
from __future__ import annotations

import json
from typing import Any

from src.edgebench.interpreters.base import Decision, Interpreter, usage_float, usage_int
from src.edgebench.interpreters.transport import make_request
from src.ranbench.interpreters.common import (
    FIELDS,
    OPTIONS,
    RanCase,
    check_policy,
    decision_state,
    field_questions,
)


class RanDecisionsClient(Interpreter):
    def __init__(
        self,
        name: str,
        model: str,
        accepted_resolved_models: set[str] | list[str] | None = None,
        base_url: str = "https://openrouter.ai",
        path: str = "/api/alpha/decisions",
        api_key: str | None = None,
        deployment: str = "hosted",
        timeout_s: float = 30.0,
    ) -> None:
        super().__init__(name=name, deployment=deployment)
        self.model = model
        self.accepted_resolved_models = frozenset(accepted_resolved_models or {model})
        self.base_url = base_url.rstrip("/")
        self.path = path
        self.api_key = api_key
        self.timeout_s = timeout_s

    def _payload(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "state": state, "questions": questions}
        if "openrouter.ai" in self.base_url:
            payload["provider"] = {"allow_fallbacks": False}
        return payload

    def build_request(self, case: RanCase) -> dict[str, Any]:
        questions = {
            q["field"]: {"type": "choice", "instructions": q["text"], "criteria": dict(q["options"])}
            for q in field_questions()
        }
        return self._payload(decision_state(case.context), questions)

    def build_probe_request(self) -> dict[str, Any]:
        """Minimal request for the no-op RTT probe: a one-line state and one two-option question."""
        return self._payload("ping", {"q": {"type": "choice", "instructions": "Answer yes.",
                                             "criteria": {"yes": "yes", "no": "no"}}})

    def post(self, payload: dict[str, Any]) -> tuple[int | None, str, float, float, float, str | None]:
        status, raw, latency_s, t_send, t_recv, err, _ = make_request(
            base_url=self.base_url, path=self.path, payload=payload, api_key=self.api_key, timeout_s=self.timeout_s
        )
        return status, raw, latency_s, t_send, t_recv, err

    def decide(self, case: RanCase) -> Decision:
        status, raw, latency_s, t_send, t_recv, err = self.post(self.build_request(case))
        timing = dict(http_status=status, latency_s=latency_s, t_send_wall=t_send, t_recv_wall=t_recv, raw_response=raw)
        if err is not None or status != 200:
            return Decision(labels=[], valid=False, error_type=err or f"HTTP_{status}", **timing)
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("expected a JSON object")
        except Exception:
            return Decision(labels=[], valid=False, error_type="json_parse_error", **timing)

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        meta = dict(
            input_tokens=usage_int(usage, "input_tokens", "prompt_tokens"),
            output_tokens=usage_int(usage, "output_tokens", "completion_tokens"),
            reasoning_tokens=usage_int(usage.get("completion_tokens_details") or {}, "reasoning_tokens"),
            cost_usd=usage_float(usage, "cost"),
            resolved_model=str(body.get("model", "")),
            provider=str(body.get("provider", "")),
        )
        if meta["resolved_model"] not in self.accepted_resolved_models:
            return Decision(labels=[], valid=False, error_type="model_mismatch", **timing, **meta)
        answers = body.get("answers")
        if not isinstance(answers, dict):
            return Decision(labels=[], valid=False, error_type="missing_answers", **timing, **meta)

        policy: dict[str, str] = {}
        probabilities: dict[str, dict[str, float]] = {}
        confidences: list[float] = []
        error_type: str | None = None
        for f in FIELDS:
            ans = answers.get(f)
            if ans is None:
                error_type = error_type or "missing_answers"
                continue
            if not isinstance(ans, dict) or ans.get("type") != "choice":
                error_type = error_type or "malformed_choice"
                continue
            choice = ans.get("choice")
            if not isinstance(choice, str) or choice not in OPTIONS[f]:
                error_type = error_type or "invalid_option"
            if isinstance(choice, str):
                policy[f] = choice
            if isinstance(ans.get("probabilities"), dict):
                probabilities[f] = ans["probabilities"]
            if isinstance(ans.get("confidence"), (int, float)):
                confidences.append(float(ans["confidence"]))

        labels, valid, schema_error = check_policy(policy)
        return Decision(
            labels=labels,
            probabilities=[probabilities] if probabilities else None,
            confidence=sum(confidences) / len(confidences) if confidences else None,
            valid=valid and error_type is None,
            error_type=error_type or schema_error,
            **timing,
            **meta,
        )
