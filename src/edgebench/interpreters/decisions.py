"""Client for System-One style decision APIs (Jev, SemIf, Laya)."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from src.edgebench.contract import Case, DEFAULT_CRITERIA, get_field_instruction
from src.edgebench.interpreters.base import Decision, Interpreter, usage_float, usage_int
from src.edgebench.interpreters.transport import make_request


class DecisionsClient(Interpreter):
    def __init__(
        self,
        name: str = "Jev-1.13.0",
        model: str = "typesafe/jev-1.13",
        accepted_resolved_models: set[str] | list[str] | None = None,
        expected_provider: str | None = "TypeSafe",
        base_url: str = "https://openrouter.ai",
        path: str = "/api/alpha/decisions",
        api_key: str | None = None,
        deployment: str = "hosted",
        timeout_s: float = 30.0,
    ) -> None:
        super().__init__(name=name, deployment=deployment)
        self.model = model
        self.accepted_resolved_models = (
            frozenset(accepted_resolved_models)
            if accepted_resolved_models is not None
            else frozenset({model, f"{model}-20260917"})
        )
        self.expected_provider = expected_provider
        self.base_url = base_url.rstrip("/")
        self.path = path
        self.api_key = api_key
        self.timeout_s = timeout_s

    def build_request(self, case: Case) -> dict[str, Any]:
        questions: dict[str, dict[str, Any]] = {}
        for req_idx in range(1, case.bundle_size + 1):
            for field in case.fields:
                qid = f"r{req_idx}__{field}"
                if field == "service_type" and case.service_options is not None:
                    criteria = case.service_options
                else:
                    criteria = DEFAULT_CRITERIA[field]
                is_catalog = field == "service_type" and case.service_options is not None
                instructions = get_field_instruction(
                    field,
                    request_index=req_idx,
                    bundle_size=case.bundle_size,
                    is_catalog=is_catalog,
                )
                questions[qid] = {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": dict(criteria),
                }

        payload: dict[str, Any] = {
            "model": self.model,
            "state": case.text,
            "questions": questions,
        }
        if "openrouter.ai" in self.base_url:
            payload["provider"] = {"allow_fallbacks": False}
        return payload

    def decide(self, case: Case) -> Decision:
        payload = self.build_request(case)
        status, raw_resp, latency_s, t_send_wall, t_recv_wall, err_type, _ = make_request(
            base_url=self.base_url,
            path=self.path,
            payload=payload,
            api_key=self.api_key,
            timeout_s=self.timeout_s,
        )

        resp_sha256 = (
            hashlib.sha256(raw_resp.encode("utf-8")).hexdigest() if raw_resp else ""
        )

        if err_type is not None or status != 200:
            error_code = err_type or f"HTTP_{status}"
            return Decision(
                labels=[{f: "unspecified" for f in case.fields} for _ in range(case.bundle_size)],
                valid=False,
                error_type=error_code,
                http_status=status,
                latency_s=latency_s,
                t_send_wall=t_send_wall,
                t_recv_wall=t_recv_wall,
                raw_response=raw_resp,
                raw_response_sha256=resp_sha256,
            )

        try:
            body = json.loads(raw_resp)
            if not isinstance(body, dict):
                raise ValueError("Expected JSON object")
        except Exception:
            return Decision(
                labels=[{f: "unspecified" for f in case.fields} for _ in range(case.bundle_size)],
                valid=False,
                error_type="json_parse_error",
                http_status=status,
                latency_s=latency_s,
                t_send_wall=t_send_wall,
                t_recv_wall=t_recv_wall,
                raw_response=raw_resp,
                raw_response_sha256=resp_sha256,
            )

        resolved_model = str(body.get("model", ""))
        provider = str(body.get("provider", ""))

        # Missing usage / cost stays None (unknown); every branch below keeps what was reported.
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        input_tokens = usage_int(usage, "input_tokens", "prompt_tokens")
        output_tokens = usage_int(usage, "output_tokens", "completion_tokens")
        reasoning_tokens = usage_int(usage.get("completion_tokens_details") or {}, "reasoning_tokens")
        cost_usd = usage_float(usage, "cost")

        # Check resolved model mismatch
        if (
            self.accepted_resolved_models
            and resolved_model not in self.accepted_resolved_models
        ):
            return Decision(
                labels=[{f: "unspecified" for f in case.fields} for _ in range(case.bundle_size)],
                valid=False,
                error_type="model_mismatch",
                http_status=status,
                latency_s=latency_s,
                t_send_wall=t_send_wall,
                t_recv_wall=t_recv_wall,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                cost_usd=cost_usd,
                resolved_model=resolved_model,
                provider=provider,
                raw_response=raw_resp,
                raw_response_sha256=resp_sha256,
            )

        answers = body.get("answers", {})
        if not isinstance(answers, dict):
            return Decision(
                labels=[{f: "unspecified" for f in case.fields} for _ in range(case.bundle_size)],
                valid=False,
                error_type="missing_answers",
                http_status=status,
                latency_s=latency_s,
                t_send_wall=t_send_wall,
                t_recv_wall=t_recv_wall,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                cost_usd=cost_usd,
                resolved_model=resolved_model,
                provider=provider,
                raw_response=raw_resp,
                raw_response_sha256=resp_sha256,
            )

        labels: list[dict[str, str]] = []
        probabilities: list[dict[str, dict[str, float]]] = []
        conf_scores: list[float] = []
        top_conf = body.get("confidence")
        if isinstance(top_conf, (int, float)):
            conf_scores.append(float(top_conf))

        valid = True
        error_type: str | None = None

        for i in range(case.bundle_size):
            req_idx = i + 1
            req_labels: dict[str, str] = {}
            req_probs: dict[str, dict[str, float]] = {}

            for field in case.fields:
                qid = f"r{req_idx}__{field}"
                # Fallback to r{i}__{field} or field if single request
                if qid not in answers:
                    if f"r{i}__{field}" in answers:
                        qid = f"r{i}__{field}"
                    elif case.bundle_size == 1 and field in answers:
                        qid = field
                    else:
                        valid = False
                        error_type = error_type or "missing_answers"
                        req_labels[field] = "unspecified"
                        continue

                ans = answers.get(qid)
                if not isinstance(ans, dict) or ans.get("type") != "choice":
                    valid = False
                    error_type = error_type or "malformed_choice"
                    req_labels[field] = "unspecified"
                    continue

                choice = ans.get("choice")
                allowed_opts = (
                    case.service_options
                    if (field == "service_type" and case.service_options is not None)
                    else DEFAULT_CRITERIA[field]
                )
                if not isinstance(choice, str) or choice not in allowed_opts:
                    valid = False
                    error_type = error_type or "invalid_option"
                    req_labels[field] = choice if isinstance(choice, str) else "unspecified"
                else:
                    req_labels[field] = choice

                if "probabilities" in ans and isinstance(ans["probabilities"], dict):
                    req_probs[field] = ans["probabilities"]
                if "confidence" in ans and isinstance(ans["confidence"], (int, float)):
                    conf_scores.append(float(ans["confidence"]))

            labels.append(req_labels)
            probabilities.append(req_probs)

        confidence_val = (
            float(sum(conf_scores) / len(conf_scores)) if conf_scores else None
        )

        return Decision(
            labels=labels,
            probabilities=probabilities if any(probabilities) else None,
            confidence=confidence_val,
            valid=valid,
            error_type=error_type,
            http_status=status,
            latency_s=latency_s,
            t_send_wall=t_send_wall,
            t_recv_wall=t_recv_wall,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=cost_usd,
            resolved_model=resolved_model,
            provider=provider,
            raw_response=raw_resp,
            raw_response_sha256=resp_sha256,
        )
