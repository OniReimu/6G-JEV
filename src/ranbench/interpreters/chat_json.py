"""Chat-completion client emitting one JSON policy object (hosted Flash LLMs on OpenRouter; Qwen3.5-4B-JSON on vLLM).

System message = the reading rules, user message = the case, both from build_interpreter_context. Structured output
via `response_format: json_schema` (strict) carrying the policy schema: OpenRouter with `require_parameters: true`
(only providers that honour response_format are routed to) and vLLM's OpenAI-compatible server (JSON-schema
constrained decoding). Same checks as src/edgebench/interpreters/chat_json.py; the output is validated against the
policy type and never repaired.
"""
from __future__ import annotations

import json
from typing import Any

from src.edgebench.interpreters.base import Decision, Interpreter, usage_float, usage_int
from src.edgebench.interpreters.transport import make_request
from src.ranbench.interpreters.common import FIELDS, RanCase, check_policy, response_schema


class RanChatJsonClient(Interpreter):
    def __init__(
        self,
        name: str,
        model: str,
        provider_slug: str | None = None,
        accepted_resolved_models: set[str] | list[str] | None = None,
        base_url: str = "https://openrouter.ai",
        path: str = "/api/v1/chat/completions",
        api_key: str | None = None,
        deployment: str = "hosted",
        timeout_s: float = 30.0,
        reasoning: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, deployment=deployment)
        self.model = model
        self.provider_slug = provider_slug
        self.accepted_resolved_models = [model] if accepted_resolved_models is None else list(accepted_resolved_models)
        self.base_url = base_url.rstrip("/")
        self.path = path
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.openrouter = "openrouter.ai" in self.base_url
        if self.openrouter and not provider_slug:
            raise ValueError(f"{name}: an OpenRouter model needs a pinned provider")
        # OpenRouter: reasoning off unless configured (GLM-5.3-Flash: effort minimal). vLLM: thinking off in the template.
        self.reasoning = ({"enabled": False} if reasoning is None else dict(reasoning)) if self.openrouter else None

    def _max_tokens(self) -> int:
        if self.reasoning and self.reasoning.get("effort") is not None:
            return 512 + 24 * len(FIELDS)
        return 64 + 24 * len(FIELDS)

    def _payload(self, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": False,
        }
        if self.openrouter:
            payload["reasoning"] = self.reasoning
            payload["provider"] = {
                "only": [self.provider_slug],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            }
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def build_request(self, case: RanCase) -> dict[str, Any]:
        ctx = case.context
        payload = self._payload(
            [{"role": "system", "content": ctx["rules"]}, {"role": "user", "content": ctx["case"]}], self._max_tokens()
        )
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "ranintent_v1_policy", "strict": True, "schema": response_schema()},
        }
        return payload

    def build_probe_request(self) -> dict[str, Any]:
        """Minimal request for the no-op RTT probe: same model, provider pin and reasoning setting, tiny prompt."""
        max_tokens = 16 if self.reasoning and self.reasoning.get("effort") is not None else 1
        return self._payload([{"role": "user", "content": "Reply with 1."}], max_tokens)

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
        completion = usage_int(usage, "completion_tokens", "output_tokens")
        reasoning_tokens = usage_int(usage.get("completion_tokens_details") or {}, "reasoning_tokens")
        meta = dict(
            input_tokens=usage_int(usage, "prompt_tokens", "input_tokens"),
            output_tokens=completion,
            completion_tokens_raw=completion,
            reasoning_tokens=reasoning_tokens,
            cost_usd=usage_float(usage, "cost"),
            resolved_model=str(body.get("model", "")),
            provider=str(body.get("provider", "")),
        )
        error_type: str | None = None
        if self.accepted_resolved_models and not any(
            meta["resolved_model"].startswith(p) for p in self.accepted_resolved_models
        ):
            error_type = "model_mismatch"
        reasoning_missing = False
        if self.reasoning is not None and self.reasoning.get("enabled") is False:
            if reasoning_tokens is None:
                reasoning_missing = True
            elif reasoning_tokens != 0:
                error_type = error_type or "reasoning_tokens_nonzero"
        meta["reasoning_tokens_missing"] = reasoning_missing

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return Decision(labels=[], valid=False, error_type=error_type or "no_choices", **timing, **meta)
        first = choices[0]
        finish_reason = first.get("finish_reason")
        if finish_reason != "stop":
            error_type = error_type or f"finish_reason_{finish_reason}"
        if body.get("reasoning_leak") is True:
            error_type = error_type or "reasoning_leak"
        content = (first.get("message") or {}).get("content") or ""
        try:
            parsed = json.loads(content)
        except Exception:
            return Decision(labels=[], valid=False, error_type=error_type or "bad_json", **timing, **meta)
        labels, valid, schema_error = check_policy(parsed)
        return Decision(
            labels=labels, valid=valid and error_type is None, error_type=error_type or schema_error, **timing, **meta
        )
