"""Client for OpenRouter chat-completion LLM baselines with strict JSON schema."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import jsonschema

from src.edgebench.contract import Case, DEFAULT_CRITERIA, READING_RULES, SERVICE_PRECEDENCE_RULE
from src.edgebench.interpreters.base import Decision, Interpreter, usage_float, usage_int
from src.edgebench.interpreters.transport import make_request


class ChatJsonClient(Interpreter):
    def __init__(
        self,
        name: str,
        model: str,
        provider_slug: str,
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
        self.accepted_resolved_models = (
            [model] if accepted_resolved_models is None else list(accepted_resolved_models)
        )
        self.base_url = base_url.rstrip("/")
        self.path = path
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.reasoning = {"enabled": False} if reasoning is None else dict(reasoning)

    def build_schema(self, case: Case) -> dict[str, Any]:
        single_req_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                field: {
                    "type": "string",
                    "enum": list(
                        (
                            case.service_options
                            if field == "service_type" and case.service_options is not None
                            else DEFAULT_CRITERIA[field]
                        ).keys()
                    ),
                }
                for field in case.fields
            },
            "required": list(case.fields),
            "additionalProperties": False,
        }

        if case.bundle_size == 1:
            return single_req_schema

        return {
            "type": "object",
            "properties": {
                "requests": {
                    "type": "array",
                    "items": single_req_schema,
                    "minItems": case.bundle_size,
                    "maxItems": case.bundle_size,
                }
            },
            "required": ["requests"],
            "additionalProperties": False,
        }

    def build_system_prompt(self, case: Case) -> str:
        policy = (
            "Extract the requested fields from the requirements stated in the text. "
            + READING_RULES
            + " Return only compact JSON with the requested fields, using the semantic values below. No explanation.\n"
        )
        for field in case.fields:
            is_cat = field == "service_type" and case.service_options is not None
            opts = (
                case.service_options
                if is_cat
                else DEFAULT_CRITERIA[field]
            )
            policy += f"{field}: " + "; ".join(f"{opt}={desc}" for opt, desc in opts.items()) + "\n"
            if is_cat:
                policy += f"Instruction for {field}: {SERVICE_PRECEDENCE_RULE}\n"

        if case.bundle_size > 1:
            policy += (
                f"\nThe text contains {case.bundle_size} numbered requests. Return a JSON object with key "
                f"'requests' containing an array of exactly {case.bundle_size} objects, one for each numbered request in order.\n"
            )
        return policy

    def build_request(self, case: Case) -> dict[str, Any]:
        schema = self.build_schema(case)
        system_prompt = self.build_system_prompt(case)
        if self.reasoning.get("effort") is not None:
            max_tokens = 512 + 24 * len(case.fields) * case.bundle_size
        else:
            max_tokens = 64 + 24 * len(case.fields) * case.bundle_size

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": case.text},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "reasoning": self.reasoning,
            "stream": False,
            "provider": {
                "only": [self.provider_slug],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            },
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "service_intent",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
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

        default_labels = [{f: "unspecified" for f in case.fields} for _ in range(case.bundle_size)]

        if err_type is not None or status != 200:
            return Decision(
                labels=default_labels,
                valid=False,
                error_type=err_type or f"HTTP_{status}",
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
                labels=default_labels,
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

        valid = True
        error_type: str | None = None

        # Check prefix match on resolved model
        if self.accepted_resolved_models:
            matched = any(
                resolved_model.startswith(prefix) for prefix in self.accepted_resolved_models
            )
            if not matched:
                valid = False
                error_type = "model_mismatch"

        # Missing usage / cost stays None (unknown), never 0.
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        input_tokens = usage_int(usage, "prompt_tokens", "input_tokens")
        completion_tokens_raw = usage_int(usage, "completion_tokens", "output_tokens")
        output_tokens = completion_tokens_raw
        cost_usd = usage_float(usage, "cost")
        reasoning_tokens = usage_int(usage.get("completion_tokens_details") or {}, "reasoning_tokens")

        # Check reasoning tokens validity: if reasoning explicitly disabled, must be 0. A missing field is
        # not verifiable: flagged (and counted in integrity.json), the row is still scored.
        reasoning_tokens_missing = False
        if self.reasoning.get("enabled") is False:
            if reasoning_tokens is None:
                reasoning_tokens_missing = True
            elif reasoning_tokens != 0:
                valid = False
                error_type = error_type or "reasoning_tokens_nonzero"

        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            valid = False
            error_type = error_type or "no_choices"
            return Decision(
                labels=default_labels,
                valid=valid,
                error_type=error_type,
                http_status=status,
                latency_s=latency_s,
                t_send_wall=t_send_wall,
                t_recv_wall=t_recv_wall,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                completion_tokens_raw=completion_tokens_raw,
                reasoning_tokens_missing=reasoning_tokens_missing,
                cost_usd=cost_usd,
                resolved_model=resolved_model,
                provider=provider,
                raw_response=raw_resp,
                raw_response_sha256=resp_sha256,
            )

        first_choice = choices[0]
        finish_reason = first_choice.get("finish_reason")
        if finish_reason != "stop":
            valid = False
            error_type = error_type or f"finish_reason_{finish_reason}"
        if body.get("reasoning_leak") is True:
            valid = False
            error_type = error_type or "reasoning_leak"

        message = first_choice.get("message") or {}
        content = message.get("content", "")

        schema = self.build_schema(case)
        try:
            parsed_content = json.loads(content)
        except Exception:
            valid = False
            error_type = error_type or "bad_json"
            return Decision(
                labels=default_labels,
                valid=valid,
                error_type=error_type,
                http_status=status,
                latency_s=latency_s,
                t_send_wall=t_send_wall,
                t_recv_wall=t_recv_wall,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                completion_tokens_raw=completion_tokens_raw,
                reasoning_tokens_missing=reasoning_tokens_missing,
                cost_usd=cost_usd,
                resolved_model=resolved_model,
                provider=provider,
                raw_response=raw_resp,
                raw_response_sha256=resp_sha256,
            )

        try:
            jsonschema.validate(instance=parsed_content, schema=schema)
        except jsonschema.ValidationError:
            valid = False
            error_type = error_type or "schema_violation"

        if valid:
            if case.bundle_size == 1:
                req_list = [parsed_content]
            else:
                req_list = parsed_content["requests"]
            labels = [
                {field: req_obj[field] for field in case.fields}
                for req_obj in req_list
            ]
        else:
            # For schema violation, extract best-effort labels if possible
            labels = default_labels
            if isinstance(parsed_content, dict):
                if case.bundle_size == 1 and not (
                    "requests" in parsed_content and isinstance(parsed_content["requests"], list)
                ):
                    labels = [{
                        field: parsed_content.get(field, "unspecified")
                        if isinstance(parsed_content.get(field), str)
                        else "unspecified"
                        for field in case.fields
                    }]
                elif case.bundle_size > 1 and isinstance(parsed_content.get("requests"), list):
                    labels = []
                    for req_obj in parsed_content["requests"][:case.bundle_size]:
                        if isinstance(req_obj, dict):
                            labels.append({
                                field: req_obj.get(field, "unspecified")
                                if isinstance(req_obj.get(field), str)
                                else "unspecified"
                                for field in case.fields
                            })
                        else:
                            labels.append({f: "unspecified" for f in case.fields})
                    while len(labels) < case.bundle_size:
                        labels.append({f: "unspecified" for f in case.fields})

        return Decision(
            labels=labels,
            valid=valid,
            error_type=error_type,
            http_status=status,
            latency_s=latency_s,
            t_send_wall=t_send_wall,
            t_recv_wall=t_recv_wall,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            completion_tokens_raw=completion_tokens_raw,
            reasoning_tokens_missing=reasoning_tokens_missing,
            cost_usd=cost_usd,
            resolved_model=resolved_model,
            provider=provider,
            raw_response=raw_resp,
            raw_response_sha256=resp_sha256,
        )
