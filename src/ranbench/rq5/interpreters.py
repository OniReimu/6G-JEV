"""RQ5 question-family adapters preserving the C1 deployment and rendering settings."""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from src.edgebench.interpreters.base import usage_float, usage_int
from src.edgebench.interpreters.transport import (
    get_post_timeout_healthy,
    get_post_timeout_wait,
    get_retry_info,
    reset_send_marker,
)
from src.ranbench.interpreters.anyjev_l0 import LABEL_FALLBACK_ERROR, AnyJevClient
from src.ranbench.interpreters.chat_json import RanChatJsonClient
from src.ranbench.interpreters.common import option_text
from src.ranbench.interpreters.decisions import RanDecisionsClient
from src.ranbench.rq5.protocol import (
    RQ5_ACTIONS,
    RQ5_CRITERIA,
    RQ5_DEADLINE_S,
    Rq5Result,
    decision_state,
    expected_pairs,
    normalise_actions,
    question_id,
    response_schema,
    rq5_context,
    rq5_questions,
)

# One {"cell":N,"class":"...","action":"..."} item: measured 21 (DeepSeek-V4.1-Flash) to 31 (Qwen3.8-Flash)
# output tokens per row on an 84-row snapshot (sim6g/rq5-ns3/smoke/hosted-single-call-probe.log); C1 budgets
# 24 per policy field. 48 per row keeps the 21-cell x 4-class answer clear of finish_reason=length.
RQ5_TOKENS_PER_ROW = 48


def _transport_failure(snapshot: dict[str, Any], error: str, metadata: dict[str, Any]) -> Rq5Result:
    result = normalise_actions(snapshot, [], error)
    return Rq5Result(result.actions, result.invalid_fields, error, metadata)


def query_decisions(client: RanDecisionsClient, snapshot: dict[str, Any]) -> Rq5Result:
    payload = client._payload(decision_state(snapshot), rq5_questions(snapshot))
    status, raw, latency_s, t_send, t_recv, error = client.post(payload)
    meta: dict[str, Any] = {
        "http_status": status, "adapter_latency_s": latency_s, "t_send_wall": t_send,
        "t_recv_wall": t_recv, "raw_response": raw,
        "post_timeout_wait_s": get_post_timeout_wait(),
        "post_timeout_healthy": get_post_timeout_healthy(),
        "retried": get_retry_info()[0], "first_error": get_retry_info()[1],
        "failed_attempt_s": get_retry_info()[2],
    }
    if error is not None or status != 200:
        return _transport_failure(snapshot, error or f"HTTP_{status}", meta)
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return _transport_failure(snapshot, "json_parse_error", meta)
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    meta.update(
        input_tokens=usage_int(usage, "input_tokens", "prompt_tokens"),
        output_tokens=usage_int(usage, "output_tokens", "completion_tokens"),
        cost_usd=usage_float(usage, "cost"), resolved_model=str(body.get("model", "")),
        provider=str(body.get("provider", "")),
    )
    if meta["resolved_model"] not in client.accepted_resolved_models:
        return _transport_failure(snapshot, "model_mismatch", meta)
    answers = body.get("answers")
    if not isinstance(answers, dict):
        return _transport_failure(snapshot, "missing_answers", meta)
    returned = []
    for cell, cls in expected_pairs(snapshot):
        answer = answers.get(question_id(cell, cls))
        if isinstance(answer, dict) and answer.get("type") == "choice":
            returned.append({"cell": cell, "class": cls, "action": answer.get("choice")})
    result = normalise_actions(snapshot, returned)
    return Rq5Result(result.actions, result.invalid_fields, None if not result.invalid_fields else "invalid_fields", meta)


def query_chat_json(client: RanChatJsonClient, snapshot: dict[str, Any]) -> Rq5Result:
    ctx = rq5_context(snapshot)
    n = len(expected_pairs(snapshot))
    payload = client._payload(
        [{"role": "system", "content": ctx["rules"]}, {"role": "user", "content": ctx["case"]}],
        (512 if client.reasoning and client.reasoning.get("effort") is not None else 64) + RQ5_TOKENS_PER_ROW * n,
    )
    payload["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": "rq5_percell", "strict": True, "schema": response_schema(snapshot)},
    }
    status, raw, latency_s, t_send, t_recv, error = client.post(payload)
    meta: dict[str, Any] = {
        "http_status": status, "adapter_latency_s": latency_s, "t_send_wall": t_send,
        "t_recv_wall": t_recv, "raw_response": raw,
        "post_timeout_wait_s": get_post_timeout_wait(),
        "post_timeout_healthy": get_post_timeout_healthy(),
        "retried": get_retry_info()[0], "first_error": get_retry_info()[1],
        "failed_attempt_s": get_retry_info()[2],
    }
    if error is not None or status != 200:
        return _transport_failure(snapshot, error or f"HTTP_{status}", meta)
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return _transport_failure(snapshot, "json_parse_error", meta)
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    reasoning = usage_int(usage.get("completion_tokens_details") or {}, "reasoning_tokens")
    meta.update(
        input_tokens=usage_int(usage, "prompt_tokens", "input_tokens"),
        output_tokens=usage_int(usage, "completion_tokens", "output_tokens"),
        reasoning_tokens=reasoning, cost_usd=usage_float(usage, "cost"),
        resolved_model=str(body.get("model", "")), provider=str(body.get("provider", "")),
    )
    model_ok = not client.accepted_resolved_models or any(
        meta["resolved_model"].startswith(prefix) for prefix in client.accepted_resolved_models
    )
    if not model_ok:
        return _transport_failure(snapshot, "model_mismatch", meta)
    if client.reasoning is not None and client.reasoning.get("enabled") is False and reasoning not in (0, None):
        return _transport_failure(snapshot, "reasoning_tokens_nonzero", meta)
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return _transport_failure(snapshot, "no_choices", meta)
    first = choices[0]
    if first.get("finish_reason") != "stop":
        return _transport_failure(snapshot, f"finish_reason_{first.get('finish_reason')}", meta)
    if body.get("reasoning_leak") is True:
        return _transport_failure(snapshot, "reasoning_leak", meta)
    try:
        parsed = json.loads((first.get("message") or {}).get("content") or "")
    except json.JSONDecodeError:
        return _transport_failure(snapshot, "bad_json", meta)
    result = normalise_actions(snapshot, parsed)
    return Rq5Result(result.actions, result.invalid_fields, None if not result.invalid_fields else "invalid_fields", meta)


def query_anyjev(client: AnyJevClient, snapshot: dict[str, Any]) -> Rq5Result:
    # One-off client setup (tokenizer load) and the post-call token count are bookkeeping, not decision time:
    # they are reported as untimed_s and the controller removes them from the scheduling latency.
    t_setup = time.perf_counter()
    client._setup()
    untimed_s = time.perf_counter() - t_setup
    from anyjev import Question

    questions = [
        Question.choice(
            rq5_questions(snapshot)[question_id(cell, cls)]["instructions"],
            [option_text(action, description) for action, description in RQ5_CRITERIA.items()],
            name=question_id(cell, cls),
        )
        for cell, cls in expected_pairs(snapshot)
    ]
    assert client._recorder is not None
    client._recorder.prompts, client._recorder.fallbacks = [], 0
    t_send = time.time()
    t0 = time.perf_counter()
    try:
        decisions = client._decider.decide(decision_state(snapshot), questions)
    except Exception as exc:  # noqa: BLE001 - adapter transports expose heterogeneous failures
        latency = time.perf_counter() - t0
        return _transport_failure(snapshot, type(exc).__name__, {
            "http_status": None, "adapter_latency_s": latency, "t_send_wall": t_send,
            "t_recv_wall": time.time(), "resolved_model": client.resolved_model, "provider": "vllm",
            "untimed_s": untimed_s,
        })
    latency = time.perf_counter() - t0
    returned = []
    for (cell, cls), decision in zip(expected_pairs(snapshot), decisions):
        probabilities = np.asarray(decision.probs, dtype=float)
        returned.append({"cell": cell, "class": cls, "action": RQ5_ACTIONS[int(probabilities.argmax())]})
    fallbacks = client._recorder.fallbacks
    result = normalise_actions(snapshot, returned, LABEL_FALLBACK_ERROR if fallbacks else None)
    t_count = time.perf_counter()
    meta = {
        "http_status": 200, "adapter_latency_s": latency, "t_send_wall": t_send,
        "t_recv_wall": time.time(), "input_tokens": sum(
            len(client._recorder.tokenizer.encode(prompt, add_special_tokens=False))
            for prompt in client._recorder.prompts
        ), "output_tokens": len(client._recorder.prompts), "resolved_model": client.resolved_model,
        "provider": "vllm", "label_logprob_fallbacks": fallbacks,
    }
    meta["untimed_s"] = untimed_s + (time.perf_counter() - t_count)
    if fallbacks:
        held = normalise_actions(snapshot, [], LABEL_FALLBACK_ERROR)
        return Rq5Result(held.actions, held.invalid_fields, LABEL_FALLBACK_ERROR, meta)
    return Rq5Result(result.actions, result.invalid_fields, result.error_type, meta)


class FakeRq5Interpreter:
    """Deterministic stand-in for smokes: seeded actions and a seeded sleep (some > 1 s, so ticks get skipped)."""

    name = "fake-rq5"

    def __init__(self, seed: int = 0, max_sleep_s: float = 1.3) -> None:
        self.seed = seed
        self.max_sleep_s = max_sleep_s

    def decide_rq5(self, snapshot: dict[str, Any]) -> Rq5Result:
        rng = np.random.default_rng([self.seed, int(round(float(snapshot["tick_s"]) * 1000))])
        time.sleep(float(rng.uniform(0.0, self.max_sleep_s)))
        actions = [
            {"cell": cell, "class": cls, "action": RQ5_ACTIONS[int(rng.integers(len(RQ5_ACTIONS)))]}
            for cell, cls in expected_pairs(snapshot)
        ]
        return Rq5Result(actions, 0, None, {"fake_seed": self.seed})


def query_interpreter(interpreter: Any, snapshot: dict[str, Any]) -> Rq5Result:
    """Dispatch every frozen interpreter adapter; test fakes may expose decide_rq5."""
    if hasattr(interpreter, "timeout_s"):
        interpreter.timeout_s = RQ5_DEADLINE_S
    reset_send_marker()
    if hasattr(interpreter, "decide_rq5"):
        returned = interpreter.decide_rq5(snapshot)
        return returned if isinstance(returned, Rq5Result) else normalise_actions(snapshot, returned)
    if isinstance(interpreter, RanDecisionsClient):
        return query_decisions(interpreter, snapshot)
    if isinstance(interpreter, RanChatJsonClient):
        return query_chat_json(interpreter, snapshot)
    if isinstance(interpreter, AnyJevClient):
        return query_anyjev(interpreter, snapshot)
    raise TypeError(f"unsupported RQ5 interpreter adapter: {type(interpreter).__name__}")
