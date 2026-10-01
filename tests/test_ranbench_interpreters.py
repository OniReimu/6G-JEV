"""RANIntent v1 C1 interpreter adapters (EXP-2026-003): inputs, schema validation, field mapping, ledger rows,
resumability, warm-ups and error accounting. No network: HTTP is faked at make_request, AnyJev runs on a fake backend."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import urllib.error

import numpy as np
import pytest

from src.edgebench.interpreters.base import Decision, Interpreter
from src.edgebench.ledger import LedgerWriter
from src.ranbench.corpus.cli import dry_run
from src.ranbench.corpus.exchange import RanExchange
from src.ranbench.corpus.prompts import build_interpreter_context
from src.ranbench.corpus.spec import FIELDS, OPTIONS, load_config
from src.ranbench.interpreters import anyjev_l0, chat_json, decisions
from src.ranbench.interpreters.anyjev_l0 import AnyJevClient
from src.ranbench.interpreters.chat_json import RanChatJsonClient
from src.ranbench.interpreters.common import (
    SCHEMA_INVALID,
    field_questions,
    RanCase,
    check_policy,
    decision_state,
    load_cases,
    response_schema,
    score_policy,
)
from src.ranbench.interpreters.decisions import RanDecisionsClient
from src.ranbench.interpreters.manifest import build_interpreter, load_manifest
from src.ranbench.interpreters.runner import run_c1, rtt_probe, verify_corpus_file
from src.ranbench.schemas import load_policy_schema

CFG = load_config()
COND, SPLIT = "c7_fresh", "dev"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    """A stub (dry-run) RANIntent v1 corpus built through the corpus package's public functions."""
    root = tmp_path_factory.mktemp("ranintent")
    dry_run(RanExchange(exchange_dir=root / "exchange", data_dir=root / "data", cfg=load_config()))
    return root / "data"


@pytest.fixture(scope="module")
def cases(corpus) -> list[RanCase]:
    return load_cases(corpus / "RQ4" / COND / f"{SPLIT}.jsonl")


@pytest.fixture
def case(cases) -> RanCase:
    return copy.deepcopy(cases[0])


def fake_http(monkeypatch, module, responses: list[tuple[int | None, Any, str | None]], sent: list | None = None):
    """Patch make_request in an adapter module: each call pops (status, body, error_type)."""
    queue = list(responses)

    def _make_request(base_url, path, payload, api_key=None, extra_headers=None, timeout_s=30.0):
        if sent is not None:
            sent.append({"base_url": base_url, "path": path, "payload": payload, "api_key": api_key})
        status, body, err = queue.pop(0)
        raw = body if isinstance(body, str) else json.dumps(body)
        return status, raw, 0.123, 1000.0, 1000.123, err, {}

    monkeypatch.setattr(module, "make_request", _make_request)


def chat_body(content: Any, model: str = "deepseek/deepseek-v4.1-flash", finish: str = "stop",
              reasoning_tokens: int | None = 0) -> dict[str, Any]:
    usage: dict[str, Any] = {"prompt_tokens": 3000, "completion_tokens": 90, "cost": 0.00042}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return {"model": model, "provider": "Together", "usage": usage,
            "choices": [{"finish_reason": finish, "message": {"content": content if isinstance(content, str)
                                                                          else json.dumps(content)}}]}


def jev_body(policy: dict[str, str], model: str = "typesafe/jev-1.13-20260917") -> dict[str, Any]:
    answers = {}
    for f, v in policy.items():
        probs = {o: (0.9 if o == v else 0.1 / (len(OPTIONS[f]) - 1)) for o in OPTIONS[f]}
        answers[f] = {"type": "choice", "choice": v, "probabilities": probs, "confidence": 0.9}
    return {"model": model, "provider": "TypeSafe", "answers": answers,
            "usage": {"input_tokens": 12000, "output_tokens": 300, "cost": 0.0011}}


# ---------------------------------------------------------------- inputs, validation, scoring


def test_context_is_the_shared_reading_rules_artifact(case):
    ctx = case.context
    assert ctx == build_interpreter_context(case.issuer, case.text, case.telemetry)
    assert decision_state(ctx) == ctx["rules"] + "\n\n" + ctx["case"]


def test_response_schema_is_the_policy_schema_without_meta_key():
    full = load_policy_schema()
    sent = response_schema()
    assert "$schema" not in sent and {k: v for k, v in full.items() if k != "$schema"} == sent
    assert sent["required"] == FIELDS and sent["additionalProperties"] is False


@pytest.mark.parametrize("obj", [
    "not a dict",
    ["list"],
    {f: "unspecified" for f in FIELDS},                                    # action/class have no `unspecified`
    {**{f: list(OPTIONS[f])[0] for f in FIELDS}, "extra": "x"},            # additional property
    {f: list(OPTIONS[f])[0] for f in FIELDS if f != "duration"},           # missing field
])
def test_check_policy_rejects_and_never_repairs(obj):
    labels, valid, err = check_policy(obj)
    assert valid is False and err == SCHEMA_INVALID
    assert labels == ([obj] if isinstance(obj, dict) else [])


def test_score_exact_match_and_invalid(case):
    ok = Decision(labels=[dict(case.truth)], valid=True)
    s = score_policy(ok, case, CFG)
    assert s["em"] and all(s["correct"].values()) and set(s["correct"]) == set(FIELDS)
    assert s["spurious_count"] == s["missed_count"] == 0
    assert s["unspecified_truth_count"] + s["specified_truth_count"] == len(FIELDS)
    bad = Decision(labels=[dict(case.truth)], valid=False, error_type=SCHEMA_INVALID)
    s = score_policy(bad, case, CFG)
    assert not s["em"] and not any(s["correct"].values()) and not s["unsafe_locality"]


def test_score_unsafe_and_unstated_counts(cases):
    tenant = next(c for c in cases if c.issuer["kind"] == "tenant" and not c.issuer["may_set_critical"])
    pred = dict(tenant.truth, priority="critical")
    s = score_policy(Decision(labels=[pred], valid=True), tenant, CFG)
    assert s["unsafe_locality"] and not s["em"]
    other = next(c for c in OPTIONS["class"] if c not in tenant.issuer["owned_classes"])
    assert score_policy(Decision(labels=[dict(tenant.truth, **{"class": other})], valid=True), tenant, CFG)["unsafe_locality"]
    named = next(c for c in cases if c.truth["target_cluster"] == "none")
    s = score_policy(Decision(labels=[dict(named.truth, target_cluster="stadium")], valid=True), named, CFG)
    assert s["spurious_count"] == 1 and s["missed_count"] == 0


# ---------------------------------------------------------------- decision models (Jev, SemIf)


def test_decisions_request_one_choice_per_field(case):
    jev = build_interpreter("Jev-1.13.0", load_manifest(), api_key="k")
    req = jev.build_request(case)
    assert req["state"] == decision_state(case.context) and req["model"] == "typesafe/jev-1.13"
    assert list(req["questions"]) == FIELDS
    for f, q in req["questions"].items():
        assert q["type"] == "choice" and q["criteria"] == OPTIONS[f] and f"`{f}`" in q["instructions"]
    assert req["provider"] == {"allow_fallbacks": False}
    semif = build_interpreter("SemIf-Qwen3.5-4B", load_manifest())
    assert "provider" not in semif.build_request(case) and semif.api_key is None
    assert max(len(o) for o in OPTIONS.values()) <= 16  # SemIf's option cap (EXP-2026-001 D-2)


def test_decisions_maps_fields_and_records_distributions(monkeypatch, case):
    fake_http(monkeypatch, decisions, [(200, jev_body(case.truth), None)])
    d = build_interpreter("Jev-1.13.0", load_manifest(), api_key="k").decide(case)
    assert d.valid and d.error_type is None and d.labels == [case.truth]
    assert set(d.probabilities[0]) == set(FIELDS)
    for f in FIELDS:
        assert set(d.probabilities[0][f]) == set(OPTIONS[f])
    assert (d.input_tokens, d.output_tokens, d.cost_usd) == (12000, 300, 0.0011)
    assert d.resolved_model == "typesafe/jev-1.13-20260917" and d.provider == "TypeSafe"


@pytest.mark.parametrize("mutate,error", [
    (lambda b: b["answers"].pop("duration"), "missing_answers"),
    (lambda b: b["answers"]["action"].update(choice="launch"), "invalid_option"),
    (lambda b: b["answers"].__setitem__("scope", "north_cluster"), "malformed_choice"),
    (lambda b: b.update(model="typesafe/jev-2"), "model_mismatch"),
    (lambda b: b.pop("answers"), "missing_answers"),
])
def test_decisions_protocol_errors(monkeypatch, case, mutate, error):
    body = jev_body(case.truth)
    mutate(body)
    fake_http(monkeypatch, decisions, [(200, body, None)])
    d = build_interpreter("Jev-1.13.0", load_manifest(), api_key="k").decide(case)
    assert not d.valid and d.error_type == error
    if error == "invalid_option":
        assert d.labels[0]["action"] == "launch"  # kept as returned, never mapped to a valid option


@pytest.mark.parametrize("status,body,err,expected", [
    (503, {"error": "busy"}, None, "HTTP_503"),
    (None, "", "TimeoutError", "TimeoutError"),
    (200, "not json", None, "json_parse_error"),
])
def test_decisions_transport_errors(monkeypatch, case, status, body, err, expected):
    fake_http(monkeypatch, decisions, [(status, body, err)])
    d = build_interpreter("Jev-1.13.0", load_manifest(), api_key="k").decide(case)
    assert not d.valid and d.error_type == expected and d.labels == [] and d.http_status == status


# ---------------------------------------------------------------- JSON-emitting LLMs (OpenRouter, vLLM)


def test_chat_request_pins_openrouter_and_constrains_output(case):
    m = load_manifest()
    ds = build_interpreter("DeepSeek-V4.1-Flash", m, api_key="k").build_request(case)
    assert ds["messages"] == [{"role": "system", "content": case.context["rules"]},
                              {"role": "user", "content": case.context["case"]}]
    assert ds["temperature"] == 0 and ds["reasoning"] == {"enabled": False} and ds["max_tokens"] == 64 + 24 * 9
    assert ds["provider"] == {"only": ["together"], "allow_fallbacks": False, "require_parameters": True,
                              "data_collection": "deny"}
    assert ds["response_format"]["json_schema"]["strict"] is True
    assert ds["response_format"]["json_schema"]["schema"] == response_schema()
    glm = build_interpreter("GLM-5.3-Flash", m, api_key="k").build_request(case)
    assert glm["reasoning"] == {"effort": "minimal"} and glm["max_tokens"] == 512 + 24 * 9
    assert build_interpreter("Qwen3.8-Flash", m, api_key="k").build_request(case)["provider"]["only"] == ["alibaba"]
    q = build_interpreter("Qwen3.5-4B-JSON", m).build_request(case)
    assert "provider" not in q and "reasoning" not in q
    assert q["chat_template_kwargs"] == {"enable_thinking": False} and q["model"] == "Qwen/Qwen3.5-4B"
    assert q["response_format"]["json_schema"]["schema"] == response_schema()


def test_chat_valid_policy(monkeypatch, case):
    sent: list = []
    fake_http(monkeypatch, chat_json, [(200, chat_body(case.truth), None)], sent)
    d = build_interpreter("DeepSeek-V4.1-Flash", load_manifest(), api_key="k").decide(case)
    assert d.valid and d.labels == [case.truth] and d.cost_usd == 0.00042 and d.reasoning_tokens == 0
    assert sent[0]["api_key"] == "k" and sent[0]["path"] == "/api/v1/chat/completions"
    assert "@" not in json.dumps(sent[0]["payload"])  # no e-mail address in any request


@pytest.mark.parametrize("content,kwargs,error", [
    ({"action": "prioritise"}, {}, SCHEMA_INVALID),
    ("{broken", {}, "bad_json"),
    (None, {"finish": "length"}, "finish_reason_length"),
    (None, {"reasoning_tokens": 12}, "reasoning_tokens_nonzero"),
    (None, {"model": "deepseek/other"}, "model_mismatch"),
])
def test_chat_invalid_outputs_are_recorded_not_repaired(monkeypatch, case, content, kwargs, error):
    body = chat_body(case.truth if content is None else content, **kwargs)
    fake_http(monkeypatch, chat_json, [(200, body, None)])
    d = build_interpreter("DeepSeek-V4.1-Flash", load_manifest(), api_key="k").decide(case)
    assert not d.valid and d.error_type == error
    if error == SCHEMA_INVALID:
        assert d.labels == [{"action": "prioritise"}]


def test_chat_missing_reasoning_count_is_flagged(monkeypatch, case):
    fake_http(monkeypatch, chat_json, [(200, chat_body(case.truth, reasoning_tokens=None), None)])
    d = RanChatJsonClient("x", "deepseek/deepseek-v4.1-flash", "together", api_key="k").decide(case)
    assert d.valid and d.reasoning_tokens_missing


def test_vllm_reference_no_cost_and_no_reasoning_check(monkeypatch, case):
    body = chat_body(case.truth, model="Qwen/Qwen3.5-4B", reasoning_tokens=None)
    body["usage"].pop("cost")
    fake_http(monkeypatch, chat_json, [(200, body, None)])
    d = build_interpreter("Qwen3.5-4B-JSON", load_manifest()).decide(case)
    assert d.valid and d.cost_usd is None and not d.reasoning_tokens_missing


def test_probe_requests_are_minimal(monkeypatch):
    m = load_manifest()
    for name in ("Jev-1.13.0", "DeepSeek-V4.1-Flash", "GLM-5.3-Flash", "Qwen3.8-Flash"):
        interp = build_interpreter(name, m, api_key="k")
        req = interp.build_probe_request()
        assert "response_format" not in req and len(json.dumps(req)) < 400
    fake_http(monkeypatch, chat_json, [(200, {"model": "deepseek/deepseek-v4.1-flash", "provider": "Together",
                                              "usage": {"prompt_tokens": 9, "completion_tokens": 1, "cost": 1e-6}}, None)])
    row = rtt_probe(build_interpreter("DeepSeek-V4.1-Flash", m, api_key="k"))
    assert row["http_status"] == 200 and row["cost_usd"] == 1e-6 and row["latency_s"] == 0.123


# ---------------------------------------------------------------- AnyJev L0 (fake backend)


class _Tok:
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        return [ord(text.strip()) - ord("A") + 1000] if re.fullmatch(r" ?[A-Z]", text) else list(range(len(text) // 4 + 1))


class FakeVLLM:
    """Letter-readout backend that prefers the option named in `answer` (a policy) for the field in the question."""

    def __init__(self, answer: dict[str, str], fail: Exception | None = None, drop_label: bool = False):
        self.name = "Qwen/Qwen3.5-4B"
        self.tokenizer = _Tok()
        self.answer = answer
        self.fail = fail
        self.drop_label = drop_label
        self.calls = 0

    def next_token_logprobs(self, prompts, token_ids):
        self.calls += 1
        if self.fail:
            raise self.fail
        out = []
        for prompt, ids in zip(prompts, token_ids):
            field_name = re.search(r"Question: Which value does the policy field `(\w+)`", prompt).group(1)
            shown = re.findall(r"^([A-Z])\. (\w+): ", prompt.split("Options:")[1], flags=re.M)
            lp = np.array([0.0 if opt == self.answer[field_name] else -4.0 for _, opt in shown])
            lp = lp - np.log(np.exp(lp).sum())
            if self.drop_label:
                lp[-1] = anyjev_l0.LABEL_FALLBACK_LOGPROB
            out.append(lp)
        return out


def test_anyjev_l0_maps_fields_and_records_order(case):
    client = AnyJevClient(backend=FakeVLLM(case.truth))
    assert client.warmup(case)["warmup"] is True
    d = client.decide(case)
    assert d.valid and d.labels == [case.truth] and d.http_status == 200
    n_prefills = sum(len(OPTIONS[f]) for f in FIELDS)
    assert n_prefills == 57 and d.output_tokens == 57 and d.input_tokens > 0
    for f in FIELDS:
        assert set(d.probabilities[0][f]) == set(OPTIONS[f])
        assert abs(sum(d.probabilities[0][f].values()) - 1) < 1e-9
    raw = json.loads(d.raw_response)
    assert raw["seq"] == 1 and raw["n_prefills"] == 57 and raw["label_logprob_fallbacks"] == 0
    assert raw["prior_running_n"]["action"] == 2 and raw["fields"]["action"]["level"] == "L0"
    assert [h["warmup"] for h in client.history] == [True, False]
    assert d.resolved_model == "anyjev-0.0.2@e172f38/L0/Qwen/Qwen3.5-4B"


def test_decision_models_get_identical_questions_and_options(case):
    """Jev, SemIf and AnyJev see the same state, question text and option descriptions; only the readout differs."""
    m = load_manifest()
    jev = build_interpreter("Jev-1.13.0", m, api_key="k").build_request(case)
    semif = build_interpreter("SemIf-Qwen3.5-4B", m).build_request(case)
    aj = AnyJevClient(backend=FakeVLLM(case.truth))
    aj._setup()
    assert jev["state"] == semif["state"] == decision_state(case.context)
    assert jev["questions"] == semif["questions"]
    assert [q.name for q in aj._questions] == list(jev["questions"]) == FIELDS
    for q in aj._questions:
        jq = jev["questions"][q.name]
        assert q.text == jq["instructions"]
        assert list(q.options) == [f"{i}: {d}" for i, d in jq["criteria"].items()]
        assert jq["criteria"] == OPTIONS[q.name]  # the reading-rules descriptions
    # the state AnyJev renders into every prefill is the same string
    prompts: list[str] = []
    inner = aj._recorder.backend
    aj._recorder.backend = type("Spy", (), {"next_token_logprobs": lambda self, p, t: (prompts.extend(p),
                                            inner.next_token_logprobs(p, t))[1]})()
    aj.decide(case)
    assert prompts and all(decision_state(case.context) in p for p in prompts)
    assert {q["text"] for q in field_questions()} <= {re.search(r"Question: (.*)", p).group(1) for p in prompts}


def test_anyjev_placeholder_logprob_is_an_error(case):
    d = AnyJevClient(backend=FakeVLLM(case.truth, drop_label=True)).decide(case)
    assert not d.valid and d.error_type == anyjev_l0.LABEL_FALLBACK_ERROR


@pytest.mark.parametrize("exc,status,error", [
    (urllib.error.URLError(ConnectionRefusedError()), None, "ConnectionRefusedError"),
    (urllib.error.HTTPError("http://x", 400, "too long", {}, None), 400, "HTTP_400"),
    (TimeoutError(), None, "TimeoutError"),
])
def test_anyjev_backend_errors(case, exc, status, error):
    client = AnyJevClient(backend=FakeVLLM(case.truth, fail=exc))
    d = client.decide(case)
    assert not d.valid and d.error_type == error and d.http_status == status and d.latency_s is not None
    assert client.history[-1]["error"] == error


def test_manifest_builds_every_interpreter_without_contacting_servers(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/hf")
    m = load_manifest()
    assert set(m) == {"Jev-1.13.0", "DeepSeek-V4.1-Flash", "GLM-5.3-Flash", "Qwen3.8-Flash", "SemIf-Qwen3.5-4B",
                      "AnyJev-L0", "Qwen3.5-4B-JSON"}
    built = {n: build_interpreter(n, m, api_key="k") for n in m}
    assert all(getattr(i, "api_key", None) in (None,) for n, i in built.items() if m[n]["deployment"] != "hosted")
    assert all(i.api_key == "k" for n, i in built.items() if m[n]["deployment"] == "hosted")
    assert built["AnyJev-L0"].tokenizer_name.startswith("/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/")
    assert built["AnyJev-L0"]._decider is None
    with pytest.raises(ValueError):
        build_interpreter("Jev-1.13.0", m, base_url="http://127.0.0.1:1")
    assert build_interpreter("AnyJev-L0", m, base_url="http://127.0.0.1:9100").base_url == "http://127.0.0.1:9100"


# ---------------------------------------------------------------- runner


class FakeInterp(Interpreter):
    """Scripted interpreter: `script(case) -> Decision`; records the call order in `log`."""

    def __init__(self, name, log, script=None, deployment="hosted", cost=0.001):
        super().__init__(name=name, deployment=deployment)
        self.platform = "hosted" if deployment == "hosted" else "H100-NVL"
        self.log = log
        self.script = script
        self.cost = cost
        self.provider_slug = "fake"

    def decide(self, case):
        self.log.append((self.name, case.case_id))
        if self.script is not None:
            return self.script(case)
        return Decision(labels=[dict(case.truth)], valid=True, http_status=200, latency_s=0.01,
                        input_tokens=10, output_tokens=5, cost_usd=self.cost, resolved_model=self.name)

    def build_probe_request(self):
        return {"probe": True}

    def post(self, payload):
        self.log.append((self.name, "probe"))
        return 200, json.dumps({"model": self.name, "usage": {"prompt_tokens": 1, "cost": 0.0}}), 0.05, 1.0, 1.05, None


def run(corpus, tmp_path, interps, **kw):
    kw.setdefault("client_location", "test")
    return run_c1(corpus_dir=corpus, condition=COND, split=SPLIT, interpreter_names=[i.name for i in interps],
                  out_dir=tmp_path, allow_dirty=True, allow_stub=True, interpreters=interps, **kw)


def rows_of(tmp_path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines() if line.strip()]


def test_runner_one_edgebench_row_per_interpreter_case(corpus, tmp_path):
    log: list = []
    interps = [FakeInterp("A", log), FakeInterp("B", log)]
    rep = run(corpus, tmp_path, interps, workers=1)
    rows = rows_of(tmp_path)
    n = len(load_cases(corpus / "RQ4" / COND / f"{SPLIT}.jsonl"))
    assert len(rows) == 2 * n and rep["duplicates"] == 0 and rep["stop_reason"] is None
    assert {(r["model"], r["case_id"]) for r in rows} == {(m, c) for m in "AB" for c in rep["case_order"]}
    # exactly the edgebench row layout
    probe_writer = LedgerWriter(tmp_path / "layout.jsonl", "r", "g", "C1", COND)
    expected = probe_writer.write_row("A", "x", 0, Decision(), correct={}, em=False)
    probe_writer.close()
    assert all(set(r) == set(expected) for r in rows)
    assert all(r["em"] and r["valid"] and set(r["correct"]) == set(FIELDS) for r in rows)
    assert rep["per_model"]["A"]["actual_rows"] == n and rep["per_model"]["A"]["total_cost_usd"] == pytest.approx(0.001 * n)
    assert rep["cases_sha256"] == verify_corpus_file(corpus, f"RQ4/{COND}/{SPLIT}.jsonl", allow_stub=True)[0]


def test_runner_seeded_interleaving(corpus, tmp_path):
    logs = []
    for sub in ("a", "b"):
        log: list = []
        run(corpus, tmp_path / sub, [FakeInterp("A", log), FakeInterp("B", log), FakeInterp("C", log)], workers=1)
        logs.append(log)
    assert logs[0] == logs[1]  # same seed -> same case order and per-case interpreter order
    per_case = [tuple(m for m, _ in logs[0][i:i + 3]) for i in range(0, len(logs[0]), 3)]
    assert len(set(per_case)) > 1  # the order is shuffled per case, not fixed
    log: list = []
    run(corpus, tmp_path / "c", [FakeInterp("A", log), FakeInterp("B", log), FakeInterp("C", log)], workers=1, seed=7)
    assert log != logs[0]


def test_runner_resumes_without_duplicates(corpus, tmp_path):
    log: list = []
    calls = {"n": 0}

    def mismatch_on_fifth(case):
        calls["n"] += 1
        if calls["n"] == 5:
            return Decision(labels=[], valid=False, error_type="model_mismatch", http_status=200)
        return Decision(labels=[dict(case.truth)], valid=True, http_status=200, cost_usd=0.0)

    first = run(corpus, tmp_path, [FakeInterp("A", log, mismatch_on_fifth)], workers=1)
    assert first["stop_reason"] == "model_mismatch" and first["per_model"]["A"]["actual_rows"] == 5
    # a torn final line (crash mid-write) is dropped and that case re-run
    with open(tmp_path / "ledger.jsonl", "a") as f:
        f.write('{"rq": "C1", "partial')
    log2: list = []
    second = run(corpus, tmp_path, [FakeInterp("A", log2)], workers=1)
    n = second["n_cases"]
    assert second["stop_reason"] is None and second["torn_line_dropped"]
    assert second["per_model"]["A"]["actual_rows"] == n and second["duplicates"] == 0
    assert len(log2) == n - 5
    log3: list = []
    third = run(corpus, tmp_path, [FakeInterp("A", log3)], workers=4)
    assert log3 == [] and third["per_model"]["A"]["actual_rows"] == n


def test_runner_error_accounting(corpus, tmp_path):
    log: list = []
    k = {"n": 0}

    def flaky(case):
        k["n"] += 1
        if k["n"] % 5 == 0:
            return Decision(labels=[], valid=False, error_type="HTTP_429", http_status=429)
        if k["n"] % 7 == 0:
            return Decision(labels=[{"action": "x"}], valid=False, error_type=SCHEMA_INVALID, http_status=200)
        return Decision(labels=[dict(case.truth)], valid=True, http_status=200, cost_usd=None)

    rep = run(corpus, tmp_path, [FakeInterp("A", log, flaky)], workers=1)
    pm = rep["per_model"]["A"]
    n = rep["n_cases"]
    assert pm["transport_error_rows"] == n // 5 and pm["error_counts_by_type"]["HTTP_429"] == n // 5
    assert pm["schema_invalid_rows"] == len([i for i in range(1, n + 1) if i % 7 == 0 and i % 5 != 0])
    assert pm["rerun_block_required"] is True and pm["cost_unknown_rows"] == n and pm["total_cost_usd"] is None
    assert pm["valid_rows"] == n - pm["transport_error_rows"] - pm["schema_invalid_rows"]


def test_runner_exception_becomes_error_row(corpus, tmp_path):
    def boom(case):
        raise ConnectionResetError("reset")

    rep = run(corpus, tmp_path, [FakeInterp("A", [], boom)], workers=1, limit=3)
    rows = rows_of(tmp_path)
    assert len(rows) == 3 and all(r["error_type"] == "ConnectionResetError" and not r["valid"] for r in rows)
    assert rep["per_model"]["A"]["transport_error_rows"] == 3


def test_runner_spend_cap_counts_ledger_and_probes(corpus, tmp_path):
    rep = run(corpus, tmp_path, [FakeInterp("A", [], cost=1.0)], workers=1, spend_cap_usd=2.5)
    assert rep["stop_reason"].startswith("spend_cap_exceeded") and rep["per_model"]["A"]["actual_rows"] == 3
    rep2 = run(corpus, tmp_path, [FakeInterp("A", [], cost=1.0)], workers=1, spend_cap_usd=2.5)
    assert rep2["stop_reason"].startswith("spend_cap_exceeded") and rep2["per_model"]["A"]["actual_rows"] == 4


def test_runner_rtt_probes_interleaved(corpus, tmp_path):
    log: list = []
    rep = run(corpus, tmp_path, [FakeInterp("A", log), FakeInterp("B", log)], workers=1, probe_every=5, limit=10)
    probes = [json.loads(line) for line in (tmp_path / "probes.jsonl").read_text().splitlines()]
    assert rep["probes_written"] == len(probes) == 4  # cases 0 and 5, both interpreters
    assert {p["model"] for p in probes} == {"A", "B"} and all(p["client_location"] == "test" for p in probes)
    # case 0 (2 calls), probes, cases 1-4 (8 calls), case 5 (2 calls), probes
    assert [i for i, (_, what) in enumerate(log) if what == "probe"] == [2, 3, 14, 15]


def test_runner_self_hosted_warmup_and_anyjev_order(corpus, tmp_path, cases):
    truth_by_id = {c.case_id: c.truth for c in cases}

    class ByCase(FakeVLLM):
        def next_token_logprobs(self, prompts, token_ids):
            cid = re.search(r"Dry-run \w+ intent (\d+)", prompts[0]).group(1)
            self.answer = truth_by_id[f"ranintent_v1_{SPLIT}_{cid}"]
            return super().next_token_logprobs(prompts, token_ids)

    aj = AnyJevClient(name="AnyJev-L0", backend=ByCase({}))
    rep = run(corpus, tmp_path, [aj], workers=8, limit=12, client_location=None)
    assert rep["warmup_cases"] == rep["case_order"][:2]
    assert [w["case_id"] for w in rep["warmup_outcomes"]["AnyJev-L0"]] == rep["case_order"][:2]
    order = [json.loads(line) for line in (tmp_path / "anyjev_order.jsonl").read_text().splitlines()]
    assert [o["case_id"] for o in order] == rep["case_order"][:2] + rep["case_order"]
    assert [o["seq"] for o in order] == list(range(14)) and [o["warmup"] for o in order[:3]] == [True, True, False]
    rows = rows_of(tmp_path)
    assert [r["case_id"] for r in rows] == rep["case_order"]  # one client, seeded order
    assert [json.loads(r["raw_response"])["seq"] for r in rows] == list(range(2, 14))
    assert all(r["valid"] for r in rows)
    # the online label prior is off for the first 7 intents (2 warm-ups + 5 cases) and on from the 8th
    methods = [json.loads(r["raw_response"])["fields"]["action"]["prior_method"] for r in rows]
    assert methods == ["none"] * 5 + ["batch"] * 7  # the 8th intent (seq 7) is the first corrected one
    assert all(r["em"] for r in rows[:5])


def test_runner_refuses_to_resume_anyjev_and_fresh_restarts(corpus, tmp_path, cases):
    truth = cases[0].truth
    calls = {"n": 0}

    class DiesAfter(FakeVLLM):
        def next_token_logprobs(self, prompts, token_ids):
            calls["n"] += 1
            if calls["n"] == 6:  # 2 warm-ups + 3 cases, then the server goes away
                raise urllib.error.URLError(ConnectionRefusedError())
            return super().next_token_logprobs(prompts, token_ids)

    first = run(corpus, tmp_path, [AnyJevClient(name="AnyJev-L0", backend=DiesAfter(truth))], limit=8,
                client_location=None)
    assert first["stop_reason"] == "self_hosted_server_down" and first["per_model"]["AnyJev-L0"]["actual_rows"] == 4
    with pytest.raises(RuntimeError, match="Refusing to resume AnyJev-L0"):
        run(corpus, tmp_path, [AnyJevClient(name="AnyJev-L0", backend=FakeVLLM(truth))], limit=8, client_location=None)
    assert len(rows_of(tmp_path)) == 4  # the refusal touched nothing
    # another interpreter's rows in the same ledger survive a fresh restart
    run(corpus, tmp_path, [FakeInterp("S", [], deployment="self-hosted")], limit=8, client_location=None)
    fresh_client = AnyJevClient(name="AnyJev-L0", backend=FakeVLLM(truth))
    rep = run(corpus, tmp_path, [fresh_client], limit=8, client_location=None, fresh=True)
    rows = rows_of(tmp_path)
    aj_rows = [r for r in rows if r["model"] == "AnyJev-L0"]
    assert rep["fresh"] and rep["stop_reason"] is None and len(aj_rows) == 8 and rep["duplicates"] == 0
    assert len([r for r in rows if r["model"] == "S"]) == 8
    assert [json.loads(r["raw_response"])["seq"] for r in aj_rows] == list(range(2, 10))  # restarted from warm-up
    order = [json.loads(line) for line in (tmp_path / "anyjev_order.jsonl").read_text().splitlines()]
    assert {o["session"] for o in order} == {fresh_client.session} and len(order) == 10
    # a hosted/order-independent interpreter still resumes normally (covered by test_runner_resumes_without_duplicates)


def test_runner_stops_on_label_fallback(corpus, tmp_path, cases):
    aj = AnyJevClient(name="AnyJev-L0", backend=FakeVLLM(cases[0].truth, drop_label=True))
    rep = run(corpus, tmp_path, [aj], limit=5, warmup=0)
    assert rep["stop_reason"] == anyjev_l0.LABEL_FALLBACK_ERROR and rep["per_model"]["AnyJev-L0"]["actual_rows"] == 1


def test_runner_refusals(corpus, tmp_path):
    with pytest.raises(ValueError, match="mix"):
        run(corpus, tmp_path, [FakeInterp("A", []), FakeInterp("S", [], deployment="self-hosted")])
    with pytest.raises(ValueError, match="client location"):
        run(corpus, tmp_path, [FakeInterp("A", [])], client_location=None)
    with pytest.raises(RuntimeError, match="stub"):
        run_c1(corpus_dir=corpus, condition=COND, split=SPLIT, interpreter_names=["A"], out_dir=tmp_path,
               allow_dirty=True, interpreters=[FakeInterp("A", [])], client_location="x")


def test_runner_refuses_changed_cases_file(corpus, tmp_path):
    copy_dir = tmp_path / "corpus"
    import shutil
    shutil.copytree(corpus, copy_dir)
    p = copy_dir / "RQ4" / COND / f"{SPLIT}.jsonl"
    p.write_text(p.read_text().replace("Dry-run", "Dry run", 1))
    with pytest.raises(RuntimeError, match="does not match"):
        run(copy_dir, tmp_path / "out", [FakeInterp("A", [])])


@pytest.mark.parametrize("failure", ["transport", "placeholder"])
def test_runner_failed_warmup_aborts_and_forces_fresh(corpus, tmp_path, cases, failure):
    """Review finding 1: a failed warm-up (transport error or -30 placeholder) stops the condition before any timed call;
    the integrity file carries the stop_reason, and the next attempt must use fresh."""
    truth = cases[0].truth
    calls = {"n": 0}

    class FailSecondWarmup(FakeVLLM):
        def next_token_logprobs(self, prompts, token_ids):
            calls["n"] += 1
            if calls["n"] == 2:
                if failure == "transport":
                    raise urllib.error.URLError(ConnectionRefusedError())
                self.drop_label = True
            else:
                self.drop_label = False
            return super().next_token_logprobs(prompts, token_ids)

    rep = run(corpus, tmp_path, [AnyJevClient(name="AnyJev-L0", backend=FailSecondWarmup(truth))], limit=5,
              client_location=None)
    expected = "ConnectionRefusedError" if failure == "transport" else anyjev_l0.LABEL_FALLBACK_ERROR
    assert rep["stop_reason"].startswith("warmup_failed") and expected in rep["stop_reason"]
    assert rep["per_model"]["AnyJev-L0"]["actual_rows"] == 0 and rows_of(tmp_path) == []
    assert [w["error"] for w in rep["warmup_outcomes"]["AnyJev-L0"]] == [None, expected]
    assert calls["n"] == 2  # no timed call after the failed warm-up
    integ = json.loads((tmp_path / "integrity.json").read_text())
    assert integ["stop_reason"] is not None  # the job script's condition_done() is false
    with pytest.raises(RuntimeError, match="order record"):
        run(corpus, tmp_path, [AnyJevClient(name="AnyJev-L0", backend=FakeVLLM(truth))], limit=5, client_location=None)
    ok = AnyJevClient(name="AnyJev-L0", backend=FakeVLLM(truth))
    rep2 = run(corpus, tmp_path, [ok], limit=5, client_location=None, fresh=True)
    assert rep2["stop_reason"] is None and rep2["per_model"]["AnyJev-L0"]["actual_rows"] == 5
    order = [json.loads(line) for line in (tmp_path / "anyjev_order.jsonl").read_text().splitlines()]
    assert {o["session"] for o in order} == {ok.session} and len(order) == 7


def test_runner_failed_warmup_of_other_self_hosted_model_aborts(corpus, tmp_path):
    k = {"n": 0}

    def down_first(case):
        k["n"] += 1
        if k["n"] == 1:
            return Decision(labels=[], valid=False, error_type="HTTP_503", http_status=503)
        return Decision(labels=[dict(case.truth)], valid=True, http_status=200)

    rep = run(corpus, tmp_path, [FakeInterp("S", [], down_first, deployment="self-hosted")], limit=4,
              client_location=None)
    assert rep["stop_reason"] == "warmup_failed (S:%s:HTTP_503)" % rep["case_order"][0]
    assert rep["per_model"]["S"]["actual_rows"] == 0
    # a schema-invalid warm-up answer is an ordinary outcome, not a warm-up failure
    invalid = lambda case: Decision(labels=[{"x": 1}], valid=False, error_type=SCHEMA_INVALID, http_status=200)
    rep = run(corpus, tmp_path / "b", [FakeInterp("S", [], invalid, deployment="self-hosted")], limit=4,
              client_location=None)
    assert rep["stop_reason"] is None and rep["per_model"]["S"]["actual_rows"] == 4


def test_fresh_with_changed_seed_deletes_all_order_records(corpus, tmp_path, cases):
    """Review finding 2: fresh removes every order record of the model+condition, whatever seed wrote it."""
    truth = cases[0].truth
    first = run(corpus, tmp_path, [AnyJevClient(name="AnyJev-L0", backend=FakeVLLM(truth))], limit=4,
                client_location=None, seed=1)
    assert first["stop_reason"] is None
    # an order record in the pre-fix layout (no `condition` field) is matched by its run_id
    with open(tmp_path / "anyjev_order.jsonl", "a") as f:
        f.write(json.dumps({"run_id": f"C1_{COND}_{SPLIT}_99", "model": "AnyJev-L0", "session": "old", "seq": 0}) + "\n")
        f.write(json.dumps({"run_id": "C1_c3_fresh_dev_1", "model": "AnyJev-L0", "condition": "c3_fresh",
                            "session": "other-cond", "seq": 0}) + "\n")
        f.write(json.dumps({"run_id": f"C1_{COND}_{SPLIT}_1", "model": "Other", "condition": COND,
                            "session": "other-model", "seq": 0}) + "\n")
    new = AnyJevClient(name="AnyJev-L0", backend=FakeVLLM(truth))
    rep = run(corpus, tmp_path, [new], limit=4, client_location=None, seed=2, fresh=True)
    assert rep["stop_reason"] is None
    order = [json.loads(line) for line in (tmp_path / "anyjev_order.jsonl").read_text().splitlines()]
    mine = [o for o in order if o["model"] == "AnyJev-L0" and o.get("condition", COND) == COND
            and o["run_id"].startswith(f"C1_{COND}_")]
    assert {o["session"] for o in mine} == {new.session} and all(o["run_id"].endswith("_2") for o in mine)
    assert {o["session"] for o in order} == {new.session, "other-cond", "other-model"}
    assert [r["run_id"] for r in rows_of(tmp_path)] == [f"C1_{COND}_{SPLIT}_2"] * 4


# ---------------------------------------------------------------- D-3 gap-fill (--only-cases)


def _write_list(path: Path, ids: list[str], text: str | None = None) -> Path:
    path.write_text(text if text is not None else "".join(i + "\n" for i in ids), encoding="utf-8")
    return path


def test_only_cases_issues_listed_cases_in_seeded_order(corpus, tmp_path):
    full = run(corpus, tmp_path / "full", [FakeInterp("A", [])], workers=1, limit=12)
    order = full["case_order"]
    listed = [order[9], order[1], order[5]]  # file order differs from the seeded order
    lst = _write_list(tmp_path / "gap.txt", listed, text=f"{listed[0]}\n\n  {listed[1]}  \n{listed[2]}")
    log: list = []
    rep = run(corpus, tmp_path / "gap", [FakeInterp("A", log)], workers=1, limit=12, probe_every=2,
              only_cases=lst)
    # issued in seeded order: positions 1, 5, 9; probes after issued index 0 and 2
    assert log == [("A", order[1]), ("A", "probe"), ("A", order[5]), ("A", order[9]), ("A", "probe")]
    assert [r["case_id"] for r in rows_of(tmp_path / "gap")] == [order[1], order[5], order[9]]
    assert rep["case_order"] == [order[1], order[5], order[9]] and rep["n_cases"] == 3
    assert rep["only_cases"] == {"path": str(lst), "sha256": hashlib.sha256(lst.read_bytes()).hexdigest(),
                                 "count": 3}
    assert rep["per_model"]["A"]["expected_rows"] == rep["per_model"]["A"]["actual_rows"] == 3
    assert rep["total_spend_usd"] == pytest.approx(0.003) and rep["probes_written"] == 2
    integrity = json.loads((tmp_path / "gap" / "integrity.json").read_text())
    assert integrity["only_cases"] == rep["only_cases"]
    assert full["only_cases"] is None
    # rerunning the same gap-fill resumes: nothing left to issue
    again: list = []
    run(corpus, tmp_path / "gap", [FakeInterp("A", again)], workers=1, limit=12, probe_every=0, only_cases=lst)
    assert again == [] and len(rows_of(tmp_path / "gap")) == 3


def test_only_cases_self_hosted_warmup_keeps_seeded_order(corpus, tmp_path):
    order = run(corpus, tmp_path / "full", [FakeInterp("S", [], deployment="self-hosted")], limit=8,
                client_location=None)["case_order"]
    log: list = []
    rep = run(corpus, tmp_path / "gap", [FakeInterp("S", log, deployment="self-hosted")], limit=8,
              client_location=None, only_cases=_write_list(tmp_path / "gap.txt", [order[6], order[3]]))
    # the 2 warm-ups are the first 2 intents of the full seeded order, then only the listed cases
    assert log == [("S", order[0]), ("S", order[1]), ("S", order[3]), ("S", order[6])]
    assert rep["warmup_cases"] == order[:2] and rep["case_order"] == [order[3], order[6]]


def test_only_cases_refusals(corpus, tmp_path, cases):
    ok_id = cases[0].case_id
    for ids, text, match in [([], "\n\n", "lists no case ids"),
                             ([ok_id, ok_id], None, "more than once"),
                             ([ok_id, "ranintent_v1_dev_9999"], None, "1 listed case id")]:
        lst = _write_list(tmp_path / "gap.txt", ids, text)
        log: list = []
        with pytest.raises(ValueError, match=match):
            run(corpus, tmp_path / "out", [FakeInterp("A", log)], only_cases=lst)
        assert log == [] and not (tmp_path / "out" / "ledger.jsonl").exists()


def test_only_cases_refused_for_anyjev(corpus, tmp_path, cases):
    lst = _write_list(tmp_path / "gap.txt", [cases[0].case_id])
    aj = AnyJevClient(name="AnyJev-L0", backend=FakeVLLM(cases[0].truth))
    with pytest.raises(ValueError, match="order-dependent interpreter\\(s\\) AnyJev-L0"):
        run(corpus, tmp_path / "out", [aj], client_location=None, only_cases=lst)
    assert not (tmp_path / "out").exists()
    # through the CLI (manifest-built AnyJev; refused before any server contact)
    from src.ranbench.interpreters.cli import main as cli_main
    with pytest.raises(ValueError, match="order-dependent"):
        cli_main(["run", "--corpus-dir", str(corpus), "--condition", COND, "--split", SPLIT,
                  "--interpreters", "AnyJev-L0", "--out", str(tmp_path / "cli"), "--base-url", "http://127.0.0.1:1",
                  "--only-cases", str(lst), "--allow-dirty", "--allow-stub"])
    assert not (tmp_path / "cli").exists()


def test_cli_forces_one_worker_for_gap_fill(monkeypatch, tmp_path):
    from src.ranbench.interpreters import cli

    captured: dict = {}

    def fake_run_c1(**kwargs):
        captured.update(kwargs)
        return {
            "run_id": "test", "n_cases": 1, "stop_reason": None, "total_spend_usd": 0.0,
            "duplicates": 0, "probes_written": 0, "per_model": {"Jev-1.13.0": {"actual_rows": 1}},
        }

    only = _write_list(tmp_path / "gap.txt", ["case-1"])
    monkeypatch.setattr(cli, "load_openrouter_key", lambda: "not-a-real-key")
    monkeypatch.setattr(cli, "run_c1", fake_run_c1)
    assert cli.main([
        "run", "--condition", "c3_fresh", "--split", "test", "--interpreters", "Jev-1.13.0",
        "--out", str(tmp_path / "out"), "--only-cases", str(only), "--workers", "9",
    ]) == 0
    assert captured["workers"] == 1
