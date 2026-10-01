"""AnyJev (PyPI anyjev==0.0.2 = git e172f38) at level L0 on a vLLM backend, for RANIntent v1 (EXP-2026-003 D-2).

One `Question.choice` per policy field with the same question text and option descriptions Jev and SemIf get
(common.field_questions; each option rendered as "id: description"), all asked on one state (the reading rules plus
the case). L0 = cyclic-shift marginalisation over the K option rotations +
label-prior correction; anyjev's VLLMBackend sends one `/v1/completions` prefill (max_tokens=1, logprobs over the
label tokens) per rotation, 57 per intent. Server: vLLM 0.30.0 `--runner generate --enable-prefix-caching
--logprobs-mode processed_logprobs`; without the last flag labels outside the top-K come back as anyjev's -30.0
placeholder, which this adapter reports as `label_logprob_fallback` (a stop condition of the runner).

The label prior is estimated online (`batch` prior, active after 8 intents), so a decision depends on the intents
this Decider answered before it. The runner therefore feeds the cases in the seeded order after the same warm-up
intents; every decide() (warm-up or timed) is appended to `history` with its session and sequence number, and each
ledger row's raw_response carries its session, sequence number and the running prior count per field.
"""
from __future__ import annotations

import json
import time
from typing import Any, Sequence
import urllib.error
import uuid

import numpy as np

from src.edgebench.interpreters.base import Decision, Interpreter
from src.ranbench.interpreters.common import (
    FIELDS,
    OPTIONS,
    RanCase,
    check_policy,
    decision_state,
    field_questions,
    option_text,
)

ANYJEV_VERSION = "0.0.2"
ANYJEV_GIT = "e172f38"
LABEL_FALLBACK_LOGPROB = -30.0  # anyjev.backends.vllm fill value for a label the server did not return
LABEL_FALLBACK_ERROR = "label_logprob_fallback"


def _error_of(exc: Exception) -> tuple[int | None, str]:
    """(http_status, error_type) in the edgebench vocabulary: HTTP_<code>, or the underlying connection error name."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code, f"HTTP_{exc.code}"
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, BaseException):
        return None, type(exc.reason).__name__
    return None, type(exc).__name__


class _Recorder:
    """Backend wrapper keeping the prompts and the placeholder-logprob count of the last call."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.tokenizer = backend.tokenizer
        self.name = backend.name
        self.prompts: list[str] = []
        self.fallbacks = 0

    def next_token_logprobs(self, prompts: Sequence[str], token_ids: Sequence[Sequence[int]]) -> list[Any]:
        self.prompts = list(prompts)
        out = self.backend.next_token_logprobs(prompts, token_ids)
        self.fallbacks = int(sum((np.asarray(x) == LABEL_FALLBACK_LOGPROB).sum() for x in out))
        return out


class AnyJevClient(Interpreter):
    order_dependent = True  # the runner never resumes it (online label prior)
    # RQ3 load runs set this: decide() then leaves input_tokens unset and attaches count_input_tokens(), so the
    # ledger's token count is done after the cell instead of occupying an interpretation slot.
    defer_input_tokens = False

    def __init__(
        self,
        name: str = "AnyJev-L0",
        backbone: str = "Qwen/Qwen3.5-4B",
        backend_url: str = "http://127.0.0.1:8000",
        tokenizer_name: str | None = None,
        level: str = "L0",
        workers: int = 16,
        timeout_s: float = 120.0,
        backend: Any = None,
        deployment: str = "self-hosted",
    ) -> None:
        super().__init__(name=name, deployment=deployment)
        if level != "L0":
            raise ValueError("RANIntent v1 reads AnyJev at level L0 only (D-2)")
        self.backbone = backbone
        self.base_url = backend_url.rstrip("/")
        self.tokenizer_name = tokenizer_name
        self.level = level
        self.workers = workers
        self.timeout_s = timeout_s
        self._backend = backend
        self._decider: Any = None
        self._recorder: _Recorder | None = None
        self._questions: list[Any] = []
        self.session = uuid.uuid4().hex[:12]
        self.seq = 0
        self.history: list[dict[str, Any]] = []
        self.resolved_model = f"anyjev-{ANYJEV_VERSION}@{ANYJEV_GIT}/{level}/{backbone}"

    def _setup(self) -> None:
        if self._decider is not None:
            return
        from anyjev import Decider, Question

        backend = self._backend
        if backend is None:
            from anyjev.backends.vllm import VLLMBackend

            backend = VLLMBackend(self.base_url, self.backbone, tokenizer_name=self.tokenizer_name,
                                  workers=self.workers, timeout=self.timeout_s)
        self._recorder = _Recorder(backend)
        # package defaults for L0: batch prior, min_prior_n 8, all K cyclic shifts, logmean combination
        self._decider = Decider(self._recorder, level=self.level)
        # same question text and option descriptions as Jev/SemIf; the letter readout shows "id: description"
        self._questions = [
            Question.choice(q["text"], [option_text(i, d) for i, d in q["options"]], name=q["field"])
            for q in field_questions()
        ]

    def warmup(self, case: RanCase) -> dict[str, Any]:
        """Untimed, unrecorded decide() that still feeds the online label prior; returns its history entry."""
        self._setup()
        assert self._recorder is not None
        entry = {"session": self.session, "seq": self.seq, "case_id": case.case_id, "warmup": True, "error": None}
        self._recorder.fallbacks = 0
        try:
            self._decider.decide(decision_state(case.context), self._questions)
            if self._recorder.fallbacks:
                entry["error"] = LABEL_FALLBACK_ERROR  # server without --logprobs-mode processed_logprobs
        except Exception as exc:
            entry["error"] = _error_of(exc)[1]
        self.seq += 1
        self.history.append(entry)
        return entry

    def decide(self, case: RanCase) -> Decision:
        self._setup()
        assert self._recorder is not None
        state = decision_state(case.context)
        self._recorder.prompts, self._recorder.fallbacks = [], 0
        seq = self.seq
        self.seq += 1
        t_send = time.time()
        t0 = time.perf_counter()
        try:
            result = self._decider.decide(state, self._questions)
        except Exception as exc:
            latency_s = time.perf_counter() - t0
            status, error_type = _error_of(exc)
            self.history.append({"session": self.session, "seq": seq, "case_id": case.case_id, "warmup": False,
                                 "error": error_type})
            return Decision(labels=[], valid=False, error_type=error_type, http_status=status, latency_s=latency_s,
                            t_send_wall=t_send, t_recv_wall=time.time(), resolved_model=self.resolved_model,
                            provider="vllm")
        latency_s = time.perf_counter() - t0
        t_recv = time.time()
        self.history.append({"session": self.session, "seq": seq, "case_id": case.case_id, "warmup": False,
                             "error": None})

        policy: dict[str, str] = {}
        probabilities: dict[str, dict[str, float]] = {}
        fields_diag: dict[str, Any] = {}
        for f, dec in zip(FIELDS, result):
            probs = np.asarray(dec.probs, dtype=float)
            option_ids = list(OPTIONS[f])  # question options are in OPTIONS order
            probabilities[f] = {o: float(p) for o, p in zip(option_ids, probs)}
            policy[f] = option_ids[int(probs.argmax())]
            diag = dec.diagnostics
            fields_diag[f] = {"level": dec.level, "prior_method": diag.get("prior_method"),
                              "permutations": diag.get("permutations"), "answer_mass": diag.get("answer_mass")}
        prior_n = {f: self._decider._running.get(q.key, (None, 0))[1] for f, q in zip(FIELDS, self._questions)}
        prompts = self._recorder.prompts
        fallbacks = self._recorder.fallbacks
        tokenizer = self._recorder.tokenizer

        def count_input_tokens() -> int:
            return sum(len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts)

        input_tokens = None if self.defer_input_tokens else count_input_tokens()
        raw = json.dumps({
            "anyjev": f"{ANYJEV_VERSION}@{ANYJEV_GIT}", "level": result.level, "backbone": self.backbone,
            "session": self.session, "seq": seq, "prior_running_n": prior_n, "n_prefills": len(prompts),
            "label_logprob_fallbacks": fallbacks, "policy": policy, "fields": fields_diag,
        })
        labels, valid, schema_error = check_policy(policy)
        error_type = LABEL_FALLBACK_ERROR if fallbacks else schema_error
        decision = Decision(
            labels=labels, probabilities=[probabilities],
            confidence=float(np.mean([max(p.values()) for p in probabilities.values()])),
            valid=valid and not fallbacks, error_type=error_type, http_status=200, latency_s=latency_s,
            t_send_wall=t_send, t_recv_wall=t_recv,
            # one prefill per rotation, each reading one next token (max_tokens=1); counted after the timed window
            input_tokens=input_tokens, output_tokens=len(prompts),
            resolved_model=self.resolved_model, provider="vllm", raw_response=raw,
        )
        if self.defer_input_tokens:
            decision.count_input_tokens = count_input_tokens
        return decision
