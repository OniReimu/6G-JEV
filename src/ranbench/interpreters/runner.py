"""C1 runner for RANIntent v1: interpreter x condition x split -> edgebench ledger (EXP-2026-003 protocol step 1).

One ledger row per (interpreter, case) in the edgebench row layout (src/edgebench/ledger.py), written as each call
returns; a rerun skips completed (rq, condition, model, case_id, repeat) keys, so an interrupted run resumes.
Cases are shuffled with a seed per (condition, split); within a case the interpreters run in a seeded random order
(hosted calls interleaved per case). Hosted and self-hosted interpreters never share a run; self-hosted runs use one
client and first answer the first `warmup` cases of the seeded order untimed and unrecorded (AnyJev's online label
prior is fed by them, D-2; every AnyJev decide() is appended to anyjev_order.jsonl with its session and sequence
number). The spend cap counts every reported cost already in the ledger and in the probe file. A no-op RTT probe
per hosted interpreter (minimal request, same model and provider pin) runs every `probe_every` cases into
probes.jsonl.
`only_cases` (D-3 gap-fill) keeps the seeded order but issues only the listed case ids; integrity.json records the
list's sha256 and count. It is refused for order-dependent interpreters.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import threading
import time
from typing import Any

from src.edgebench.interpreters.base import Decision, Interpreter, usage_float, usage_int
from src.edgebench.interpreters.transport import (
    get_post_timeout_healthy,
    get_post_timeout_wait,
    get_send_marker,
    reset_send_marker,
)
from src.edgebench.ledger import LedgerWriter, completed_keys, drop_torn_tail, resolve_git_provenance
from src.edgebench.provenance import sha256_file
from src.ranbench.corpus.spec import load_config
from src.ranbench.interpreters.anyjev_l0 import LABEL_FALLBACK_ERROR
from src.ranbench.interpreters.common import SCHEMA_INVALID, RanCase, load_cases, score_policy
from src.ranbench.interpreters.manifest import build_interpreter, load_manifest

# Same values as src/edgebench/runner.py (pinned by a test); not imported from there because that module pulls in
# scipy through edgebench.scoring, which the vLLM client environment does not have.
SERVER_DOWN_ERRORS = frozenset(
    {"ConnectionRefusedError", "RemoteDisconnected", "ConnectionResetError", "BrokenPipeError"}
)
SERVER_ERROR_STREAK_STOP = 3
RQ = "C1"
ORDER_FILE = "anyjev_order.jsonl"
DEFAULT_SEED = 20260925
ERROR_RATE_RERUN_THRESHOLD = 0.05  # EXP-2026-001 rule, protocol.md "Quality controls"
STOP_ERRORS = ("model_mismatch", "reasoning_tokens_nonzero", LABEL_FALLBACK_ERROR)
SELF_HOSTED_WARMUP = 2  # D-2: the same 2 warm-up intents per condition


def verify_corpus_file(corpus_dir: str | Path, rel: str, allow_stub: bool = False) -> tuple[str, str]:
    """(sha256 of the cases file, sha256 of manifest.json); refuses an unfrozen, changed or (unless allowed) stub corpus."""
    corpus_dir = Path(corpus_dir)
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Refusing to start: {manifest_path} missing (RANIntent v1 must be frozen before any call)")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("stub") and not allow_stub:
        raise RuntimeError("Refusing to start: the corpus is a stub dry run; pass allow_stub for a smoke test")
    expected = manifest.get("files", {}).get(rel)
    actual = sha256_file(corpus_dir / rel)
    if expected != actual:
        raise RuntimeError(f"Refusing to start: {rel} sha256 {actual} does not match manifest.json ({expected})")
    return actual, hashlib.sha256(manifest_bytes).hexdigest()


def rtt_probe(interp: Interpreter) -> dict[str, Any]:
    """One no-op request to the interpreter's endpoint (same model and provider pin), with its timing and cost."""
    status, raw, latency_s, t_send, t_recv, err = interp.post(interp.build_probe_request())
    row: dict[str, Any] = {
        "model": interp.name, "http_status": status, "error_type": err or (None if status == 200 else f"HTTP_{status}"),
        "latency_s": latency_s, "t_send_wall": t_send, "t_recv_wall": t_recv,
        "input_tokens": None, "output_tokens": None, "cost_usd": None, "resolved_model": "", "provider": "",
    }
    if status == 200:
        try:
            body = json.loads(raw)
            usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
            row.update(
                input_tokens=usage_int(usage, "prompt_tokens", "input_tokens"),
                output_tokens=usage_int(usage, "completion_tokens", "output_tokens"),
                cost_usd=usage_float(usage, "cost"),
                resolved_model=str(body.get("model", "")), provider=str(body.get("provider", "")),
            )
        except Exception:
            row["error_type"] = "json_parse_error"
    return row


def _rewrite_without(path: Path, drop) -> int:
    """Rewrite a JSONL file without the rows for which drop(row) is true; returns how many were removed."""
    if not path.exists():
        return 0
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keep = [line for line in lines if not drop(json.loads(line))]
    path.write_text("".join(line + "\n" for line in keep), encoding="utf-8")
    return len(lines) - len(keep)


def _count(path: Path, pred) -> int:
    if not path.exists():
        return 0
    return sum(pred(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _guard_order_dependent(name: str, condition: str, out: Path, fresh: bool) -> None:
    """Refuse to resume an order-dependent interpreter; with fresh, delete its ledger rows and order records for this
    condition first (any seed). Order records alone (e.g. an aborted warm-up) also count as a started run."""
    ledger_path, order_path = out / "ledger.jsonl", out / ORDER_FILE
    mine = lambda r: r.get("rq") == RQ and r.get("condition") == condition and r.get("model") == name  # noqa: E731
    # order records carry `condition`; older ones only the run_id "C1_<condition>_<split>_<seed>"
    my_order = lambda r: r.get("model") == name and (  # noqa: E731
        r.get("condition") == condition or str(r.get("run_id", "")).startswith(f"{RQ}_{condition}_"))
    rows, records = _count(ledger_path, mine), _count(order_path, my_order)
    if not rows and not records:
        return
    if not fresh:
        raise RuntimeError(
            f"Refusing to resume {name} on {condition}: its decisions depend on the order of the intents before them "
            f"(online label prior) and {rows} ledger row(s) / {records} order record(s) already exist; rerun with "
            "fresh to delete them and restart from the warm-up."
        )
    _rewrite_without(ledger_path, mine)
    _rewrite_without(order_path, my_order)


def _is_transport_error(row: dict[str, Any]) -> bool:
    return row.get("http_status") != 200


def read_only_cases(path: str | Path) -> tuple[list[str], str]:
    """(case ids, sha256 of the file) from a gap-fill list: one case_id per line, blank lines ignored."""
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"Refusing to start: {path} lists no case ids")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Refusing to start: {path} lists a case id more than once")
    return ids, sha256_file(path)


def run_c1(
    corpus_dir: str | Path,
    condition: str,
    split: str,
    interpreter_names: list[str],
    out_dir: str | Path,
    workers: int = 4,
    seed: int = DEFAULT_SEED,
    spend_cap_usd: float = 20.0,
    allow_dirty: bool = False,
    allow_stub: bool = False,
    manifest_path: str | Path | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    probe_every: int = 0,
    limit: int | None = None,
    client_location: str | None = None,
    warmup: int = SELF_HOSTED_WARMUP,
    fresh: bool = False,
    interpreters: list[Interpreter] | None = None,
    only_cases: str | Path | None = None,
) -> dict[str, Any]:
    """Run one (condition, split) for the named interpreters; returns (and writes) integrity.json.

    An order-dependent interpreter (AnyJev: online label prior) is never resumed: with rows of it for this condition
    already in the ledger the run is refused unless `fresh`, which first deletes them (and its order records) so the
    run starts again from the warm-up.

    `only_cases` is a gap-fill list (one case_id per line, D-3): the seeded order is kept (warm-ups included) but
    only the listed cases are issued. Every listed id must be a case of this condition and split. Refused for an
    order-dependent interpreter, whose decisions depend on the intents before them.

    `interpreters` injects prebuilt adapters (tests); otherwise they come from the manifest.
    """
    git_sha, is_dirty, git_source, provenance_tree_sha256 = resolve_git_provenance()
    if is_dirty and not allow_dirty:
        raise RuntimeError("Refusing to start: git working tree has modified tracked files. Pass allow_dirty to proceed.")
    rel = f"RQ4/{condition}/{split}.jsonl"
    cases_sha256, corpus_manifest_sha256 = verify_corpus_file(corpus_dir, rel, allow_stub=allow_stub)
    cfg = load_config(Path(corpus_dir) / "config.json")

    manifest = load_manifest(manifest_path)
    if interpreters is None:
        interpreters = [build_interpreter(n, manifest, api_key=api_key, base_url=base_url) for n in interpreter_names]
    names = [m.name for m in interpreters]
    deployments = {m.deployment for m in interpreters}
    if deployments == {"hosted", "self-hosted"}:
        raise ValueError("Refusing to mix self-hosted and hosted interpreters in one run.")
    hosted = "hosted" in deployments
    if only_cases is not None:
        dependent = [m.name for m in interpreters if getattr(m, "order_dependent", False)]
        if dependent:
            raise ValueError(f"Refusing only_cases for order-dependent interpreter(s) {', '.join(dependent)}.")
    if hosted and not client_location:
        raise ValueError("Hosted runs record the client location (protocol.md): pass client_location.")
    if hosted:
        warmup = 0
    else:
        workers = 1

    cases: list[RanCase] = load_cases(Path(corpus_dir) / rel)
    scope = f"{condition}/{split}"
    rng = random.Random((seed + int(hashlib.sha256(scope.encode()).hexdigest()[:8], 16)) & 0x7FFFFFFF)
    rng.shuffle(cases)
    if limit is not None:
        cases = cases[:limit]
    seeded = cases
    only_record = None
    if only_cases is not None:
        only_ids, only_sha256 = read_only_cases(only_cases)
        wanted = set(only_ids)
        unknown = sorted(wanted - {c.case_id for c in cases})
        if unknown:
            raise ValueError(
                f"Refusing to start: {len(unknown)} listed case id(s) not in {rel}: {', '.join(unknown[:5])}")
        cases = [c for c in cases if c.case_id in wanted]
        only_record = {"path": str(only_cases), "sha256": only_sha256, "count": len(only_ids)}
    case_ids = {c.case_id for c in cases}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "ledger.jsonl"
    probes_path = out / "probes.jsonl"
    run_id = f"{RQ}_{condition}_{split}_{seed}"
    torn_before = drop_torn_tail(ledger_path)
    for m in interpreters:
        if getattr(m, "order_dependent", False):
            _guard_order_dependent(m.name, condition, out, fresh)
    writer = LedgerWriter(
        path=ledger_path, run_id=run_id, git_sha=git_sha, rq=RQ, condition=condition, allow_dirty=allow_dirty,
        git_source=git_source, provenance_tree_sha256=provenance_tree_sha256, cases_sha256=cases_sha256,
    )
    completed = completed_keys(ledger_path)

    def _spend(path: Path) -> float:
        total = 0.0
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        cost = json.loads(line).get("cost_usd")
                    except json.JSONDecodeError:
                        continue
                    total += float(cost) if cost is not None else 0.0
        return total

    lock = threading.Lock()
    state: dict[str, Any] = {"spend": _spend(ledger_path) + _spend(probes_path), "stop": None, "probes": 0}
    streak: dict[str, int] = {}

    def check_stop(model: Interpreter, decision: Decision, unhealthy: bool) -> str | None:
        if state["spend"] >= spend_cap_usd:
            return f"spend_cap_exceeded (${state['spend']:.4f} >= ${spend_cap_usd})"
        if decision.http_status in (401, 403):
            return f"HTTP_{decision.http_status}"
        if decision.error_type in STOP_ERRORS:
            return str(decision.error_type)
        if model.deployment == "self-hosted":
            if (decision.http_status is None and decision.error_type in SERVER_DOWN_ERRORS) or unhealthy:
                return "self_hosted_server_down"
            if streak[model.name] >= SERVER_ERROR_STREAK_STOP:
                return "self_hosted_server_error"
        return None

    def run_probes(index: int) -> None:
        order = [m for m in interpreters if m.deployment == "hosted"]
        random.Random(f"{seed}:{scope}:probe:{index}").shuffle(order)
        for m in order:
            row = {"run_id": run_id, "condition": condition, "split": split, "case_index": index,
                   "client_location": client_location, **rtt_probe(m)}
            with lock:
                with open(probes_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                state["probes"] += 1
                if row["cost_usd"] is not None:
                    state["spend"] += row["cost_usd"]
                if state["spend"] >= spend_cap_usd and not state["stop"]:
                    state["stop"] = f"spend_cap_exceeded (${state['spend']:.4f} >= ${spend_cap_usd})"

    warmups: dict[str, list[dict[str, Any]]] = {}
    if warmup > 0 and any((RQ, condition, m.name, c.case_id, 0) not in completed for m in interpreters for c in cases):
        for m in interpreters:
            warmups[m.name] = []
            for c in seeded[:warmup]:
                reset_send_marker()
                if hasattr(m, "warmup"):
                    entry = m.warmup(c)
                    warmups[m.name].append({"case_id": c.case_id, "error": entry["error"]})
                else:
                    try:
                        d = m.decide(c)
                        # a transport/protocol failure is a warm-up error; an ordinary (e.g. schema-invalid) answer is not
                        failed = d.http_status != 200 or d.error_type == LABEL_FALLBACK_ERROR
                        warmups[m.name].append({"case_id": c.case_id, "error": d.error_type if failed else None,
                                                "outcome_error_type": d.error_type, "latency_s": d.latency_s})
                    except Exception as exc:
                        warmups[m.name].append({"case_id": c.case_id, "error": type(exc).__name__})
        # The warm-ups are part of the protocol (D-2): any failed one aborts the condition before a timed call.
        failed = [f"{n}:{w['case_id']}:{w['error']}" for n, ws in warmups.items() for w in ws if w["error"]]
        if failed:
            state["stop"] = "warmup_failed (" + ", ".join(failed) + ")"

    def process_case(index: int, case: RanCase) -> None:
        order = list(interpreters)
        random.Random(f"{seed}:{scope}:{case.case_id}").shuffle(order)
        pending = [m for m in order if (RQ, condition, m.name, case.case_id, 0) not in completed]
        if not pending:
            return
        for model in pending:
            with lock:
                if state["stop"]:
                    return
            reset_send_marker()
            try:
                decision = model.decide(case)
            except Exception as exc:
                sent = get_send_marker()
                decision = Decision(
                    labels=[], valid=False, error_type=type(exc).__name__,
                    latency_s=(time.perf_counter() - sent[1]) if sent else None,
                    t_send_wall=sent[0] if sent else 0.0, t_recv_wall=time.time() if sent else 0.0,
                    provider=getattr(model, "provider_slug", "") or "",
                )
            decision.post_timeout_wait_s = get_post_timeout_wait()
            unhealthy = get_post_timeout_healthy() is False
            scores = score_policy(decision, case, cfg)
            with lock:
                writer.write_row(model=model.name, case_id=case.case_id, repeat=0, decision=decision,
                                 platform=model.platform, **scores)
                completed.add((RQ, condition, model.name, case.case_id, 0))
                if decision.cost_usd is not None:
                    state["spend"] += decision.cost_usd
                is_5xx = decision.http_status is not None and 500 <= decision.http_status < 600
                streak[model.name] = streak.get(model.name, 0) + 1 if is_5xx else 0
                if not state["stop"]:
                    state["stop"] = check_stop(model, decision, unhealthy)
        if probe_every > 0 and index % probe_every == 0 and hosted:
            with lock:
                if state["stop"]:
                    return
            run_probes(index)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_case, i, c) for i, c in enumerate(cases)]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    with lock:
                        state["stop"] = state["stop"] or f"exception_{type(exc).__name__}"
    else:
        for i, c in enumerate(cases):
            process_case(i, c)
            if state["stop"]:
                break
    writer.close()
    for m in interpreters:
        if getattr(m, "history", None):
            with open(out / ORDER_FILE, "a", encoding="utf-8") as f:
                for entry in m.history:
                    f.write(json.dumps({"run_id": run_id, "model": m.name, "condition": condition, **entry}) + "\n")

    rows = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("rq") == RQ and r.get("condition") == condition and r.get("case_id") in case_ids:
                rows.append(r)
    keys = [(r["model"], r["case_id"], r["repeat"]) for r in rows]
    per_model: dict[str, Any] = {}
    for name in names:
        mr = [r for r in rows if r["model"] == name]
        transport = sum(_is_transport_error(r) for r in mr)
        errors: dict[str, int] = {}
        for r in mr:
            if r.get("error_type"):
                errors[r["error_type"]] = errors.get(r["error_type"], 0) + 1
        costs = [r["cost_usd"] for r in mr if r.get("cost_usd") is not None]
        per_model[name] = {
            "expected_rows": len(cases),
            "actual_rows": len(mr),
            "valid_rows": sum(bool(r["valid"]) for r in mr),
            "schema_invalid_rows": sum(r.get("error_type") == SCHEMA_INVALID for r in mr),
            "transport_error_rows": transport,
            "transport_error_rate": transport / len(mr) if mr else None,
            "rerun_block_required": bool(mr) and transport / len(mr) > ERROR_RATE_RERUN_THRESHOLD,
            "error_counts_by_type": errors,
            "resolved_model_ids_seen": sorted({r["resolved_model"] for r in mr if r.get("resolved_model")}),
            "total_cost_usd": sum(costs) if costs else None,
            "cost_unknown_rows": len(mr) - len(costs),
            "usage_unreported_rows": sum(r.get("usage_reported") is False for r in mr),
            "reasoning_tokens_missing_rows": sum(bool(r.get("reasoning_tokens_missing")) for r in mr),
        }
    report = {
        "run_id": run_id,
        "rq": RQ,
        "condition": condition,
        "split": split,
        "seed": seed,
        "git_sha": git_sha,
        "git_source": git_source,
        "allow_dirty": allow_dirty,
        "provenance_tree_sha256": provenance_tree_sha256,
        "cases_path": rel,
        "cases_sha256": cases_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "n_cases": len(cases),
        "client_location": client_location,
        "interpreters": {m.name: {**manifest.get(m.name, {}), "effective_base_url": getattr(m, "base_url", None)}
                         for m in interpreters},
        "stop_reason": state["stop"],
        "torn_line_dropped": torn_before or writer.torn_line_dropped,
        "fresh": fresh,
        "total_spend_usd": state["spend"],
        "spend_cap_usd": spend_cap_usd,
        "duplicates": len(keys) - len(set(keys)),
        "probes_written": state["probes"],
        "warmup_cases": [c.case_id for c in seeded[:warmup]] if warmups else [],
        "warmup_outcomes": warmups,
        "case_order": [c.case_id for c in cases],
        "only_cases": only_record,
        "per_model": per_model,
    }
    (out / "integrity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
