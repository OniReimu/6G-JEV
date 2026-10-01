"""Tests for RANIntent v1 C1 analysis pipeline (EXP-2026-003).

Verifies:
1. Near-RT feasibility phi with known counts and bootstrap CI.
2. Availability table, error breakdowns, and > 5% error threshold flagging.
3. Handling of missing interpreters and conditions without errors.
4. Confirmatory H3 hypothesis testing: true for one interpreter, false for another.
5. Multi-interpreter comparisons at c57_fresh (Cochran's Q and McNemar).
6. Quality and cost metrics (field accuracy, actuated-vector match, EM, net latency, fees/1k correct).
7. Sensitivity block comparisons and delta calculation.
8. Deterministic, byte-identical results.md output.
9. End-to-end CLI invocation.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import pytest
import numpy as np

from src.ranbench.c1_analysis import (
    DEFAULT_SEED,
    DELTA_E2_S,
    EXPECTED_CONDITIONS,
    NEAR_RT_DEADLINE_S,
    canonical_model_name,
    find_run_files,
    load_probes,
    load_all_c1_runs,
    load_corpus_cases,
    compute_availability_table,
    compute_near_rt_feasibility_table,
    compute_quality_and_cost_table,
    evaluate_h3_confirmatory,
    compute_exploratory_contrasts,
    compute_cell_count_curves,
    compute_sensitivity_table,
    generate_results_md,
    run_c1_analysis,
)
from scripts.rb_c1_analyze import main as cli_main


def _make_case_record(case_id: str, condition: str, half: str = "state_dependent") -> dict:
    return {
        "case_id": case_id,
        "condition": condition,
        "n_cells": int(condition.split("_")[0][1:]),
        "quality": condition.split("_")[1],
        "split": "test",
        "half": half,
        "issuer": {"id": "operator", "kind": "operator", "owned_classes": ["video_streaming"]},
        "text": f"Intent text for {case_id}",
        "telemetry": "sample telemetry",
        "truth": {
            "action": "prioritise",
            "class": "video_streaming",
            "scope": "north_cluster",
            "target_cluster": "north_cluster" if half == "state_dependent" else "none",
            "priority": "high",
            "prb_share": "20",
            "edge_site": "on_site",
            "latency_target": "10",
            "duration": "1h",
        },
        "meta": {},
    }


def _make_ledger_row(
    model: str,
    condition: str,
    case_id: str,
    latency_s: float | None = 0.4,
    valid: bool = True,
    error_type: str | None = None,
    correct_all: bool = True,
    tc_correct: bool = True,
    cost_usd: float | None = 0.0001,
    in_tokens: int | None = 500,
    out_tokens: int | None = 50,
    provider: str = "TypeSafe",
) -> dict:
    correct_dict = {
        "action": correct_all,
        "class": correct_all,
        "scope": correct_all,
        "target_cluster": tc_correct if valid else False,
        "priority": correct_all,
        "prb_share": correct_all,
        "edge_site": correct_all,
        "latency_target": correct_all,
        "duration": correct_all,
    }
    em = valid and all(correct_dict.values())
    return {
        "run_id": f"run_{model}_{condition}",
        "rq": "C1",
        "condition": condition,
        "model": model,
        "platform": "hosted",
        "case_id": case_id,
        "repeat": 0,
        "labels": [{"action": "prioritise"}],
        "valid": valid,
        "error_type": error_type,
        "http_status": 200 if error_type is None else 429,
        "latency_s": latency_s,
        "t_send_wall": 1000.0,
        "t_recv_wall": 1000.0 + (latency_s if latency_s is not None else 1.0),
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "cost_usd": cost_usd,
        "provider": provider,
        "correct": correct_dict,
        "em": em,
        "unsafe_locality": False,
        "spurious_count": 0,
        "missed_count": 0,
        "unspecified_truth_count": 0,
        "specified_truth_count": 9,
    }


# ---------------------------------------------------------------------------
# Test 1: Near-RT Feasibility with Known Count
# ---------------------------------------------------------------------------

def test_near_rt_feasibility_known_count(tmp_path: Path):
    """Near-RT feasibility phi = P(latency + delta_e2 <= 1.0s) on a known count."""
    run_dir = tmp_path / "run_primary"
    run_dir.mkdir(parents=True)
    ledger_file = run_dir / "ledger.jsonl"

    # 100 cases: exactly 80 meet deadline (lat=0.5s), 20 fail (lat=1.2s)
    rows = []
    for i in range(100):
        lat = 0.5 if i < 80 else 1.2
        rows.append(_make_ledger_row("Jev-1.13.0", "c3_fresh", f"c_{i:04d}", latency_s=lat))
    ledger_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    loaded = load_all_c1_runs([ledger_file])
    res_rows, csv_str = compute_near_rt_feasibility_table(
        loaded,
        expected_models=["Jev-1.13.0"],
        expected_conditions=["c3_fresh"],
        seed=DEFAULT_SEED,
        n_resamples=1000,
    )

    assert len(res_rows) == 1
    r = res_rows[0]
    assert r["n"] == 100
    assert r["n_feasible"] == 80
    assert pytest.approx(r["phi"], abs=1e-5) == 0.80
    assert r["phi_ci_low"] <= 0.80 <= r["phi_ci_high"]
    assert 0.70 < r["phi_ci_low"] < 0.80
    assert 0.80 < r["phi_ci_high"] < 0.90


# ---------------------------------------------------------------------------
# Test 2: Availability & >5% Error Flagging
# ---------------------------------------------------------------------------

def test_availability_and_429_flag_gt_5pct(tmp_path: Path):
    """Availability table records error breakdown and flags > 5% transport/protocol error rate."""
    run_dir = tmp_path / "run_primary"
    run_dir.mkdir(parents=True)
    ledger_file = run_dir / "ledger.jsonl"

    rows = []
    # Arm 1: GLM-5.3-Flash with 10 HTTP_429 errors out of 100 calls = 10% (> 5%)
    for i in range(100):
        err = "HTTP_429" if i < 10 else None
        valid = err is None
        lat = None if err else 0.4
        rows.append(_make_ledger_row("GLM-5.3-Flash", "c57_contradictory", f"c_{i:04d}", latency_s=lat, valid=valid, error_type=err))

    # Arm 2: DeepSeek-V4.1-Flash with 2 TimeoutErrors out of 100 calls = 2% (<= 5%)
    for i in range(100):
        err = "TimeoutError" if i < 2 else None
        valid = err is None
        lat = None if err else 0.4
        rows.append(_make_ledger_row("DeepSeek-V4.1-Flash", "c57_contradictory", f"c_{i:04d}", latency_s=lat, valid=valid, error_type=err))

    ledger_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    loaded = load_all_c1_runs([ledger_file])
    res_rows, csv_str = compute_availability_table(
        loaded,
        expected_models=["GLM-5.3-Flash", "DeepSeek-V4.1-Flash"],
        expected_conditions=["c57_contradictory"],
    )

    r_glm = next(r for r in res_rows if r["model"] == "GLM-5.3-Flash")
    assert r_glm["primary_n"] == 100
    assert r_glm["primary_errors"] == 10
    assert r_glm["primary_error_types"] == "HTTP_429: 10"
    assert pytest.approx(r_glm["primary_avail_rate"]) == 0.90
    assert r_glm["error_gt_5pct"] is True

    r_ds = next(r for r in res_rows if r["model"] == "DeepSeek-V4.1-Flash")
    assert r_ds["primary_n"] == 100
    assert r_ds["primary_errors"] == 2
    assert r_ds["primary_error_types"] == "TimeoutError: 2"
    assert pytest.approx(r_ds["primary_avail_rate"]) == 0.98
    assert r_ds["error_gt_5pct"] is False


# ---------------------------------------------------------------------------
# Test 3: Missing Interpreters & Incomplete Conditions
# ---------------------------------------------------------------------------

def test_missing_interpreters_handling(tmp_path: Path):
    """Pipeline gracefully handles missing models and conditions without crashing."""
    run_dir = tmp_path / "run_primary"
    run_dir.mkdir(parents=True)
    ledger_file = run_dir / "ledger.jsonl"

    # Only Jev-1.13.0 on c3_fresh
    rows = [_make_ledger_row("Jev-1.13.0", "c3_fresh", f"c_{i:04d}") for i in range(50)]
    ledger_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    loaded = load_all_c1_runs([ledger_file])
    avail_rows, _ = compute_availability_table(loaded)

    # Check Jev status is partial (50 < 300)
    jev_c3 = next(r for r in avail_rows if r["model"] == "Jev-1.13.0" and r["condition"] == "c3_fresh")
    assert jev_c3["status"] == "partial"
    assert jev_c3["primary_n"] == 50

    # Check missing model status is missing
    qwen_c3 = next(r for r in avail_rows if r["model"] == "Qwen3.8-Flash" and r["condition"] == "c3_fresh")
    assert qwen_c3["status"] == "missing"
    assert qwen_c3["primary_n"] == 0

    # Near-RT feasibility reports NaN for missing models
    near_rt, _ = compute_near_rt_feasibility_table(loaded, expected_models=["Qwen3.8-Flash"], expected_conditions=["c3_fresh"])
    assert np.isnan(near_rt[0]["phi"])


# ---------------------------------------------------------------------------
# Test 4: H3 True for One Interpreter, False for Another
# ---------------------------------------------------------------------------

def test_h3_confirmatory_paired_contrast(tmp_path: Path):
    """H3 is resolved (True) for an interpreter with c3 - c57 > 0 and False for another."""
    run_dir = tmp_path / "run_primary"
    run_dir.mkdir(parents=True)
    ledger_file = run_dir / "ledger.jsonl"

    # Create dummy corpus cases
    corpus_dir = tmp_path / "corpus"
    c3_dir = corpus_dir / "RQ4" / "c3_fresh"
    c57_dir = corpus_dir / "RQ4" / "c57_fresh"
    c3_dir.mkdir(parents=True)
    c57_dir.mkdir(parents=True)

    c3_cases = [_make_case_record(f"ranintent_v1_test_{i:04d}", "c3_fresh", "state_dependent") for i in range(150)]
    c57_cases = [_make_case_record(f"ranintent_v1_test_{i:04d}", "c57_fresh", "state_dependent") for i in range(150)]

    (c3_dir / "test.jsonl").write_text("\n".join(json.dumps(c) for c in c3_cases) + "\n", encoding="utf-8")
    (c57_dir / "test.jsonl").write_text("\n".join(json.dumps(c) for c in c57_cases) + "\n", encoding="utf-8")

    corpus = load_corpus_cases(corpus_dir)

    rows = []
    # Model 1: Jev-1.13.0 -> huge degradation from 3 to 57 cells (acc=1.0 at c3, acc=0.5 at c57)
    for i in range(150):
        cid = f"ranintent_v1_test_{i:04d}"
        rows.append(_make_ledger_row("Jev-1.13.0", "c3_fresh", cid, tc_correct=True))
        rows.append(_make_ledger_row("Jev-1.13.0", "c57_fresh", cid, tc_correct=(i < 75)))

    # Model 2: DeepSeek-V4.1-Flash -> no degradation (acc=0.8 at c3, acc=0.82 at c57)
    for i in range(150):
        cid = f"ranintent_v1_test_{i:04d}"
        rows.append(_make_ledger_row("DeepSeek-V4.1-Flash", "c3_fresh", cid, tc_correct=(i < 120)))
        rows.append(_make_ledger_row("DeepSeek-V4.1-Flash", "c57_fresh", cid, tc_correct=(i < 123)))

    ledger_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    loaded = load_all_c1_runs([ledger_file])

    h3_rows, comp_rows, _, _ = evaluate_h3_confirmatory(
        loaded,
        corpus,
        seed=DEFAULT_SEED,
        n_resamples=1000,
    )

    r_jev = next(r for r in h3_rows if r["model"] == "Jev-1.13.0")
    assert pytest.approx(r_jev["acc_c3_fresh"]) == 1.0000
    assert pytest.approx(r_jev["acc_c57_fresh"]) == 0.5000
    assert pytest.approx(r_jev["delta_c3_minus_c57"]) == +0.5000
    assert r_jev["ci_low"] > 0.35
    assert r_jev["p_holm"] < 0.05
    assert r_jev["resolved"] is True

    r_ds = next(r for r in h3_rows if r["model"] == "DeepSeek-V4.1-Flash")
    assert r_ds["delta_c3_minus_c57"] <= 0.0
    assert r_ds["ci_low"] <= 0.0
    assert r_ds["resolved"] is False

    # Cochran's Q needs all six -> incomplete; McNemar DeepSeek vs Jev is tested
    q_row = next(c for c in comp_rows if c["test_type"] == "Cochran_Q")
    assert q_row["status"] == "incomplete"
    ds_mcn = next(c for c in comp_rows if c["test_type"] == "McNemar" and c["model_1"] == "DeepSeek-V4.1-Flash")
    assert ds_mcn["status"] == "tested"


# ---------------------------------------------------------------------------
# Test 5: Quality, Cost, and Net Latency Subtraction
# ---------------------------------------------------------------------------

def test_quality_cost_and_net_latency(tmp_path: Path):
    """Quality and cost metrics: field accuracy, actuated-vector match, tokens, fees, net latency."""
    run_dir = tmp_path / "run_primary"
    run_dir.mkdir(parents=True)
    ledger_file = run_dir / "ledger.jsonl"
    probes_file = run_dir / "probes.jsonl"

    # Probes for (Jev-1.13.0, c3_fresh) in this block: median 0.200s
    probe_rows = [
        {"model": "Jev-1.13.0", "condition": "c3_fresh", "provider": "TypeSafe", "http_status": 200, "error_type": None, "latency_s": 0.200},
        {"model": "Jev-1.13.0", "condition": "c3_fresh", "provider": "TypeSafe", "http_status": 200, "error_type": None, "latency_s": 0.200},
    ]
    probes_file.write_text("\n".join(json.dumps(p) for p in probe_rows) + "\n", encoding="utf-8")
    probe_med = load_probes([probes_file])
    assert pytest.approx(probe_med[("run_primary", "Jev-1.13.0", "c3_fresh")]) == 0.200

    # 10 cases with raw latency 0.500s -> net latency = 0.300s
    # 8 cases full correct, 2 cases wrong action
    rows = []
    for i in range(10):
        corr_all = (i < 8)
        rows.append(_make_ledger_row(
            "Jev-1.13.0", "c3_fresh", f"c_{i:04d}",
            latency_s=0.500,
            correct_all=corr_all,
            cost_usd=0.0010,
            in_tokens=300,
            out_tokens=50,
            provider="TypeSafe",
        ))
    ledger_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    loaded = load_all_c1_runs([ledger_file])
    qc_rows, _ = compute_quality_and_cost_table(
        loaded,
        corpus_cases={},
        probe_medians=probe_med,
        expected_models=["Jev-1.13.0"],
        expected_conditions=["c3_fresh"],
    )

    r = qc_rows[0]
    assert r["n_valid"] == 10
    assert pytest.approx(r["full_policy_match"]) == 0.80
    assert pytest.approx(r["actuated_vector_match"]) == 0.80
    assert pytest.approx(r["action_acc"]) == 0.80
    assert pytest.approx(r["target_cluster_acc"]) == 1.00
    assert pytest.approx(r["latency_p50"]) == 0.500
    assert pytest.approx(r["latency_net_p50"]) == 0.300
    assert pytest.approx(r["mean_input_tokens"]) == 300.0
    assert pytest.approx(r["mean_output_tokens"]) == 50.0

    # Total cost = 10 * 0.0010 = $0.0100 for 8 correct -> fees per 1k = (0.0100 / 8) * 1000 = $1.2500
    assert pytest.approx(r["cost_per_1000_correct_usd"]) == 1.2500


# ---------------------------------------------------------------------------
# Test 6: Sensitivity Re-Run Comparison
# ---------------------------------------------------------------------------

def test_sensitivity_rerun_comparison(tmp_path: Path):
    """Sensitivity table correctly matches primary block with rerun block and computes deltas."""
    primary_dir = tmp_path / "c1_hosted"
    rerun_dir = tmp_path / "c1_rerun" / "c57_contradictory__GLM-5.3-Flash__rerun1"
    primary_dir.mkdir(parents=True)
    rerun_dir.mkdir(parents=True)

    # Primary block: 20 calls, 2 errors (avail = 0.90)
    p_rows = []
    for i in range(20):
        err = "HTTP_429" if i < 2 else None
        p_rows.append(_make_ledger_row(
            "GLM-5.3-Flash", "c57_contradictory", f"c_{i:04d}",
            valid=(err is None), error_type=err, latency_s=None if err else 0.400
        ))
    (primary_dir / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in p_rows) + "\n", encoding="utf-8")

    # Rerun block: 20 calls, 0 errors (avail = 1.00)
    s_rows = [
        _make_ledger_row("GLM-5.3-Flash", "c57_contradictory", f"c_{i:04d}", valid=True, error_type=None, latency_s=0.350)
        for i in range(20)
    ]
    (rerun_dir / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in s_rows) + "\n", encoding="utf-8")

    files = find_run_files([primary_dir, rerun_dir])
    loaded = load_all_c1_runs(files["ledgers"])
    sens_rows, _ = compute_sensitivity_table(loaded, corpus_cases={})

    assert len(sens_rows) > 0
    avail_metric = next(r for r in sens_rows if r["metric_id"] == "availability_rate")
    assert pytest.approx(float(avail_metric["primary_value"])) == 0.9000
    assert pytest.approx(float(avail_metric["sensitivity_value"])) == 1.0000
    assert avail_metric["delta"] == "+0.1000"


# ---------------------------------------------------------------------------
# Test 7: Deterministic Output & CLI Invocation
# ---------------------------------------------------------------------------

def test_results_md_determinism_and_cli(tmp_path: Path):
    """Pipeline produces byte-identical outputs across runs and CLI executes successfully."""
    # Setup corpus
    corpus_dir = tmp_path / "corpus"
    for cond in EXPECTED_CONDITIONS:
        cdir = corpus_dir / "RQ4" / cond
        cdir.mkdir(parents=True)
        cases = [
            _make_case_record(f"ranintent_v1_test_{i:04d}", cond, "state_dependent" if i < 150 else "named_scope")
            for i in range(300)
        ]
        (cdir / "test.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")

    # Setup run ledgers
    run_dir = tmp_path / "runs" / "c1_hosted"
    run_dir.mkdir(parents=True)
    rows = []
    for cond in ["c3_fresh", "c57_fresh"]:
        for m in ["Jev-1.13.0", "DeepSeek-V4.1-Flash"]:
            for i in range(300):
                rows.append(_make_ledger_row(m, cond, f"ranintent_v1_test_{i:04d}", latency_s=0.3))
    (run_dir / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    res1 = run_c1_analysis([run_dir], corpus_dir, out1, seed=DEFAULT_SEED, n_resamples=500)
    res2 = run_c1_analysis([run_dir], corpus_dir, out2, seed=DEFAULT_SEED, n_resamples=500)

    # Check byte-identical results.md
    md1 = (out1 / "results.md").read_bytes()
    md2 = (out2 / "results.md").read_bytes()
    assert md1 == md2, "results.md must be byte-identical on repeated runs with same seed"

    # Check byte-identical CSVs
    for f in res1["files_written"]:
        if f.endswith(".csv"):
            assert (out1 / f).read_bytes() == (out2 / f).read_bytes()

    # Test CLI invocation via main()
    out_cli = tmp_path / "out_cli"
    ret = cli_main(["--runs", str(run_dir), "--corpus", str(corpus_dir), "--out", str(out_cli), "--n-resamples", "500"])
    assert ret == 0
    assert (out_cli / "results.md").exists()
    assert (out_cli / "h3_confirmatory.csv").exists()


def test_energy_trace_integration(tmp_path: Path):
    """Energy per decision is integrated from GPU power trace when present."""
    run_dir = tmp_path / "c1_selfhosted" / "SemIf-Qwen3.5-4B" / "c3_fresh" / "test"
    run_dir.mkdir(parents=True)
    ledger_file = run_dir / "ledger.jsonl"
    power_file = run_dir / "power.test.csv"

    # 10 decisions starting at epoch 1000.0, ending at epoch 1010.0 (10 s window)
    rows = []
    for i in range(10):
        r = _make_ledger_row("SemIf-Qwen3.5-4B", "c3_fresh", f"c_{i:04d}", latency_s=0.5)
        r["t_send_wall"] = 1000.0 + i * 1.0
        r["t_recv_wall"] = 1000.0 + i * 1.0 + 0.5
        rows.append(r)
    ledger_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    # Power trace sampled every 1 s from 00:00:00 to 00:00:10 (11 samples at 100 W)
    trace_lines = ["# gpu=0 tz=+0000 fields=timestamp,power.draw[W],utilization.gpu[%],memory.used[MiB]"]
    for s in range(11):
        trace_lines.append(f"2026/09/25 00:00:{s:02d}.000, 100.0, 50, 2000")
    power_file.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")
    # The datetime corresponds to epoch 1790294400; align ledger rows to this epoch
    t0 = 1790294400.0
    rows = []
    for i in range(10):
        r = _make_ledger_row("SemIf-Qwen3.5-4B", "c3_fresh", f"c_{i:04d}", latency_s=0.5)
        r["t_send_wall"] = t0 + i * 1.0
        r["t_recv_wall"] = t0 + i * 1.0 + 0.5
        rows.append(r)
    ledger_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    files = find_run_files([run_dir])
    loaded = load_all_c1_runs(files["ledgers"])
    qc_rows, _ = compute_quality_and_cost_table(
        loaded,
        corpus_cases={},
        power_traces=files["power_traces"],
        expected_models=["SemIf-Qwen3.5-4B"],
        expected_conditions=["c3_fresh"],
    )

    r = qc_rows[0]
    assert r["energy_j_per_decision"] != ""
    val = float(r["energy_j_per_decision"])
    # ~9.5s * 100W = 950J / 10 decisions = 95 J/dec
    assert 80.0 <= val <= 110.0


def _write_power_trace(path: Path, watts: float) -> None:
    """11 samples, 1 s apart, from 2026/09/25 00:00:00 UTC (epoch 1790294400) at constant power."""
    lines = ["# gpu=0 tz=+0000 fields=timestamp,power.draw[W],utilization.gpu[%],memory.used[MiB]"]
    lines += [f"2026/09/25 00:00:{s:02d}.000, {watts}, 50, 2000" for s in range(11)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_timed_ledger(path: Path, model: str, cost_usd: float | None) -> None:
    """10 decisions inside the power-trace window, one per second, 0.5 s each."""
    t0 = 1790294400.0
    rows = []
    for i in range(10):
        r = _make_ledger_row(model, "c3_fresh", f"c_{i:04d}", latency_s=0.5, cost_usd=cost_usd)
        r["t_send_wall"] = t0 + i * 1.0
        r["t_recv_wall"] = t0 + i * 1.0 + 0.5
        rows.append(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _energy_cost_rows(tmp_path: Path) -> dict[str, dict]:
    """Hosted Jev and two self-hosted interpreters run over the same time window. Each
    self-hosted job has its own trace (SemIf 100 W, AnyJev 300 W); a third trace sits beside
    the hosted run. Only the AnyJev ledger reports a cost (None), SemIf reports 0.0."""
    root = tmp_path / "runs"
    _write_timed_ledger(root / "c1_hosted" / "c3_fresh" / "test" / "ledger.jsonl", "Jev-1.13.0", 0.0002)
    _write_power_trace(root / "c1_hosted" / "power.hosted.csv", 500.0)
    _write_timed_ledger(root / "c1_selfhosted" / "semif" / "c3_fresh" / "test" / "ledger.jsonl",
                        "SemIf-Qwen3.5-4B", 0.0)
    _write_power_trace(root / "c1_selfhosted" / "semif" / "power.1.csv", 100.0)
    _write_timed_ledger(root / "c1_selfhosted" / "anyjev" / "c3_fresh" / "test" / "ledger.jsonl",
                        "AnyJev-L0", None)
    _write_power_trace(root / "c1_selfhosted" / "anyjev" / "power.0.csv", 300.0)
    files = find_run_files([root / "c1_hosted", root / "c1_selfhosted"])
    assert len(files["power_traces"]) == 3
    loaded = load_all_c1_runs(files["ledgers"], files["block_of"])
    qc_rows, _ = compute_quality_and_cost_table(
        loaded,
        corpus_cases={},
        power_traces=files["power_traces"],
        expected_models=["Jev-1.13.0", "SemIf-Qwen3.5-4B", "AnyJev-L0"],
        expected_conditions=["c3_fresh"],
    )
    return {r["model"]: r for r in qc_rows}


def test_energy_is_na_for_hosted_interpreter(tmp_path: Path):
    """A hosted interpreter has no GPU power trace, even when traces overlap its time window."""
    rows = _energy_cost_rows(tmp_path)
    assert rows["Jev-1.13.0"]["n_attempts"] == 10
    assert rows["Jev-1.13.0"]["energy_j_per_decision"] == ""


def test_energy_selfhosted_uses_own_trace_only(tmp_path: Path):
    """Each self-hosted interpreter integrates the trace of its own job directory:
    samples at 0..9 s lie in [t0, t0 + 9.5], so 9 s x W / 10 decisions = 0.9 x W J per decision."""
    rows = _energy_cost_rows(tmp_path)
    assert float(rows["SemIf-Qwen3.5-4B"]["energy_j_per_decision"]) == pytest.approx(90.0)
    assert float(rows["AnyJev-L0"]["energy_j_per_decision"]) == pytest.approx(270.0)


def test_energy_na_when_selfhosted_trace_is_ambiguous(tmp_path: Path):
    """Two traces in the job directory -> no unique trace -> NA."""
    root = tmp_path / "c1_selfhosted" / "semif"
    _write_timed_ledger(root / "c3_fresh" / "test" / "ledger.jsonl", "SemIf-Qwen3.5-4B", 0.0)
    _write_power_trace(root / "power.1.csv", 100.0)
    _write_power_trace(root / "power.2.csv", 200.0)
    files = find_run_files([root])
    loaded = load_all_c1_runs(files["ledgers"], files["block_of"])
    qc_rows, _ = compute_quality_and_cost_table(
        loaded, corpus_cases={}, power_traces=files["power_traces"],
        expected_models=["SemIf-Qwen3.5-4B"], expected_conditions=["c3_fresh"],
    )
    assert qc_rows[0]["energy_j_per_decision"] == ""


def test_selfhosted_cost_is_zero(tmp_path: Path):
    """Self-hosted fees are 0 whether the ledger reports cost_usd 0.0 or None; hosted is priced."""
    rows = _energy_cost_rows(tmp_path)
    for m in ("SemIf-Qwen3.5-4B", "AnyJev-L0"):
        assert rows[m]["cost_per_1000_correct_usd"] == 0.0
        assert rows[m]["n_cost_rows"] == 10
    assert rows["Jev-1.13.0"]["cost_per_1000_correct_usd"] == pytest.approx(0.0002 * 10 / 10 * 1000.0)


def test_exploratory_and_curves(tmp_path: Path):
    """Exploratory contrasts and accuracy vs cell count curves compute properly."""
    corpus_dir = tmp_path / "corpus"
    for cond in EXPECTED_CONDITIONS:
        cdir = corpus_dir / "RQ4" / cond
        cdir.mkdir(parents=True)
        cases = [
            _make_case_record(f"ranintent_v1_test_{i:04d}", cond, "state_dependent" if i < 150 else "named_scope")
            for i in range(300)
        ]
        (cdir / "test.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")

    run_dir = tmp_path / "c1_hosted"
    run_dir.mkdir(parents=True)
    rows = []
    for cond in EXPECTED_CONDITIONS:
        for i in range(300):
            rows.append(_make_ledger_row("Jev-1.13.0", cond, f"ranintent_v1_test_{i:04d}", tc_correct=(i < 200)))
    (run_dir / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    corpus = load_corpus_cases(corpus_dir)
    loaded = load_all_c1_runs([run_dir / "ledger.jsonl"])

    exp_rows, exp_csv = compute_exploratory_contrasts(loaded, corpus, expected_models=["Jev-1.13.0"], n_resamples=200)
    assert len(exp_rows) == 3  # stale, noisy, contradictory
    assert all(r["status"] == "tested" for r in exp_rows)

    curves_rows, curves_csv = compute_cell_count_curves(loaded, corpus, expected_models=["Jev-1.13.0"])
    assert len(curves_rows) == 16  # 4 qualities * 4 cell counts



# ---------------------------------------------------------------------------
# Review fixes (one test per finding; expected values computed by hand)
# ---------------------------------------------------------------------------

from src.ranbench.c1_analysis import CorpusCase, THE_SIX_MODELS, holm_fixed_family


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _corpus(conds: dict[str, list[str]], n_named: int = 0) -> dict:
    """condition -> list of state-dependent ids (plus n_named named-scope ids)."""
    out = {}
    for cond, sd_ids in conds.items():
        cases = {}
        for cid in sd_ids:
            cases[cid] = CorpusCase(cid, cond, 0, "fresh", "test", "state_dependent", {})
        for i in range(n_named):
            cid = f"named_{i:04d}"
            cases[cid] = CorpusCase(cid, cond, 0, "fresh", "test", "named_scope", {})
        out[cond] = cases
    return out


SD_IDS = [f"ranintent_v1_test_{i:04d}" for i in range(150)]


def _h3_rows_for(model: str, c3_correct: set[int], c57_correct: set[int]) -> list[dict]:
    rows = []
    for i, cid in enumerate(SD_IDS):
        rows.append(_make_ledger_row(model, "c3_fresh", cid, tc_correct=(i in c3_correct)))
        rows.append(_make_ledger_row(model, "c57_fresh", cid, tc_correct=(i in c57_correct)))
    return rows


def test_holm_family_fixed_literal():
    """[1] Holm over the six; untested members stay in the family (p=1) and come back NaN."""
    raw = {"Jev-1.13.0": 0.01, "DeepSeek-V4.1-Flash": 0.02, "GLM-5.3-Flash": 0.04}
    adj = holm_fixed_family(raw, THE_SIX_MODELS)
    # m=6: 0.01*6=0.06, 0.02*5=0.10, 0.04*4=0.16 (a shrunken family of 3 would give 0.03/0.04/0.04)
    assert adj["Jev-1.13.0"] == pytest.approx(0.06)
    assert adj["DeepSeek-V4.1-Flash"] == pytest.approx(0.10)
    assert adj["GLM-5.3-Flash"] == pytest.approx(0.16)
    for m in ("Qwen3.8-Flash", "SemIf-Qwen3.5-4B", "AnyJev-L0"):
        assert np.isnan(adj[m])


def test_h3_family_fixed_and_comparisons_incomplete(tmp_path: Path):
    """[1][2] Qwen3.8-Flash lacks c57_fresh: untested, not resolved, family stays 6; the
    reference is reported outside the family; Cochran's Q incomplete; McNemar Qwen incomplete."""
    corpus = _corpus({"c3_fresh": SD_IDS, "c57_fresh": SD_IDS})
    all_i = set(range(150))
    designs = {
        # +1 on 10 cases, -1 on 3 -> intermediate p
        "Jev-1.13.0": (set(range(100)), set(range(90)) | {140, 141, 142}),
        "DeepSeek-V4.1-Flash": (set(range(100)), set(range(88)) | {140, 141, 142}),
        "GLM-5.3-Flash": (all_i, set(range(75))),
        "SemIf-Qwen3.5-4B": (set(range(50)), set(range(50))),
        "AnyJev-L0": (set(range(60)), set(range(58)) | {149}),
        "Qwen3.5-4B-JSON": (all_i, set()),  # reference, huge effect
    }
    rows = []
    for m, (c3, c57) in designs.items():
        rows += _h3_rows_for(m, c3, c57)
    rows += [_make_ledger_row("Qwen3.8-Flash", "c3_fresh", cid) for cid in SD_IDS]  # no c57
    lf = _write_jsonl(tmp_path / "c1_hosted" / "ledger.jsonl", rows)
    loaded = load_all_c1_runs([lf])
    h3_rows, comp_rows, _, _ = evaluate_h3_confirmatory(loaded, corpus, n_resamples=2000)
    by = {r["model"]: r for r in h3_rows}

    assert [r["model"] for r in h3_rows] == THE_SIX_MODELS + ["Qwen3.5-4B-JSON"]
    q = by["Qwen3.8-Flash"]
    assert q["status"] == "untested_incomplete" and q["resolved"] is False and np.isnan(q["p_holm"])
    ref = by["Qwen3.5-4B-JSON"]
    assert ref["status"] == "reference" and np.isnan(ref["p_holm"]) and ref["delta_c3_minus_c57"] == pytest.approx(1.0)
    assert by["GLM-5.3-Flash"]["delta_c3_minus_c57"] == pytest.approx(0.5)
    assert by["Jev-1.13.0"]["delta_c3_minus_c57"] == pytest.approx(7 / 150)

    # Hand Holm over 6 members, Qwen3.8-Flash at p=1.0, reference excluded
    tested = {m: by[m]["p_raw"] for m in THE_SIX_MODELS if by[m]["status"] == "tested"}
    full = dict(tested, **{"Qwen3.8-Flash": 1.0})
    order = sorted(full, key=lambda k: full[k])
    running, expected = 0.0, {}
    for rank, k in enumerate(order):
        running = max(running, (6 - rank) * full[k])
        expected[k] = min(1.0, running)
    for m in tested:
        assert by[m]["p_holm"] == pytest.approx(expected[m])
    # The smallest raw p is multiplied by 6 (family of six), not 5 (shrunk) or 7 (reference added)
    smallest = order[0]
    assert by[smallest]["p_holm"] == pytest.approx(min(1.0, 6 * full[smallest]))

    q_row = next(r for r in comp_rows if r["test_type"] == "Cochran_Q")
    assert q_row["status"] == "incomplete" and "Qwen3.8-Flash" in q_row["details"]
    mcn = {r["model_1"]: r for r in comp_rows if r["test_type"] == "McNemar"}
    assert sorted(mcn) == sorted(m for m in THE_SIX_MODELS if m != "Jev-1.13.0")
    assert mcn["Qwen3.8-Flash"]["status"] == "incomplete"
    assert all(mcn[m]["status"] == "tested" for m in mcn if m != "Qwen3.8-Flash")
    assert "Qwen3.5-4B-JSON" not in mcn


def test_cochran_q_exactly_six_and_mcnemar(tmp_path: Path):
    """[2] All six complete: Jev correct on all 150 at c57, the other five on the first 75.
    Q = (k-1)(k*sum C^2 - N^2)/(k*N - sum R^2) = 5*(6*50625 - 525^2)/(3150 - 2775) = 375, df 5.
    McNemar vs Jev: b=0, c=75 -> exact two-sided p = 2 * 0.5^75. The reference is not in Q."""
    corpus = _corpus({"c3_fresh": SD_IDS, "c57_fresh": SD_IDS})
    all_i, half = set(range(150)), set(range(75))
    rows = _h3_rows_for("Jev-1.13.0", all_i, all_i)
    for m in THE_SIX_MODELS[1:]:
        rows += _h3_rows_for(m, all_i, half)
    rows += _h3_rows_for("Qwen3.5-4B-JSON", all_i, set())
    lf = _write_jsonl(tmp_path / "c1_hosted" / "ledger.jsonl", rows)
    _, comp_rows, _, _ = evaluate_h3_confirmatory(load_all_c1_runs([lf]), corpus, n_resamples=200)

    q_row = next(r for r in comp_rows if r["test_type"] == "Cochran_Q")
    assert q_row["status"] == "tested"
    assert q_row["statistic_value"] == pytest.approx(375.0)
    assert q_row["df"] == 5
    mcn = [r for r in comp_rows if r["test_type"] == "McNemar"]
    assert len(mcn) == 5
    for r in mcn:
        assert r["statistic_value"] == "0/75"
        assert r["p_value"] == pytest.approx(2 * 0.5 ** 75, rel=1e-9)


def test_h3_corpus_state_dependent_ids_must_match():
    """[3] c3_fresh and c57_fresh must define the same 150 state-dependent ids."""
    loaded = load_all_c1_runs([])
    other = SD_IDS[:-1] + ["ranintent_v1_test_9999"]
    with pytest.raises(ValueError, match="state-dependent ids differ"):
        evaluate_h3_confirmatory(loaded, _corpus({"c3_fresh": SD_IDS, "c57_fresh": other}))
    with pytest.raises(ValueError, match="expected 150"):
        evaluate_h3_confirmatory(loaded, _corpus({"c3_fresh": SD_IDS[:149], "c57_fresh": SD_IDS[:149]}))
    with pytest.raises(ValueError, match="c57_fresh"):
        evaluate_h3_confirmatory(loaded, _corpus({"c3_fresh": SD_IDS}))


def test_quality_over_all_attempts_and_cost_same_rows(tmp_path: Path):
    """[4][5] 10 attempts: 6 correct (one without cost), 2 wrong action, 1 schema-invalid, 1 HTTP 429.
    Correctness over 10; schema-valid 8/10; latency over the 10 rows reporting it; tokens over 9;
    fees over the 8 cost-reporting rows (5 correct among them): 0.008 / 5 * 1000 = 1.6."""
    rows = []
    for i in range(6):
        rows.append(_make_ledger_row("Jev-1.13.0", "c3_fresh", f"c_{i}", latency_s=0.5,
                                     cost_usd=None if i == 0 else 0.001, in_tokens=100, out_tokens=10))
    for i in (6, 7):
        r = _make_ledger_row("Jev-1.13.0", "c3_fresh", f"c_{i}", latency_s=0.5, cost_usd=0.001,
                             in_tokens=100, out_tokens=10)
        r["correct"]["action"] = False
        r["em"] = False
        rows.append(r)
    r8 = _make_ledger_row("Jev-1.13.0", "c3_fresh", "c_8", latency_s=0.5, cost_usd=0.001,
                          in_tokens=100, out_tokens=10)
    r8["valid"] = False
    r8["correct"] = {k: True for k in r8["correct"]}
    r8["em"] = True  # schema-invalid: must still score false
    rows.append(r8)
    r9 = _make_ledger_row("Jev-1.13.0", "c3_fresh", "c_9", latency_s=2.0, valid=False, error_type="HTTP_429",
                          cost_usd=None, in_tokens=None, out_tokens=None)
    r9["correct"] = {k: True for k in r9["correct"]}
    rows.append(r9)
    lf = _write_jsonl(tmp_path / "c1_hosted" / "ledger.jsonl", rows)

    qc, _ = compute_quality_and_cost_table(load_all_c1_runs([lf]), corpus_cases={},
                                           expected_models=["Jev-1.13.0"], expected_conditions=["c3_fresh"])
    r = qc[0]
    assert r["n_attempts"] == 10 and r["n_valid"] == 8
    assert r["schema_valid_rate"] == pytest.approx(0.8)
    assert r["full_policy_match"] == pytest.approx(0.6)
    assert r["actuated_vector_match"] == pytest.approx(0.6)
    assert r["action_acc"] == pytest.approx(0.6)
    assert r["class_acc"] == pytest.approx(0.8)
    assert r["target_cluster_acc"] == pytest.approx(0.8)
    assert r["n_latency"] == 10
    assert r["latency_p50"] == pytest.approx(0.5)
    assert r["latency_p95"] == pytest.approx(1.325)  # 0.5 + 0.55 * (2.0 - 0.5)
    assert r["mean_input_tokens"] == pytest.approx(100.0)
    assert r["n_cost_rows"] == 8 and r["n_correct_cost_rows"] == 5
    assert r["cost_per_1000_correct_usd"] == pytest.approx(1.6)
    assert np.isnan(r["target_cluster_state_dep_acc"])  # no corpus -> no fabricated ids


def test_primary_block_rule_and_duplicates(tmp_path: Path):
    """[6] Main run first; a stopped main arm yields to the earliest complete re-run block;
    every other block is sensitivity-only; duplicate keys within a block raise."""
    ids = [f"c_{i}" for i in range(5)]
    expected = {"c3_fresh": set(ids)}

    def rows(model, n, t0, err_at=None):
        out = []
        for i in range(n):
            e = "HTTP_429" if i == err_at else None
            r = _make_ledger_row(model, "c3_fresh", ids[i], valid=e is None, error_type=e,
                                 latency_s=0.4)
            r["t_send_wall"] = t0 + i
            out.append(r)
        return out

    _write_jsonl(tmp_path / "c1_hosted" / "ledger.jsonl",
                 rows("DeepSeek-V4.1-Flash", 3, 100, err_at=0) + rows("GLM-5.3-Flash", 5, 100, err_at=1))
    _write_jsonl(tmp_path / "c1_rerun" / "c3_fresh__GLM-5.3-Flash__rerun1" / "ledger.jsonl",
                 rows("GLM-5.3-Flash", 5, 500))
    _write_jsonl(tmp_path / "c1_hosted_qwen_block2" / "ledger.jsonl", rows("DeepSeek-V4.1-Flash", 2, 200))
    dirs = [tmp_path / d for d in ("c1_hosted", "c1_rerun", "c1_hosted_qwen_block2")]

    # Without a completing block, DeepSeek-V4.1-Flash has no primary block: incomplete
    files = find_run_files(dirs)
    loaded = load_all_c1_runs(files["ledgers"], files["block_of"], expected)
    assert loaded.primary_block_by_cell[("DeepSeek-V4.1-Flash", "c3_fresh")] is None
    avail, _ = compute_availability_table(loaded, ["DeepSeek-V4.1-Flash", "GLM-5.3-Flash"], ["c3_fresh"])
    a = {r["model"]: r for r in avail}
    assert a["DeepSeek-V4.1-Flash"]["status"] == "incomplete"
    assert a["DeepSeek-V4.1-Flash"]["primary_n"] == 0 and a["DeepSeek-V4.1-Flash"]["sensitivity_n"] == 5
    # The stopped main arm (1 error in 3 = 33.3%) stays flagged; block2 (0/2) does not
    assert a["DeepSeek-V4.1-Flash"]["blocks_error_gt_5pct"] == "c1_hosted (33.3%)"
    assert a["GLM-5.3-Flash"]["primary_block"] == "c1_hosted"
    assert a["GLM-5.3-Flash"]["primary_errors"] == 1
    assert a["GLM-5.3-Flash"]["sensitivity_block_names"] == "c1_rerun/c3_fresh__GLM-5.3-Flash__rerun1"

    # Two complete re-run blocks: the earlier one (by send time, not by name) is primary
    _write_jsonl(tmp_path / "c1_hosted_qwen_blockZ" / "ledger.jsonl", rows("DeepSeek-V4.1-Flash", 5, 300))
    _write_jsonl(tmp_path / "c1_hosted_qwen_blockA" / "ledger.jsonl", rows("DeepSeek-V4.1-Flash", 5, 400))
    dirs += [tmp_path / "c1_hosted_qwen_blockZ", tmp_path / "c1_hosted_qwen_blockA"]
    files = find_run_files(dirs)
    loaded = load_all_c1_runs(files["ledgers"], files["block_of"], expected)
    key = ("DeepSeek-V4.1-Flash", "c3_fresh")
    assert loaded.primary_block_by_cell[key] == "c1_hosted_qwen_blockZ"
    assert len(loaded.primary_by_cell[key]) == 5
    assert {r["_block_name"] for r in loaded.sensitivity_by_cell[key]} == {
        "c1_hosted", "c1_hosted_qwen_block2", "c1_hosted_qwen_blockA"}
    avail, _ = compute_availability_table(loaded, ["DeepSeek-V4.1-Flash"], ["c3_fresh"])
    assert avail[0]["status"] == "complete" and avail[0]["primary_n"] == 5 and avail[0]["primary_errors"] == 0

    # Duplicate (model, condition, case_id, repeat) within a block raises
    dup = _write_jsonl(tmp_path / "dup" / "c1_hosted" / "ledger.jsonl",
                       rows("GLM-5.3-Flash", 2, 100) + rows("GLM-5.3-Flash", 1, 900))
    with pytest.raises(ValueError, match="duplicate primary key"):
        load_all_c1_runs([dup])


def test_malformed_jsonl_raises_with_file_and_line(tmp_path: Path):
    """[7] A malformed line raises with file:line; a torn last line is named as such."""
    good = json.dumps(_make_ledger_row("Jev-1.13.0", "c3_fresh", "c_0"))
    mid = tmp_path / "a" / "ledger.jsonl"
    mid.parent.mkdir()
    mid.write_text(good + "\n{not json\n" + good.replace("c_0", "c_1") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"ledger\.jsonl:2: malformed JSON"):
        load_all_c1_runs([mid])

    torn = tmp_path / "b" / "ledger.jsonl"
    torn.parent.mkdir()
    torn.write_text(good + "\n" + good[:40], encoding="utf-8")
    with pytest.raises(ValueError, match=r"ledger\.jsonl:2: torn last line"):
        load_all_c1_runs([torn])

    probes = tmp_path / "c" / "probes.jsonl"
    probes.parent.mkdir()
    probes.write_text('{"model": "Jev-1.13.0"}\n{"model": \n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"probes\.jsonl:2"):
        load_probes([probes])


def test_net_latency_uses_matching_probe(tmp_path: Path):
    """[8][9] Net latency subtracts the probe median of the same (block, model, condition);
    no probe -> NaN (hosted); self-hosted -> inapplicable."""
    hosted = tmp_path / "c1_hosted"
    rows = []
    for m, c in [("Jev-1.13.0", "c3_fresh"), ("Jev-1.13.0", "c57_fresh"), ("DeepSeek-V4.1-Flash", "c57_fresh")]:
        rows += [_make_ledger_row(m, c, f"c_{i}", latency_s=0.5) for i in range(3)]
    _write_jsonl(hosted / "ledger.jsonl", rows)
    probe = lambda m, c, lat, st=200: {"model": m, "condition": c, "provider": "TypeSafe", "http_status": st,
                                       "error_type": None if st == 200 else "HTTP_429", "latency_s": lat}
    _write_jsonl(hosted / "probes.jsonl", [
        probe("Jev-1.13.0", "c3_fresh", 0.1), probe("Jev-1.13.0", "c3_fresh", 0.3),
        probe("Jev-1.13.0", "c3_fresh", 0.2), probe("Jev-1.13.0", "c3_fresh", 5.0, st=429),
        probe("Jev-1.13.0", "c57_fresh", 0.4), probe("Jev-1.13.0", "c57_fresh", 0.4),
        probe("DeepSeek-V4.1-Flash", "c3_fresh", 0.05),  # other condition: must not be used
    ])
    # A probe for DeepSeek c57_fresh in a different block must not be used either
    _write_jsonl(tmp_path / "c1_rerun" / "x" / "probes.jsonl", [probe("DeepSeek-V4.1-Flash", "c57_fresh", 0.3)])
    _write_jsonl(tmp_path / "c1_selfhosted" / "semif" / "c3_fresh" / "test" / "ledger.jsonl",
                 [_make_ledger_row("SemIf-Qwen3.5-4B", "c3_fresh", f"c_{i}", latency_s=0.5) for i in range(3)])

    files = find_run_files([hosted, tmp_path / "c1_rerun", tmp_path / "c1_selfhosted"])
    loaded = load_all_c1_runs(files["ledgers"], files["block_of"])
    probe_med = load_probes(files["probes"], files["block_of"])
    qc, _ = compute_quality_and_cost_table(
        loaded, {}, probe_med,
        expected_models=["Jev-1.13.0", "DeepSeek-V4.1-Flash", "SemIf-Qwen3.5-4B"],
        expected_conditions=["c3_fresh", "c57_fresh"],
    )
    q = {(r["model"], r["condition"]): r for r in qc}
    assert q[("Jev-1.13.0", "c3_fresh")]["rtt_probe_median"] == pytest.approx(0.2)
    assert q[("Jev-1.13.0", "c3_fresh")]["latency_net_p50"] == pytest.approx(0.3)
    assert q[("Jev-1.13.0", "c57_fresh")]["rtt_probe_median"] == pytest.approx(0.4)
    assert q[("Jev-1.13.0", "c57_fresh")]["latency_net_p50"] == pytest.approx(0.1)
    ds = q[("DeepSeek-V4.1-Flash", "c57_fresh")]
    assert ds["net_latency_status"] == "no_probe" and np.isnan(ds["latency_net_p50"]) and np.isnan(ds["rtt_probe_median"])
    assert ds["latency_p50"] == pytest.approx(0.5)
    sh = q[("SemIf-Qwen3.5-4B", "c3_fresh")]
    assert sh["net_latency_status"] == "inapplicable_selfhosted" and np.isnan(sh["latency_net_p50"])


def test_results_md_reports_actual_bootstrap_settings():
    """[10] results.md states the seed and resample count; non-default values are labelled sensitivity."""
    default_md = generate_results_md([], [], [], [], [], [], [], [])
    assert "seed 20260925, 10,000 bootstrap resamples (pre-registered)" in default_md
    assert "SENSITIVITY" not in default_md
    md = generate_results_md([], [], [], [], [], [], [], [], seed=123, n_resamples=500)
    assert "seed 123, 500 bootstrap resamples -- SENSITIVITY RUN" in md
    assert "20260925, 10,000 resamples" in md  # names the pre-registered values it departs from
    assert "seed 20260925, 10,000 bootstrap resamples" not in md


def test_d3_qwen_per_case_primary_rule(tmp_path: Path):
    """[D-3] Qwen3.8-Flash: primary row per case = first HTTP 200 attempt across blocks in block order
    (c1_hosted first, then by first send time); never succeeding -> its first attempt, a failure; every
    attempt in availability with per-block and per-cell HTTP 429 counts. GLM keeps the block rule."""
    ids = [f"c_{i}" for i in range(6)]
    expected = {"c3_fresh": set(ids)}
    Q = "Qwen3.8-Flash"

    def att(cid, t, kind, lat=0.4, model=Q):
        if kind == "ok":
            r = _make_ledger_row(model, "c3_fresh", cid, latency_s=lat)
        elif kind == "invalid":  # schema-invalid answer: HTTP 200, an outcome
            r = _make_ledger_row(model, "c3_fresh", cid, latency_s=lat, valid=False, error_type="schema_invalid",
                                 correct_all=False)
            r["http_status"] = 200
        else:
            r = _make_ledger_row(model, "c3_fresh", cid, latency_s=0.1, valid=False, error_type="HTTP_429")
        r["t_send_wall"] = float(t)
        return r

    _write_jsonl(tmp_path / "c1_hosted" / "ledger.jsonl", [
        att("c_0", 100, "ok"), att("c_1", 101, "429"), att("c_2", 102, "429"), att("c_3", 103, "429"),
        att("c_5", 105, "429"),
        *[att(c, 100 + i, "429" if i == 2 else "ok", model="GLM-5.3-Flash") for i, c in enumerate(ids)],
    ])
    _write_jsonl(tmp_path / "c1_hosted_qwen_block2" / "ledger.jsonl", [att("c_0", 200, "ok"), att("c_1", 201, "429")])
    _write_jsonl(tmp_path / "c1_hosted_qwen_block3" / "ledger.jsonl", [
        att("c_2", 302, "invalid"), att("c_3", 303, "429"), att("c_4", 304, "ok"), att("c_5", 305, "429")])
    _write_jsonl(tmp_path / "c1_hosted_qwen_gapfill1" / "c3_fresh" / "ledger.jsonl", [
        att("c_1", 501, "ok", lat=0.7), att("c_3", 503, "429"), att("c_5", 505, "429")])
    # gapfill10 sorts before gapfill2 by name; the earlier send time (gapfill2) wins
    _write_jsonl(tmp_path / "c1_hosted_qwen_gapfill2" / "c3_fresh" / "ledger.jsonl", [
        att("c_3", 603, "ok", lat=0.8), att("c_5", 605, "429")])
    _write_jsonl(tmp_path / "c1_hosted_qwen_gapfill10" / "c3_fresh" / "ledger.jsonl", [
        att("c_3", 703, "ok", lat=0.9), att("c_5", 705, "429")])
    # A different repeat for an expected case and an out-of-corpus row are
    # evidence, but neither may create an extra primary observation.
    repeat = att("c_0", 706, "ok", lat=9.9)
    repeat["repeat"] = 1
    rogue = att("not_in_corpus", 707, "ok", lat=9.9)
    with (tmp_path / "c1_hosted_qwen_gapfill10" / "c3_fresh" / "ledger.jsonl").open("a") as handle:
        handle.write(json.dumps(repeat) + "\n")
        handle.write(json.dumps(rogue) + "\n")
    _write_jsonl(tmp_path / "c1_rerun" / "glm" / "ledger.jsonl",
                 [att(c, 900 + i, "ok", model="GLM-5.3-Flash") for i, c in enumerate(ids)])
    probe = lambda lat: {"model": Q, "condition": "c3_fresh", "http_status": 200, "error_type": None, "latency_s": lat}
    for d, lat in [("c1_hosted", 0.1), ("c1_hosted_qwen_block3", 0.2), ("c1_hosted_qwen_gapfill1/c3_fresh", 0.3),
                   ("c1_hosted_qwen_gapfill2/c3_fresh", 0.05), ("c1_hosted_qwen_gapfill10/c3_fresh", 0.6)]:
        _write_jsonl(tmp_path / d / "probes.jsonl", [probe(lat)])
    dirs = [tmp_path / d for d in ("c1_hosted", "c1_hosted_qwen_block2", "c1_hosted_qwen_block3",
                                   "c1_hosted_qwen_gapfill1", "c1_hosted_qwen_gapfill2", "c1_hosted_qwen_gapfill10",
                                   "c1_rerun")]
    files = find_run_files(dirs)
    loaded = load_all_c1_runs(files["ledgers"], files["block_of"], expected)
    key = (Q, "c3_fresh")

    primary = {r["case_id"]: r["_block_name"] for r in loaded.primary_by_cell[key]}
    assert primary == {
        "c_0": "c1_hosted",                              # main block first; block2's success is sensitivity
        "c_1": "c1_hosted_qwen_gapfill1/c3_fresh",       # failed in c1_hosted and block2, succeeded in gap-fill
        "c_2": "c1_hosted_qwen_block3",                  # schema-invalid at HTTP 200: done, not re-issued
        "c_3": "c1_hosted_qwen_gapfill2/c3_fresh",       # first success by send time, not by name
        "c_4": "c1_hosted_qwen_block3",
        "c_5": "c1_hosted",                              # never succeeded: first attempt, a failure
    }
    assert loaded.primary_block_by_cell[key] == "per-case (D-3)" and loaded.primary_complete_by_cell[key]
    assert len(loaded.primary_by_cell[key]) == len(expected["c3_fresh"]) == 6
    assert len(loaded.sensitivity_by_cell[key]) == 20 - 6

    avail, _ = compute_availability_table(loaded, [Q, "GLM-5.3-Flash"], ["c3_fresh"])
    a = {r["model"]: r for r in avail}
    q = a[Q]
    assert q["status"] == "complete" and q["primary_rule"] == "per_case" and q["primary_block"] == "per-case (D-3)"
    assert q["primary_n"] == 6 and q["primary_errors"] == 2
    assert q["primary_error_types"] == "HTTP_429: 1; schema_invalid: 1"
    assert q["sensitivity_n"] == 14 and q["total_n"] == 20
    assert q["total_http_429"] == 11 and q["total_http_429_rate"] == pytest.approx(11 / 20)
    assert q["block_attempts"] == (
        "c1_hosted: 5 attempts, 4 HTTP 429 (80.0%); "
        "c1_hosted_qwen_block2: 2 attempts, 1 HTTP 429 (50.0%); "
        "c1_hosted_qwen_block3: 4 attempts, 2 HTTP 429 (50.0%); "
        "c1_hosted_qwen_gapfill1/c3_fresh: 3 attempts, 2 HTTP 429 (66.7%); "
        "c1_hosted_qwen_gapfill10/c3_fresh: 4 attempts, 1 HTTP 429 (25.0%); "
        "c1_hosted_qwen_gapfill2/c3_fresh: 2 attempts, 1 HTTP 429 (50.0%)")
    g = a["GLM-5.3-Flash"]  # block rule unchanged: the complete main block stays primary with its 429
    assert g["primary_rule"] == "block" and g["primary_block"] == "c1_hosted" and g["primary_errors"] == 1
    assert g["sensitivity_block_names"] == "c1_rerun/glm" and g["total_http_429"] == 1

    qc, _ = compute_quality_and_cost_table(loaded, {}, load_probes(files["probes"], files["block_of"]),
                                           expected_models=[Q], expected_conditions=["c3_fresh"])
    # valid: c_0, c_1, c_3, c_4; latencies .4 .7 .4 .8 .4 .1 -> p50 .4
    # net, each row minus its own block's probe: .3 .4 .2 .75 .2 .0 -> p50 .25; RTTs .1 .3 .2 .05 .2 .1 -> .15
    assert qc[0]["n_attempts"] == 6 and qc[0]["n_valid"] == 4
    assert qc[0]["full_policy_match"] == pytest.approx(4 / 6)
    assert qc[0]["latency_p50"] == pytest.approx(0.4)
    assert qc[0]["net_latency_status"] == "ok"
    assert qc[0]["latency_net_p50"] == pytest.approx(0.25) and qc[0]["rtt_probe_median"] == pytest.approx(0.15)
    nrt, _ = compute_near_rt_feasibility_table(loaded, [Q], ["c3_fresh"], n_resamples=200)
    assert nrt[0]["n"] == 6 and nrt[0]["n_feasible"] == 4

    sens, _ = compute_sensitivity_table(loaded, {})
    assert {r["model"] for r in sens} == {"GLM-5.3-Flash"}

    # Without block3, c_4 was never attempted: no primary rows, the cell is incomplete; attempts still count
    files = find_run_files([d for d in dirs if d.name != "c1_hosted_qwen_block3"])
    loaded = load_all_c1_runs(files["ledgers"], files["block_of"], expected)
    assert loaded.primary_block_by_cell[key] is None and loaded.primary_by_cell[key] == []
    avail, _ = compute_availability_table(loaded, [Q], ["c3_fresh"])
    assert avail[0]["status"] == "incomplete" and avail[0]["primary_n"] == 0
    assert avail[0]["total_n"] == 16 and avail[0]["total_http_429"] == 9
