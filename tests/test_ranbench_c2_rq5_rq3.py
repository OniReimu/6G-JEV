"""RQ5 division of labour and RQ3 radio replay units of the C2 analysis (exploratory, no Holm).

Unit computations run on hand-built inputs; the dependency rule runs analysis.analyse() on the synthetic matrix of
test_ranbench_c2_partial extended by the RQ5 (b) rows and RQ3 replay rows, with the trace readers replaced.
"""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import test_ranbench_c2_partial as partial
from src.ranbench.c2 import analysis, stats
from src.ranbench.c2.analysis import BASE_POINT, RQ5_ARM, RQ5_CLASSES
from src.ranbench.c2.matrix import INTERPRETERS
from src.ranbench.c2.outcomes import SlotTable, class_jain, jain_index

rid = partial.rid
BASE_MATRIX_ROWS = partial.matrix_rows


# ---------------------------------------------------------------- Jain's index


def test_jain_index_on_known_vectors() -> None:
    assert jain_index(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(1.0)
    assert jain_index(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(0.25)
    assert jain_index(np.array([1.0, 3.0])) == pytest.approx(16.0 / 20.0)
    assert jain_index(np.array([])) is None
    assert jain_index(np.zeros(3)) is None


def test_class_jain_uses_per_ue_mean_throughput_per_class() -> None:
    # Two slots after t_from=0.1. video UEs 1, 2 average 1 and 3 kb per slot; xr UE 3 alone; the first slot
    # (t=0.1) is outside (t_from, t_to] and would change the video means if it were counted.
    rows = [(0.1, 1, "video", 1000), (0.1, 2, "video", 0), (0.1, 3, "xr", 5),
            (0.2, 1, "video", 100), (0.2, 2, "video", 300), (0.2, 3, "xr", 7),
            (0.3, 1, "video", 100), (0.3, 2, "video", 300), (0.3, 3, "xr", 9)]
    ue_slots = pd.DataFrame({
        "t": [r[0] for r in rows], "imsi": [r[1] for r in rows], "cell": 1, "cls": [r[2] for r in rows],
        "rx_bytes": [r[3] for r in rows], "n_rx": 1, "mean_delay_ms": 1.0, "p95_delay_ms": 1.0})
    trace = SimpleNamespace(ue_slots=ue_slots, class_targets={}, clusters={})
    got = class_jain(SlotTable(trace), 0.1, 0.3)
    assert got == {"video": pytest.approx(jain_index(np.array([1.0, 3.0]))), "xr": pytest.approx(1.0)}


# ---------------------------------------------------------------- RQ5 row


def _context(n: int) -> dict[str, Any]:
    blocking = lambda block_s: {"block_s": block_s, "n_blocks": n, "blocks": np.arange(n), "counts": stats.draws(n)}
    return {"assignments": {"primary": blocking(48.8), "sensitivity_10s": blocking(10.0)}}


def _arm(aff: list[float], net: list[float], thr: float, jain: float) -> dict[str, Any]:
    return {"aff": np.array(aff, float), "net": np.array(net, float), "throughput_mean_mbps": thr,
            "jain": {cls: jain for cls in RQ5_CLASSES}}


def test_rq5_row_levels_contrasts_and_aggregates() -> None:
    n = 12
    rng = np.random.default_rng(3)
    a = _arm(list(rng.uniform(0, 1, n)), list(rng.uniform(0, 1, n)), 5.0, 0.8)
    b = _arm(list(rng.uniform(0, 1, n)), list(rng.uniform(0, 1, n)), 6.0, 0.9)
    c = _arm(list(rng.uniform(0, 1, n)), list(rng.uniform(0, 1, n)), 7.0, 0.95)
    b["aff"][0] = np.nan  # an event without affected-class slots drops out of every contrast with (b)
    context = _context(n)
    row = analysis._rq5_row({"design_point": BASE_POINT}, "Jev-1.13.0", {"a": "A", "b": "B", "c": "C"},
                            {"a": a, "b": b, "c": c}, context)
    assert row["run_id_a"] == "A" and row["run_id_b"] == "B" and row["run_id_c"] == "C"
    assert row["affected_b_point"] == pytest.approx(np.nanmean(b["aff"]))
    assert row["affected_b_n"] == n - 1 and row["network_b_n"] == n
    assert row["affected_b_minus_a_point"] == pytest.approx(np.nanmean(b["aff"] - a["aff"]))
    assert row["affected_b_minus_a_n"] == n - 1
    assert row["network_b_minus_c_point"] == pytest.approx(np.mean(b["net"] - c["net"]))
    for prefix in ("affected_b_minus_a", "network_b_minus_c", "affected_b_minus_a_10s"):
        assert row[f"{prefix}_ci_low"] <= row[f"{prefix}_point"] <= row[f"{prefix}_ci_high"]
    # The two blockings share the point estimate; intervals come from the base point's B and from 10 s.
    assert row["network_b_minus_a_point"] == row["network_b_minus_a_10s_point"]
    assert row["throughput_mean_mbps_b"] == 6.0 and row["jain_xr_c"] == 0.95
    assert row["block_s"] == 48.8 and row["block_10s_s"] == 10.0
    assert not any(key.endswith("p_raw") or "holm" in key or "resolved" in key for key in row)


def test_rq5_row_refuses_a_run_with_a_missing_class() -> None:
    arms = {arm: _arm([0.1] * 4, [0.1] * 4, 1.0, 1.0) for arm in ("a", "b", "c")}
    del arms["b"]["jain"]["be"]
    with pytest.raises(RuntimeError, match="classes"):
        analysis._rq5_row({}, "Jev-1.13.0", {"a": "A", "b": "B", "c": "C"}, arms, _context(4))


def test_rq5_events_are_the_n_arm_events_and_need_identical_inputs(tmp_path: Path, monkeypatch) -> None:
    manifest = {"schedule_sha256": "s", "stream_sha256": "t", "config_sha256": "c", "sim_time_s": 345.3, "rng_run": 1}
    located = {}
    for run_id, extra in (("a", {}), ("b", {"rq5_arm": RQ5_ARM, "mode": "live"}), ("b2", {"schedule_sha256": "x"}),
                          ("b3", {}), ("c", {}), ("c_bad", {})):
        (tmp_path / run_id).mkdir()
        (tmp_path / run_id / "manifest.json").write_text(json.dumps(manifest | extra), encoding="utf-8")
        (tmp_path / run_id / "positions.csv").write_text("t,imsi,x,y,serving_cell\n0.1,1,0.0,0.0,1\n")
        (tmp_path / run_id / "tx_trace.csv").write_text(f"t_slot,imsi,tx_bytes,n_tx_pkts\n0.1,1,{7 if run_id in ('b3', 'c_bad') else 5},1\n")
        located[run_id] = analysis.LocatedRun(tmp_path / run_id, tmp_path)
    monkeypatch.setattr(analysis, "_events", lambda row: [{"j": 0, "from": row["run_id"]}])
    a, c = {"run_id": "a"}, {"run_id": "c"}
    assert analysis._rq5_events(a, {"run_id": "b"}, c, located) == [{"j": 0, "from": "a"}]
    with pytest.raises(RuntimeError, match="schedule_sha256 differs"):
        analysis._rq5_events(a, {"run_id": "b2"}, c, located)
    with pytest.raises(RuntimeError, match="not paired with a"):  # traffic arrivals differ: no paired b - a
        analysis._rq5_events(a, {"run_id": "b3"}, c, located)
    with pytest.raises(RuntimeError, match="not paired with c_bad"):  # (b) pairs with (a) but not the oracle
        analysis._rq5_events(a, {"run_id": "b"}, {"run_id": "c_bad"}, located)


def test_arm_outcomes_refuses_events_that_differ_from_the_stream(tmp_path: Path) -> None:
    stream = {"events": [{"j": 0, "intent_id": "i0", "t_issue_s": 2.0}]}
    for events in ([{"j": 0, "intent_id": "i0", "t_issue_s": 22.0}], [{"j": 0, "intent_id": "i9", "t_issue_s": 2.0}]):
        with pytest.raises(RuntimeError, match="events differ from its stream"):
            analysis._arm_outcomes(tmp_path, events, stream)


# ---------------------------------------------------------------- RQ3 radio replay


def _rq3_inputs(tmp: Path, rho: float, non_stationary: bool) -> dict[str, str]:
    inputs = tmp / "inputs"
    (inputs / "streams").mkdir(parents=True)
    integrity = tmp / "trace" / "integrity.json"
    integrity.parent.mkdir()
    integrity.write_text(json.dumps({"rho": rho, "non_stationary": non_stationary}), encoding="utf-8")
    entry = {"integrity_path": str(integrity), "integrity_sha256": hashlib.sha256(integrity.read_bytes()).hexdigest()}
    (inputs / "input-manifest.json").write_text(
        json.dumps({"load_traces": {"GLM-5.3-Flash@2": entry}}), encoding="utf-8")
    return {"run_id": "r", "interpreter": "GLM-5.3-Flash", "rate_per_s": "2.0",
            "stream_file": str(inputs / "streams" / "s.json")}


def test_rq3_load_reads_the_pinned_integrity(tmp_path: Path) -> None:
    row = _rq3_inputs(tmp_path, 1.2, True)
    assert analysis._rq3_load(row) == {"rho": 1.2, "non_stationary": True}
    integrity = tmp_path / "trace" / "integrity.json"
    integrity.write_text(json.dumps({"rho": 1.2, "non_stationary": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from the input manifest"):
        analysis._rq3_load(row)


def test_rq3_load_refuses_a_flag_that_disagrees_with_rho(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="non_stationary"):
        analysis._rq3_load(_rq3_inputs(tmp_path, 1.0, False))


@pytest.mark.parametrize("non_stationary", [False, True])
def test_rq3_row_blocks_on_10s_and_drops_intervals_when_non_stationary(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, non_stationary: bool) -> None:
    n, n_blocks = 20, 10
    t_issue = [2.0 + 5.0 * k for k in range(n)]
    stream = {"meta": {"block_s": 10.0, "t0_s": 2.0, "n_blocks": n_blocks},
              "events": [{"j": k, "t_issue_s": t} for k, t in enumerate(t_issue)]}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "stream.json").write_text(json.dumps(stream), encoding="utf-8")
    # 20 events: installed within 1 s (8), installed after 1 s (6), never installed (6)
    events = [{"j": k, "t_issue_s": t, "install": None if k >= 14 else ["s", "c", "high"],
               "e_s": None if k >= 14 else t + (0.5 if k < 8 else 2.0)} for k, t in enumerate(t_issue)]
    aff = np.linspace(0.0, 0.5, n)
    net = np.linspace(0.1, 0.3, n)
    monkeypatch.setattr(analysis, "_events", lambda row: events)
    monkeypatch.setattr(analysis, "_arm_outcomes",
                        lambda *_: {"aff": aff, "net": net, "prb_util_mean": 0.4})
    monkeypatch.setattr(analysis, "_rq3_load", lambda row: {"rho": 1.1 if non_stationary else 0.3,
                                                            "non_stationary": non_stationary})
    row = {"run_id": "r", "design_point": "dp_rq3_glm_5_3_flash_r2_u5", "rqs": "RQ3", "rate_per_s": "2.0",
           "speed_kmh": "30.0", "ues_per_cell": "5", "interpreter": "GLM-5.3-Flash"}
    out = analysis._rq3_row(row, {"r": analysis.LocatedRun(run_dir, tmp_path)})
    assert out["block_s"] == 10.0 and out["n_blocks"] == n_blocks
    assert out["affected_point"] == pytest.approx(aff.mean()) and out["network_n"] == n
    assert out["share_enforced_within_1s"] == pytest.approx(8 / 20)
    assert out["prb_util_mean"] == 0.4 and out["n_events"] == n
    assert out["non_stationary"] is non_stationary and out["block_inference"] is (not non_stationary)
    blocking = stats.block_assignment(np.array(t_issue), 2.0, 10.0, n_blocks)
    want = stats.level(aff, blocking["blocks"], blocking["counts"])
    for metric in ("affected", "network"):
        if non_stationary:
            assert out[f"{metric}_ci_low"] is None and out[f"{metric}_ci_high"] is None
        else:
            assert out[f"{metric}_ci_low"] < out[f"{metric}_point"] < out[f"{metric}_ci_high"]
    if not non_stationary:
        assert (out["affected_ci_low"], out["affected_ci_high"]) == (want["ci_low"], want["ci_high"])


# ---------------------------------------------------------------- dependency rule (analyse)

RQ3_POINTS = [(model, rate, ues) for model in INTERPRETERS for rate, ues in ((0.1, 5), (2.0, 5), (1.0, 20))]


def _rq3_point(model: str, rate: float, ues: int) -> str:
    return f"dp_rq3_{analysis.re.sub(r'[^a-z0-9]+', '_', model.lower()).strip('_')}_r{rate:g}_u{ues}".replace(".", "p")


def _extended_rows() -> list[dict[str, str]]:
    rows = BASE_MATRIX_ROWS()
    base = {"rate_per_s": "0.3", "speed_kmh": "30", "ues_per_cell": "5", "schedule_file": "/unused.csv"}
    for model in INTERPRETERS:
        rows.append(base | {"run_id": rid(BASE_POINT, RQ5_ARM, model), "design_point": BASE_POINT, "rqs": "RQ5",
                            "arm": RQ5_ARM, "mode": "E", "interpreter": model,
                            "reused_arm_a_run_id": rid(BASE_POINT, "N", model),
                            "reused_arm_c_run_id": rid(BASE_POINT, "oracle", "control")})
    for model, rate, ues in RQ3_POINTS:
        point = _rq3_point(model, rate, ues)
        rows.append(base | {"run_id": f"{point}__N", "design_point": point, "rqs": "RQ3", "arm": "N", "mode": "E",
                            "interpreter": model, "rate_per_s": str(rate), "ues_per_cell": str(ues)})
    # the partial harness writes the matrix with the first row's columns
    fields = dict.fromkeys(key for row in rows for key in row)
    return [{key: row.get(key, "") for key in fields} for row in rows]


def _run(tmp: Path, monkeypatch: pytest.MonkeyPatch, absent: set[str] = frozenset(),
         refs: bool = True) -> dict[str, Any]:
    arm_calls: list[str] = []
    roots: dict[str, str] = {}  # run id -> name of the root it was read from
    scored_on: dict[str, str] = {}  # run id -> run id whose events it was scored on

    def arm_outcomes(run_dir: Path, events: Any, stream: Any) -> dict[str, Any]:
        assert run_dir.parent.name == "refs" or run_dir.name not in absent, f"read absent run {run_dir.name}"
        arm_calls.append(run_dir.name)
        roots[run_dir.name] = run_dir.parent.name
        scored_on[run_dir.name] = events[0]["events_of"]
        return {**partial._arrays(run_dir.name), "throughput_mean_mbps": 1.0, "prb_util_mean": 0.5,
                "jain": {cls: 0.9 for cls in RQ5_CLASSES}}

    def rq3_row(row: dict[str, str], located: Any) -> dict[str, Any]:
        assert row["run_id"] not in absent
        return analysis._point_meta(row["design_point"], [row]) | {"interpreter": row["interpreter"],
                                                                   "run_id": row["run_id"]}

    monkeypatch.setattr(partial, "matrix_rows", _extended_rows)
    monkeypatch.setattr(analysis, "_arm_outcomes", arm_outcomes)
    monkeypatch.setattr(analysis, "_events", lambda row: [{"events_of": row["run_id"]}])
    # the real _rq5_events, on manifests and C-4 traces that agree
    monkeypatch.setattr(analysis, "_manifest", lambda run_dir: {"schedule_sha256": "s", "stream_sha256": "t"})
    monkeypatch.setattr(analysis, "pairing_identical", lambda a, b: {"positions.csv": True, "tx_trace.csv": True})
    monkeypatch.setattr(analysis, "_rq5_references", lambda root, rows, b_row, located: {
        row["run_id"]: analysis.LocatedRun(root / row["run_id"], root) for row in rows})
    monkeypatch.setattr(analysis, "analyse", functools.partial(
        analysis.analyse, rq5_refs_root=tmp / "refs" if refs else None))
    monkeypatch.setattr(analysis, "_rq3_row", rq3_row)
    for row in _extended_rows():  # present runs need a directory: the new units read from located[run_id]
        if row["run_id"] not in absent:
            (tmp / "runs" / row["run_id"]).mkdir(parents=True, exist_ok=True)
    result = partial.run_analysis(analysis, tmp, monkeypatch, set(absent))
    result["arm_calls"] = arm_calls
    result["scored_on"] = scored_on
    result["roots"] = roots
    for name in ("rq5", "rq3_radio"):
        result[name] = pd.read_csv(result["out"] / f"{name}.csv", keep_default_na=False).to_dict("records")
    return result


def test_complete_matrix_has_every_rq5_and_rq3_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run(tmp_path, monkeypatch)
    assert result["skipped"] == {"analysis_units": [], "unavailable_runs": []}
    assert [row["interpreter"] for row in result["rq5"]] == list(INTERPRETERS)
    assert {row["rqs"] for row in result["rq5"]} == {"RQ5"}
    assert {(row["run_id_a"], row["run_id_c"]) for row in result["rq5"] if row["interpreter"] == "GLM-5.3-Flash"} \
        == {(rid(BASE_POINT, "N", "GLM-5.3-Flash"), rid(BASE_POINT, "oracle", "control"))}
    assert len(result["rq3_radio"]) == len(RQ3_POINTS)
    # the oracle arm is read once for all seven interpreters
    oracle = rid(BASE_POINT, "oracle", "control")
    assert result["arm_calls"].count(oracle) == 1
    # (b) is scored on its arm-(a) events, (a) on its own, the oracle on the oracle's
    for model in INTERPRETERS:
        a = rid(BASE_POINT, "N", model)
        assert result["scored_on"][rid(BASE_POINT, RQ5_ARM, model)] == a
        assert result["scored_on"][a] == a
    assert result["scored_on"][oracle] == oracle
    # (a) and (c) are read from the RQ5 reference root, (b) from the production selection
    for model in INTERPRETERS:
        assert result["roots"][rid(BASE_POINT, "N", model)] == "refs"
        assert result["roots"][rid(BASE_POINT, RQ5_ARM, model)] == "runs"
    assert result["roots"][oracle] == "refs"
    assert {row["reference_root"] for row in result["rq5"]} == {str(tmp_path / "refs")}


def test_absent_b_run_pends_only_its_rq5_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = rid(BASE_POINT, RQ5_ARM, "Qwen3.8-Flash")
    result = _run(tmp_path, monkeypatch, {missing})
    assert partial._units(result) == {
        ("RQ5 division of labour", BASE_POINT, "Qwen3.8-Flash", None, None): [missing],
        ("radio KPIs", BASE_POINT, "Qwen3.8-Flash", None, missing): [missing],
    }
    assert [row["interpreter"] for row in result["rq5"]] == [m for m in INTERPRETERS if m != "Qwen3.8-Flash"]
    assert len(result["rq3_radio"]) == len(RQ3_POINTS)


def test_absent_production_arm_a_leaves_rq5_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # arm (a) comes from the reference root, so the production N-arm copy is not an RQ5 dependency
    missing = rid(BASE_POINT, "N", "AnyJev-L0")
    result = _run(tmp_path, monkeypatch, {missing})
    assert not [key for key in partial._units(result) if key[0] == "RQ5 division of labour"]
    assert len(result["rq5"]) == len(INTERPRETERS)


def test_rq5_without_a_reference_root_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="--rq5-refs-root"):
        _run(tmp_path, monkeypatch, refs=False)


def _reference_fixture(tmp_path: Path, binary: str) -> tuple[dict[str, analysis.LocatedRun], dict[str, str]]:
    b_dir = tmp_path / "runs" / "b"
    b_dir.mkdir(parents=True)
    (b_dir / "manifest.json").write_text(json.dumps({"binary_sha256": "4d46"}), encoding="utf-8")
    ref = tmp_path / "refs" / "a"
    ref.mkdir(parents=True)
    (ref / "manifest.json").write_text(json.dumps({"binary_sha256": binary}), encoding="utf-8")
    return {"b": analysis.LocatedRun(b_dir, tmp_path / "runs")}, {"run_id": "b"}


@pytest.mark.parametrize("bad", ["binary", "verify"])
def test_every_rq5_reference_is_checked_not_only_the_first(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    located, b_row = _reference_fixture(tmp_path, "4d46")
    oracle = tmp_path / "refs" / "oracle"
    oracle.mkdir()
    (oracle / "manifest.json").write_text(
        json.dumps({"binary_sha256": "foreign" if bad == "binary" else "4d46"}), encoding="utf-8")
    monkeypatch.setattr(analysis, "verify_run", lambda run_dir, row, root: (
        ("FAIL", ["x"]) if bad == "verify" and row["run_id"] == "oracle" else ("PASS", [])))
    with pytest.raises(RuntimeError, match="RQ5 reference .*oracle"):
        analysis._rq5_references(tmp_path / "refs", [{"run_id": "a"}, {"run_id": "oracle"}], b_row, located)


def test_rq5_reference_that_is_the_production_copy_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    located, b_row = _reference_fixture(tmp_path, "4d46")
    (tmp_path / "selected").mkdir()
    (tmp_path / "selected" / "a").symlink_to(tmp_path / "refs" / "a")  # selection links to the same copy
    located["a"] = analysis.LocatedRun(tmp_path / "selected" / "a", tmp_path / "selected")
    monkeypatch.setattr(analysis, "verify_run", lambda run_dir, row, root: ("PASS", []))
    with pytest.raises(RuntimeError, match="is the production copy"):
        analysis._rq5_references(tmp_path / "refs", [{"run_id": "a"}], b_row, located)


def test_rq5_reference_on_the_b_binary_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    located, b_row = _reference_fixture(tmp_path, "4d46")
    monkeypatch.setattr(analysis, "verify_run", lambda run_dir, row, root: ("PASS", []))
    got = analysis._rq5_references(tmp_path / "refs", [{"run_id": "a"}], b_row, located)
    assert got == {"a": analysis.LocatedRun(tmp_path / "refs" / "a", tmp_path / "refs")}


def test_rq5_reference_on_another_binary_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    located, b_row = _reference_fixture(tmp_path, "foreign")
    monkeypatch.setattr(analysis, "verify_run", lambda run_dir, row, root: ("PASS", []))
    with pytest.raises(RuntimeError, match="binary_sha256 foreign differs from"):
        analysis._rq5_references(tmp_path / "refs", [{"run_id": "a"}], b_row, located)


def test_rq5_reference_that_fails_verification_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    located, b_row = _reference_fixture(tmp_path, "4d46")
    monkeypatch.setattr(analysis, "verify_run", lambda run_dir, row, root: ("INCOMPLETE", ["no manifest.json"]))
    with pytest.raises(RuntimeError, match="RQ5 reference .*INCOMPLETE"):
        analysis._rq5_references(tmp_path / "refs", [{"run_id": "a"}], b_row, located)


def test_absent_n_arm_elsewhere_leaves_rq5_and_rq3_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # the production state: rq2_e_r0p1_s120_n_anyjev_l0 (an N arm at an RQ2 point) is still absent
    missing = rid(partial.INELIGIBLE, "N", "AnyJev-L0")
    result = _run(tmp_path, monkeypatch, {missing})
    assert {key[0] for key in partial._units(result)} == {"A/N decomposition", "radio KPIs"}
    assert len(result["rq5"]) == len(INTERPRETERS) and len(result["rq3_radio"]) == len(RQ3_POINTS)


def test_absent_control_pends_rq5_but_not_rq3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = rid(BASE_POINT, "oracle", "control")
    result = _run(tmp_path, monkeypatch, {missing})
    units = partial._units(result)
    for model in INTERPRETERS:
        assert units[("RQ5 division of labour", BASE_POINT, model, None, None)] == [missing]
    assert result["rq5"] == [] and len(result["rq3_radio"]) == len(RQ3_POINTS)


def test_absent_replay_pends_only_its_rq3_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    point = _rq3_point("AnyJev-L0", 1.0, 20)
    missing = f"{point}__N"
    result = _run(tmp_path, monkeypatch, {missing})
    assert partial._units(result) == {
        ("RQ3 radio replay", point, "AnyJev-L0", None, None): [missing],
        ("radio KPIs", point, "AnyJev-L0", None, missing): [missing],
    }
    assert len(result["rq3_radio"]) == len(RQ3_POINTS) - 1 and len(result["rq5"]) == len(INTERPRETERS)
