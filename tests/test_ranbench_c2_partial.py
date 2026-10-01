"""analysis.py --allow-incomplete: absent runs pend their dependent units; a FAIL run always stops the analysis.

The end-to-end tests run analysis.analyse() on a synthetic matrix with the trace-reading steps replaced
(verify_run, _control_analysis, _controls_row, _outcome_arrays, _radio_and_provenance), so the dependency rule,
the skipped.json record and the Holm-family wiring are exercised on the production driver code.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from src.ranbench.c2 import analysis, stats
from src.ranbench.c2.analysis import BASE_POINT, _verify, discover_run_dirs
from src.ranbench.c2.matrix import INTERPRETERS
from src.ranbench.c2.pilot import PILOT_ARMS
from test_ranbench_c2_return import make_inputs, make_run

RQ2 = [f"dp_r{rate}_s{speed}_u5" for rate in ("0p1", "0p3", "1") for speed in ("3", "60", "120")]
POINTS = [BASE_POINT, *RQ2]
INELIGIBLE = "dp_r0p1_s120_u5"
N_EVENTS = 12


def rid(point: str, arm: str, interpreter: str, mode: str = "E") -> str:
    return f"{point}__{arm}__{mode}__{interpreter}"


def matrix_rows() -> list[dict[str, str]]:
    rows = []
    for point in POINTS:
        meta = {"design_point": point, "rqs": "RQ1" if point == BASE_POINT else "RQ2", "rate_per_s": "0.3",
                "speed_kmh": "30", "ues_per_cell": "5", "schedule_file": "/unused.csv"}
        for arm in PILOT_ARMS:
            rows.append(meta | {"run_id": rid(point, arm, "control"), "arm": arm, "mode": "E",
                                "interpreter": "control"})
        for model in INTERPRETERS:
            for arm in ("L", "A", "N"):
                rows.append(meta | {"run_id": rid(point, arm, model), "arm": arm, "mode": "E", "interpreter": model})
            if point == BASE_POINT:
                for mode in ("P-1", "P-10"):
                    rows.append(meta | {"run_id": rid(point, "L", model, mode), "arm": "L", "mode": mode,
                                        "interpreter": model})
    return rows


def _assignment(block_s: float) -> dict[str, Any]:
    return {"block_s": block_s, "n_blocks": N_EVENTS, "blocks": np.arange(N_EVENTS), "counts": stats.draws(N_EVENTS)}


def _arrays(run_id: str) -> dict[str, np.ndarray]:
    seed = int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return {"aff": rng.integers(0, 2, N_EVENTS).astype(float), "net": rng.integers(0, 2, N_EVENTS).astype(float)}


def run_analysis(module: ModuleType, tmp: Path, monkeypatch: pytest.MonkeyPatch, absent: set[str] = frozenset(),
                 allow_incomplete: bool = True) -> dict[str, Any]:
    """analyse() on the synthetic matrix; `absent` run ids have no run directory yet (verify_run: MISSING)."""
    rows = matrix_rows()
    tmp.mkdir(parents=True, exist_ok=True)
    matrix = tmp / "matrix.csv"
    with matrix.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tmp / "runs").mkdir(exist_ok=True)
    loaded: list[str] = []
    control_calls: list[str] = []

    def verify(run_dir: Path, row: dict[str, str], _root: Any, *_: Any) -> tuple[str, list[str]]:
        return ("MISSING", ["run directory absent"]) if row["run_id"] in absent else ("PASS", [])

    def control_analysis(point: str, point_rows: list[dict[str, str]], _located: Any) -> dict[str, Any]:
        control_calls.append(point)
        for arm in PILOT_ARMS:  # the real function reads all five; fail loudly if one is absent
            assert rid(point, arm, "control") not in absent
        eligible = point != INELIGIBLE
        return {"stream": {"events": []}, "actuation": {"intents": []},
                "assignments": {"primary": _assignment(12.0), "sensitivity_10s": _assignment(10.0)},
                **{flag: eligible for flag in ("eligible", "eligible_10s", "eligible_d10", "eligible_original_5pp",
                                               "eligible_original_5pp_10s", "eligible_original_5pp_d10")}}

    def outcome_arrays(row: dict[str, str], _located: Any, _stream: Any) -> dict[str, np.ndarray]:
        assert row["run_id"] not in absent, f"statistic computed from absent run {row['run_id']}"
        loaded.append(row["run_id"])
        return _arrays(row["run_id"])

    def radio(rows_: list[dict[str, str]], _located: Any, status: dict[str, Any], actuating: dict[str, Any]):
        return ([{"run_id": row["run_id"], "design_point": row["design_point"],
                  "stale_exposure_ue_s_per_intent": 1.0 if row["design_point"] in actuating else None}
                 for row in rows_ if status[row["run_id"]][0] == "PASS"], [])

    monkeypatch.setattr(module, "verify_run", verify)
    monkeypatch.setattr(module, "_control_analysis", control_analysis)
    monkeypatch.setattr(module, "_controls_row", lambda meta, context: meta | {"eligible_d10": context["eligible_d10"]})
    monkeypatch.setattr(module, "_outcome_arrays", outcome_arrays)
    monkeypatch.setattr(module, "_radio_and_provenance", radio)
    out = tmp / "out"
    module.analyse(matrix, [tmp / "runs"], out, allow_incomplete=allow_incomplete)

    def read(name: str) -> list[dict[str, str]]:
        with (out / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    return {"out": out, "loaded": loaded, "control_calls": control_calls,
            "controls": read("controls.csv"), "l": read("l_arm_contrasts.csv"), "an": read("an_decomposition.csv"),
            "modes": read("rq1_modes.csv"), "radio": read("radio_kpis.csv"),
            "hypotheses": json.loads((out / "hypotheses.json").read_text(encoding="utf-8")),
            "skipped": json.loads((out / "skipped.json").read_text(encoding="utf-8"))}


HOLM_COLUMNS = {f"{prefix}_{key}" for prefix in ("affected_gap", "network_gap") for key in analysis.HOLM_KEYS}


def _keyed(rows: list[dict[str, str]], *keys: str, drop: set[str] = frozenset()) -> dict[tuple, dict[str, str]]:
    return {tuple(row[key] for key in keys): {k: v for k, v in row.items() if k not in drop} for row in rows}


def _units(result: dict[str, Any]) -> dict[tuple, list[str]]:
    return {(unit["analysis"], unit["design_point"], unit.get("interpreter"), unit.get("rule"), unit.get("run_id")):
            [item["run_id"] for item in unit["required_runs_unavailable"]]
            for unit in result["skipped"]["analysis_units"]}


@pytest.fixture
def complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return run_analysis(analysis, tmp_path / "complete", monkeypatch)


# ---------------------------------------------------------------- verify: absent is skipped, FAIL stops


def test_absent_runs_are_skipped_and_recorded(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    (root / "still-running").mkdir(parents=True)  # directory present, no manifest.json yet
    rows = [{"run_id": "not-started"}, {"run_id": "still-running"}]
    located = discover_run_dirs(rows, [root])
    status = _verify(rows, located, allow_incomplete=True)
    assert status["not-started"][0] == "MISSING"
    assert status["still-running"][0] == "INCOMPLETE"
    with pytest.raises(RuntimeError, match="missing or incomplete"):
        _verify(rows, located, allow_incomplete=False)


def test_absent_run_is_named_in_skipped_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = rid(RQ2[0], "N", "Jev-1.13.0")
    result = run_analysis(analysis, tmp_path, monkeypatch, {missing})
    assert [run["run_id"] for run in result["skipped"]["unavailable_runs"]] == [missing]
    assert _units(result) == {("A/N decomposition", RQ2[0], "Jev-1.13.0", None, None): [missing],
                              ("radio KPIs", RQ2[0], "Jev-1.13.0", None, missing): [missing]}


@pytest.mark.parametrize("allow_incomplete", [False, True])
def test_present_run_that_fails_verification_stops_the_analysis(tmp_path: Path, allow_incomplete: bool) -> None:
    inputs, root = tmp_path / "inputs", tmp_path / "runs"
    rows = []
    for run_id, complete in (("good", True), ("bad-manifest", False)):
        make_inputs(inputs, run_id)
        rows.append({"run_id": run_id, "ues": "3", "simulated_seconds": "0.3",
                     "config_file": f"{inputs}/configs/speed_3_ues_1.json",
                     "schedule_file": f"{inputs}/schedules/{run_id}.csv",
                     "stream_file": f"{inputs}/streams/rate_0p1_speed_3.json"})
        run_dir = make_run(root, inputs, run_id)
        if not complete:  # a manifest that is present but not complete is a FAIL, never an absent run
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest |= {"complete": False, "status": "failed"}
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows.append({**rows[0], "run_id": "not-started"})
    located = discover_run_dirs(rows, [root])
    with pytest.raises(RuntimeError, match=r"invalid completed run\(s\):\nbad-manifest: .*manifest not complete"):
        _verify(rows, located, allow_incomplete=allow_incomplete)


# ---------------------------------------------------------------- dependency rule


def test_complete_matrix_has_no_pending_units(complete: dict[str, Any]) -> None:
    assert complete["skipped"] == {"analysis_units": [], "unavailable_runs": []}
    assert len(complete["controls"]) == len(POINTS)
    assert len(complete["l"]) == len(complete["an"]) == len(POINTS) * len(INTERPRETERS)
    assert len(complete["modes"]) == 3 * len(INTERPRETERS)
    for rule in ("primary_d10", "sensitivity_original_5pp"):
        assert complete["hypotheses"][rule]["H1"]["result"] is not None
        assert complete["hypotheses"][rule]["H2"]["result"] is not None
        assert "pending_missing_runs" not in complete["hypotheses"][rule]["H2"]


@pytest.mark.parametrize("point", [RQ2[4], BASE_POINT])
def test_one_absent_control_pends_every_unit_of_its_design_point(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: dict[str, Any], point: str) -> None:
    missing = rid(point, "fixed-1", "control")
    result = run_analysis(analysis, tmp_path / "partial", monkeypatch, {missing})

    # Nothing is computed at the point: no control analysis, no outcome arrays, no rows.
    assert point not in result["control_calls"]
    assert not [run for run in result["loaded"] if run.startswith(point + "__")]
    for name in ("controls", "l", "an", "modes"):
        assert not [row for row in result[name] if row["design_point"] == point], name

    units = _units(result)
    assert units[("controls", point, None, None, None)] == [missing]
    for model in INTERPRETERS:
        assert units[("L-arm contrast", point, model, None, None)] == [missing]
        assert units[("A/N decomposition", point, model, None, None)] == [missing]
        stale = units[("stale-policy exposure", point, model, None, rid(point, "L", model))]
        assert stale == [missing]
        if point == BASE_POINT:
            assert units[("RQ1 modes", point, model, None, None)] == [missing]
    family = "H1" if point == BASE_POINT else "H2"
    for rule in ("primary_d10", "sensitivity_original_5pp"):
        entry = result["hypotheses"][rule][family]
        assert entry["result"] is None and entry["pending_missing_runs"] == [missing]
        assert units[(f"{family} Holm family", point if family == "H1" else None, None, rule, None)] == [missing]
    assert [row["stale_exposure_ue_s_per_intent"] for row in result["radio"] if row["design_point"] == point
            ] == [""] * (len([r for r in matrix_rows() if r["design_point"] == point]) - 1)

    # Every other design point is untouched, apart from Holm fields of the pending family.
    drop = HOLM_COLUMNS
    for name, keys in (("controls", ("design_point",)), ("l", ("design_point", "interpreter")),
                       ("an", ("design_point", "interpreter")), ("modes", ("interpreter", "mode"))):
        got = _keyed(result[name], *keys, drop=drop)
        want = {key: row for key, row in _keyed(complete[name], *keys, drop=drop).items() if key[0] != point}
        if name == "modes" and point == BASE_POINT:
            want = {}
        assert got == want, name
    other = "H2" if family == "H1" else "H1"
    assert result["hypotheses"]["primary_d10"][other] == complete["hypotheses"]["primary_d10"][other]


def test_absent_challenger_l_arm_pends_only_its_contrast(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: dict[str, Any]) -> None:
    point, model = RQ2[4], "GLM-5.3-Flash"
    missing = rid(point, "L", model)
    result = run_analysis(analysis, tmp_path / "partial", monkeypatch, {missing})
    assert missing not in result["loaded"]
    assert _units(result) == {
        ("L-arm contrast", point, model, None, None): [missing],
        ("A/N decomposition", point, model, None, None): [missing],
        ("radio KPIs", point, model, None, missing): [missing],
        ("H2 Holm family", None, None, "primary_d10", None): [missing],
        ("H2 Holm family", None, None, "sensitivity_original_5pp", None): [missing],
    }
    gone = (point, model)
    assert _keyed(result["controls"], "design_point") == _keyed(complete["controls"], "design_point")
    for name in ("l", "an"):
        want = {k: v for k, v in _keyed(complete[name], "design_point", "interpreter", drop=HOLM_COLUMNS).items()
                if k != gone}
        assert _keyed(result[name], "design_point", "interpreter", drop=HOLM_COLUMNS) == want
    # The H2 family waits for the run: no Holm p on the eight points at hand, and no H2 row carries one.
    assert result["hypotheses"]["primary_d10"]["H2"]["result"] is None
    assert all(row["affected_gap_p_holm"] == "" for row in result["l"] if row["design_point"] != BASE_POINT)
    # H1 at the base point is unaffected.
    assert result["hypotheses"]["primary_d10"]["H1"] == complete["hypotheses"]["primary_d10"]["H1"]


def test_absent_anchor_l_arm_pends_its_challengers(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    point = RQ2[2]
    missing = rid(point, "L", "Jev-1.13.0")
    result = run_analysis(analysis, tmp_path, monkeypatch, {missing})
    pending = {key[2] for key, runs in _units(result).items() if key[0] == "L-arm contrast"}
    assert pending == {"Jev-1.13.0", "DeepSeek-V4.1-Flash", "GLM-5.3-Flash", "Qwen3.8-Flash"}
    present = {row["interpreter"] for row in result["l"] if row["design_point"] == point}
    assert present == {"SemIf-Qwen3.5-4B", "AnyJev-L0", "Qwen3.5-4B-JSON"}


def test_absent_arm_outside_every_holm_family_leaves_the_families_computed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: dict[str, Any]) -> None:
    for point, model in ((RQ2[3], "AnyJev-L0"), (INELIGIBLE, "GLM-5.3-Flash")):
        result = run_analysis(analysis, tmp_path / point, monkeypatch, {rid(point, "L", model)})
        assert result["hypotheses"] == complete["hypotheses"]
        assert {key[0] for key in _units(result)} == {"L-arm contrast", "A/N decomposition", "radio KPIs"}
