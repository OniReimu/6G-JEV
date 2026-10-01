"""Synthetic C2 analysis output (the files JFC src.ranbench.c2.analysis writes) for paper_assets tests.

`write_complete` writes a complete analysis directory. `drop_control` and `drop_l_arm` turn it into the partial
analysis that JFC analysis.py --allow-incomplete writes when one control run, or one mode-E L-arm run, is absent:
the dependent rows are removed and skipped.json names each pending unit with its missing run ids.
"""
from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
BASE = "dp_r0p3_s30_u5"
RATES = (0.1, 0.3, 1.0)
SPEEDS = (3.0, 60.0, 120.0)
MODES = ("E", "P-1", "P-10")
CONTROL_ARMS = ("oracle", "no-update", "fixed-0.1", "fixed-1", "fixed-5")
MODELS = ("Jev-1.13.0", "SemIf-Qwen3.5-4B", "AnyJev-L0", "DeepSeek-V4.1-Flash", "GLM-5.3-Flash",
          "Qwen3.8-Flash", "Qwen3.5-4B-JSON")
ANCHOR = {"DeepSeek-V4.1-Flash": "Jev-1.13.0", "GLM-5.3-Flash": "Jev-1.13.0", "Qwen3.8-Flash": "Jev-1.13.0",
          "Qwen3.5-4B-JSON": "SemIf-Qwen3.5-4B"}
INELIGIBLE = "dp_r0p1_s120_u5"  # one RQ2 point below the control floor: its Holm fields are a genuine n/a
RULES = ("primary_d10", "sensitivity_original_5pp")
RQ3_CELLS = ((0.1, 5), (0.5, 5), (1.0, 5), (2.0, 5), (1.0, 20))
NON_STATIONARY = {("AnyJev-L0", 2.0, 5), ("Qwen3.8-Flash", 2.0, 5)}
JAIN_CLASSES = ("video", "xr", "iot", "be")
SHA = {"cluster-A": "7e2268c1" + "0" * 56, "cluster-B": "bd1738b0" + "0" * 56, "M4": "32976335" + "0" * 56,
       "M4-RQ5": "4d46bc2e" + "0" * 56}
RQ3_PLATFORM = {(0.1, 5): "M4", (0.5, 5): "cluster-A", (1.0, 5): "cluster-A", (2.0, 5): "cluster-A",
                (1.0, 20): "cluster-B"}


def point(rate: float, speed: float) -> str:
    return f"dp_r{rate:g}_s{speed:g}_u5".replace(".", "p")


POINTS = (BASE, *(point(rate, speed) for rate in RATES for speed in SPEEDS))
RQ2_POINTS = POINTS[1:]


def rq3_point(model: str, rate: float, ues: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    return f"dp_rq3_{slug}_r{rate:g}_u{ues}".replace(".", "p")


def load_paper_assets() -> Any:
    spec = importlib.util.spec_from_file_location("paper_assets", SCRIPTS / "paper_assets.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["paper_assets"] = module
    spec.loader.exec_module(module)
    return module


def run_id(design_point: str, interpreter: str, arm: str, mode: str = "E") -> str:
    return f"{design_point}__{arm}__{mode}__{interpreter}"


def _meta(design_point: str) -> dict[str, Any]:
    if design_point == BASE:
        return {"design_point": design_point, "rqs": "RQ1", "rate_per_s": 0.3, "speed_kmh": 30.0, "ues_per_cell": 5}
    rate, speed = re.fullmatch(r"dp_r([\dp]+)_s(\d+)_u5", design_point).groups()
    return {"design_point": design_point, "rqs": "RQ2", "rate_per_s": float(rate.replace("p", ".")),
            "speed_kmh": float(speed), "ues_per_cell": 5}


def _stat(rng: np.random.Generator, prefix: str, signed: bool = False, p: bool = False) -> dict[str, Any]:
    value = rng.uniform(-0.1, 0.2) if signed else rng.uniform(0.05, 0.6)
    width = rng.uniform(0.01, 0.05)
    out = {f"{prefix}_point": value, f"{prefix}_ci_low": value - width, f"{prefix}_ci_high": value + width,
           f"{prefix}_n": 120}
    if p:
        out[f"{prefix}_p_raw"] = rng.uniform(0.0, 0.2)
    return out


def _blocks(rng: np.random.Generator) -> dict[str, Any]:
    return {"block_s": float(rng.integers(10, 20)), "n_blocks": 30, "block_10s_s": 10.0, "n_blocks_10s": 40}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_complete(directory: Path, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    controls, l_arm, an, modes, radio = [], [], [], [], []
    for design_point in POINTS:
        meta = _meta(design_point)
        eligible = design_point != INELIGIBLE
        blocks = _blocks(rng)
        controls.append(meta | {
            "eligible_d10": eligible, "eligible_original_5pp_d10": eligible,
            **_stat(rng, "c1", True, True), "c1_resolved_both": True,
            **_stat(rng, "c2", True, True), "c2_resolved_both": bool(rng.integers(0, 2)),
            "c2_mean_fixed_0p1": rng.uniform(0.1, 0.3), "c2_mean_fixed_1": rng.uniform(0.2, 0.4),
            "c2_mean_fixed_5": rng.uniform(0.3, 0.5), "c2_monotone": True, "tau_s": rng.uniform(1, 9), **blocks,
        })
        for model in MODELS:
            anchor = ANCHOR.get(model)
            row = meta | {"interpreter": model, "comparator": anchor,
                          **_stat(rng, "affected"), **_stat(rng, "network")}
            for prefix in ("affected_gap", "network_gap"):
                family = None
                if anchor is not None and eligible and (design_point == BASE or prefix == "affected_gap"):
                    family = "primary_d10:H1" if design_point == BASE else "primary_d10:H2"
                if anchor is None:
                    row |= {f"{prefix}_{key}": None for key in ("point", "ci_low", "ci_high", "n", "p_raw")}
                else:
                    row |= _stat(rng, prefix, True, True)
                row |= {
                    f"{prefix}_holm_family": family,
                    f"{prefix}_p_holm": None if family is None else rng.uniform(0.0, 0.1),
                    f"{prefix}_p_holm_10s": None if family is None else rng.uniform(0.0, 0.1),
                    f"{prefix}_resolved_both": None if family is None else bool(rng.integers(0, 2)),
                }
            l_arm.append(row | blocks)
            an.append(meta | {"interpreter": model, "outcome": "affected_class_W2",
                              **_stat(rng, "n_minus_l", True, True), "n_minus_l_resolved_both": bool(rng.integers(0, 2)),
                              **_stat(rng, "n_minus_a", True, True), "n_minus_a_resolved_both": bool(rng.integers(0, 2)),
                              **blocks})
            for mode in (MODES if design_point == BASE else ("E",)):
                if design_point == BASE:
                    modes.append(meta | {"interpreter": model, "mode": mode,
                                         **_stat(rng, "affected"), **_stat(rng, "network"), **blocks})
                radio.append({
                    "run_id": run_id(design_point, model, "L", mode), "design_point": design_point,
                    "rqs": meta["rqs"], "mode": mode, "arm": "L", "interpreter": model,
                    "throughput_mean_mbps": rng.uniform(5, 20), "throughput_p5_mbps": rng.uniform(0.5, 3),
                    "delay_p95_ms": rng.uniform(20, 90), "outage_share": rng.uniform(0, 0.05),
                    "handover_rate_per_ue_s": rng.uniform(0.001, 0.01),
                    "handover_total_time_p95_ms": rng.uniform(30, 60), "rlf_count": int(rng.integers(0, 5)),
                    "stale_exposure_ue_s_per_intent": rng.uniform(1, 9),
                })
    _write_csv(directory / "controls.csv", controls)
    _write_csv(directory / "l_arm_contrasts.csv", l_arm)
    _write_csv(directory / "an_decomposition.csv", an)
    _write_csv(directory / "rq1_modes.csv", modes)
    _write_csv(directory / "radio_kpis.csv", radio)
    hypotheses = {}
    for rule in RULES:
        holds = {f"{challenger} vs {anchor}": {"share_resolved": rng.uniform(0.5, 1.0), "n_reversed_resolved": 0,
                                                "n_points": 8, "holds": bool(rng.integers(0, 2))}
                 for challenger, anchor in ANCHOR.items()}
        hypotheses[rule] = {"H1": {"tested": True, "base_point": BASE, "result": {}},
                            "H2": {"complete": True, "result": {"n_eligible": 8, "holds": holds}}}
    (directory / "hypotheses.json").write_text(json.dumps(hypotheses, indent=2) + "\n", encoding="utf-8")
    (directory / "skipped.json").write_text(
        json.dumps({"analysis_units": [], "unavailable_runs": []}, indent=2) + "\n", encoding="utf-8")
    _write_rq5_rq3(directory, np.random.default_rng(seed + 1))  # own stream: the pre-RQ5 files stay byte-identical
    _write_runs(directory)
    return directory


def _write_runs(directory: Path) -> None:
    """runs.csv (D-17 binaries) and the control rows of radio_kpis.csv, appended after the L-arm rows.
    Every C2 design point runs on cluster-A; each RQ3 rate on its D-17 platform; RQ5 (b) on the M4 RQ5 build."""
    radio = _read_csv(directory / "radio_kpis.csv")
    for design_point in POINTS:
        for arm in CONTROL_ARMS:
            radio.append({"run_id": run_id(design_point, "control", arm), "design_point": design_point,
                          "rqs": _meta(design_point)["rqs"], "mode": "E", "arm": arm, "interpreter": "control"})
    _write_csv(directory / "radio_kpis.csv", radio)
    runs = [{"run_id": row["run_id"], "design_point": row["design_point"], "platform_root": "/cluster-a",
             "binary_sha256": SHA["cluster-A"]} for row in radio]
    for row in _read_csv(directory / "rq3_radio.csv"):
        platform = RQ3_PLATFORM[(float(row["rate_per_s"]), int(row["ues_per_cell"]))]
        runs.append({"run_id": row["run_id"], "design_point": row["design_point"], "platform_root": f"/{platform}",
                     "binary_sha256": SHA[platform]})
    for row in _read_csv(directory / "rq5.csv"):
        runs.append({"run_id": row["run_id_b"], "design_point": BASE, "platform_root": "/m4",
                     "binary_sha256": SHA["M4-RQ5"]})
    _write_csv(directory / "runs.csv", runs)


def set_binary(directory: Path, run: str, sha: str) -> None:
    """Move one run to another build, as a D-11 rerun or a foreign-platform replay would appear in runs.csv."""
    rows = _read_csv(directory / "runs.csv")
    assert run in {row["run_id"] for row in rows}, run
    for row in rows:
        if row["run_id"] == run:
            row["binary_sha256"] = sha
    _write_csv(directory / "runs.csv", rows)


def _write_rq5_rq3(directory: Path, rng: np.random.Generator) -> None:
    rq5, rq3 = [], []
    blocks = _blocks(rng)
    for model in MODELS:
        row = _meta(BASE) | {"rqs": "RQ5", "interpreter": model, "run_id_a": run_id(BASE, model, "N"),
                             "run_id_b": run_id(BASE, model, "b-requery-1s"),
                             "run_id_c": run_id(BASE, "control", "oracle")}
        for arm in ("a", "b", "c"):
            row |= _stat(rng, f"affected_{arm}") | _stat(rng, f"network_{arm}")
            row[f"throughput_mean_mbps_{arm}"] = rng.uniform(5, 20)
            row |= {f"jain_{cls}_{arm}": rng.uniform(0.3, 1.0) for cls in JAIN_CLASSES}
        for ref in ("a", "c"):
            row |= _stat(rng, f"affected_b_minus_{ref}", True) | _stat(rng, f"network_b_minus_{ref}", True)
        rq5.append(row | blocks)
        for rate, ues in RQ3_CELLS:
            flagged = (model, rate, ues) in NON_STATIONARY
            cell = {"design_point": rq3_point(model, rate, ues), "rqs": "RQ3", "rate_per_s": rate, "speed_kmh": 30.0,
                    "ues_per_cell": ues, "interpreter": model, "run_id": f"rq3__{model}__{rate:g}__{ues}",
                    "rho": rng.uniform(1.0, 1.2) if flagged else rng.uniform(0.0, 0.9), "non_stationary": flagged,
                    "block_inference": not flagged, "block_s": 10.0, "n_blocks": 10}
            for metric in ("affected", "network"):
                cell |= _stat(rng, metric)
                if flagged:
                    cell |= {f"{metric}_ci_low": None, f"{metric}_ci_high": None}
            rq3.append(cell | {"prb_util_mean": rng.uniform(0.3, 0.9), "share_enforced_within_1s": rng.uniform(0, 1),
                               "n_events": 100})
    _write_csv(directory / "rq5.csv", rq5)
    _write_csv(directory / "rq3_radio.csv", rq3)


def _unit(analysis: str, design_point: str | None, missing: list[str], **extra: Any) -> dict[str, Any]:
    return {"analysis": analysis, "design_point": design_point, **extra,
            "required_runs_unavailable": [{"run_id": rid, "status": "MISSING", "detail": "run directory absent"}
                                          for rid in missing]}


def _filter(directory: Path, name: str, keep: Any) -> None:
    rows = _read_csv(directory / name)
    _write_csv(directory / name, [row for row in rows if keep(row)])


def _pend(directory: Path, units: list[dict[str, Any]], absent: list[tuple[str, str]],
          hypotheses_edit: Any = None) -> None:
    skipped = json.loads((directory / "skipped.json").read_text(encoding="utf-8"))
    skipped["analysis_units"] += units
    skipped["unavailable_runs"] += [{"run_id": rid, "design_point": dp, "status": "MISSING",
                                     "detail": "run directory absent"} for rid, dp in absent]
    (directory / "skipped.json").write_text(json.dumps(skipped, indent=2) + "\n", encoding="utf-8")
    if hypotheses_edit is not None:
        hypotheses = json.loads((directory / "hypotheses.json").read_text(encoding="utf-8"))
        hypotheses_edit(hypotheses)
        (directory / "hypotheses.json").write_text(json.dumps(hypotheses, indent=2) + "\n", encoding="utf-8")


def drop_control(directory: Path, design_point: str, arm: str = "fixed-1") -> str:
    """As analysis.py --allow-incomplete writes it when one control run of an RQ2 design point is absent."""
    missing = run_id(design_point, "control", arm)
    _filter(directory, "controls.csv", lambda row: row["design_point"] != design_point)
    for name in ("l_arm_contrasts.csv", "an_decomposition.csv", "rq1_modes.csv"):
        _filter(directory, name, lambda row: row["design_point"] != design_point)
    units = [_unit("controls", design_point, [missing])]
    for model in MODELS:
        units.append(_unit("L-arm contrast", design_point, [missing], interpreter=model))
        units.append(_unit("A/N decomposition", design_point, [missing], interpreter=model))
        units.append(_unit("stale-policy exposure", design_point, [missing], interpreter=model, mode="E",
                           arm="L", run_id=run_id(design_point, model, "L")))
    radio = [row for row in _read_csv(directory / "radio_kpis.csv") if row["run_id"] != missing]
    for row in radio:
        if row["design_point"] == design_point and row["arm"] == "L":
            row["stale_exposure_ue_s_per_intent"] = ""
    _write_csv(directory / "radio_kpis.csv", radio)

    def edit(hypotheses: dict[str, Any]) -> None:
        for rule in RULES:
            hypotheses[rule]["H2"]["result"] = None
            hypotheses[rule]["H2"]["pending_missing_runs"] = [missing]

    units += [_unit("H2 Holm family", None, [missing], rule=rule) for rule in RULES]
    _pend(directory, units, [(missing, design_point)], edit)
    return missing


def drop_rq5(directory: Path, interpreter: str) -> str:
    """As analysis.py --allow-incomplete writes it when one RQ5 (b) run is absent."""
    missing = run_id(BASE, interpreter, "b-requery-1s")
    _filter(directory, "rq5.csv", lambda row: row["interpreter"] != interpreter)
    _pend(directory, [_unit("RQ5 division of labour", BASE, [missing], interpreter=interpreter),
                      _unit("radio KPIs", BASE, [missing], interpreter=interpreter, mode="E", arm="b-requery-1s",
                            run_id=missing)], [(missing, BASE)])
    return missing


def drop_rq3(directory: Path, interpreter: str, rate: float, ues: int) -> str:
    """As analysis.py --allow-incomplete writes it when one RQ3 replay run is absent."""
    point = rq3_point(interpreter, rate, ues)
    missing = f"rq3__{interpreter}__{rate:g}__{ues}"
    _filter(directory, "rq3_radio.csv", lambda row: row["design_point"] != point)
    _pend(directory, [_unit("RQ3 radio replay", point, [missing], interpreter=interpreter),
                      _unit("radio KPIs", point, [missing], interpreter=interpreter, mode="E", arm="N",
                            run_id=missing)], [(missing, point)])
    return missing


def drop_l_arm(directory: Path, design_point: str, interpreter: str) -> str:
    """As analysis.py --allow-incomplete writes it when one mode-E L arm (not an H1 anchor) is absent."""
    missing = run_id(design_point, interpreter, "L")
    gone = lambda row: row["design_point"] == design_point and row["interpreter"] == interpreter
    _filter(directory, "l_arm_contrasts.csv", lambda row: not gone(row))
    _filter(directory, "an_decomposition.csv", lambda row: not gone(row))
    _filter(directory, "radio_kpis.csv", lambda row: row["run_id"] != missing)
    units = [_unit("L-arm contrast", design_point, [missing], interpreter=interpreter),
             _unit("A/N decomposition", design_point, [missing], interpreter=interpreter),
             _unit("radio KPIs", design_point, [missing], interpreter=interpreter, mode="E", arm="L",
                   run_id=missing)]
    edit = None
    if interpreter in ANCHOR and design_point != INELIGIBLE:
        family = "H1" if design_point == BASE else "H2"

        def edit(hypotheses: dict[str, Any]) -> None:
            for rule in RULES:
                hypotheses[rule][family]["result"] = None
                hypotheses[rule][family]["pending_missing_runs"] = [missing]

        units += [_unit(f"{family} Holm family", design_point if family == "H1" else None, [missing], rule=rule)
                  for rule in RULES]
    _pend(directory, units, [(missing, design_point)], edit)
    return missing


def render(pa: Any, c2_dir: Path, root: Path, allow_partial: bool = False) -> dict[str, bytes]:
    """The C2 part of paper_assets.build, into root; returns every written file (tmp paths normalised)."""
    pa.check_palette()
    pa.setup_mpl()
    c2 = pa.C2Data(c2_dir, allow_partial=allow_partial)
    out = pa.Out(root / "paper", root / "analysis", png=False)
    if hasattr(pa, "build_c2"):
        pa.build_c2(c2, out)
    else:  # the pre-change build() sequence, kept to reproduce the golden strict output
        out.write_analysis_csv("c2_h2_summary.csv", c2.h2_csv)
        for step in (pa.fig_c2_h1, pa.fig_c2_grid, pa.fig_c2_an, pa.tab_c2_h1, pa.tab_c2_grid, pa.tab_c2_radio,
                     pa.tab_c2_controls):
            step(c2, out)
    (root / "analysis" / "c2_numbers.md").write_text("\n".join(pa.c2_numbers(c2)) + "\n", encoding="utf-8")
    artifacts = [dataclasses.asdict(artifact) for artifact in out.artifacts]
    (root / "artifacts.json").write_text(json.dumps(artifacts, indent=1) + "\n", encoding="utf-8")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(root))] = path.read_bytes().replace(str(c2_dir.resolve()).encode(), b"<C2>")
    return files
