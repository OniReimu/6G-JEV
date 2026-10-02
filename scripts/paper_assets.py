#!/usr/bin/env python3
"""Build the paper's figures, tables and number ledger from the released results.

Reported values come from the analysis outputs in experiments/*/results/. The C1 latency CDF is rebuilt from the
per-call C1 ledger fields in experiments/c1-interpretation/results/ledger-latency/ through the C1 selection code, and
the RQ3 and RQ6 per-intent summaries from the released traces and records. A provenance block is maintained in
paper_assets/figure-manifest.yml.

    python scripts/paper_assets.py            # figures and tables into paper_assets/, ledgers into experiments/
    python scripts/paper_assets.py --check    # rebuild in a temporary directory; compare SHA-256 with the paper
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_assets"
ANALYSIS = ROOT / "experiments"
C1 = ROOT / "experiments" / "c1-interpretation" / "results"
C1_RUNS = C1 / "ledger-latency"
C1_CORPUS = ROOT / "data" / "ranbench" / "ranintent-v1"
C2_RESULTS = ROOT / "experiments" / "c2-closed-loop" / "results" / "analysis"
JFC_SOURCE = ROOT
PNG_DIR = Path("/tmp/jev6g_paper_assets_png")
FIGSTAR_W, FIG_W, PANEL_H = 1.70, 1.68, 1.30
MANIFEST_BEGIN = "  # >>> scripts/paper_assets.py: generated block; do not edit by hand"
MANIFEST_END = "  # <<< scripts/paper_assets.py"
SCRIPT_REL = "../scripts/paper_assets.py"
ANALYSIS_REL = "../experiments"
ANALYSIS_GROUP = {'c1_latency_cdf_summary.csv': 'c1-interpretation/results',
    'rq3_load.csv': 'c1-interpretation/results/rq3-load',
    'rq3_trace_latency_summary.csv': 'c1-interpretation/results/rq3-load',
    'c3_path_decomposition.csv': 'c3-real-stack/results',
    'c3_path_samples.csv': 'c3-real-stack/results',
    'c2_h2_summary.csv': 'c2-closed-loop/results',
    'c2_pending.csv': 'c2-closed-loop/results'}


def _rel(path: Path) -> str:
    """Path relative to paper_assets/, as the manifest and the number ledger record it."""
    return os.path.relpath(path, PAPER)
OBSOLETE_PAPER_FILES = (
    "figures/fig_c1_cells.pdf",
    "figures/fig_c1_latency.pdf",
    "figures/fig_c3_path.pdf",
    "figures/fig_rq3_load.pdf",
    "tables/tab_c1_h3.tex",
    "tables/tab_rq3_availability.tex",
)

if str(JFC_SOURCE) not in sys.path:
    sys.path.insert(0, str(JFC_SOURCE))

from src.ranbench.c1_analysis import (  # noqa: E402
    DELTA_E2_S,
    EXPECTED_CONDITIONS,
    NEAR_RT_DEADLINE_S,
    PER_CASE_PRIMARY_LABEL,
    find_run_files,
    load_all_c1_runs,
    load_corpus_cases,
)
from src.ranbench.c2.stats import H2_MIN_ELIGIBLE  # noqa: E402

PALETTE = {
    "ebBlue": "DCE3F0",
    "ebAmber": "F4CC79",
    "ebBeige": "ECD7BC",
    "ebCyan": "CCFDFE",
    "ebCoral": "ED8683",
    "ebPink": "F7D3D2",
    "ebWarm": "F2F2EE",
    "ebDeepBlue": "B9C6E0",
    "ebInk": "2E5BE6",
}

MODELS = (
    "Jev-1.13.0",
    "SemIf-Qwen3.5-4B",
    "AnyJev-L0",
    "DeepSeek-V4.1-Flash",
    "GLM-5.3-Flash",
    "Qwen3.8-Flash",
    "Qwen3.5-4B-JSON",
)
HOSTED = {
    "Jev-1.13.0",
    "DeepSeek-V4.1-Flash",
    "GLM-5.3-Flash",
    "Qwen3.8-Flash",
}
SELF_HOSTED = set(MODELS) - HOSTED
FILL = {
    "Jev-1.13.0": PALETTE["ebCoral"],
    "SemIf-Qwen3.5-4B": PALETTE["ebBlue"],
    "AnyJev-L0": PALETTE["ebCyan"],
    "DeepSeek-V4.1-Flash": PALETTE["ebBeige"],
    "GLM-5.3-Flash": PALETTE["ebAmber"],
    "Qwen3.8-Flash": PALETTE["ebDeepBlue"],
    "Qwen3.5-4B-JSON": PALETTE["ebPink"],
}
MARKER = {
    "Jev-1.13.0": "o",
    "SemIf-Qwen3.5-4B": "s",
    "AnyJev-L0": "^",
    "DeepSeek-V4.1-Flash": "D",
    "GLM-5.3-Flash": "v",
    "Qwen3.8-Flash": "P",
    "Qwen3.5-4B-JSON": "X",
}
PRINT_LINESTYLE = {
    "Jev-1.13.0": "-",
    "SemIf-Qwen3.5-4B": "--",
    "AnyJev-L0": ":",
    "DeepSeek-V4.1-Flash": "-.",
    "GLM-5.3-Flash": (0, (5, 1)),
    "Qwen3.8-Flash": (0, (3, 1, 1, 1)),
    "Qwen3.5-4B-JSON": (0, (1, 1, 3, 1)),
}
PRINT_MARKER_EDGE = "#202020"
SHORT = {
    "Jev-1.13.0": "Jev",
    "SemIf-Qwen3.5-4B": "SemIf",
    "AnyJev-L0": "AnyJev",
    "DeepSeek-V4.1-Flash": "DeepSeek",
    "GLM-5.3-Flash": "GLM",
    "Qwen3.8-Flash": "Qwen3.8",
    "Qwen3.5-4B-JSON": "Qwen-JSON",
}
PLOT_CODE = {
    "Jev-1.13.0": "JEV",
    "SemIf-Qwen3.5-4B": "SIF",
    "AnyJev-L0": "ANY",
    "DeepSeek-V4.1-Flash": "DSK",
    "GLM-5.3-Flash": "GLM",
    "Qwen3.8-Flash": "Q38",
    "Qwen3.5-4B-JSON": "QJS",
}
TEX_NAME = {
    "Jev-1.13.0": r"\jev{}",
    "SemIf-Qwen3.5-4B": r"\semif{}",
    "AnyJev-L0": r"AnyJev-L0",
    "DeepSeek-V4.1-Flash": r"\deepseek{}",
    "GLM-5.3-Flash": r"\glm{}",
    "Qwen3.8-Flash": r"\qwenflash{}",
    "Qwen3.5-4B-JSON": r"\qwenjson{} (ref.)",
}
PDF_META = {
    "Creator": "scripts/paper_assets.py",
    "Producer": None,
    "CreationDate": None,
    "ModDate": None,
}


def hexcol(value: str) -> str:
    return "#" + value


def dark(value: str, amount: float = 0.34) -> str:
    r, g, b = (int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    hue, light, sat = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(hue, max(0.0, light - amount), sat)
    return "#{:02X}{:02X}{:02X}".format(round(r * 255), round(g * 255), round(b * 255))


def linestyle(model: str) -> str:
    return "-" if model in HOSTED else "--"


def check_palette() -> None:
    if not (PAPER / "palette.tex").exists():  # the paper's LaTeX sources are not distributed with the code
        return
    text = (PAPER / "palette.tex").read_text(encoding="utf-8")
    found = dict(re.findall(r"\\definecolor\{(\w+)\}\{HTML\}\{([0-9A-Fa-f]{6})\}", text))
    found = {key: value.upper() for key, value in found.items()}
    if found != PALETTE:
        raise SystemExit(f"PALETTE mismatch with paper/palette.tex:\n  tex={found}\n  py ={PALETTE}")


def setup_mpl() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.45,
            "axes.axisbelow": True,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
            "axes.labelpad": 2.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.hashsalt": "jev6g-paper-assets",
        }
    )


def with_csv_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.insert(0, "_csv_row", np.arange(1, len(frame) + 1))
    return frame


@dataclass
class C1LatencyData:
    latencies: dict[str, np.ndarray]
    summary: pd.DataFrame
    summary_csv: str
    reference_mismatches: list[str]


def c1_analysis_run_dirs() -> list[Path]:
    """Return the same C1 run-directory family used by the C2 record exporter."""
    fixed = ("c1_hosted", "c1_selfhosted", "c1_rerun")
    run_dirs = [C1_RUNS / name for name in fixed if (C1_RUNS / name).is_dir()]
    for pattern in ("c1_hosted_qwen_block*", "c1_hosted_qwen_gapfill*"):
        run_dirs.extend(sorted(path for path in C1_RUNS.glob(pattern) if path.is_dir()))
    if not run_dirs:
        raise FileNotFoundError(f"no C1 analysis run directories found under {C1_RUNS}")
    return run_dirs


def c1_latency_data(quality: pd.DataFrame, feasibility: pd.DataFrame) -> C1LatencyData:
    """Select primary C1 rows, reproduce frozen cell metrics, and pool all 16 conditions."""
    corpus = load_corpus_cases(C1_CORPUS)
    missing = [condition for condition in EXPECTED_CONDITIONS if not corpus.get(condition)]
    if missing:
        raise ValueError(f"C1 corpus is missing conditions: {', '.join(missing)}")
    expected_case_ids = {condition: set(corpus[condition]) for condition in EXPECTED_CONDITIONS}
    discovered = find_run_files(c1_analysis_run_dirs())
    loaded = load_all_c1_runs(
        discovered["ledgers"], block_of=discovered["block_of"], expected_case_ids=expected_case_ids
    )

    quality_ref = quality.set_index(["model", "condition"])
    feasibility_ref = feasibility.set_index(["model", "condition"])
    latency_by_model: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    metric_names = ("latency_p50", "latency_p95", "latency_p99")

    for model in MODELS:
        pooled_rows: list[dict[str, Any]] = []
        percentile_matches = {metric: 0 for metric in metric_names}
        phi_matches = 0
        primary_rules: set[str] = set()

        for condition in EXPECTED_CONDITIONS:
            key = (model, condition)
            primary_rows = loaded.primary_by_cell.get(key, [])
            expected_n = len(expected_case_ids[condition])
            if len(primary_rows) != expected_n:
                raise ValueError(
                    f"{model} {condition}: selected {len(primary_rows)} primary rows, expected {expected_n}"
                )
            pooled_rows.extend(primary_rows)
            primary_rules.add(str(loaded.primary_block_by_cell.get(key)))

            cell_latencies = np.asarray(
                [float(row["latency_s"]) for row in primary_rows if row.get("latency_s") is not None],
                dtype=float,
            )
            if not len(cell_latencies):
                raise ValueError(f"{model} {condition}: no primary row reports latency_s")
            cell_percentiles = np.percentile(cell_latencies, [50, 95, 99])
            reference_quality = quality_ref.loc[key]
            for metric, actual in zip(metric_names, cell_percentiles):
                expected = float(reference_quality[metric])
                if np.isclose(actual, expected, rtol=0.0, atol=1e-12):
                    percentile_matches[metric] += 1
                else:
                    mismatches.append(
                        f"{model} {condition} {metric}: raw={actual:.17g}, reference={expected:.17g}"
                    )

            feasible = [
                1.0
                if (
                    row.get("latency_s") is not None
                    and row.get("error_type") is None
                    and float(row["latency_s"]) + DELTA_E2_S <= NEAR_RT_DEADLINE_S
                )
                else 0.0
                for row in primary_rows
            ]
            cell_phi = float(np.mean(feasible))
            expected_phi = float(feasibility_ref.loc[key]["phi"])
            if np.isclose(cell_phi, expected_phi, rtol=0.0, atol=1e-12):
                phi_matches += 1
            else:
                mismatches.append(
                    f"{model} {condition} phi: raw={cell_phi:.17g}, reference={expected_phi:.17g}"
                )

        pooled_latencies = np.asarray(
            [float(row["latency_s"]) for row in pooled_rows if row.get("latency_s") is not None], dtype=float
        )
        if np.any(pooled_latencies <= 0.0):
            raise ValueError(f"{model}: non-positive latency cannot be plotted on a log axis")
        latency_by_model[model] = np.sort(pooled_latencies)
        pooled_percentiles = np.percentile(pooled_latencies, [50, 95, 99])
        pooled_feasible = sum(
            row.get("latency_s") is not None
            and row.get("error_type") is None
            and float(row["latency_s"]) + DELTA_E2_S <= NEAR_RT_DEADLINE_S
            for row in pooled_rows
        )
        all_reference_cells_match = (
            all(count == len(EXPECTED_CONDITIONS) for count in percentile_matches.values())
            and phi_matches == len(EXPECTED_CONDITIONS)
        )
        primary_rule = (
            PER_CASE_PRIMARY_LABEL
            if primary_rules == {PER_CASE_PRIMARY_LABEL}
            else "primary-block"
        )
        summary_rows.append(
            {
                "model": model,
                "primary_rule": primary_rule,
                "conditions_pooled": len(EXPECTED_CONDITIONS),
                "n_primary": len(pooled_rows),
                "n_latency": len(pooled_latencies),
                "n_feasible": pooled_feasible,
                "delta_e2_s": DELTA_E2_S,
                "deadline_s": NEAR_RT_DEADLINE_S,
                "phi": pooled_feasible / len(pooled_rows),
                "latency_p50": float(pooled_percentiles[0]),
                "latency_p95": float(pooled_percentiles[1]),
                "latency_p99": float(pooled_percentiles[2]),
                "quality_p50_cells_matched": percentile_matches["latency_p50"],
                "quality_p95_cells_matched": percentile_matches["latency_p95"],
                "quality_p99_cells_matched": percentile_matches["latency_p99"],
                "feasibility_phi_cells_matched": phi_matches,
                "reference_cells_total": len(EXPECTED_CONDITIONS),
                "reference_status": "match" if all_reference_cells_match else "mismatch",
            }
        )

    summary_plain = pd.DataFrame(summary_rows)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(summary_plain.columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary_plain.to_dict(orient="records"))
    summary_with_rows = summary_plain.copy()
    summary_with_rows.insert(0, "_csv_row", np.arange(1, len(summary_with_rows) + 1))
    return C1LatencyData(latency_by_model, summary_with_rows, buffer.getvalue(), mismatches)


def frame_csv(frame: pd.DataFrame) -> str:
    """Serialize a generated analysis frame without its ledger-only row index."""
    return frame.drop(columns=["_csv_row"], errors="ignore").to_csv(
        index=False, lineterminator="\n", na_rep=""
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_row"] = line_number
            rows.append(row)
    return rows


def assert_percentile(actual: float, expected: Any, context: str) -> None:
    if not np.isclose(actual, float(expected), rtol=0.0, atol=1e-10):
        raise ValueError(f"{context}: raw={actual:.17g}, CSV={float(expected):.17g}")


def rq3_latency_data(load: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Aggregate per-intent interpretation latency from each primary RQ3 trace."""
    rows: list[dict[str, Any]] = []
    for _, cell in load.iterrows():
        source = Path(str(cell["source_trace"]))
        records = read_jsonl(ROOT / source)
        if len(records) != int(cell["n"]):
            raise ValueError(f"{source}: {len(records)} trace rows, expected n={int(cell['n'])}")
        if any("latency_s" not in record or record["latency_s"] is None for record in records):
            raise ValueError(f"{source}: per-intent latency_s is missing")
        if any(str(record.get("model")) != str(cell["model"]) for record in records):
            raise ValueError(f"{source}: trace model does not match rq3_load.csv")
        latencies = np.asarray([float(record["latency_s"]) for record in records], dtype=float)
        if np.any(latencies <= 0.0):
            raise ValueError(f"{source}: non-positive latency_s")
        p50, p95 = np.percentile(latencies, [50, 95])
        rows.append(
            {
                "model": str(cell["model"]),
                "deployment": str(cell["deployment"]),
                "rate_per_s": float(cell["rate_per_s"]),
                "ell_median_s": float(p50),
                "ell_p95_s": float(p95),
                "n_latency": len(latencies),
                "source_trace": str(source),
                "source_trace_rows": f"1-{max(record['_source_row'] for record in records)}",
                "latency_field": "latency_s",
            }
        )
    frame = pd.DataFrame(rows)
    frame.insert(0, "_csv_row", np.arange(1, len(frame) + 1))
    return frame, frame_csv(frame)


def c3_sample_data(summary: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Load C3 record samples and reproduce every aggregate used by the new assets."""
    samples: list[dict[str, Any]] = []
    checks = (
        ("ell_s", "n", "ell_median_s", "ell_p95_s", 1.0),
        ("delta_a1_s", "n_delta_a1", "delta_a1_median_ms", "delta_a1_p95_ms", 1000.0),
        ("delta_e2_s", "n_delta_e2", "delta_e2_median_ms", "delta_e2_p95_ms", 1000.0),
        ("kpm_change_upper_bound_s", "n_kpm_observed", "kpm_upper_bound_median_s",
         "kpm_upper_bound_p95_s", 1.0),
    )
    for _, row in summary.iterrows():
        source = Path(str(row["source_records"]))
        records = read_jsonl(ROOT / source)
        if len(records) != int(row["n"]):
            raise ValueError(f"{source}: {len(records)} records, expected n={int(row['n'])}")
        if any(str(record.get("interpreter")) != str(row["model"]) for record in records):
            raise ValueError(f"{source}: record interpreter does not match c3_path_decomposition.csv")
        for field, n_col, median_col, p95_col, factor in checks:
            values = np.asarray(
                [float(record[field]) * factor for record in records if record.get(field) is not None], dtype=float
            )
            if len(values) != int(row[n_col]):
                raise ValueError(f"{source} {field}: n={len(values)}, CSV {n_col}={int(row[n_col])}")
            p50, p95 = np.percentile(values, [50, 95])
            assert_percentile(float(p50), row[median_col], f"{source} {median_col}")
            assert_percentile(float(p95), row[p95_col], f"{source} {p95_col}")
        for record in records:
            samples.append(
                {
                    "model": str(row["model"]),
                    "deployment": str(row["deployment"]),
                    "source_records": str(source),
                    "source_record_row": int(record["_source_row"]),
                    "ell_s": record.get("ell_s"),
                    "delta_a1_ms": (None if record.get("delta_a1_s") is None
                                    else 1000.0 * float(record["delta_a1_s"])),
                    "delta_e2_ms": (None if record.get("delta_e2_s") is None
                                    else 1000.0 * float(record["delta_e2_s"])),
                    "kpm_upper_bound_s": record.get("kpm_change_upper_bound_s"),
                }
            )
    frame = pd.DataFrame(samples)
    frame.insert(0, "_csv_row", np.arange(1, len(frame) + 1))
    return frame, frame_csv(frame)


class Data:
    def __init__(self) -> None:
        self.curves = with_csv_rows(C1 / "cell_count_curves.csv")
        self.h3 = with_csv_rows(C1 / "h3_confirmatory.csv")
        self.exploratory = with_csv_rows(C1 / "exploratory_contrasts.csv")
        self.h3_tests = with_csv_rows(C1 / "h3_c57_comparisons.csv")
        self.quality = with_csv_rows(C1 / "quality_and_cost.csv")
        self.feasibility = with_csv_rows(C1 / "near_rt_feasibility.csv")
        latency = c1_latency_data(self.quality, self.feasibility)
        self.c1_latencies = latency.latencies
        self.c1_latency_summary = latency.summary
        self.c1_latency_summary_csv = latency.summary_csv
        self.c1_latency_reference_mismatches = latency.reference_mismatches
        self.c3 = with_csv_rows(ANALYSIS / "c3-real-stack/results/c3_path_decomposition.csv")
        self.rq3 = with_csv_rows(ANALYSIS / "c1-interpretation/results/rq3-load/rq3_load.csv")
        self.rq3_latency, self.rq3_latency_csv = rq3_latency_data(self.rq3)
        self.c3_samples, self.c3_samples_csv = c3_sample_data(self.c3)
        for name, frame in (
            ("cell_count_curves", self.curves),
            ("quality_and_cost", self.quality),
            ("c3_path_decomposition", self.c3),
        ):
            roster = set(frame["model"])
            if roster != set(MODELS):
                raise ValueError(f"{name}: roster mismatch, got {sorted(roster)}")
        if len(self.rq3) != 28 or set(self.rq3["model"]) != set(MODELS):
            raise ValueError("RQ3 needs 28 primary interpreter-rate cells")

    @staticmethod
    def one(frame: pd.DataFrame, **filters: Any) -> pd.Series:
        selected = frame
        for column, value in filters.items():
            selected = selected[selected[column] == value]
        if len(selected) != 1:
            raise ValueError(f"expected one row for {filters}, found {len(selected)}")
        return selected.iloc[0]

    def n_cases(self, model: str) -> int:
        return int(self.one(self.h3, model=model)["n_cases"])


@dataclass
class Artifact:
    artifact_id: str
    paper_path: str
    source_data: list[str]
    columns: list[str]
    filters: str
    plot_type: str


class Out:
    def __init__(self, paper: Path, analysis: Path, png: bool) -> None:
        self.paper = paper
        self.analysis = analysis
        self.png = png
        self.files: list[tuple[Path, Path]] = []
        self.artifacts: list[Artifact] = []
        self.reference_mismatches: list[str] = []
        (paper / "figures").mkdir(parents=True, exist_ok=True)
        (paper / "tables").mkdir(parents=True, exist_ok=True)
        analysis.mkdir(parents=True, exist_ok=True)

    def save_fig(self, fig: Any, name: str) -> None:
        rel = Path("figures") / name
        target = self.paper / rel
        fig.savefig(target, metadata=PDF_META)
        if self.png:
            PNG_DIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(PNG_DIR / (Path(name).stem + ".png"), dpi=150, metadata={})
        plt.close(fig)
        self.files.append((target, PAPER / rel))

    def write_table(self, name: str, text: str) -> None:
        rel = Path("tables") / name
        target = self.paper / rel
        target.write_text(text, encoding="utf-8", newline="\n")
        self.files.append((target, PAPER / rel))

    def write_figure(self, name: str, text: str) -> None:
        rel = Path("figures") / name
        target = self.paper / rel
        target.write_text(text, encoding="utf-8", newline="\n")
        self.files.append((target, PAPER / rel))

    def write_numbers(self, text: str) -> None:
        target = self.analysis / "paper_numbers.md"
        target.write_text(text, encoding="utf-8", newline="\n")
        self.files.append((target, ANALYSIS / "paper_numbers.md"))

    def write_analysis_csv(self, name: str, text: str) -> None:
        if self.analysis == ANALYSIS:  # the release keeps each analysis file with its experiment group
            name = f"{ANALYSIS_GROUP[name]}/{name}"
        target = self.analysis / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        self.files.append((target, ANALYSIS / name))

    def entry(self, artifact: Artifact) -> None:
        self.artifacts.append(artifact)


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], color=dark(FILL[model]), linestyle=PRINT_LINESTYLE[model], marker=MARKER[model],
            markerfacecolor=hexcol(FILL[model]), markeredgecolor=PRINT_MARKER_EDGE, markeredgewidth=0.7,
            markersize=3.4, linewidth=1.3 if model == "Jev-1.13.0" else 0.9,
        )
        for model in MODELS
    ]


def new_panel(width: float) -> tuple[Any, Any]:
    return plt.subplots(figsize=(width, PANEL_H), layout="constrained")


def model_line(
    ax: Any,
    model: str,
    x: Iterable[float],
    y: Iterable[float],
    *,
    hollow_indices: Iterable[int] = (),
) -> None:
    xs, ys = list(x), list(y)
    line_colour = dark(FILL[model])
    zorder = 5 if model == "Jev-1.13.0" else 3
    ax.plot(
        xs, ys, color=line_colour, linestyle=PRINT_LINESTYLE[model], marker=MARKER[model],
        markerfacecolor=hexcol(FILL[model]), markeredgecolor=PRINT_MARKER_EDGE, markeredgewidth=0.7,
        markersize=3.6 if model == "Jev-1.13.0" else 3.1,
        linewidth=1.3 if model == "Jev-1.13.0" else 0.9, zorder=zorder,
    )
    hollow = list(hollow_indices)
    if hollow:
        ax.plot(
            [xs[index] for index in hollow], [ys[index] for index in hollow], linestyle="none",
            marker=MARKER[model], markerfacecolor="white", markeredgecolor=line_colour,
            markeredgewidth=0.9, markersize=4.3, zorder=zorder + 2,
        )


def wilson_interval(proportion: float, n: int) -> tuple[float, float]:
    """Deterministic 95% Wilson interval from a frozen proportion and case count."""
    z = 1.959963984540054
    successes = int(round(float(proportion) * n))
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def grouped_bars(
    ax: Any, groups: list[str], getter: Any, ci_getter: Any | None = None,
    models: tuple[str, ...] = MODELS,
) -> dict[tuple[str, int], float]:
    width = 0.84 / len(models)
    positions: dict[tuple[str, int], float] = {}
    for group_index, _ in enumerate(groups):
        for model_index, model in enumerate(models):
            value = float(getter(model, group_index))
            x = group_index + (model_index - (len(models) - 1) / 2) * width
            if not math.isfinite(value):
                continue  # a cell absent from a partial C2 analysis is left empty
            positions[(model, group_index)] = x
            ax.bar(
                x, value, width=width, color=hexcol(FILL[model]), edgecolor="black",
                linewidth=0.8 if model == "Jev-1.13.0" else 0.5,
                hatch="////" if model in SELF_HOSTED else None, zorder=4 if model == "Jev-1.13.0" else 3,
            )
            if ci_getter is not None:
                low, high = ci_getter(model, group_index)
                ax.errorbar(
                    x, value, yerr=[[max(0.0, value - low)], [max(0.0, high - value)]], fmt="none",
                    ecolor="black", elinewidth=0.45, capsize=0.8, capthick=0.45, zorder=6,
                )
    ax.set_xlim(-0.5, len(groups) - 0.5)
    return positions


def legend_strip(out: Out, name: str, handles: list[Any], labels: list[str], width: float, ncol: int) -> None:
    fig = plt.figure(figsize=(width, 0.25))
    columns = ncol
    for columns in range(ncol, 0, -1):
        legend = fig.legend(
            handles, labels, loc="center", ncol=columns, frameon=False, handlelength=1.8,
            columnspacing=0.8, handletextpad=0.35, borderaxespad=0.0, labelspacing=0.3,
        )
        fig.canvas.draw()
        if legend.get_window_extent().width <= 0.99 * fig.bbox.width:
            break
        legend.remove()
    fig.set_size_inches(width, 0.07 + 0.155 * math.ceil(len(handles) / columns))
    out.save_fig(fig, name)


def figure_wrapper(
    out: Out,
    *,
    name: str,
    star: bool,
    panels: list[tuple[str, str]],
    caption: str,
) -> None:
    environment = "figure*" if star else "figure"
    panel_width = FIGSTAR_W if star else FIG_W
    legend_width = 7.00 if star else 3.36
    lines = [
        "% Generated by scripts/paper_assets.py; do not edit by hand.",
        f"\\begin{{{environment}}}[!t]",
        r"\centering",
        f"\\includegraphics[width={legend_width:.2f}in]{{figures/{name}_legend.pdf}}\\\\[-2pt]",
    ]
    subfloats = [
        f"\\subfloat[{subcaption}]{{\\includegraphics[width={panel_width:.2f}in]"
        f"{{figures/{name}_{letter}.pdf}}\\label{{fig:{name[4:]}_{letter}}}}}"
        for letter, subcaption in panels
    ]
    lines.append("\\hfill\n".join(subfloats))
    lines.extend([f"\\caption{{{caption}}}", f"\\label{{fig:{name[4:]}}}", f"\\end{{{environment}}}", ""])
    out.write_figure(f"{name}.tex", "\n".join(lines))


def add_figure_entries(
    out: Out,
    *,
    name: str,
    panel_specs: list[tuple[str, str, list[str], list[str], str]],
    star: bool,
) -> None:
    for letter, plot_type, sources, columns, filters in panel_specs:
        out.entry(Artifact(f"{name}_{letter}", f"figures/{name}_{letter}.pdf", sources, columns, filters, plot_type))
    out.entry(Artifact(f"{name}_legend", f"figures/{name}_legend.pdf", [], [], "", "legend strip"))
    out.entry(Artifact(name, f"figures/{name}.tex", [], [], "", f"{'figure*' if star else 'figure'} wrapper"))


def accuracy_line_panel(data: Data, out: Out, *, quality: str, letter: str) -> None:
    cells, xs = [3, 7, 21, 57], np.arange(4)
    fig, ax = new_panel(FIGSTAR_W)
    for model in MODELS:
        values = [
            float(data.one(data.curves, model=model, quality=quality, n_cells=count)["target_cluster_state_dep_acc"])
            for count in cells
        ]
        intervals = np.asarray([wilson_interval(value, data.n_cases(model)) for value in values])
        ax.fill_between(xs, intervals[:, 0], intervals[:, 1], color=hexcol(FILL[model]), alpha=0.35,
                        linewidth=0, zorder=1)
        model_line(ax, model, xs, values)
    ax.set_xticks(xs, [str(cell) for cell in cells])
    ax.set_xlabel("Telemetry cells")
    ax.set_ylabel("Target-cluster acc.")
    ax.set_ylim(-0.03, 1.03)
    out.save_fig(fig, f"fig_c1_{letter}.pdf")


def fig_c1(data: Data, out: Out) -> None:
    accuracy_line_panel(data, out, quality="fresh", letter="a")
    accuracy_line_panel(data, out, quality="contradictory", letter="b")

    qualities = ["fresh", "stale", "noisy", "contradictory"]
    quality_labels = ["Fresh", "Stale", "Noisy", "Contrad."]
    fig, ax = new_panel(FIGSTAR_W)
    value = lambda model, index: float(data.one(
        data.curves, model=model, quality=qualities[index], n_cells=57
    )["target_cluster_state_dep_acc"])
    grouped_bars(
        ax, qualities, value,
        lambda model, index: wilson_interval(value(model, index), data.n_cases(model)),
    )
    ax.set_xticks(np.arange(4), quality_labels, rotation=20, ha="right")
    ax.set_xlabel("Telemetry quality at 57 cells")
    ax.set_ylabel("Target-cluster acc.")
    ax.set_ylim(-0.03, 1.03)
    out.save_fig(fig, "fig_c1_c.pdf")

    fig, ax = new_panel(FIGSTAR_W)
    for model in MODELS:
        latencies = data.c1_latencies[model]
        probabilities = np.arange(1, len(latencies) + 1, dtype=float) / len(latencies)
        ax.step(
            np.concatenate(([latencies[0]], latencies)), np.concatenate(([0.0], probabilities)), where="post",
            color=dark(FILL[model]), linestyle=PRINT_LINESTYLE[model],
            linewidth=1.3 if model == "Jev-1.13.0" else 0.9, zorder=5 if model == "Jev-1.13.0" else 3,
        )
    ax.axvline(NEAR_RT_DEADLINE_S, color="black", linestyle=":", linewidth=0.8, zorder=2)
    ax.annotate("1 s", xy=(NEAR_RT_DEADLINE_S, 0.06), xytext=(-2, 1), textcoords="offset points",
                rotation=90, ha="right", va="bottom", fontsize=6.5)
    ax.set_xscale("log")
    all_latencies = np.concatenate([data.c1_latencies[model] for model in MODELS])
    ax.set_xlim(float(all_latencies.min()) / 1.15, float(all_latencies.max()) * 1.15)
    ax.set_ylim(0.0, 1.005)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel(r"Decision $\ell$ (s, log)")
    ax.set_ylabel("Empirical CDF")
    out.save_fig(fig, "fig_c1_d.pdf")

    legend_strip(out, "fig_c1_legend.pdf", legend_handles(), [SHORT[model] for model in MODELS], 7.00, 7)
    figure_wrapper(
        out, name="fig_c1", star=True,
        panels=[("a", "Fresh telemetry"), ("b", "Contradictory telemetry"),
                ("c", "Telemetry quality at 57 cells"), ("d", "Decision-latency ECDF")],
        caption=("(a) Target-cluster accuracy is plotted against telemetry cell count under fresh telemetry with "
                 "95\\% Wilson bands. (b) Target-cluster accuracy is plotted against telemetry cell count under "
                 "contradictory telemetry with 95\\% Wilson bands. (c) Grouped bars encode target-cluster accuracy "
                 "at 57 cells across four telemetry qualities. (d) Empirical cumulative distributions encode "
                 "decision latency pooled over 16 conditions with a vertical 1 s reference."),
    )
    curve_source = _rel(C1 / "cell_count_curves.csv")
    add_figure_entries(
        out, name="fig_c1", star=True,
        panel_specs=[
            ("a", "line with 95% Wilson bands", [curve_source, _rel(C1 / "h3_confirmatory.csv")],
             ["model", "quality", "n_cells", "target_cluster_state_dep_acc", "n_cases"], "quality=fresh"),
            ("b", "line with 95% Wilson bands", [curve_source, _rel(C1 / "h3_confirmatory.csv")],
             ["model", "quality", "n_cells", "target_cluster_state_dep_acc", "n_cases"],
             "quality=contradictory"),
            ("c", "grouped bar with 95% Wilson intervals", [curve_source, _rel(C1 / "h3_confirmatory.csv")],
             ["model", "quality", "n_cells", "target_cluster_state_dep_acc", "n_cases"], "n_cells=57"),
            ("d", "empirical CDF line (log-x)",
             [_rel(C1_RUNS), _rel(C1_CORPUS), f"{ANALYSIS_REL}/c1-interpretation/results/c1_latency_cdf_summary.csv"],
             ["model", "condition", "case_id", "latency_s"], "primary rows pooled over all 16 conditions"),
        ],
    )


def rq3_panel_setup(ax: Any, rates: list[float]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(rates, ["0.1", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_xlabel(r"Offered rate $\lambda$ (intents/s)")


def fig_rq3(data: Data, out: Out) -> None:
    rates = [0.1, 0.5, 1.0, 2.0]
    specs = (
        ("a", "share_enforced_within_1s", "Enforced within 1 s", False),
        ("b", "rho", r"Slot utilization $\eta$", False),
        ("c", "queue_wait_p95_s", "Queue p95 (s)", True),
    )
    for letter, column, ylabel, log_y in specs:
        fig, ax = new_panel(FIGSTAR_W)
        for model in MODELS:
            subset = data.rq3[data.rq3["model"] == model].sort_values("rate_per_s")
            if list(subset["rate_per_s"].astype(float)) != rates:
                raise ValueError(f"RQ3 rate grid mismatch for {model}")
            hollow = [index for index, value in enumerate(subset["rho"]) if letter == "b" and float(value) > 1.0]
            model_line(ax, model, rates, [float(value) for value in subset[column]], hollow_indices=hollow)
        if letter == "a":
            ax.set_ylim(-0.03, 1.03)
        if letter == "b":
            ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
            ax.set_ylim(bottom=0.0)
        if log_y:
            ax.set_yscale("log")
            ylabel = "Queue p95 (s, log)"
        rq3_panel_setup(ax, rates)
        ax.set_ylabel(ylabel)
        out.save_fig(fig, f"fig_rq3_{letter}.pdf")

    fig, ax = new_panel(FIGSTAR_W)
    for model in MODELS:
        subset = data.rq3_latency[data.rq3_latency["model"] == model].sort_values("rate_per_s")
        model_line(ax, model, rates, [float(value) for value in subset["ell_median_s"]])
    rq3_panel_setup(ax, rates)
    ax.set_ylabel(r"Median $\ell$ (s)")
    out.save_fig(fig, "fig_rq3_d.pdf")

    legend_strip(out, "fig_rq3_legend.pdf", legend_handles(), [SHORT[model] for model in MODELS], 7.00, 7)
    figure_wrapper(
        out, name="fig_rq3", star=True,
        panels=[("a", "Enforced within 1 s"), ("b", "Interpretation-slot utilization"),
                ("c", "Queue wait p95"), ("d", "Median decision latency")],
        caption=("(a) The share of intents enforced within 1 s is plotted against offered rate. "
                 "(b) Interpretation-slot utilization is plotted against offered rate with a horizontal "
                 "$\\eta=1$ reference and hollow markers for non-stationary cells. (c) The p95 queue wait is "
                 "plotted against offered rate on a logarithmic vertical axis. (d) Median per-intent decision "
                 "latency is plotted against offered rate."),
    )
    load_source = f"{ANALYSIS_REL}/c1-interpretation/results/rq3-load/rq3_load.csv"
    latency_source = f"{ANALYSIS_REL}/c1-interpretation/results/rq3-load/rq3_trace_latency_summary.csv"
    add_figure_entries(
        out, name="fig_rq3", star=True,
        panel_specs=[
            ("a", "line", [load_source], ["model", "rate_per_s", "share_enforced_within_1s"],
             "all 28 primary cells"),
            ("b", "line with hollow non-stationary markers", [load_source],
             ["model", "rate_per_s", "rho", "non_stationary"], "all 28 primary cells; hollow when rho>1"),
            ("c", "line (log-y)", [load_source], ["model", "rate_per_s", "queue_wait_p95_s"],
             "all 28 primary cells"),
            ("d", "line", [latency_source], ["model", "rate_per_s", "ell_median_s", "latency_field"],
             "all 28 primary traces; latency_field=latency_s"),
        ],
    )


def recolour_boxes(boxplot: dict[str, Any], colour: str) -> None:
    for box in boxplot["boxes"]:
        box.set_facecolor(hexcol(colour))
        box.set_edgecolor("black")
        box.set_linewidth(0.6)
    for key in ("whiskers", "caps", "medians"):
        for artist in boxplot[key]:
            artist.set_color("black")
            artist.set_linewidth(0.6 if key != "medians" else 0.9)


def fig_c3(data: Data, out: Out) -> None:
    positions = np.arange(len(MODELS), dtype=float)
    labels = [PLOT_CODE[model] for model in MODELS]

    fig, ax = new_panel(FIG_W)
    ell = [
        data.c3_samples[data.c3_samples["model"] == model]["ell_s"].dropna().astype(float).to_numpy()
        for model in MODELS
    ]
    boxes = ax.boxplot(ell, positions=positions, widths=0.58, patch_artist=True, showfliers=False)
    recolour_boxes(boxes, PALETTE["ebCoral"])
    ax.axhline(NEAR_RT_DEADLINE_S, color="black", linestyle=":", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.5, 1.0, 2.0], ["0.2", "0.5", "1", "2"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(positions, labels, rotation=45, ha="right")
    ax.set_ylabel(r"Decision $\ell$ (s, log)")
    out.save_fig(fig, "fig_c3_a.pdf")

    fig, ax = new_panel(FIG_W)
    a1 = [
        data.c3_samples[data.c3_samples["model"] == model]["delta_a1_ms"].dropna().astype(float).to_numpy()
        for model in MODELS
    ]
    e2 = [
        data.c3_samples[data.c3_samples["model"] == model]["delta_e2_ms"].dropna().astype(float).to_numpy()
        for model in MODELS
    ]
    a1_boxes = ax.boxplot(a1, positions=positions - 0.17, widths=0.28, patch_artist=True, showfliers=False)
    e2_boxes = ax.boxplot(e2, positions=positions + 0.17, widths=0.28, patch_artist=True, showfliers=False)
    recolour_boxes(a1_boxes, PALETTE["ebAmber"])
    recolour_boxes(e2_boxes, PALETTE["ebBlue"])
    ax.set_xticks(positions, labels, rotation=45, ha="right")
    ax.set_ylabel("Protocol latency (ms)")
    out.save_fig(fig, "fig_c3_b.pdf")

    component_handles = [
        Patch(facecolor=hexcol(PALETTE["ebCoral"]), edgecolor="black"),
        Patch(facecolor=hexcol(PALETTE["ebAmber"]), edgecolor="black"),
        Patch(facecolor=hexcol(PALETTE["ebBlue"]), edgecolor="black"),
    ]
    legend_strip(out, "fig_c3_legend.pdf", component_handles,
                 [r"$\ell$", r"$\delta_{\mathrm{A1}}$", r"$\delta_{\mathrm{E2}}$"], 3.36, 3)
    figure_wrapper(
        out, name="fig_c3", star=False,
        panels=[("a", "Decision latency"), ("b", "A1 and E2 latency")],
        caption=("(a) Box plots encode per-intent interpreter decision latency on a logarithmic axis with a "
                 "horizontal 1 s reference. (b) Paired box plots encode per-intent A1 and E2 latency in milliseconds."),
    )
    source = f"{ANALYSIS_REL}/c3-real-stack/results/c3_path_samples.csv"
    add_figure_entries(
        out, name="fig_c3", star=False,
        panel_specs=[
            ("a", "box plot (log-y)", [source], ["model", "ell_s", "source_records", "source_record_row"],
             "30 record rows per interpreter"),
            ("b", "paired box plot", [source],
             ["model", "delta_a1_ms", "delta_e2_ms", "source_records", "source_record_row"],
             "null component samples dropped"),
        ],
    )


def tex_float(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def tex_p(value: Any) -> str:
    if pd.isna(value):
        return "--"
    number = float(value)
    if number == 0.0:
        return "$<10^{-4}$"
    if number < 0.001:
        mantissa, exponent = f"{number:.1e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return f"{number:.3f}"


@dataclass
class TableCell:
    text: str
    value: float | None = None
    suffix: str = ""
    best: bool = False
    pending: bool = False

    def render(self) -> str:
        content = f"\\textbf{{{self.text}}}" if self.best else self.text
        prefix = r"\cellcolor{ebCoral!35}" if self.best else ""
        return prefix + content + self.suffix


def mark_best(rows: list[tuple[str, list[TableCell]]], directions: list[str | None]) -> None:
    for column, direction in enumerate(directions):
        if direction is None or any(cells[column].pending for _, cells in rows):
            continue  # no best mark over a column with a pending cell: the ranking would cover a subset
        candidates = [cells[column] for _, cells in rows if cells[column].value is not None]
        if not candidates:
            continue
        values = [float(cell.value) for cell in candidates if cell.value is not None]
        best_value = max(values) if direction == "max" else min(values)
        best_texts = {cell.text for cell in candidates if float(cell.value) == best_value}
        if sum(cell.text in best_texts for cell in candidates) > 2:
            continue  # a tie shared by most rows carries no ranking information
        for cell in candidates:
            if cell.text in best_texts:
                cell.best = True


def table_header(cells: list[str]) -> str:
    rendered: list[str] = []
    for cell in cells:
        if "|" in cell:
            name, unit = cell.split("|", 1)
            rendered.append(f"\\ebh{{{name}}}{{{unit}}}")
        else:
            rendered.append(cell)
    return r"\rowcolor{ebBlue}" + " & ".join(rendered) + r" \\"


def render_big_table(
    *,
    name: str,
    caption: str,
    colspec: str,
    header: str,
    blocks: list[tuple[str, list[tuple[str, list[TableCell]]]]],
    source: str,
    tabcolsep: str = "3pt",
    note: str | None = None,
    size: str = r"\footnotesize",
) -> str:
    ncols = 1 + len(blocks[0][1][0][1])
    lines = [
        f"% Generated by scripts/paper_assets.py from {source}; do not edit by hand.",
        r"\begin{table*}[!t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{name[4:]}}}",
        size,
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
        r"\setlength{\aboverulesep}{0pt}\setlength{\belowrulesep}{0pt}\setlength{\extrarowheight}{0pt}",
        r"\renewcommand{\arraystretch}{0.94}",
        r"\def\ebtab{%",
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for title, rows in blocks:
        if title:
            lines.append(
                f"\\rowcolor{{ebAmber!45}}\\multicolumn{{{ncols}}}{{l}}"
                f"{{\\textbf{{\\textit{{{title}}}}}}} \\\\"
            )
        for model, cells in rows:
            prefix = r"\rowcolor{ebCoral!10}" if model == "Jev-1.13.0" else ""
            lines.append(prefix + " & ".join([TEX_NAME.get(model, model)] + [cell.render() for cell in cells]) + r" \\ ")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\def\ebh#1#2{#1 #2}\sbox0{\ebtab}",
            r"\ifdim\wd0>\dimexpr\linewidth-1pt\relax\def\ebh#1#2{\shortstack{#1\\{}#2}}\sbox0{\ebtab}\fi",
            f"\\setlength{{\\tabcolsep}}{{\\dimexpr\\tabcolsep+(\\linewidth-\\wd0-1pt)/{2 * ncols}\\relax}}",
            r"\ifdim\tabcolsep<1.5pt\setlength{\tabcolsep}{1.5pt}\fi",
            f"\\typeout{{EBTAB {name} natural=\\the\\wd0\\space line=\\the\\linewidth\\space sep=\\the\\tabcolsep}}",
            r"\ebtab",
        ]
    )
    if note is not None:
        lines.append(f"\\par\\smallskip{{{size} {note}\\par}}")
    lines.extend([r"\end{table*}", ""])
    return "\n".join(lines)


def tab_c1(data: Data, out: Out) -> None:
    qualities = [("fresh", "Fresh"), ("stale", "Stale"), ("noisy", "Noisy"),
                 ("contradictory", "Contradictory")]
    blocks: list[tuple[str, list[tuple[str, list[TableCell]]]]] = []
    for quality, title in qualities:
        rows: list[tuple[str, list[TableCell]]] = []
        for model in MODELS:
            curves = [data.one(data.curves, model=model, quality=quality, n_cells=count) for count in (3, 7, 21, 57)]
            cell57 = curves[-1]
            condition = f"c57_{quality}"
            quality_row = data.one(data.quality, model=model, condition=condition)
            feasibility = data.one(data.feasibility, model=model, condition=condition)
            if not np.isclose(float(cell57["full_policy_match"]), float(quality_row["full_policy_match"]),
                              rtol=0.0, atol=1e-12):
                raise ValueError(f"{model} {condition}: full_policy_match source mismatch")
            contrast = data.one(data.h3, model=model) if quality == "fresh" else data.one(
                data.exploratory, model=model, quality=quality
            )
            delta = float(contrast["delta_c3_minus_c57"])
            ci = (f"\\,{{\\color{{black!60}}[{tex_float(contrast['ci_low'])}, "
                  f"{tex_float(contrast['ci_high'])}]}}")
            cells = [
                *[TableCell(tex_float(row["target_cluster_state_dep_acc"]),
                            float(row["target_cluster_state_dep_acc"])) for row in curves],
                TableCell(tex_float(delta), delta, ci),
                TableCell(tex_float(cell57["full_policy_match"]), float(cell57["full_policy_match"])),
                TableCell(tex_float(quality_row["schema_valid_rate"]), float(quality_row["schema_valid_rate"])),
                TableCell(tex_float(quality_row["unsafe_policy_rate"]), float(quality_row["unsafe_policy_rate"])),
                TableCell(tex_float(feasibility["phi"]), float(feasibility["phi"])),
                TableCell(tex_float(quality_row["latency_p50"], 2), float(quality_row["latency_p50"])),
                TableCell(tex_float(quality_row["latency_p95"], 2), float(quality_row["latency_p95"])),
            ]
            rows.append((model, cells))
        mark_best(rows, ["max", "max", "max", "max", "min", "max", "max", "min", "max", "min", "min"])
        blocks.append((title, rows))
    caption = ("RQ4 telemetry grounding by telemetry quality. The fresh $\\Delta_{3\\to57}$ is the "
               "confirmatory H3 contrast from the Holm family of six with the JSON model outside the family, "
               "and the other contrasts are exploratory.")
    header = table_header([
        "Interpreter", "Target|3 $\\uparrow$", "Target|7 $\\uparrow$", "Target|21 $\\uparrow$",
        "Target|57 $\\uparrow$", "$\\Delta_{3\\to57}$|[95\\% CI] $\\downarrow$",
        "Full policy|57 $\\uparrow$", "Schema-valid|57 $\\uparrow$", "Unsafe|57 $\\downarrow$",
        "Near-RT $\\phi$|57 $\\uparrow$", "p50 57|(s) $\\downarrow$", "p95 57|(s) $\\downarrow$",
    ])
    text = render_big_table(
        name="tab_c1", caption=caption, colspec="l" + "r" * 11, header=header, blocks=blocks,
        source="cell_count_curves.csv + h3_confirmatory.csv + exploratory_contrasts.csv + quality_and_cost.csv + near_rt_feasibility.csv",
        tabcolsep="2pt", note=(r"Only the $\Delta_{3\to57}$ contrasts carry intervals, because H3 and the exploratory "
                                r"comparisons test these contrasts."),
    )
    out.write_table("tab_c1.tex", text)
    out.entry(Artifact(
        "tab_c1", "tables/tab_c1.tex",
        [_rel(C1 / name) for name in ("cell_count_curves.csv", "h3_confirmatory.csv",
                                    "exploratory_contrasts.csv", "quality_and_cost.csv", "near_rt_feasibility.csv")],
        ["target_cluster_state_dep_acc", "delta_c3_minus_c57", "ci_low", "ci_high", "full_policy_match",
         "schema_valid_rate", "unsafe_policy_rate", "phi", "latency_p50", "latency_p95"],
        "four telemetry-quality groups; all seven interpreters; c57 metric rows", "table* (28 rows)",
    ))


def tex_count(value: Any) -> str:
    if pd.isna(value):
        return "--"
    return f"{round(float(value)):,}".replace(",", "{,}")


def tab_c1_cost(data: Data, out: Out) -> None:
    """RQ4 cost and energy per decision at fresh telemetry, 3 and 57 cells (quality_and_cost.csv)."""
    rows: list[tuple[str, list[TableCell]]] = []
    for model in MODELS:
        cells: list[TableCell] = []
        for count in (3, 57):
            row = data.one(data.quality, model=model, condition=f"c{count}_fresh")
            usd, energy = float(row["cost_per_1000_correct_usd"]), float(row["energy_j_per_decision"])
            if model in HOSTED and not np.isnan(energy):
                raise ValueError(f"{model} c{count}_fresh: hosted interpreter with an energy value")
            if model in SELF_HOSTED and (usd != 0.0 or np.isnan(energy)):
                raise ValueError(f"{model} c{count}_fresh: self-hosted needs 0 USD and an energy value")
            cells += [
                TableCell(tex_count(row["mean_input_tokens"]), float(row["mean_input_tokens"])),
                TableCell(tex_count(row["mean_output_tokens"]), float(row["mean_output_tokens"])),
                TableCell("0" if usd == 0.0 else tex_float(usd), usd),
                TableCell(tex_float(energy, 1), None if np.isnan(energy) else energy),
            ]
        rows.append((model, cells))
    mark_best(rows, ["min"] * 8)
    caption = ("RQ4 cost and energy per decision with fresh telemetry at 3 and 57 cells. Tokens are means per "
               "decision, fees are US dollars per 1,000 correct policies over every call of the condition, and "
               "energy is integrated from the GPU power trace of each self-hosted interpreter. Hosted interpreters have "
               "no power trace, and self-hosted interpreters carry no API fee.")
    header = table_header([
        "Interpreter",
        *[cell for count in (3, 57) for cell in (
            f"Input tok.|{count} $\\downarrow$", f"Output tok.|{count} $\\downarrow$",
            f"USD/1k correct|{count} $\\downarrow$", f"J/decision|{count} $\\downarrow$")],
    ])
    text = render_big_table(
        name="tab_c1_cost", caption=caption, colspec="l" + "r" * 8, header=header, blocks=[("", rows)],
        source="quality_and_cost.csv",
    )
    out.write_table("tab_c1_cost.tex", text)
    out.entry(Artifact(
        "tab_c1_cost", "tables/tab_c1_cost.tex", [_rel(C1 / "quality_and_cost.csv")],
        ["mean_input_tokens", "mean_output_tokens", "cost_per_1000_correct_usd", "energy_j_per_decision"],
        "c3_fresh and c57_fresh; all seven interpreters", "table* (7 rows)",
    ))


def tab_rq3(data: Data, out: Out) -> None:
    rates = [0.1, 0.5, 1.0, 2.0]
    blocks: list[tuple[str, list[tuple[str, list[TableCell]]]]] = []
    for rate in rates:
        rows: list[tuple[str, list[TableCell]]] = []
        for model in MODELS:
            row = data.one(data.rq3, model=model, rate_per_s=rate)
            latency = data.one(data.rq3_latency, model=model, rate_per_s=rate)
            flagged = float(row["rho"]) > 1.0
            if flagged != bool(row["non_stationary"]):
                raise ValueError(f"{model} rate={rate}: non_stationary does not equal eta>1")
            dagger = r"$^{\dagger}$" if flagged else ""
            cells = [
                TableCell(str(int(row["n"])), float(row["n"])),
                TableCell(tex_float(row["share_enforced_within_1s"]), float(row["share_enforced_within_1s"])),
                TableCell(tex_float(row["rho"]), float(row["rho"]), dagger),
                TableCell(tex_float(row["queue_wait_p50_s"]), float(row["queue_wait_p50_s"])),
                TableCell(tex_float(row["queue_wait_p95_s"]), float(row["queue_wait_p95_s"])),
                TableCell(tex_float(row["queue_wait_p99_s"]), float(row["queue_wait_p99_s"])),
                TableCell(tex_float(latency["ell_median_s"]), float(latency["ell_median_s"])),
                TableCell(tex_float(latency["ell_p95_s"]), float(latency["ell_p95_s"])),
            ]
            rows.append((model, cells))
        mark_best(rows, [None, "max", "min", "min", "min", "min", "min", "min"])
        blocks.append((f"{rate:g} intents/s", rows))
    header = table_header([
        "Interpreter", "$n$", "Within 1 s|$\\uparrow$", "$\\eta$|$\\downarrow$",
        "Wait p50|(s) $\\downarrow$", "Wait p95|(s) $\\downarrow$", "Wait p99|(s) $\\downarrow$",
        "$\\ell$ p50|(s) $\\downarrow$", "$\\ell$ p95|(s) $\\downarrow$",
    ])
    text = render_big_table(
        name="tab_rq3", caption="RQ3 load metrics for every interpreter at each offered intent rate.",
        colspec="l" + "r" * 8, header=header, blocks=blocks,
        source="rq3_load.csv + rq3_trace_latency_summary.csv", tabcolsep="4pt",
        note=r"$^{\dagger}$Non-stationary cell with $\eta>1$.",
    )
    out.write_table("tab_rq3.tex", text)
    out.entry(Artifact(
        "tab_rq3", "tables/tab_rq3.tex",
        [f"{ANALYSIS_REL}/c1-interpretation/results/rq3-load/rq3_load.csv", f"{ANALYSIS_REL}/c1-interpretation/results/rq3-load/rq3_trace_latency_summary.csv"],
        ["n", "share_enforced_within_1s", "rho", "non_stationary", "queue_wait_p50_s",
         "queue_wait_p95_s", "queue_wait_p99_s", "ell_median_s", "ell_p95_s", "latency_field"],
        "four offered-rate groups; all seven interpreters; latency_field=latency_s", "table* (28 rows)",
    ))


def tab_c3(data: Data, out: Out) -> None:
    rows: list[tuple[str, list[TableCell]]] = []
    for model in MODELS:
        row = data.one(data.c3, model=model)
        placement = str(row["deployment"]).replace("self-hosted", "self-hosted")
        cells = [
            TableCell(placement),
            TableCell(tex_float(row["ell_median_s"]), float(row["ell_median_s"])),
            TableCell(tex_float(row["ell_p95_s"]), float(row["ell_p95_s"])),
            TableCell(tex_float(row["delta_a1_median_ms"], 1), float(row["delta_a1_median_ms"])),
            TableCell(tex_float(row["delta_a1_p95_ms"], 1), float(row["delta_a1_p95_ms"])),
            TableCell(tex_float(row["delta_e2_median_ms"], 1), float(row["delta_e2_median_ms"])),
            TableCell(tex_float(row["delta_e2_p95_ms"], 1), float(row["delta_e2_p95_ms"])),
            TableCell(tex_float(row["kpm_upper_bound_median_s"]), float(row["kpm_upper_bound_median_s"])),
            TableCell(tex_float(row["kpm_upper_bound_p95_s"]), float(row["kpm_upper_bound_p95_s"])),
            TableCell(str(int(row["n"])), float(row["n"])),
        ]
        rows.append((model, cells))
    mark_best(rows, [None, "min", "min", "min", "min", "min", "min", "min", "min", None])
    header = table_header([
        "Interpreter", "Placement", "$\\ell$ p50|(s) $\\downarrow$", "$\\ell$ p95|(s) $\\downarrow$",
        "$\\delta_{\\mathrm{A1}}$ p50|(ms) $\\downarrow$", "$\\delta_{\\mathrm{A1}}$ p95|(ms) $\\downarrow$",
        "$\\delta_{\\mathrm{E2}}$ p50|(ms) $\\downarrow$", "$\\delta_{\\mathrm{E2}}$ p95|(ms) $\\downarrow$",
        "KPM upper p50|(s) $\\downarrow$", "KPM upper p95|(s) $\\downarrow$", "$n$",
    ])
    text = render_big_table(
        name="tab_c3", caption="RQ6 real-stack path metrics for every interpreter and placement.",
        colspec="ll" + "r" * 9, header=header, blocks=[("", rows)], source="c3_path_decomposition.csv",
        tabcolsep="3pt",
    )
    out.write_table("tab_c3.tex", text)
    out.entry(Artifact(
        "tab_c3", "tables/tab_c3.tex", [f"{ANALYSIS_REL}/c3-real-stack/results/c3_path_decomposition.csv"],
        ["model", "deployment", "ell_median_s", "ell_p95_s", "delta_a1_median_ms", "delta_a1_p95_ms",
         "delta_e2_median_ms", "delta_e2_p95_ms", "kpm_upper_bound_median_s", "kpm_upper_bound_p95_s", "n"],
        "all seven C3 main runs", "table* (7 rows)",
    ))


# ---------------------------------------------------------------- C2 closed-loop radio simulation
#
# Inputs are the files that ``src/ranbench/c2/analysis.py`` writes into one ``--out`` directory.
# Every SLA share is the per-intent violation share at W = 2 s; plotted and tabulated in percent, gaps in pp.
# Intervals are the primary-B (D-10) 95% block-bootstrap intervals; a contrast is marked resolved only from
# the analysis field ``*_resolved_both`` (Holm, resolved under both B and the 10 s blocking).

C2_BASE = "dp_r0p3_s30_u5"
C2_RATES = (0.1, 0.3, 1.0)
C2_SPEEDS = (3.0, 60.0, 120.0)
C2_MODES = ("E", "P-1", "P-10")
H1_ANCHOR = {
    "DeepSeek-V4.1-Flash": "Jev-1.13.0",
    "GLM-5.3-Flash": "Jev-1.13.0",
    "Qwen3.8-Flash": "Jev-1.13.0",
    "Qwen3.5-4B-JSON": "SemIf-Qwen3.5-4B",
}
C2_CHALLENGERS = tuple(H1_ANCHOR)
H2_SHARE = 0.80
C2_FILES = ("controls.csv", "l_arm_contrasts.csv", "an_decomposition.csv", "rq1_modes.csv",
            "radio_kpis.csv", "rq5.csv", "rq3_radio.csv", "runs.csv", "hypotheses.json", "skipped.json")
# D-17: platform of each C2 build, keyed by the first 8 hex digits of binary_sha256. Patched builds on one
# platform reproduce its unpatched output until a guard fires, so they count as the same binary.
C2_PLATFORM = {
    "7e2268c1": "cluster-A", "025d3dc4": "cluster-A", "74d32e46": "cluster-A",   # production, D-7, D-7b
    "bd1738b0": "cluster-B", "a4b41532": "cluster-B",                   # production, D-7
    "70abfed6": "M4", "32976335": "M4", "4d46bc2e": "M4",         # production, D-11 (d7b), RQ5 build
    "a59ba72a": "L40",                                            # production (rate-1 design points)
}


C2_RQ3_CELLS = ((0.1, 5), (0.5, 5), (1.0, 5), (2.0, 5), (1.0, 20))  # (intents/s, UEs per cell); last = scale point
C2_JAIN_CLASSES = (("video", "video"), ("xr", "XR"), ("iot", "IoT"), ("be", "best effort"))


def c2_point(rate: float, speed: float) -> str:
    """Design-point id as ``src.ranbench.c2.matrix._row`` builds it."""
    return f"dp_r{rate:g}_s{speed:g}_u5".replace(".", "p")


C2_POINTS = (C2_BASE, *(c2_point(rate, speed) for rate in C2_RATES for speed in C2_SPEEDS))


def c2_rq3_point(model: str, rate: float, ues_per_cell: int) -> str:
    """RQ3 replay design-point id as ``src.ranbench.c2.matrix._row`` builds it."""
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    return f"dp_rq3_{slug}_r{rate:g}_u{ues_per_cell}".replace(".", "p")


def is_true(value: Any) -> bool | None:
    """Tri-state read of a CSV boolean: None when the analysis left the field empty."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value).strip().lower() == "true"


def c2_h2_summary(hypotheses: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Flatten the H2 verdicts of both eligibility rules from hypotheses.json into one CSV."""
    rows: list[dict[str, Any]] = []
    for rule in ("primary_d10", "sensitivity_original_5pp"):
        result = hypotheses[rule]["H2"]["result"]  # None: the family waits for absent runs (partial analysis)
        testable = result is None or result["n_eligible"] >= H2_MIN_ELIGIBLE
        for challenger, anchor in H1_ANCHOR.items():
            verdict = None if result is None else result["holds"].get(f"{challenger} vs {anchor}")
            rows.append({
                "rule": rule,
                "challenger": challenger,
                "anchor": anchor,
                "n_eligible": None if result is None else result["n_eligible"],
                "n_points": None if verdict is None else verdict["n_points"],
                "share_resolved": None if verdict is None else verdict["share_resolved"],
                "n_reversed_resolved": None if verdict is None else verdict["n_reversed_resolved"],
                "holds": (None if verdict is None else verdict["holds"]) if testable else "not testable",
                "source_key": f"{rule}.H2.result.{'holds' if testable else 'n_eligible'}",
            })
    frame = pd.DataFrame(rows)
    frame.insert(0, "_csv_row", np.arange(1, len(frame) + 1))
    return frame, frame_csv(frame)


PENDING = r"\pending{pending}"
PENDING_MARK = r"\pending{$^{?}$}"  # resolution mark of a contrast whose Holm family is pending
# skipped.json analysis unit -> (pending kind, key fields after design_point); src/ranbench/c2/analysis.py
C2_PENDING_KINDS = {
    "controls": ("controls", ()),
    "L-arm contrast": ("l", ("interpreter",)),
    "A/N decomposition": ("an", ("interpreter",)),
    "RQ1 modes": ("modes", ("interpreter",)),
    "radio KPIs": ("radio", ("interpreter", "mode", "arm")),
    "stale-policy exposure": ("stale", ("interpreter", "mode", "arm")),
    "H1 Holm family": ("holm", ("rule",)),
    "H2 Holm family": ("holm", ("rule",)),
    "RQ5 division of labour": ("rq5", ("interpreter",)),
    "RQ3 radio replay": ("rq3radio", ("interpreter",)),
}


def c2_pending_units(skipped: dict[str, Any]) -> dict[tuple[str, ...], dict[str, Any]]:
    """Pending units of a partial analysis, keyed (kind, design_point, ...); H1/H2 key on (holm, family, rule)."""
    units: dict[tuple[str, ...], dict[str, Any]] = {}
    for unit in skipped.get("analysis_units", []):
        if unit["analysis"] not in C2_PENDING_KINDS:
            raise ValueError(f"skipped.json: unknown analysis unit {unit['analysis']!r}")
        kind, fields = C2_PENDING_KINDS[unit["analysis"]]
        head = unit["analysis"][:2] if kind == "holm" else unit["design_point"]
        key = (kind, head, *(str(unit[field]) for field in fields))
        runs = unit["required_runs_unavailable"]
        if not runs:
            raise ValueError(f"skipped.json: pending unit {key} names no absent run")
        entry = units.setdefault(key, {"analysis": unit["analysis"], "runs": {}})
        entry["runs"].update({run["run_id"]: run["status"] for run in runs})
    return units


class C2Data:
    def __init__(self, directory: Path, *, allow_partial: bool) -> None:
        missing_files = [name for name in C2_FILES if not (directory / name).is_file()]
        if missing_files:
            raise FileNotFoundError(f"C2 analysis output {directory} lacks {', '.join(missing_files)}")
        self.dir = directory
        self.controls = with_csv_rows(directory / "controls.csv")
        self.l_arm = with_csv_rows(directory / "l_arm_contrasts.csv")
        self.an = with_csv_rows(directory / "an_decomposition.csv")
        self.modes = with_csv_rows(directory / "rq1_modes.csv")
        self.radio = with_csv_rows(directory / "radio_kpis.csv")
        self.rq5 = with_csv_rows(directory / "rq5.csv")
        self.rq3_radio = with_csv_rows(directory / "rq3_radio.csv")
        self.runs = with_csv_rows(directory / "runs.csv")
        unknown = sorted({str(sha) for sha in self.runs["binary_sha256"] if str(sha)[:8] not in C2_PLATFORM})
        if unknown:
            raise ValueError(f"C2 runs.csv: binary_sha256 without a platform (D-17): {unknown}")
        self.platform = {str(run): C2_PLATFORM[str(sha)[:8]]
                         for run, sha in zip(self.runs["run_id"], self.runs["binary_sha256"])}
        self.hypotheses = json.loads((directory / "hypotheses.json").read_text(encoding="utf-8"))
        self.skipped = json.loads((directory / "skipped.json").read_text(encoding="utf-8"))
        self.h2, self.h2_csv = c2_h2_summary(self.hypotheses)
        for name, frame in (("l_arm_contrasts", self.l_arm), ("an_decomposition", self.an),
                            ("rq1_modes", self.modes), ("rq5", self.rq5), ("rq3_radio", self.rq3_radio)):
            extra = set(frame["interpreter"]) - set(MODELS)
            if extra:
                raise ValueError(f"C2 {name}: interpreters outside the paper roster: {sorted(extra)}")
        for _, row in self.l_arm.iterrows():
            expected = H1_ANCHOR.get(str(row["interpreter"]))
            found = None if pd.isna(row["comparator"]) else str(row["comparator"])
            if found != expected:
                raise ValueError(f"C2 {row['design_point']} {row['interpreter']}: comparator {found}, expected {expected}")
        self.allow_partial = allow_partial
        self.pending_units = c2_pending_units(self.skipped) if allow_partial else {}
        present = [" ".join(key) for key in self.pending_units
                   if (key[0] == "rq5" and self.find(self.rq5, design_point=key[1], interpreter=key[2]) is not None)
                   or (key[0] == "rq3radio" and self.find(self.rq3_radio, design_point=key[1], interpreter=key[2])
                       is not None)]
        if present:
            raise SystemExit("skipped.json names pending unit(s) whose rows are present:\n  " + "\n  ".join(present))
        missing = self.coverage()
        unexplained = [label for label, key in missing if allow_partial and key not in self.pending_units]
        if unexplained:
            raise SystemExit("C2 analysis lacks unit(s) that skipped.json does not name pending:\n  "
                             + "\n  ".join(unexplained))
        self.missing = [label for label, _ in missing]
        if self.missing:
            print(f"C2 COVERAGE INCOMPLETE: {len(self.missing)} unit(s) absent from {directory}", file=sys.stderr)
            for unit in self.missing:
                print(f"  missing: {unit}", file=sys.stderr)
            skipped = len(self.skipped.get("analysis_units", []))
            unavailable = len(self.skipped.get("unavailable_runs", []))
            print(f"  skipped.json: {skipped} analysis unit(s), {unavailable} unavailable run(s)", file=sys.stderr)
            if not allow_partial:
                raise SystemExit("refusing to render C2 assets from a partial analysis (use --c2-allow-partial)")

    def coverage(self) -> list[tuple[str, tuple[str, ...]]]:
        """Absent units as (label, pending key)."""
        missing: list[tuple[str, tuple[str, ...]]] = []
        for point in C2_POINTS:
            if self.find(self.controls, design_point=point) is None:
                missing.append((f"controls.csv {point}", ("controls", point)))
            for model in MODELS:
                if self.find(self.l_arm, design_point=point, interpreter=model) is None:
                    missing.append((f"l_arm_contrasts.csv {point} {model}", ("l", point, model)))
                if self.find(self.an, design_point=point, interpreter=model) is None:
                    missing.append((f"an_decomposition.csv {point} {model}", ("an", point, model)))
                if self.radio_row(point, model, "E") is None:
                    missing.append((f"radio_kpis.csv {point} {model} E/L", ("radio", point, model, "E", "L")))
        for model in MODELS:
            for mode in C2_MODES:
                if self.find(self.modes, design_point=C2_BASE, interpreter=model, mode=mode) is None:
                    missing.append((f"rq1_modes.csv {model} {mode}", ("modes", C2_BASE, model)))
                if self.radio_row(C2_BASE, model, mode) is None:
                    missing.append((f"radio_kpis.csv {C2_BASE} {model} {mode}/L",
                                    ("radio", C2_BASE, model, mode, "L")))
            if self.find(self.rq5, design_point=C2_BASE, interpreter=model) is None:
                missing.append((f"rq5.csv {model}", ("rq5", C2_BASE, model)))
            for rate, ues in C2_RQ3_CELLS:
                if self.rq3_row(model, rate, ues) is None:
                    point = c2_rq3_point(model, rate, ues)
                    missing.append((f"rq3_radio.csv {point}", ("rq3radio", point, model)))
        return missing

    def pending(self, *key: str) -> bool:
        """True when the unit waits for an absent run (only a --c2-allow-partial render has pending units)."""
        return key in self.pending_units

    def holm_pending(self, point: str, model: str, metric: str) -> bool:
        """The Holm family of this L-arm gap (primary D-10 rule) waits for an absent run."""
        if model not in H1_ANCHOR:
            return False
        if point == C2_BASE:
            return self.pending("holm", "H1", "primary_d10")
        return metric == "affected" and self.eligible(point) is True and self.pending("holm", "H2", "primary_d10")

    def pending_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["unit", "reason", "missing_run_ids"])
        for key, entry in self.pending_units.items():
            statuses = ", ".join(sorted(set(entry["runs"].values())))
            writer.writerow([" ".join(key), f"{entry['analysis']}: waits for absent run(s) ({statuses})",
                             ";".join(entry["runs"])])
        return buffer.getvalue()

    @staticmethod
    def find(frame: pd.DataFrame, **filters: Any) -> pd.Series | None:
        selected = frame
        for column, value in filters.items():
            selected = selected[selected[column] == value]
        if len(selected) > 1:
            raise ValueError(f"expected at most one row for {filters}, found {len(selected)}")
        return None if selected.empty else selected.iloc[0]

    def radio_row(self, point: str, model: str, mode: str) -> pd.Series | None:
        return self.find(self.radio, design_point=point, interpreter=model, mode=mode, arm="L")

    def rq3_row(self, model: str, rate: float, ues_per_cell: int) -> pd.Series | None:
        return self.find(self.rq3_radio, design_point=c2_rq3_point(model, rate, ues_per_cell), interpreter=model)

    def run_platform(self, point: str, interpreter: str, arm: str, mode: str = "E") -> str:
        row = self.find(self.radio, design_point=point, interpreter=interpreter, mode=mode, arm=arm)
        if row is None or str(row["run_id"]) not in self.platform:
            raise ValueError(f"C2 {point} {interpreter} {mode}/{arm}: no run with a binary in runs.csv")
        return self.platform[str(row["run_id"])]

    def has_reference(self, point: str) -> bool:
        return any(self.find(self.radio, design_point=point, interpreter="control", mode="E", arm=arm) is not None
                   for arm in ("oracle", "no-update"))

    def reference_platform(self, point: str) -> str:
        """D-17: the platform of the point's oracle; with the oracle absent (partial render), of its C-4 pairing
        partner, the no-update control."""
        for arm in ("oracle", "no-update"):
            if self.find(self.radio, design_point=point, interpreter="control", mode="E", arm=arm) is not None:
                return self.run_platform(point, "control", arm)
        raise ValueError(f"C2 {point}: no oracle or no-update run fixes the reference binary")

    def control(self, point: str) -> pd.Series | None:
        return self.find(self.controls, design_point=point)

    def eligible(self, point: str) -> bool | None:
        row = self.control(point)
        return None if row is None else is_true(row["eligible_d10"])

    def h2_row(self, challenger: str, rule: str = "primary_d10") -> pd.Series:
        return Data.one(self.h2, rule=rule, challenger=challenger)

    def h2_not_testable(self, rule: str = "primary_d10") -> str | None:
        """Note sentence when too few design points are eligible for H2; None when H2 is testable or pending."""
        family = self.hypotheses[rule]["H2"]
        n_eligible = None if family["result"] is None else int(family["result"]["n_eligible"])
        if n_eligible is None or n_eligible >= H2_MIN_ELIGIBLE:
            return None
        return (f"H2 is not testable, because only {n_eligible} of the {family['n_prespecified_points']} design "
                f"points {'is' if n_eligible == 1 else 'are'} eligible and it needs {H2_MIN_ELIGIBLE}.")


def c2_value(row: pd.Series | None, column: str, scale: float = 100.0) -> float:
    if row is None or pd.isna(row[column]):
        return math.nan
    return scale * float(row[column])


def c2_interval(row: pd.Series | None, prefix: str, scale: float = 100.0) -> tuple[float, float]:
    return c2_value(row, f"{prefix}_ci_low", scale), c2_value(row, f"{prefix}_ci_high", scale)


def tex_signed(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:+.{digits}f}".replace("-", "$-$")


def tex_interval(low: float, high: float, digits: int = 2) -> str:
    if not (math.isfinite(low) and math.isfinite(high)):
        return ""
    text = f"[{low:.{digits}f}, {high:.{digits}f}]".replace("-", "$-$")
    return f"\\,{{\\color{{black!60}}{text}}}"


def resolved_marks(ax: Any, x: float, value: float, low: float, high: float, flag: bool | None) -> None:
    """Asterisk beyond the interval end of a contrast that the analysis marks resolved under both blockings."""
    if not flag or not math.isfinite(value):
        return
    upward = value >= 0
    end = high if upward and math.isfinite(high) else low if math.isfinite(low) else value
    ax.annotate("*", xy=(x, end), xytext=(0, 0.5 if upward else -4.5), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, zorder=7)


def model_bars(
    ax: Any, models: tuple[str, ...], values: list[float], intervals: list[tuple[float, float]] | None,
    flags: list[bool | None] | None = None,
) -> None:
    positions = np.arange(len(models), dtype=float)
    for index, model in enumerate(models):
        value = values[index]
        if not math.isfinite(value):
            continue
        ax.bar(
            positions[index], value, width=0.66, color=hexcol(FILL[model]), edgecolor="black",
            linewidth=0.8 if model == "Jev-1.13.0" else 0.5,
            hatch="////" if model in SELF_HOSTED else None, zorder=4 if model == "Jev-1.13.0" else 3,
        )
        low, high = intervals[index] if intervals is not None else (math.nan, math.nan)
        if math.isfinite(low) and math.isfinite(high):
            ax.errorbar(
                positions[index], value, yerr=[[max(0.0, value - low)], [max(0.0, high - value)]], fmt="none",
                ecolor="black", elinewidth=0.45, capsize=0.8, capthick=0.45, zorder=6,
            )
        if flags is not None:
            resolved_marks(ax, positions[index], value, low, high, flags[index])
    ax.set_xticks(positions, [PLOT_CODE[model] for model in models], rotation=45, ha="right")
    ax.set_xlim(-0.6, len(models) - 0.4)


def bar_legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=hexcol(FILL[model]), edgecolor="black", linewidth=0.5,
              hatch="////" if model in SELF_HOSTED else None)
        for model in MODELS
    ]


def c2_source(c2: C2Data, name: str) -> str:
    path = (c2.dir / name).resolve()
    return os.path.relpath(path, PAPER) if ROOT in path.parents else str(path)


def fig_c2_h1(c2: C2Data, out: Out) -> None:
    def mode_row(model: str, index: int) -> pd.Series | None:
        return c2.find(c2.modes, design_point=C2_BASE, interpreter=model, mode=C2_MODES[index])

    for letter, metric, ylabel in (("a", "affected", "Affected viol. (%)"),
                                   ("b", "network", "Network viol. (%)")):
        fig, ax = new_panel(FIGSTAR_W)
        grouped_bars(
            ax, list(C2_MODES), lambda model, index: c2_value(mode_row(model, index), f"{metric}_point"),
            lambda model, index: c2_interval(mode_row(model, index), metric),
        )
        ax.set_xticks(np.arange(len(C2_MODES)), list(C2_MODES))
        ax.set_xlabel("Loop mode")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0.0)
        out.save_fig(fig, f"fig_c2_h1_{letter}.pdf")

    metrics = ("affected_gap", "network_gap")
    base_row = lambda model: c2.find(c2.l_arm, design_point=C2_BASE, interpreter=model)
    fig, ax = new_panel(FIGSTAR_W)
    positions = grouped_bars(
        ax, ["Affected", "Network"],
        lambda model, index: c2_value(base_row(model), f"{metrics[index]}_point"),
        lambda model, index: c2_interval(base_row(model), metrics[index]), models=C2_CHALLENGERS,
    )
    for (model, index), x in positions.items():
        row = base_row(model)
        value = c2_value(row, f"{metrics[index]}_point")
        resolved_marks(ax, x, value, *c2_interval(row, metrics[index]),
                       is_true(row[f"{metrics[index]}_resolved_both"]))
    ax.axhline(0.0, color="black", linewidth=0.6, zorder=2)
    ax.set_xticks([0, 1], ["Affected", "Network"])
    ax.set_xlabel("H1 contrast (mode E)")
    ax.set_ylabel("Gap to baseline (pp)")
    ax.margins(y=0.12)
    out.save_fig(fig, "fig_c2_h1_c.pdf")

    fig, ax = new_panel(FIGSTAR_W)
    grouped_bars(
        ax, list(C2_MODES),
        lambda model, index: c2_value(c2.radio_row(C2_BASE, model, C2_MODES[index]), "throughput_p5_mbps", 1.0),
    )
    ax.set_xticks(np.arange(len(C2_MODES)), list(C2_MODES))
    ax.set_xlabel("Loop mode")
    ax.set_ylabel("Edge p5 thr. (Mb/s)")
    ax.set_ylim(bottom=0.0)
    out.save_fig(fig, "fig_c2_h1_d.pdf")

    legend_strip(out, "fig_c2_h1_legend.pdf", bar_legend_handles(), [SHORT[model] for model in MODELS], 7.00, 7)
    figure_wrapper(
        out, name="fig_c2_h1", star=True,
        panels=[("a", "Affected-class SLA violation"), ("b", "Network-wide SLA violation"),
                ("c", "H1 latency gap to baseline"), ("d", "Cell-edge throughput")],
        caption=("(a) Grouped bars encode the latency-only affected-class SLA violation at $W=2$~s per loop mode at "
                 "0.3 intents/s and 30 km/h with primary-block 95\\% intervals. (b) Grouped bars encode the "
                 "network-wide SLA violation in the same layout. (c) Bars encode each H1 contrast, challenger minus "
                 "the baseline of its deployment class (\\jev{} or \\semif{}) in mode E, with 95\\% intervals and an asterisk where the Holm-adjusted contrast is "
                 "resolved under both block lengths. (d) Grouped bars encode the 5th-percentile UE throughput of "
                 "each latency-only run per loop mode."),
    )
    modes_src, l_src, radio_src = (c2_source(c2, name) for name in
                                   ("rq1_modes.csv", "l_arm_contrasts.csv", "radio_kpis.csv"))
    add_figure_entries(
        out, name="fig_c2_h1", star=True,
        panel_specs=[
            ("a", "grouped bar with primary-B 95% intervals", [modes_src],
             ["interpreter", "mode", "affected_point", "affected_ci_low", "affected_ci_high"],
             f"design_point={C2_BASE}; L arms; modes E, P-1, P-10"),
            ("b", "grouped bar with primary-B 95% intervals", [modes_src],
             ["interpreter", "mode", "network_point", "network_ci_low", "network_ci_high"],
             f"design_point={C2_BASE}; L arms; modes E, P-1, P-10"),
            ("c", "grouped bar with primary-B 95% intervals and resolved marks", [l_src],
             ["interpreter", "comparator", "affected_gap_point", "affected_gap_ci_low", "affected_gap_ci_high",
              "affected_gap_resolved_both", "network_gap_point", "network_gap_ci_low", "network_gap_ci_high",
              "network_gap_resolved_both"], f"design_point={C2_BASE}; four H1 challengers"),
            ("d", "grouped bar", [radio_src], ["interpreter", "mode", "arm", "throughput_p5_mbps"],
             f"design_point={C2_BASE}; arm=L; modes E, P-1, P-10"),
        ],
    )


def fig_c2_grid(c2: C2Data, out: Out) -> None:
    xs = np.arange(len(C2_SPEEDS), dtype=float)
    offsets = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2) * 0.06
    series: dict[tuple[float, str], tuple[list[float], list[float], list[float]]] = {}
    for rate in C2_RATES:
        for model in MODELS:
            rows = [c2.find(c2.l_arm, design_point=c2_point(rate, speed), interpreter=model) for speed in C2_SPEEDS]
            series[(rate, model)] = (
                [c2_value(row, "affected_point") for row in rows],
                [c2_value(row, "affected_ci_low") for row in rows],
                [c2_value(row, "affected_ci_high") for row in rows],
            )
    finite = [value for low_high in series.values() for value in low_high[1] + low_high[2] if math.isfinite(value)]
    span = (max(finite) - min(finite)) if finite else 1.0
    y_limits = (max(0.0, min(finite) - 0.06 * span), max(finite) + 0.06 * span) if finite else (0.0, 1.0)
    for letter, rate in zip("abc", C2_RATES):
        fig, ax = new_panel(FIGSTAR_W)
        hollow = [index for index, speed in enumerate(C2_SPEEDS) if c2.eligible(c2_point(rate, speed)) is False]
        for model_index, model in enumerate(MODELS):
            values, lows, highs = (np.asarray(part, dtype=float) for part in series[(rate, model)])
            x = xs + offsets[model_index]
            ax.errorbar(x, values, yerr=[np.maximum(0.0, values - lows), np.maximum(0.0, highs - values)],
                        fmt="none", ecolor=dark(FILL[model]), elinewidth=0.5, capsize=0.8, capthick=0.5, zorder=2)
            model_line(ax, model, x, values, hollow_indices=hollow)
        ax.set_xticks(xs, [f"{speed:g}" for speed in C2_SPEEDS])
        ax.set_xlim(-0.35, len(C2_SPEEDS) - 0.65)
        ax.set_ylim(*y_limits)
        ax.set_xlabel("UE speed (km/h)")
        ax.set_ylabel("Affected viol. (%)")
        out.save_fig(fig, f"fig_c2_grid_{letter}.pdf")

    fig, ax = new_panel(FIGSTAR_W)
    rows = [c2.h2_row(model) for model in C2_CHALLENGERS]
    model_bars(ax, C2_CHALLENGERS, [c2_value(row, "share_resolved", 1.0) for row in rows], None)
    for index, row in enumerate(rows):
        reversed_count = 0 if pd.isna(row["n_reversed_resolved"]) else int(row["n_reversed_resolved"])
        if reversed_count:
            share = c2_value(row, "share_resolved", 1.0)
            ax.annotate(f"{reversed_count} rev.", xy=(index, 0.0 if not math.isfinite(share) else share),
                        xytext=(0, 1), textcoords="offset points", ha="center", va="bottom", fontsize=6)
    ax.axhline(H2_SHARE, color="black", linestyle=":", linewidth=0.8, zorder=2)
    n_eligible = "pending" if c2.pending("holm", "H2", "primary_d10") else int(rows[0]["n_eligible"])
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Share resolved")
    h2_note = c2.h2_not_testable()
    ax.set_xlabel(f"Eligible: {n_eligible} (H2 not testable)" if h2_note else f"Eligible points: {n_eligible}")
    out.save_fig(fig, "fig_c2_grid_d.pdf")

    legend_strip(out, "fig_c2_grid_legend.pdf", legend_handles(), [SHORT[model] for model in MODELS], 7.00, 7)
    figure_wrapper(
        out, name="fig_c2_grid", star=True,
        panels=[("a", "0.1 intents/s"), ("b", "0.3 intents/s"), ("c", "1 intent/s"),
                ("d", "H2 share of resolved points")],
        caption=("(a) The latency-only affected-class SLA violation at $W=2$~s in mode E is plotted against UE "
                 "speed at 0.1 intents/s with primary-block 95\\% intervals and hollow markers at design points "
                 "outside the control floor. (b) The same quantity is plotted at 0.3 intents/s. (c) The same "
                 "quantity is plotted at 1 intent/s. (d) Bars encode, per H2 challenger, the share of eligible "
                 "design points whose Holm-adjusted gap is resolved under both block lengths, with a horizontal "
                 "0.8 reference and a count of reversed-and-resolved points where one occurs.")
        + (f" {h2_note}" if h2_note else ""),
    )
    l_src, controls_src = c2_source(c2, "l_arm_contrasts.csv"), c2_source(c2, "controls.csv")
    h2_src = f"{ANALYSIS_REL}/c2-closed-loop/results/c2_h2_summary.csv"
    line_columns = ["interpreter", "rate_per_s", "speed_kmh", "affected_point", "affected_ci_low", "affected_ci_high"]
    add_figure_entries(
        out, name="fig_c2_grid", star=True,
        panel_specs=[
            *[(letter, "line with primary-B 95% error bars; hollow = ineligible point", [l_src, controls_src],
               line_columns + ["eligible_d10"], f"rate_per_s={rate:g}; RQ2 grid; mode E; L arms")
              for letter, rate in zip("abc", C2_RATES)],
            ("d", "bar with 0.8 reference", [h2_src, c2_source(c2, "hypotheses.json")],
             ["challenger", "anchor", "share_resolved", "n_reversed_resolved", "n_points", "n_eligible"],
             "rule=primary_d10"),
        ],
    )


def fig_c2_an(c2: C2Data, out: Out) -> None:
    for letter, prefix, ylabel in (("a", "n_minus_l", r"N $-$ L (pp)"), ("b", "n_minus_a", r"N $-$ A (pp)")):
        rows = [c2.find(c2.an, design_point=C2_BASE, interpreter=model) for model in MODELS]
        fig, ax = new_panel(FIG_W)
        model_bars(
            ax, MODELS, [c2_value(row, f"{prefix}_point") for row in rows],
            [c2_interval(row, prefix) for row in rows],
            [None if row is None else is_true(row[f"{prefix}_resolved_both"]) for row in rows],
        )
        ax.axhline(0.0, color="black", linewidth=0.6, zorder=2)
        ax.set_ylabel(ylabel)
        ax.margins(y=0.12)
        out.save_fig(fig, f"fig_c2_an_{letter}.pdf")
    legend_strip(out, "fig_c2_an_legend.pdf", bar_legend_handles(), [SHORT[model] for model in MODELS], 3.36, 4)
    figure_wrapper(
        out, name="fig_c2_an", star=False,
        panels=[("a", "Accuracy cost at own latency"), ("b", "Latency cost at own accuracy")],
        caption=("(a) Bars encode the affected-class difference between the net and latency-only arms at 0.3 "
                 "intents/s and 30 km/h with primary-block 95\\% intervals and an asterisk where it is resolved "
                 "under both block lengths. (b) Bars encode the difference between the net and accuracy-only arms "
                 "in the same layout."),
    )
    source = c2_source(c2, "an_decomposition.csv")
    add_figure_entries(
        out, name="fig_c2_an", star=False,
        panel_specs=[
            (letter, "bar with primary-B 95% intervals and resolved marks", [source],
             ["interpreter", f"{prefix}_point", f"{prefix}_ci_low", f"{prefix}_ci_high", f"{prefix}_resolved_both"],
             f"design_point={C2_BASE}; outcome=affected_class_W2")
            for letter, prefix in (("a", "n_minus_l"), ("b", "n_minus_a"))
        ],
    )


def c2_blocks_text(row: pd.Series | None) -> str:
    if row is None:
        return "n/a"
    return f"{int(row['n_blocks'])}/{int(row['n_blocks_10s'])}"


def pending_cell() -> TableCell:
    return TableCell(PENDING, pending=True)


def c2_gap_cell(row: pd.Series | None, prefix: str, model: str, pending: bool = False,
                holm_pending: bool = False) -> TableCell:
    """pending: the L-arm row waits for an absent run; holm_pending: its Holm family does."""
    if model not in H1_ANCHOR:
        if pending and model in H1_ANCHOR.values():
            return pending_cell()
        return TableCell("--")  # H1 baselines and unpaired interpreters carry no gap
    if pending:
        return pending_cell()
    value = c2_value(row, f"{prefix}_point")
    mark = r"$^{\ast}$" if row is not None and is_true(row[f"{prefix}_resolved_both"]) else ""
    if holm_pending:
        mark = PENDING_MARK
    return TableCell(tex_signed(value), None, tex_interval(*c2_interval(row, prefix)) + mark)


def c2_level_cell(row: pd.Series | None, prefix: str, with_interval: bool, pending: bool = False) -> TableCell:
    if pending:
        return pending_cell()
    value = c2_value(row, f"{prefix}_point")
    suffix = tex_interval(*c2_interval(row, prefix)) if with_interval else ""
    return TableCell(tex_float(value, 2), value if math.isfinite(value) else None, suffix)


RESOLVED_NOTE = (r"$^{\ast}$Holm-adjusted contrast resolved under both the primary block length and the 10~s "
                 r"sensitivity blocking.")


def tab_c2_h1(c2: C2Data, out: Out) -> None:
    control = c2.control(C2_BASE)
    if c2.has_reference(C2_BASE):  # H1 runs on one binary (D-4, D-17): this table carries no marker
        reference = c2.reference_platform(C2_BASE)
        for model in MODELS:
            for mode in C2_MODES:
                if c2.radio_row(C2_BASE, model, mode) is not None \
                        and c2.run_platform(C2_BASE, model, "L", mode) != reference:
                    raise ValueError(f"tab_c2_h1: {model} {mode}/L is on another platform binary than the oracle")
    blocks: list[tuple[str, list[tuple[str, list[TableCell]]]]] = []
    for metric, title in (("affected", "Affected class"), ("network", "Network-wide")):
        rows: list[tuple[str, list[TableCell]]] = []
        for model in MODELS:
            mode_rows = [c2.find(c2.modes, design_point=C2_BASE, interpreter=model, mode=mode) for mode in C2_MODES]
            l_row = c2.find(c2.l_arm, design_point=C2_BASE, interpreter=model)
            l_pending = c2.pending("l", C2_BASE, model)
            holm_pending = c2.holm_pending(C2_BASE, model, metric)
            p_cell = TableCell("--")
            if model in H1_ANCHOR and (l_pending or holm_pending):
                p_cell = pending_cell()
            elif model in H1_ANCHOR and l_row is not None:
                p_cell = TableCell(
                    f"{tex_p(l_row[f'{metric}_gap_p_holm'])} / {tex_p(l_row[f'{metric}_gap_p_holm_10s'])}")
            modes_pending = c2.pending("modes", C2_BASE, model)
            rows.append((model, [
                *[c2_level_cell(row, metric, True, modes_pending) for row in mode_rows],
                c2_gap_cell(l_row, f"{metric}_gap", model, l_pending, holm_pending),
                p_cell,
            ]))
        mark_best(rows, ["min", "min", "min", None, None])
        block_text = ("n/a" if control is None else
                      f"$B={float(control['block_s']):.1f}$~s, $n_B/n_{{10}}={c2_blocks_text(control)}$ blocks")
        if c2.pending("controls", C2_BASE):
            block_text = f"$B$ {PENDING}"
        blocks.append((f"{title}; {block_text}", rows))
    header = table_header([
        "Interpreter", "Mode E [95\\% CI]|(\\%) $\\downarrow$", "Mode P-1 [95\\% CI]|(\\%) $\\downarrow$",
        "Mode P-10 [95\\% CI]|(\\%) $\\downarrow$", "H1 gap, E [95\\% CI]|(pp)", "$p_{\\mathrm{Holm}}$|$B$ / 10 s",
    ])
    text = render_big_table(
        name="tab_c2_h1",
        caption=("Latency-only SLA violation at $W=2$~s per loop mode at the base point (0.3 intents/s, 30 km/h), "
                 "and the H1 contrasts in mode E against the baseline of each deployment class, \\jev{} for the hosted LLMs "
                 "and \\semif{} for \\qwenjson{}. Baseline and unpaired rows carry no gap."),
        colspec="l" + "r" * 5, header=header, blocks=blocks, source="rq1_modes.csv + l_arm_contrasts.csv + controls.csv",
        tabcolsep="3pt", note=RESOLVED_NOTE, size=r"\footnotesize",
    )
    out.write_table("tab_c2_h1.tex", text)
    out.entry(Artifact(
        "tab_c2_h1", "tables/tab_c2_h1.tex",
        [c2_source(c2, name) for name in ("rq1_modes.csv", "l_arm_contrasts.csv", "controls.csv")],
        ["affected_point", "affected_ci_low", "affected_ci_high", "network_point", "network_ci_low",
         "network_ci_high", "affected_gap_point", "affected_gap_p_holm", "affected_gap_p_holm_10s",
         "affected_gap_resolved_both", "network_gap_point", "network_gap_p_holm", "network_gap_p_holm_10s",
         "network_gap_resolved_both", "block_s", "n_blocks", "n_blocks_10s"],
        f"design_point={C2_BASE}; L arms; two metric groups; all seven interpreters", "table* (14 rows)",
    ))


def tab_c2_grid(c2: C2Data, out: Out) -> None:
    blocks: list[tuple[str, list[tuple[str, list[TableCell]]]]] = []
    for rate in C2_RATES:
        points = [c2_point(rate, speed) for speed in C2_SPEEDS]
        controls = [c2.control(point) for point in points]
        block_parts = [
            f"{speed:g} km/h {PENDING if c2.pending('controls', point) else c2_blocks_text(row)}"
            + (r"$^{\ddagger}$" if c2.eligible(point) is False else "")
            for speed, point, row in zip(C2_SPEEDS, points, controls)
        ]
        for metric, title in (("affected", "affected class"), ("network", "network-wide")):
            rows: list[tuple[str, list[TableCell]]] = []
            for model in MODELS:
                cells: list[TableCell] = []
                for point in points:
                    row = c2.find(c2.l_arm, design_point=point, interpreter=model)
                    pending = c2.pending("l", point, model)
                    level = c2_level_cell(row, metric, False, pending)
                    gap = c2_gap_cell(row, f"{metric}_gap", model, pending, c2.holm_pending(point, model, metric))
                    cells += [level, gap]
                rows.append((model, cells))
            mark_best(rows, ["min", None] * len(C2_SPEEDS))
            rate_text = f"{rate:g} intent/s" if rate == 1.0 else f"{rate:g} intents/s"
            blocks.append((f"{rate_text}, {title}; blocks $n_B/n_{{10}}$: {', '.join(block_parts)}", rows))
    speeds = " & ".join(f"\\multicolumn{{2}}{{c}}{{{speed:g} km/h}}" for speed in C2_SPEEDS)
    sub = " & ".join(["SLA (\\%) $\\downarrow$ & Gap [95\\% CI] (pp)"] * len(C2_SPEEDS))
    header = (f"\\rowcolor{{ebBlue}}Interpreter & {speeds} \\\\\n"
              f"\\rowcolor{{ebBlue}} & {sub} \\\\")
    text = render_big_table(
        name="tab_c2_grid",
        caption=("Latency-only SLA violation at $W=2$~s in mode E over the RQ2 rate-by-speed grid, with each "
                 "challenger's gap to the baseline of its deployment class, \\jev{} for the hosted LLMs and \\semif{} for "
                 "\\qwenjson{}. Baseline and unpaired rows carry no gap. Affected-class gaps at eligible points form "
                 "the H2 family, and network-wide gaps on the grid are descriptive."),
        colspec="l" + "rr" * len(C2_SPEEDS), header=header, blocks=blocks,
        source="l_arm_contrasts.csv + controls.csv", tabcolsep="3pt", size=r"\footnotesize",
        note=("Intervals are on the paired gaps only, because all arms see identical UE positions and traffic. "
              + RESOLVED_NOTE + r" $^{\ddagger}$Design point below the control floor and outside the H2 family."
              + (f" {c2.h2_not_testable()}" if c2.h2_not_testable() else "")
              ),
    )
    out.write_table("tab_c2_grid.tex", text)
    out.entry(Artifact(
        "tab_c2_grid", "tables/tab_c2_grid.tex",
        [c2_source(c2, "l_arm_contrasts.csv"), c2_source(c2, "controls.csv"), c2_source(c2, "runs.csv")],
        ["affected_point", "network_point", "affected_gap_point", "affected_gap_ci_low", "affected_gap_ci_high",
         "affected_gap_resolved_both", "network_gap_point", "network_gap_ci_low", "network_gap_ci_high",
         "network_gap_resolved_both", "n_blocks", "n_blocks_10s", "eligible_d10"],
        "nine RQ2 design points; mode E; L arms; six rate-metric groups; all seven interpreters", "table* (42 rows)",
    ))


C2_RADIO_METRICS = (
    ("throughput_p5_mbps", "Cell-edge (p5) UE throughput (Mb/s)", 1.0, 3, "max"),
    ("delay_p95_ms", "p95 packet delay (ms)", 1.0, 1, "min"),
    ("handover_rate_per_ue_s", "Handover rate (per UE per minute)", 60.0, 3, None),
    ("handover_total_time_p95_ms", "p95 handover interruption (ms)", 1.0, 1, "min"),
    ("rlf_count", "Radio link failures (count per run)", 1.0, 0, "min"),
    ("stale_exposure_ue_s_per_intent", "Stale-policy exposure (UE$\\cdot$s per intent)", 1.0, 2, "min"),
)


def tab_c2_radio(c2: C2Data, out: Out) -> None:
    columns = [(rate, speed) for rate in C2_RATES
               for speed in ((3.0, 30.0, 60.0, 120.0) if rate == 0.3 else C2_SPEEDS)]
    points = [C2_BASE if (rate, speed) == (0.3, 30.0) else c2_point(rate, speed) for rate, speed in columns]
    blocks: list[tuple[str, list[tuple[str, list[TableCell]]]]] = []
    for column, title, scale, digits, direction in C2_RADIO_METRICS:
        rows: list[tuple[str, list[TableCell]]] = []
        for model in MODELS:
            cells = []
            for point in points:
                if c2.pending("radio", point, model, "E", "L") or (
                        column == "stale_exposure_ue_s_per_intent" and c2.pending("stale", point, model, "E", "L")):
                    cells.append(pending_cell())
                    continue
                if c2.pending("controls", point) and not c2.has_reference(point):
                    cells.append(pending_cell())  # the platform comparison waits for the point's absent controls
                    continue
                value = c2_value(c2.radio_row(point, model, "E"), column, scale)
                cells.append(TableCell(tex_float(value, digits), value if math.isfinite(value) else None))
            rows.append((model, cells))
        mark_best(rows, [direction] * len(points))
        blocks.append((title, rows))
    header_top = " & ".join(
        f"\\multicolumn{{{sum(r == rate for r, _ in columns)}}}{{c}}{{{rate:g} "
        f"intent{'s' if rate != 1.0 else ''}/s}}" for rate in C2_RATES)
    header_speed = " & ".join(f"{speed:g}" for _, speed in columns)
    header = (f"\\rowcolor{{ebBlue}}Interpreter & {header_top} \\\\\n"
              f"\\rowcolor{{ebBlue}}UE speed (km/h) & {header_speed} \\\\")
    text = render_big_table(
        name="tab_c2_radio",
        caption=("Radio KPIs of the latency-only runs in mode E at every C2 design point, including the base "
                 "point at 30 km/h."),
        colspec="l" + "r" * len(columns), header=header, blocks=blocks, source="radio_kpis.csv",
        tabcolsep="3pt", size=r"\footnotesize", note=None,
    )
    out.write_table("tab_c2_radio.tex", text)
    out.entry(Artifact(
        "tab_c2_radio", "tables/tab_c2_radio.tex", [c2_source(c2, "radio_kpis.csv"), c2_source(c2, "runs.csv")],
        [column for column, *_ in C2_RADIO_METRICS],
        "ten design points; mode=E; arm=L; six KPI groups; all seven interpreters", "table* (42 rows)",
    ))


def tab_c2_controls(c2: C2Data, out: Out) -> None:
    rows: list[tuple[str, list[TableCell]]] = []
    labels = {C2_BASE: "Base: 0.3/s, 30 km/h"} | {
        c2_point(rate, speed): f"{rate:g}/s, {speed:g} km/h" for rate in C2_RATES for speed in C2_SPEEDS}
    yes_no = lambda value: "--" if value is None else ("yes" if value else "no")
    for point in C2_POINTS:
        row = c2.control(point)
        if c2.pending("controls", point):
            rows.append((labels[point], [pending_cell() for _ in range(11)]))
            continue
        if row is None:
            rows.append((labels[point], [TableCell("--") for _ in range(11)]))
            continue
        c1_mark = r"$^{\ast}$" if is_true(row["c1_resolved_both"]) else ""
        c2_mark = r"$^{\ast}$" if is_true(row["c2_resolved_both"]) else ""
        means = [c2_value(row, f"c2_mean_fixed_{tag}") for tag in ("0p1", "1", "5")]
        rows.append((labels[point], [
            TableCell(tex_signed(c2_value(row, "c1_point")), None, tex_interval(*c2_interval(row, "c1")) + c1_mark),
            TableCell(tex_signed(c2_value(row, "c2_point")), None, tex_interval(*c2_interval(row, "c2")) + c2_mark),
            *[TableCell(tex_float(value, 2)) for value in means],
            TableCell(yes_no(is_true(row["c2_monotone"]))),
            TableCell(tex_float(row["tau_s"], 1)),
            TableCell(tex_float(row["block_s"], 1)),
            TableCell(c2_blocks_text(row)),
            TableCell(yes_no(is_true(row["eligible_d10"]))),
            TableCell(yes_no(is_true(row["eligible_original_5pp_d10"]))),
        ]))
    header = table_header([
        "Design point", "C-1 [95\\% CI]|(pp)", "C-2 [95\\% CI]|(pp)", "$d=0.1$ s|(\\%)", "$d=1$ s|(\\%)",
        "$d=5$ s|(\\%)", "Mono-|tone", "$\\tau$|(s)", "$B$|(s)", "$n_B/n_{10}$|blocks", "Eligible|C-1 $>0$",
        "Eligible|original rule",
    ])
    text = render_big_table(
        name="tab_c2_controls",
        caption=("Signed affected-class oracle headroom (C-1), fixed-latency sweep (C-2), block length, and eligibility "
                 "per C2 design point."),
        colspec="l" + "r" * 11, header=header, blocks=[("", rows)], source="controls.csv",
        tabcolsep="3pt", size=r"\footnotesize",
        note=(r"$^{\ast}$Resolved under both the primary block length and the 10~s sensitivity blocking."
              + r" The $d$ columns are point estimates for the C-2 monotonicity check and carry no interval."
              ),
    )
    out.write_table("tab_c2_controls.tex", text)
    out.entry(Artifact(
        "tab_c2_controls", "tables/tab_c2_controls.tex", [c2_source(c2, "controls.csv"), c2_source(c2, "runs.csv")],
        ["c1_point", "c1_ci_low", "c1_ci_high", "c1_resolved_both", "c2_point", "c2_ci_low", "c2_ci_high",
         "c2_resolved_both", "c2_mean_fixed_0p1", "c2_mean_fixed_1", "c2_mean_fixed_5", "c2_monotone", "tau_s",
         "block_s", "n_blocks", "n_blocks_10s", "eligible_d10", "eligible_original_5pp_d10"],
        "ten design points", "table* (10 rows)",
    ))


def tab_c2_rq5(c2: C2Data, out: Out) -> None:
    blocks: list[tuple[str, list[tuple[str, list[TableCell]]]]] = []
    first = c2.rq5.iloc[0] if len(c2.rq5) else None
    block_text = ("n/a" if first is None else
                  f"$B={float(first['block_s']):.1f}$~s, $n_B/n_{{10}}={c2_blocks_text(first)}$ blocks")
    if c2.pending("controls", C2_BASE):
        block_text = f"$B$ {PENDING}"
    groups = [(metric, f"{title} SLA violation (\\%), contrasts (pp); {block_text}", 100.0, 2, "min", True)
              for metric, title in (("affected", "Affected-class"), ("network", "Network-wide"))]
    groups.append(("throughput_mean_mbps", "Mean UE throughput (Mb/s)", 1.0, 3, "max", False))
    groups += [(f"jain_{cls}", f"Jain's index of per-UE mean throughput, {label}", 1.0, 3, "max", False)
               for cls, label in C2_JAIN_CLASSES]
    for column, title, scale, digits, direction, is_sla in groups:
        rows: list[tuple[str, list[TableCell]]] = []
        for model in MODELS:
            if c2.pending("rq5", C2_BASE, model):
                rows.append((model, [pending_cell() for _ in range(5)]))
                continue
            row = c2.find(c2.rq5, design_point=C2_BASE, interpreter=model)
            cells = []
            for arm in ("a", "b", "c"):
                value = c2_value(row, f"{column}_{arm}_point" if is_sla else f"{column}_{arm}", scale)
                cells.append(TableCell(tex_float(value, digits), value if math.isfinite(value) else None))
            for ref in ("a", "c"):
                if is_sla:
                    prefix = f"{column}_b_minus_{ref}"
                    cells.append(TableCell(tex_signed(c2_value(row, f"{prefix}_point")), None,
                                           tex_interval(*c2_interval(row, prefix))))
                else:
                    diff = c2_value(row, f"{column}_b", scale) - c2_value(row, f"{column}_{ref}", scale)
                    cells.append(TableCell(tex_signed(diff, digits)))
            rows.append((model, cells))
        mark_best(rows, [direction, direction, None, None, None])
        blocks.append((title, rows))
    header = table_header([
        "Interpreter", "(a) Policy|+ xApp", "(b) Direct|control", "(c) Oracle|+ xApp",
        "(b)$-$(a)|[95\\% CI]", "(b)$-$(c)|[95\\% CI]",
    ])
    text = render_big_table(
        name="tab_c2_rq5",
        caption=("RQ5 division of labor at the base point (0.3 intents/s, 30 km/h): (a) the interpreter's policy "
                 "with the numerical xApp, (b) the interpreter re-queried every 1~s for per-cell class priorities, "
                 "and (c) the oracle policy with the xApp."),
        colspec="l" + "r" * 5, header=header, blocks=blocks, source="rq5.csv", tabcolsep="3pt",
        size=r"\footnotesize",
        note=("Exploratory contrasts without multiplicity adjustment. SLA intervals are time-block bootstrap "
              "95\\% CIs with block length $B$ on the paired contrasts only, because all arms see identical UE "
              "positions and traffic. Arm (c) does not depend on the interpreter. "
              "Throughput and Jain's index are per-run values and carry no interval."),
    )
    out.write_table("tab_c2_rq5.tex", text)
    out.entry(Artifact(
        "tab_c2_rq5", "tables/tab_c2_rq5.tex", [c2_source(c2, "rq5.csv")],
        [*[f"{metric}_{arm}_point" for metric in ("affected", "network") for arm in ("a", "b", "c")],
         *[f"{metric}_b_minus_{ref}_{key}" for metric in ("affected", "network") for ref in ("a", "c")
           for key in ("point", "ci_low", "ci_high")],
         *[f"throughput_mean_mbps_{arm}" for arm in ("a", "b", "c")],
         *[f"jain_{cls}_{arm}" for cls, _ in C2_JAIN_CLASSES for arm in ("a", "b", "c")],
         "block_s", "n_blocks", "n_blocks_10s"],
        f"design_point={C2_BASE}; seven outcome groups; all seven interpreters", "table* (49 rows)",
    ))


C2_RQ3_METRICS = (
    ("network", "Network-wide SLA violation (\\%)", True),
    ("affected", "Affected-class SLA violation (\\%)", True),
    ("prb_util_mean", "PRB utilization (\\%)", False),
)


def tab_c2_rq3(c2: C2Data, out: Out) -> None:
    blocks: list[tuple[str, list[tuple[str, list[TableCell]]]]] = []
    for column, title, is_sla in C2_RQ3_METRICS:
        rows: list[tuple[str, list[TableCell]]] = []
        for model in MODELS:
            cells = []
            for rate, ues in C2_RQ3_CELLS:
                if c2.pending("rq3radio", c2_rq3_point(model, rate, ues), model):
                    cells.append(pending_cell())
                    continue
                row = c2.rq3_row(model, rate, ues)
                flagged = row is not None and is_true(row["non_stationary"]) is True
                if flagged == (row is not None and is_true(row["block_inference"]) is True):
                    raise ValueError(f"{model} {rate:g}/s {ues} UEs: block_inference does not negate non_stationary")
                value = c2_value(row, f"{column}_point" if is_sla else column)
                if flagged and is_sla and any(math.isfinite(bound) for bound in c2_interval(row, column)):
                    raise ValueError(f"{model} {rate:g}/s {ues} UEs: non-stationary cell carries a {column} interval")
                suffix = tex_interval(*c2_interval(row, column)) if is_sla else ""
                cell = TableCell(tex_float(value, 2), value if math.isfinite(value) and not flagged else None,
                                 suffix + (r"$^{\dagger}$" if flagged else ""))  # flagged: not ranked
                cells.append(cell)
            rows.append((model, cells))
        mark_best(rows, ["min" if is_sla else None] * len(C2_RQ3_CELLS))
        blocks.append((title, rows))
    n5 = sum(ues == 5 for _, ues in C2_RQ3_CELLS)
    header = (f"\\rowcolor{{ebBlue}}Interpreter & \\multicolumn{{{n5}}}{{c}}{{5 UEs per cell}} & "
              f"\\multicolumn{{{len(C2_RQ3_CELLS) - n5}}}{{c}}{{20 UEs per cell}} \\\\\n"
              f"\\rowcolor{{ebBlue}}Intent rate & "
              + " & ".join(f"{rate:g}/s" for rate, _ in C2_RQ3_CELLS) + r" \\")
    text = render_big_table(
        name="tab_c2_rq3",
        caption=("RQ3 radio outcomes of each interpreter's load trace replayed in mode E, per offered intent rate "
                 "at 5 UEs per cell and at the 20-UE scale point. SLA violation is given at $W=2$~s with its 95\\% CI."),
        colspec="l" + "r" * len(C2_RQ3_CELLS), header=header, blocks=blocks, source="rq3_radio.csv",
        tabcolsep="3pt", size=r"\footnotesize",
        note=("Because replays run without control arms, intervals use the registered 10~s time blocks of each replay "
              "stream. PRB utilization is a whole-run mean and carries no interval. $^{\\dagger}$With $\\eta\\ge1$ the "
              "interpreter's queue does not settle, so the SLA series is non-stationary and has no valid "
              "interval."),
    )
    out.write_table("tab_c2_rq3.tex", text)
    out.entry(Artifact(
        "tab_c2_rq3", "tables/tab_c2_rq3.tex", [c2_source(c2, "rq3_radio.csv"), c2_source(c2, "runs.csv")],
        ["network_point", "network_ci_low", "network_ci_high", "affected_point", "affected_ci_low",
         "affected_ci_high", "prb_util_mean", "non_stationary", "block_inference"],
        "RQ3 replays: rates 0.1/0.5/1/2 at 5 UEs per cell and 1 at 20 UEs per cell; three metric groups; "
        "all seven interpreters", "table* (21 rows)",
    ))


def c2_numbers(c2: C2Data) -> list[str]:
    lines = ["## C2 controls per design point (D-9 floor, D-10 blockings)", ""]
    ledger_table(
        lines, c2.controls, c2_source(c2, "controls.csv"), ["design_point"],
        ["c1_point", "c1_ci_low", "c1_ci_high", "c1_p_raw", "c1_resolved_both", "c2_point", "c2_ci_low",
         "c2_ci_high", "c2_p_raw", "c2_resolved_both", "c2_mean_fixed_0p1", "c2_mean_fixed_1", "c2_mean_fixed_5",
         "c2_monotone", "tau_s", "block_s", "n_blocks", "block_10s_s", "n_blocks_10s", "eligible_d10",
         "eligible_original_5pp_d10"],
    )
    lines += ["## C2 latency-only arms and H1/H2 contrasts (mode E)", ""]
    ledger_table(
        lines, c2.l_arm, c2_source(c2, "l_arm_contrasts.csv"), ["design_point", "interpreter", "comparator"],
        ["affected_point", "affected_ci_low", "affected_ci_high", "network_point", "network_ci_low",
         "network_ci_high", "affected_gap_point", "affected_gap_ci_low", "affected_gap_ci_high",
         "affected_gap_p_holm", "affected_gap_p_holm_10s", "affected_gap_resolved_both", "network_gap_point",
         "network_gap_ci_low", "network_gap_ci_high", "network_gap_p_holm", "network_gap_p_holm_10s",
         "network_gap_resolved_both", "n_blocks", "n_blocks_10s"],
    )
    lines += ["## C2 H2 verdicts (flattened from hypotheses.json)", ""]
    ledger_table(
        lines, c2.h2, f"{ANALYSIS_REL}/c2-closed-loop/results/c2_h2_summary.csv", ["rule", "challenger", "anchor"],
        ["n_eligible", "n_points", "share_resolved", "n_reversed_resolved", "holds", "source_key"],
    )
    lines += ["## C2 RQ1 loop modes at the base point (L arms)", ""]
    ledger_table(
        lines, c2.modes, c2_source(c2, "rq1_modes.csv"), ["interpreter", "mode"],
        ["affected_point", "affected_ci_low", "affected_ci_high", "network_point", "network_ci_low",
         "network_ci_high", "n_blocks", "n_blocks_10s"],
    )
    lines += ["## C2 A/N decomposition (affected class)", ""]
    ledger_table(
        lines, c2.an, c2_source(c2, "an_decomposition.csv"), ["design_point", "interpreter"],
        ["n_minus_l_point", "n_minus_l_ci_low", "n_minus_l_ci_high", "n_minus_l_resolved_both",
         "n_minus_a_point", "n_minus_a_ci_low", "n_minus_a_ci_high", "n_minus_a_resolved_both"],
    )
    used = c2.radio[(c2.radio["arm"] == "L") & (c2.radio["design_point"].isin(C2_POINTS))]
    lines += ["## C2 radio KPIs of the latency-only runs", ""]
    ledger_table(
        lines, used, c2_source(c2, "radio_kpis.csv"), ["design_point", "interpreter", "mode", "run_id"],
        ["throughput_mean_mbps", "throughput_p5_mbps", "delay_p95_ms", "outage_share", "handover_rate_per_ue_s",
         "handover_total_time_p95_ms", "rlf_count"],
    )
    lines += ["## C2 RQ5 division of labour at the base point (exploratory)", ""]
    ledger_table(
        lines, c2.rq5, c2_source(c2, "rq5.csv"), ["design_point", "interpreter", "run_id_a", "run_id_b", "run_id_c"],
        [*[f"{metric}_{arm}_point" for metric in ("affected", "network") for arm in ("a", "b", "c")],
         *[f"{metric}_b_minus_{ref}_{key}" for metric in ("affected", "network") for ref in ("a", "c")
           for key in ("point", "ci_low", "ci_high")],
         *[f"throughput_mean_mbps_{arm}" for arm in ("a", "b", "c")],
         *[f"jain_{cls}_{arm}" for cls, _ in C2_JAIN_CLASSES for arm in ("a", "b", "c")],
         "block_s", "n_blocks", "n_blocks_10s"],
    )
    lines += ["## C2 RQ3 radio replays (exploratory)", ""]
    ledger_table(
        lines, c2.rq3_radio, c2_source(c2, "rq3_radio.csv"),
        ["design_point", "interpreter", "rate_per_s", "ues_per_cell", "run_id"],
        ["rho", "non_stationary", "block_inference", "block_s", "n_blocks", "network_point", "network_ci_low",
         "network_ci_high", "affected_point", "affected_ci_low", "affected_ci_high", "prb_util_mean",
         "share_enforced_within_1s", "n_events"],
    )
    return lines


def md_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return str(value).replace("|", r"\|")


def ledger_table(lines: list[str], frame: pd.DataFrame, source: Path | str, selectors: list[str], columns: list[str]) -> None:
    header = selectors + columns + ["source", "CSV row"]
    lines += ["|" + "|".join(header) + "|", "|" + "|".join(["---"] * len(header)) + "|"]
    for _, row in frame.iterrows():
        values = [md_value(row[column]) for column in selectors + columns]
        values += [f"`{source}`", str(int(row["_csv_row"]))]
        lines.append("|" + "|".join(values) + "|")
    lines.append("")


def paper_numbers(data: Data, c2: C2Data | None = None) -> str:
    lines = [
        "# EXP-2026-003 paper number ledger",
        "",
        "Generated by `scripts/paper_assets.py`; do not edit by hand. CSV row numbers are one-based data rows "
        "(the physical file line is CSV row + 1 for the header). Values marked as Wilson limits are deterministic "
        "95% Wilson intervals computed from the cited frozen accuracy and `n_cases`. Raw-sample summaries retain "
        "their source path and source row information.",
        "",
        "## C1 H3 confirmatory results",
        "",
    ]
    ledger_table(
        lines, data.h3, _rel(C1 / "h3_confirmatory.csv"), ["model"],
        ["n_cases", "acc_c3_fresh", "acc_c57_fresh", "delta_c3_minus_c57", "ci_low", "ci_high", "p_raw",
         "p_holm", "resolved", "status"],
    )
    lines += ["## C1 exploratory telemetry-quality contrasts", ""]
    ledger_table(
        lines, data.exploratory, _rel(C1 / "exploratory_contrasts.csv"), ["quality", "model"],
        ["n_cases", "acc_c3", "acc_c57", "delta_c3_minus_c57", "ci_low", "ci_high", "p_value", "status"],
    )
    lines += ["## C1 paired multi-interpreter tests at 57 cells", ""]
    ledger_table(
        lines, data.h3_tests, _rel(C1 / "h3_c57_comparisons.csv"), ["test_type", "model_1", "model_2"],
        ["statistic_name", "statistic_value", "df", "p_value", "n_cases", "status", "details"],
    )

    lines += ["## C1 target-cluster curves and plotted 95% Wilson limits", ""]
    curve_rows: list[dict[str, Any]] = []
    for _, row in data.curves.iterrows():
        n = data.n_cases(str(row["model"]))
        low, high = wilson_interval(float(row["target_cluster_state_dep_acc"]), n)
        curve_rows.append({**row.to_dict(), "n_cases": n, "wilson95_low": low, "wilson95_high": high})
    curve_frame = pd.DataFrame(curve_rows)
    ledger_table(
        lines, curve_frame, _rel(C1 / "cell_count_curves.csv"), ["model", "quality", "n_cells"],
        ["target_cluster_state_dep_acc", "actuated_vector_match", "full_policy_match", "n_cases", "wilson95_low",
         "wilson95_high"],
    )

    lines += ["## C1 empirical latency CDF summary (primary rows pooled over all 16 conditions)", ""]
    ledger_table(
        lines, data.c1_latency_summary, f"{ANALYSIS_REL}/c1-interpretation/results/c1_latency_cdf_summary.csv", ["model"],
        ["primary_rule", "conditions_pooled", "n_primary", "n_latency", "n_feasible", "delta_e2_s",
         "deadline_s", "phi", "latency_p50", "latency_p95", "latency_p99",
         "quality_p50_cells_matched", "quality_p95_cells_matched", "quality_p99_cells_matched",
         "feasibility_phi_cells_matched", "reference_cells_total", "reference_status"],
    )

    lines += ["## C1 interpretation quality, latency, and cost", ""]
    ledger_table(
        lines, data.quality, _rel(C1 / "quality_and_cost.csv"), ["model", "condition"],
        ["n_attempts", "n_valid", "schema_valid_rate", "unsafe_policy_rate", "actuated_vector_match",
         "full_policy_match", "target_cluster_state_dep_acc", "latency_p50", "latency_p95", "latency_p99",
         "mean_input_tokens", "mean_output_tokens", "cost_per_1000_correct_usd", "energy_j_per_decision"],
    )
    lines += ["## C1 near-RT feasibility", ""]
    ledger_table(
        lines, data.feasibility, _rel(C1 / "near_rt_feasibility.csv"), ["model", "condition"],
        ["n", "n_feasible", "delta_e2_s", "deadline_s", "phi", "phi_ci_low", "phi_ci_high"],
    )
    lines += ["## C3 real-stack path decomposition", ""]
    ledger_table(
        lines, data.c3, f"{ANALYSIS_REL}/c3-real-stack/results/c3_path_decomposition.csv", ["model", "deployment"],
        ["ell_median_s", "ell_p95_s", "delta_a1_median_ms", "delta_a1_p95_ms", "delta_e2_median_ms",
         "delta_e2_p95_ms", "kpm_upper_bound_median_s", "kpm_upper_bound_p95_s", "n", "n_delta_a1",
         "n_delta_e2", "n_kpm_observed", "kpm_null_count", "tunnel_rtt_before_median_ms",
         "tunnel_rtt_after_median_ms", "tunnel_rtt_note"],
    )
    lines += ["## C3 per-intent path samples", ""]
    ledger_table(
        lines, data.c3_samples, f"{ANALYSIS_REL}/c3-real-stack/results/c3_path_samples.csv",
        ["model", "deployment", "source_records", "source_record_row"],
        ["ell_s", "delta_a1_ms", "delta_e2_ms", "kpm_upper_bound_s"],
    )
    lines += ["## RQ3 primary load cells", ""]
    ledger_table(
        lines, data.rq3, f"{ANALYSIS_REL}/c1-interpretation/results/rq3-load/rq3_load.csv", ["model", "deployment", "rate_per_s"],
        ["rho", "queue_wait_p50_s", "queue_wait_p95_s", "queue_wait_p99_s", "share_enforced_within_1s",
         "non_stationary", "n", "delta_a1_s", "delta_e2_s"],
    )
    lines += ["## RQ3 per-trace decision-latency summary", ""]
    ledger_table(
        lines, data.rq3_latency, f"{ANALYSIS_REL}/c1-interpretation/results/rq3-load/rq3_trace_latency_summary.csv",
        ["model", "deployment", "rate_per_s"],
        ["ell_median_s", "ell_p95_s", "n_latency", "source_trace", "source_trace_rows", "latency_field"],
    )
    if c2 is not None:
        lines += c2_numbers(c2)
    return "\n".join(lines).rstrip() + "\n"


def git_sha(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_manifest(out: Out, manifest_path: Path) -> None:
    code_sha, paper_sha = git_sha(ROOT), git_sha(PAPER)
    lines = [MANIFEST_BEGIN]
    for artifact in out.artifacts:
        lines += [
            f"  - artifact_id: {yaml_quote(artifact.artifact_id)}",
            f"    paper_path: {yaml_quote(artifact.paper_path)}",
            '    experiment_id: "EXP-2026-003"',
            '    run_id: ""',
            f"    source_data: [{', '.join(yaml_quote(value) for value in artifact.source_data)}]",
            f"    columns: [{', '.join(yaml_quote(value) for value in artifact.columns)}]",
            f"    filters: {yaml_quote(artifact.filters)}",
            f"    plot_type: {yaml_quote(artifact.plot_type)}",
            f"    generator: {yaml_quote(SCRIPT_REL)}",
            f"    code_git_sha: {yaml_quote(code_sha)}",
            f"    paper_git_sha: {yaml_quote(paper_sha)}",
            '    status: "draft"',
            "",
        ]
    lines.append(MANIFEST_END)
    block = "\n".join(lines) + "\n"
    text = manifest_path.read_text(encoding="utf-8")
    if MANIFEST_BEGIN in text:
        before, rest = text.split(MANIFEST_BEGIN, 1)
        after = rest.split(MANIFEST_END + "\n", 1)[1]
        text = before + block + after
    else:
        text = text.rstrip("\n") + "\n\n" + block
    manifest_path.write_text(text, encoding="utf-8", newline="\n")


def build_c2(c2: C2Data, out: Out) -> None:
    out.write_analysis_csv("c2_h2_summary.csv", c2.h2_csv)
    fig_c2_h1(c2, out)
    fig_c2_grid(c2, out)
    fig_c2_an(c2, out)
    tab_c2_h1(c2, out)
    tab_c2_grid(c2, out)
    tab_c2_radio(c2, out)
    tab_c2_controls(c2, out)
    tab_c2_rq5(c2, out)
    tab_c2_rq3(c2, out)
    if c2.allow_partial:  # placeholders to close as the absent runs land
        out.write_analysis_csv("c2_pending.csv", c2.pending_csv())


def build(paper_dir: Path, analysis_dir: Path, *, png: bool, c2_dir: Path | None = None,
          c2_allow_partial: bool = False) -> Out:
    check_palette()
    setup_mpl()
    data = Data()
    c2 = None if c2_dir is None else C2Data(c2_dir, allow_partial=c2_allow_partial)
    out = Out(paper_dir, analysis_dir, png)
    out.reference_mismatches = data.c1_latency_reference_mismatches
    out.write_analysis_csv("c1_latency_cdf_summary.csv", data.c1_latency_summary_csv)
    out.write_analysis_csv("rq3_trace_latency_summary.csv", data.rq3_latency_csv)
    out.write_analysis_csv("c3_path_samples.csv", data.c3_samples_csv)
    fig_c1(data, out)
    fig_rq3(data, out)
    fig_c3(data, out)
    tab_c1(data, out)
    tab_c1_cost(data, out)
    tab_rq3(data, out)
    tab_c3(data, out)
    if c2 is not None:
        build_c2(c2, out)
    out.write_numbers(paper_numbers(data, c2))
    write_manifest(out, paper_dir / "figure-manifest.yml")
    return out


def remove_obsolete_outputs() -> None:
    for relative in OBSOLETE_PAPER_FILES:
        path = PAPER / relative
        if path.exists():
            path.unlink()


SHA_RE = re.compile(r'(code_git_sha|paper_git_sha): "[^"]*"')


def report_reference_reproduction(out: Out) -> bool:
    if out.reference_mismatches:
        print("C1 LATENCY REPRODUCTION MISMATCH", file=sys.stderr)
        for mismatch in out.reference_mismatches:
            print(f"  {mismatch}", file=sys.stderr)
        return False
    comparisons = len(MODELS) * len(EXPECTED_CONDITIONS)
    print(
        "C1 latency reproduction: "
        f"p50/p95/p99 and phi match all {comparisons} model-condition reference cells"
    )
    return True


def c2_pending_units_named(directory: Path) -> int:
    """Number of C2 analysis units that skipped.json names as waiting for a run (0 when the analysis is complete)."""
    skipped = directory / "skipped.json"
    if not skipped.is_file():
        return 0
    return len(json.loads(skipped.read_text(encoding="utf-8")).get("analysis_units", []))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="rebuild in a temporary directory and compare SHA-256 hashes with the paper's files")
    parser.add_argument("--png", action="store_true", help=f"also render panel previews under {PNG_DIR}")
    parser.add_argument("--c2", type=Path, metavar="DIR", default=C2_RESULTS,
                        help="C2 analysis directory (default: experiments/c2-closed-loop/results/analysis)")
    parser.add_argument("--c2-allow-partial", action="store_true",
                        help="render C2 from a partial analysis (implied when skipped.json names pending units): "
                             "cells of pending units render as \\pending{pending} and c2_pending.csv lists them")
    args = parser.parse_args()
    pending = c2_pending_units_named(args.c2)
    if pending:
        print(f"C2: skipped.json names {pending} unit(s) that wait for a run; they render as placeholders")
    c2_options = {"c2_dir": args.c2, "c2_allow_partial": args.c2_allow_partial or pending > 0}
    if not args.check:
        out = build(PAPER, ANALYSIS, png=args.png, **c2_options)
        if not report_reference_reproduction(out):
            return 2
        message = f"wrote {len(out.files)} files + generated figure-manifest.yml block"
        if args.png:
            message += f"; previews in {PNG_DIR}"
        print(message)
        return 0

    expected = json.loads((PAPER / "expected_sha256.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="jev6g-paper-assets-") as tmp_name:
        tmp = Path(tmp_name)
        tmp_paper, tmp_analysis = tmp / "paper", tmp / "analysis"
        tmp_paper.mkdir(parents=True)
        shutil.copy2(PAPER / "figure-manifest.yml", tmp_paper / "figure-manifest.yml")
        out = build(tmp_paper, tmp_analysis, png=False, **c2_options)
        if not report_reference_reproduction(out):
            return 2
        got = {str(generated.relative_to(tmp_paper)): hashlib.sha256(generated.read_bytes()).hexdigest()
               for generated, _ in out.files if tmp_paper in generated.parents}
    differs = sorted(name for name in expected if got.get(name) != expected[name])
    unexpected = sorted(set(got) - set(expected))
    if differs or unexpected:
        print("CHECK FAILED")
        for name in differs:
            print("  differs from the paper:", name)
        for name in unexpected:
            print("  not in the paper:", name)
        return 1
    print(f"CHECK OK: all {len(expected)} figure and table files are byte-identical to the paper (SHA-256)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
