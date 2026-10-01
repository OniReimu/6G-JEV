"""The one place that knows the ns-3 C2 trace formats (src/ranbench/ns3/ran-closed-loop.cc).

Every other module uses the internal names on the left of COLUMNS. A renamed column in the scenario is a one-line
change here: add the new file name to the alias tuple. Extra columns and extra files are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
import filecmp
from io import StringIO
import json
from pathlib import Path
from typing import Any

import pandas as pd

SLOT_S = 0.1  # ue_slots/cell_prb rows cover (t_slot - 0.1, t_slot]

# internal name -> accepted file column names (first found wins); OPTIONAL columns may be absent
COLUMNS: dict[str, tuple[str, dict[str, tuple[str, ...]]]] = {
    "ue_slots": ("ue_slots.csv", {
        "t": ("t_slot",), "imsi": ("imsi",), "cell": ("serving_cell",), "cls": ("class",),
        "rx_bytes": ("rx_bytes",), "n_rx": ("n_rx_pkts",),
        "mean_delay_ms": ("mean_delay_ms",), "p95_delay_ms": ("p95_delay_ms",),
    }),
    "cell_prb": ("cell_prb.csv", {"t": ("t_slot",), "cell": ("cell",), "prb_share": ("prb_share",)}),
    "handover": ("handover.csv", {
        "t_start": ("t_start",), "t_end": ("t_end_ok", "t_end"), "imsi": ("imsi",),
        "source": ("source_cell",), "target": ("target_cell",), "total_ms": ("handover_total_time_ms",),
        "outcome": ("outcome",),
    }),
    "rlf": ("rlf.csv", {"t": ("t",), "imsi": ("imsi",), "cell": ("cell",), "cause": ("cause",)}),
    "enforcement": ("enforcement.csv", {
        "t": ("t",), "cell": ("cell",), "cls": ("class",), "old": ("old",), "new": ("new",), "source": ("source",),
    }),
}
OPTIONAL = {("handover", "outcome")}
PAIRING_FILES = ("positions.csv", "tx_trace.csv")  # control C-4


def _read_text(path: Path, tolerate_torn_tail: bool) -> str:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if text and not text.endswith("\n"):
        if not tolerate_torn_tail:
            raise ValueError(f"{path}: last line has no newline (run still writing or killed)")
        text = text[: text.rfind("\n") + 1]
    return text


def load_table(run_dir: str | Path, name: str, tolerate_torn_tail: bool = False) -> pd.DataFrame:
    fname, spec = COLUMNS[name]
    text = _read_text(Path(run_dir) / fname, tolerate_torn_tail)
    if not text.strip():
        return pd.DataFrame({k: pd.Series(dtype=object) for k in spec})
    df = pd.read_csv(StringIO(text), keep_default_na=True)
    out = {}
    for key, aliases in spec.items():
        col = next((a for a in aliases if a in df.columns), None)
        if col is None:
            if (name, key) in OPTIONAL:
                continue
            raise ValueError(f"{fname}: none of {aliases} in columns {list(df.columns)}")
        out[key] = df[col]
    return pd.DataFrame(out)


@dataclass
class RunTraces:
    run_dir: Path
    config: dict[str, Any]
    ue_slots: pd.DataFrame
    cell_prb: pd.DataFrame
    handover: pd.DataFrame
    rlf: pd.DataFrame
    enforcement: pd.DataFrame

    @property
    def clusters(self) -> dict[str, list[int]]:
        return {k: list(v) for k, v in self.config["clusters"].items() if not k.startswith("_")}

    @property
    def class_targets(self) -> dict[str, dict[str, Any]]:
        return self.config["traffic"]["classes"]

    @property
    def sim_time_s(self) -> float:
        return float(self.config["simulation"]["sim_time_s"])


def load_run(run_dir: str | Path, tolerate_torn_tail: bool = False) -> RunTraces:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    tabs = {n: load_table(run_dir, n, tolerate_torn_tail) for n in COLUMNS}
    return RunTraces(run_dir=run_dir, config=cfg, **tabs)


# positions.csv also carries serving_cell, which depends on the arm (handover completion can shift by a few ms across
# a 100 ms sample); C-4 concerns UE positions, so only its t, imsi, x, y columns are compared.
POSITION_COLUMNS = 4


def _position_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [",".join(line.rstrip("\n").split(",")[:POSITION_COLUMNS]) for line in f]


def pairing_identical(dir_a: str | Path, dir_b: str | Path) -> dict[str, bool]:
    """C-4: byte identity of the traffic-arrival trace and of the UE positions (t, imsi, x, y) of two runs."""
    out = {}
    for f in PAIRING_FILES:
        a, b = Path(dir_a) / f, Path(dir_b) / f
        out[f] = (_position_lines(a) == _position_lines(b)) if f == "positions.csv" else filecmp.cmp(a, b, shallow=False)
    return out
