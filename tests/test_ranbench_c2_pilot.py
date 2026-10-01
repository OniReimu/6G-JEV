from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ranbench.c2.pilot import PILOT_ARMS, TRAFFIC_PROFILES, prepare

REPO = Path(__file__).resolve().parents[1]


def test_prepare_pilot_changes_only_load_targets_and_duration(tmp_path: Path) -> None:
    base_path = REPO / "src/ranbench/ns3/c2-default.json"
    pool_path = REPO / "data/ranbench/ranintent-v1/pool/c2c3_pool.jsonl"
    out = tmp_path / "pilot"
    spec = prepare(out, base_path, pool_path, "balanced-a", 120.0, 0.004300475120544434)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    got = json.loads((out / "config.json").read_text(encoding="utf-8"))
    for section in ("scenario", "ues", "clusters", "priorities", "xapp"):
        assert got[section] == base[section]
    assert got["simulation"] | {"sim_time_s": base["simulation"]["sim_time_s"]} == base["simulation"]
    for cls, (rate, target) in TRAFFIC_PROFILES["balanced-a"].items():
        before, after = base["traffic"]["classes"][cls], got["traffic"]["classes"][cls]
        assert after | {"rate_kbps": before["rate_kbps"], "target_value": before["target_value"]} == before
        assert after["rate_kbps"] == rate and after["target_value"] == target
    assert spec["rng_run"] == 1 and 0 < spec["n_events"] < 100
    assert spec["a1_put_ack_median_s"] == pytest.approx(0.004300475120544434)
    assert spec["delta_a1_s"] == pytest.approx(0.009300475120544434)
    assert spec["delta_e2_s"] == pytest.approx(0.005)
    for arm in PILOT_ARMS:
        assert (out / "arms" / arm / "schedule.csv").is_file()
        assert (out / "arms" / arm / "events.json").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare(out, base_path, pool_path, "balanced-a", 120.0, 0.004300475120544434)
