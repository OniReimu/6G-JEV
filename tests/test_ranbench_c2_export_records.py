"""Selected C1 ledger export for EXP-2026-003 C2."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.rb_c2_export_c1_records import export_records
from src.ranbench.c1_analysis import ACTUATED_FIELDS
from src.ranbench.c2.matrix import INTERPRETERS, slug
from src.ranbench.c2.schedule import load_records


def _pool(path: Path) -> list[dict]:
    rows = [
        {
            "intent_id": f"case_{i}",
            "source_condition": "c21_fresh",
            "actuated": {
                "action": "prioritise",
                "class": "video_streaming",
                "scope": "north_cluster",
                "priority": "high",
            },
        }
        for i in range(2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def _row(model: str, case_id: str, *, status: int = 200, latency: float = 0.1,
         sent: float = 1.0) -> dict:
    ok = status == 200
    labels = {
        "action": "prioritise",
        "class": "video_streaming",
        "scope": "north_cluster",
        "target_cluster": "none",
        "priority": "high",
        "prb_share": "unspecified",
        "edge_site": "unspecified",
        "latency_target": "unspecified",
        "duration": "unspecified",
    }
    return {
        "rq": "C1",
        "condition": "c21_fresh",
        "model": model,
        "case_id": case_id,
        "repeat": 0,
        "labels": [labels] if ok else None,
        "valid": ok,
        "error_type": None if ok else "HTTP_429",
        "http_status": status,
        "latency_s": latency,
        "t_send_wall": sent,
        "correct": {field: ok for field in ACTUATED_FIELDS},
    }


def _ledger(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    runs, pool_path = tmp_path / "runs", tmp_path / "pool.jsonl"
    _pool(pool_path)
    hosted = []
    for model in INTERPRETERS[:3]:
        hosted.extend(_row(model, f"case_{i}") for i in range(2))
    hosted.extend([
        _row("Qwen3.8-Flash", "case_0", status=429, latency=0.01, sent=1.0),
        _row("Qwen3.8-Flash", "case_1", status=429, latency=0.02, sent=1.1),
    ])
    _ledger(runs / "c1_hosted" / "ledger.jsonl", hosted)
    selfhosted = [
        _row(model, f"case_{i}")
        for model in INTERPRETERS[4:]
        for i in range(2)
    ]
    _ledger(runs / "c1_selfhosted" / "ledger.jsonl", selfhosted)
    _ledger(runs / "c1_hosted_qwen_block2" / "ledger.jsonl", [
        _row("Qwen3.8-Flash", "case_0", latency=0.2, sent=2.0),
        _row("Qwen3.8-Flash", "case_1", status=429, latency=0.03, sent=2.1),
    ])
    _ledger(runs / "c1_hosted_qwen_block3" / "ledger.jsonl", [
        _row("Qwen3.8-Flash", "case_0", latency=0.3, sent=3.0),
        _row("Qwen3.8-Flash", "case_1", status=429, latency=0.04, sent=3.1),
    ])
    return runs, pool_path


def test_one_record_per_intent_and_metrics_match_c1_selection(tmp_path: Path):
    runs, pool = _fixture(tmp_path)
    out = tmp_path / "out"
    manifest = export_records(runs, pool, out)

    assert set(manifest["models"]) == set(INTERPRETERS)
    assert all(info["records"] == 2 for info in manifest["models"].values())
    assert all(comp["equal"] for comp in manifest["comparison"].values())
    assert manifest["comparison"]["Qwen3.8-Flash"]["exported"] == {
        "records": 2,
        "failures": 1,
        "actuated_vector_accuracy": 0.5,
        "latency_median_s": pytest.approx(0.11),
    }
    for model in INTERPRETERS:
        rows = [json.loads(line) for line in (out / f"{slug(model)}.jsonl").read_text().splitlines()]
        assert [row["case_id"] for row in rows] == ["case_0", "case_1"]
        assert all(row["model"] == model for row in rows)
        assert all(set(row) == {"model", "case_id", "labels", "valid", "latency_s"} for row in rows)


def test_qwen_d3_first_transport_ok_and_all_429_is_failure(tmp_path: Path):
    runs, pool = _fixture(tmp_path)
    out = tmp_path / "out"
    manifest = export_records(runs, pool, out)
    rows = {row["case_id"]: row for row in load_records(
        out / f"{slug('Qwen3.8-Flash')}.jsonl", "Qwen3.8-Flash"
    ).values()}

    assert rows["case_0"]["valid"] is True
    assert rows["case_0"]["latency_s"] == 0.2
    assert isinstance(rows["case_0"]["labels"], dict)
    assert rows["case_1"] == {
        "model": "Qwen3.8-Flash",
        "case_id": "case_1",
        "labels": None,
        "valid": False,
        "latency_s": 0.02,
    }
    assert "D-3" in manifest["models"]["Qwen3.8-Flash"]["selection_rule"]


def test_fail_closed_when_pool_intent_has_no_selected_record(tmp_path: Path):
    runs, pool = _fixture(tmp_path)
    ledger = runs / "c1_selfhosted" / "ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows = [row for row in rows if not (row["model"] == "AnyJev-L0" and row["case_id"] == "case_1")]
    _ledger(ledger, rows)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match=r"AnyJev-L0: no selected C1 record for 2 pool intent"):
        export_records(runs, pool, out)
    assert not out.exists()


def test_outputs_load_via_schedule_load_records(tmp_path: Path):
    runs, pool = _fixture(tmp_path)
    out = tmp_path / "out"
    export_records(runs, pool, out)

    for model in INTERPRETERS:
        records = load_records(out / f"{slug(model)}.jsonl", model)
        assert set(records) == {"case_0", "case_1"}
