from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from src.ranbench.c2.materialize import materialize
from src.ranbench.c2.matrix import generate_matrix
from src.ranbench.c2.streams import load_stream

REPO = Path(__file__).resolve().parents[1]


def test_materialize_controls_and_defer_missing_prerequisites(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    all_rows = generate_matrix(root)
    selected = [
        next(row for row in all_rows if row.rqs == "RQ2" and row.arm == "oracle"),
        next(row for row in all_rows if row.rqs == "RQ2" and row.arm == "fixed-1"),
        next(row for row in all_rows if row.rqs == "RQ2" and row.arm == "L"),
        next(row for row in all_rows if row.rqs == "RQ3"),
        next(row for row in all_rows if row.rqs == "RQ5"),
    ]
    rows = [{key: str(value) for key, value in asdict(row).items()} for row in selected]
    summary = materialize(
        rows,
        root,
        REPO / "src/ranbench/ns3/c2-default.json",
        REPO / "data/ranbench/ranintent-v1/pool/c2c3_pool.jsonl",
        10.0,
        0.004300475120544434,
    )
    assert summary["row_status_counts"] == {
        "exploratory_not_materialised": 1,
        "ready": 2,
        "waiting_for_c1_ledger": 1,
        "waiting_for_load_trace": 1,
    }
    for row in rows[:2]:
        assert Path(row["config_file"]).is_file()
        assert Path(row["schedule_file"]).is_file()
        assert Path(row["schedule_file"]).with_suffix(".events.json").is_file()
        assert load_stream(row["stream_file"])["meta"]["n_events"] == 100
    assert not Path(rows[2]["schedule_file"]).exists()
    manifest = json.loads((root / "input-manifest.json").read_text(encoding="utf-8"))
    assert manifest["delta_a1_s"] == pytest.approx(0.009300475120544434)
    assert manifest["delta_e2_s"] == pytest.approx(0.005)
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize(rows, root, REPO / "src/ranbench/ns3/c2-default.json",
                    REPO / "data/ranbench/ranintent-v1/pool/c2c3_pool.jsonl", 10.0, 0.004300475120544434)


def test_materialize_rejects_matrix_path_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    row = asdict(generate_matrix(root)[0])
    row["schedule_file"] = str(tmp_path / "escape.csv")
    with pytest.raises(ValueError, match="escapes"):
        materialize([{key: str(value) for key, value in row.items()}], root,
                    REPO / "src/ranbench/ns3/c2-default.json",
                    REPO / "data/ranbench/ranintent-v1/pool/c2c3_pool.jsonl", 10.0, 0.004300475120544434)


def test_materialize_rq3_trace(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    row = next(row for row in generate_matrix(root) if row.rqs == "RQ3")
    model, rate = row.interpreter, row.rate_per_s
    pool = [json.loads(line) for line in
            (REPO / "data/ranbench/ranintent-v1/pool/c2c3_pool.jsonl").read_text().splitlines()]
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps({
        "status": "decision_recorded", "model": row.interpreter,
        "intent_id": item["intent_id"], "arrival_s": float(index),
        "latency_s": 0.2, "queue_wait_s": 0.1, "labels": item["actuated"], "valid": True,
    }) + "\n" for index, item in enumerate((pool * 2)[:300])), encoding="utf-8")
    (tmp_path / "integrity.json").write_text(json.dumps({
        "complete": True, "model": row.interpreter,
        "completed_intents": 300, "rate_per_s": row.rate_per_s,
    }), encoding="utf-8")
    row = next(candidate for candidate in generate_matrix(
        root, rq3_trace_paths={f"{model}@{rate:g}": trace},
    ) if candidate.rqs == "RQ3" and candidate.interpreter == model and candidate.rate_per_s == rate)
    summary = materialize(
        [{key: str(value) for key, value in asdict(row).items()}], root,
        REPO / "src/ranbench/ns3/c2-default.json",
        REPO / "data/ranbench/ranintent-v1/pool/c2c3_pool.jsonl",
        10.0, 0.004300475120544434, trace_paths={f"{row.interpreter}@{row.rate_per_s:g}": trace},
    )
    assert summary["row_status_counts"] == {"ready": 1}
    assert load_stream(row.stream_file)["meta"]["n_events"] == 100
    assert Path(row.schedule_file).is_file()
    assert summary["load_traces"][f"{row.interpreter}@{row.rate_per_s:g}"]["sha256"]


def test_materialize_pins_a1_delay_and_records_explicit_override(tmp_path: Path) -> None:
    def one_control(root: Path) -> list[dict[str, str]]:
        row = next(row for row in generate_matrix(root) if row.arm == "oracle")
        return [{key: str(value) for key, value in asdict(row).items()}]

    common = (
        REPO / "src/ranbench/ns3/c2-default.json",
        REPO / "data/ranbench/ranintent-v1/pool/c2c3_pool.jsonl",
    )
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="0.0043.*override"):
        materialize(one_control(rejected), rejected, *common, 10.0, 0.0042)

    root = tmp_path / "override"
    summary = materialize(
        one_control(root), root, *common, 10.0, 0.0042,
        allow_a1_put_ack_override=True,
    )
    assert summary["a1_put_ack_override"] is True
    assert summary["delta_a1_s"] == pytest.approx(0.0092)
    assert summary["delta_e2_s"] == pytest.approx(0.005)
    events_path = next((root / "schedules").glob("*.events.json"))
    meta = json.loads(events_path.read_text(encoding="utf-8"))["meta"]
    assert meta["delta_a1_s"] == pytest.approx(0.0092)
    assert meta["delta_e2_s"] == pytest.approx(0.005)
