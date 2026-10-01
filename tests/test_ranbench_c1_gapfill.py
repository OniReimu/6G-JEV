"""D-3 gap-fill lists for Qwen3.8-Flash (scripts/rb_c1_gapfill.py, EXP-2026-003 deviations.md D-3)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.rb_c1_gapfill import main as gapfill_main
from src.ranbench.c1_analysis import EXPECTED_CONDITIONS

IDS = ["t_0", "t_1", "t_2"]


def _corpus(root: Path) -> Path:
    for cond in EXPECTED_CONDITIONS:
        d = root / "RQ4" / cond
        d.mkdir(parents=True)
        (d / "test.jsonl").write_text("".join(json.dumps({"case_id": c, "condition": cond}) + "\n" for c in IDS))
    return root


def _row(cond, cid, status, model="Qwen3.8-Flash", error_type=None):
    return {"rq": "C1", "model": model, "condition": cond, "case_id": cid, "repeat": 0, "http_status": status,
            "error_type": error_type if status == 200 else (error_type or f"HTTP_{status}")}


def _ledger(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_gapfill_lists_per_condition(tmp_path, capsys):
    corpus = _corpus(tmp_path / "corpus")
    runs = tmp_path / "runs"
    _ledger(runs / "c1_hosted" / "ledger.jsonl", [
        _row("c3_fresh", "t_0", 200), _row("c3_fresh", "t_1", 429),
        _row("c3_fresh", "t_2", 429, model="GLM-5.3-Flash"),  # another interpreter: ignored
    ])
    _ledger(runs / "c1_hosted_qwen_block2" / "ledger.jsonl", [_row("c3_fresh", "t_1", 429)])
    block3 = [
        _row("c3_fresh", "t_2", 200, error_type="schema_invalid"),  # an outcome: done
        _row("c7_fresh", "t_0", 200), _row("c7_fresh", "t_1", 200),
        _row("c7_fresh", "t_2", None, error_type="TimeoutError"),
        _row("c7_fresh", "d_0", 429),  # not a test case of the condition: ignored
    ]
    # By the end of fixed block 3 every test case has been attempted.  Failed
    # initial attempts are precisely what the generated gap lists may reissue.
    block3 += [
        _row(cond, cid, 429)
        for cond in EXPECTED_CONDITIONS
        for cid in IDS
        if (cond, cid) not in {
            ("c3_fresh", "t_0"), ("c3_fresh", "t_1"), ("c3_fresh", "t_2"),
            ("c7_fresh", "t_0"), ("c7_fresh", "t_1"), ("c7_fresh", "t_2"),
        }
    ]
    _ledger(runs / "c1_hosted_qwen_block3" / "ledger.jsonl", block3)
    _ledger(runs / "c1_hosted_qwen_gapfill1" / "c3_fresh" / "ledger.jsonl", [_row("c3_fresh", "t_1", 429)])
    _ledger(runs / "c1_hosted_qwen_gapfill1" / "c7_fresh" / "ledger.jsonl", [_row("c7_fresh", "t_2", 200)])
    out = tmp_path / "lists"

    assert gapfill_main(["--runs-root", str(runs), "--corpus-dir", str(corpus), "--out", str(out)]) == 0
    files = {p.name: p.read_text() for p in out.iterdir()}
    # c3_fresh: t_0 ok, t_2 schema-invalid -> only t_1 left; c7_fresh complete -> no file; 14 untouched -> all 3
    assert files["c3_fresh.txt"] == "t_1\n"
    assert "c7_fresh.txt" not in files and len(files) == 15
    assert all(files[f"{c}.txt"] == "t_0\nt_1\nt_2\n" for c in EXPECTED_CONDITIONS if c not in ("c3_fresh", "c7_fresh"))
    printed = capsys.readouterr().out
    assert [ln.split() for ln in printed.splitlines() if ln.startswith(("c3_fresh ", "c7_fresh ", "total "))] == [
        ["c3_fresh", "3", "3", "2", "1"], ["c7_fresh", "3", "3", "3", "0"], ["total", "48", "5", "43"]]
    assert "c1_hosted_qwen_gapfill1/c7_fresh/ledger.jsonl" in printed

    # a non-empty output directory is refused (stale lists)
    assert gapfill_main(["--runs-root", str(runs), "--corpus-dir", str(corpus), "--out", str(out)]) == 2


def test_gapfill_requires_the_fixed_blocks(tmp_path):
    corpus = _corpus(tmp_path / "corpus")
    _ledger(tmp_path / "runs" / "c1_hosted" / "ledger.jsonl", [])
    with pytest.raises(FileNotFoundError, match="c1_hosted_qwen_block2, c1_hosted_qwen_block3"):
        gapfill_main(["--runs-root", str(tmp_path / "runs"), "--corpus-dir", str(corpus), "--out", str(tmp_path / "o")])


def test_gapfill_refuses_until_fixed_blocks_attempt_every_case(tmp_path):
    corpus = _corpus(tmp_path / "corpus")
    runs = tmp_path / "runs"
    _ledger(runs / "c1_hosted" / "ledger.jsonl", [])
    _ledger(runs / "c1_hosted_qwen_block2" / "ledger.jsonl", [])
    rows = [_row(cond, cid, 429) for cond in EXPECTED_CONDITIONS for cid in IDS]
    rows.remove(next(r for r in rows if r["condition"] == "c57_contradictory" and r["case_id"] == "t_2"))
    _ledger(runs / "c1_hosted_qwen_block3" / "ledger.jsonl", rows)

    with pytest.raises(ValueError, match=r"c57_contradictory: 1 unattempted"):
        gapfill_main(["--runs-root", str(runs), "--corpus-dir", str(corpus), "--out", str(tmp_path / "lists")])
    assert not (tmp_path / "lists").exists()
