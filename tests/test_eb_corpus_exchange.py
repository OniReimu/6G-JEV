"""Tests for file-exchange transport (export, ingest, accept/reject, re-export, manifest, lints)."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.edgebench.contract import DEFAULT_CRITERIA, load_cases_jsonl
from src.edgebench.corpus.assemble import generate_manifest
from src.edgebench.corpus.catalog import save_catalogs
from src.edgebench.corpus.exchange import ALL_CONDITIONS, CorpusExchange, TargetItem


@pytest.fixture
def exchange_setup(tmp_path: Path):
    exchange_dir = tmp_path / "exchange"
    data_dir = tmp_path / "data"
    cat_dir = data_dir / "catalog"
    save_catalogs(cat_dir, seed=20260924)

    exchange = CorpusExchange(
        exchange_dir=exchange_dir,
        data_dir=data_dir,
        seed=20260924,
    )
    return exchange, exchange_dir, data_dir


def test_round_trip_export_ingest_reexport_and_blind_verification(exchange_setup):
    exchange, exchange_dir, data_dir = exchange_setup
    test_conditions = ["clean", "negation", "K15"]

    # 1. Export gen stage
    gen_files = exchange.export_gen(
        conditions=test_conditions,
        splits=["test"],
        gen_batch_size=10,
        n_test=10,
        n_dev=0,
    )
    assert len(gen_files) > 0

    # Verify gen/todo file format
    for gf in gen_files:
        assert gf.exists()
        with open(gf, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
            assert len(lines) <= 10
            for obj in lines:
                assert "item_id" in obj
                assert "condition" in obj
                assert "split" in obj
                assert "prompt" in obj
                assert "max_words" in obj
                assert obj["condition"] in test_conditions

    # 2. Fake generator writes gen/done files
    # Deliberately make one item in 'clean' fail catalog leak lint
    # All other items produce valid text
    clean_failed_item_id = None
    for gf in gen_files:
        done_f = exchange.gen_done_dir / gf.name
        with open(gf, "r", encoding="utf-8") as f_in, open(done_f, "w", encoding="utf-8") as f_out:
            for idx, line in enumerate(f_in):
                obj = json.loads(line)
                it_id = obj["item_id"]
                cond = obj["condition"]

                if cond == "clean" and clean_failed_item_id is None:
                    # Violates catalog leak lint for non-RQ4
                    clean_failed_item_id = it_id
                    text = "Please run the generic counting fallback for this camera."
                else:
                    text = f"Valid natural edge request {idx} requiring specific edge capability cleanly."

                f_out.write(json.dumps({
                    "item_id": it_id,
                    "text": text,
                    "generator": "fake-generator-v1@agy",
                }) + "\n")

    # 3. Ingest gen stage
    gen_res = exchange.ingest_gen()
    assert gen_res["files_processed"] == len(gen_files)
    assert gen_res["items_rejected"] >= 1
    assert gen_res["rejections"]["lint_catalog_leak"] >= 1
    assert exchange.items[clean_failed_item_id].attempt == 2
    assert exchange.items[clean_failed_item_id].status == "pending_gen"

    # 4. Export ver stage
    ver_files = exchange.export_ver(
        conditions=test_conditions,
        splits=["test"],
        ver_batch_size=10,
    )
    assert len(ver_files) > 0

    # 5. Assert ver files contain NO target information and NO condition names
    jsonl_ver_files = [f for f in ver_files if f.name.endswith(".jsonl")]
    context_ver_files = [f for f in ver_files if f.name.endswith(".context.md")]
    assert len(jsonl_ver_files) == len(context_ver_files)

    for jf in jsonl_ver_files:
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                # Lines contain ONLY item_id and text!
                assert set(obj.keys()) == {"item_id", "text"}
                # No condition name in item_id
                for c in ALL_CONDITIONS:
                    assert c not in obj["item_id"]
                # No condition name in file name
                for c in ALL_CONDITIONS:
                    assert c not in jf.name

    for cf in context_ver_files:
        content = cf.read_text(encoding="utf-8")
        # Context file contains evaluator instructions and field criteria
        assert "Evaluator Instructions" in content
        assert "Field Criteria" in content
        # No condition name in file name
        for c in ALL_CONDITIONS:
            assert c not in cf.name
        # No target tuple values or condition names revealing targets
        for c in test_conditions:
            # Condition name must not appear
            assert f"Condition: {c}" not in content
            assert f"condition: {c}" not in content

    # 6. Fake verifier writes ver/done files
    # One valid item is given wrong labels (mismatch), others match targets
    mismatch_item_id = None
    for jf in jsonl_ver_files:
        done_ver = exchange.ver_done_dir / jf.name
        with open(jf, "r", encoding="utf-8") as f_in, open(done_ver, "w", encoding="utf-8") as f_out:
            for idx, line in enumerate(f_in):
                obj = json.loads(line)
                it_id = obj["item_id"]
                target_labels = exchange.items[it_id].tuple_labels

                if mismatch_item_id is None and exchange.items[it_id].condition == "negation":
                    mismatch_item_id = it_id
                    # Corrupt one label to test mismatch rejection
                    bad_labels = dict(target_labels)
                    first_field = list(target_labels.keys())[0]
                    bad_labels[first_field] = "unspecified" if target_labels[first_field] != "unspecified" else "standard"
                    labels_to_write = bad_labels
                else:
                    labels_to_write = dict(target_labels)

                f_out.write(json.dumps({
                    "item_id": it_id,
                    "labels": labels_to_write,
                    "verifier": "fake-verifier-opus@subagent",
                }) + "\n")

    # 7. Ingest ver stage
    ver_res = exchange.ingest_ver()
    assert ver_res["items_rejected"] >= 1
    assert ver_res["rejections"]["verifier_mismatch"] >= 1
    assert exchange.items[mismatch_item_id].attempt == 2
    assert exchange.items[mismatch_item_id].status == "pending_gen"
    assert ver_res["items_accepted"] > 0

    # 8. Re-export only rejected items
    # Calling export_gen should re-export ONLY the 2 rejected items (attempt=2)
    retry_gen_files = exchange.export_gen(
        conditions=test_conditions,
        splits=["test"],
        gen_batch_size=10,
        n_test=10,
        n_dev=0,
    )
    assert len(retry_gen_files) > 0

    reexported_ids: set[str] = set()
    for rf in retry_gen_files:
        with open(rf, "r", encoding="utf-8") as f:
            for line in f:
                reexported_ids.add(json.loads(line)["item_id"])

    # Must contain the 2 rejected items
    assert clean_failed_item_id in reexported_ids
    assert mismatch_item_id in reexported_ids
    # Must NOT contain any already accepted items
    for it_id, it in exchange.items.items():
        if it.status == "accepted":
            assert it_id not in reexported_ids


def test_malformed_done_files_rejected_with_clear_error(exchange_setup):
    exchange, exchange_dir, data_dir = exchange_setup

    # Export 5 items
    gen_files = exchange.export_gen(
        conditions=["clean"],
        splits=["test"],
        gen_batch_size=5,
        n_test=5,
        n_dev=0,
    )
    gf = gen_files[0]
    done_f = exchange.gen_done_dir / gf.name

    # 1. Missing required key 'generator'
    with open(done_f, "w", encoding="utf-8") as f:
        f.write(json.dumps({"item_id": "it_1", "text": "Some text"}) + "\n")

    with pytest.raises(ValueError, match="missing 'generator'"):
        exchange.ingest_gen()

    # 2. Unknown item_id
    with open(done_f, "w", encoding="utf-8") as f:
        f.write(json.dumps({"item_id": "it_non_existent", "text": "Some text", "generator": "gen1"}) + "\n")

    with pytest.raises(ValueError, match="unknown item_id"):
        exchange.ingest_gen()

    # 3. Duplicate item_id
    with open(gf, "r", encoding="utf-8") as f:
        first_id = json.loads(f.readline())["item_id"]

    with open(done_f, "w", encoding="utf-8") as f:
        f.write(json.dumps({"item_id": first_id, "text": "Text 1", "generator": "gen1"}) + "\n")
        f.write(json.dumps({"item_id": first_id, "text": "Text 2", "generator": "gen1"}) + "\n")

    with pytest.raises(ValueError, match="duplicate item_id"):
        exchange.ingest_gen()

    # 4. Extra item ID in batch
    with open(gf, "r", encoding="utf-8") as f:
        valid_ids = [json.loads(line)["item_id"] for line in f]

    # Create another target item that belongs to a different batch
    other_item = TargetItem(
        item_id="it_different_batch",
        tuple_id="clean_test_9999",
        condition="clean",
        split="test",
        tuple_labels={"service_type": "ocr"},
        meta={},
    )
    exchange.items[other_item.item_id] = other_item

    with open(done_f, "w", encoding="utf-8") as f:
        for vid in valid_ids:
            f.write(json.dumps({"item_id": vid, "text": "Good text", "generator": "gen1"}) + "\n")
        f.write(json.dumps({"item_id": other_item.item_id, "text": "Extra text", "generator": "gen1"}) + "\n")

    with pytest.raises(ValueError, match="contains extra item IDs"):
        exchange.ingest_gen()


def test_idempotency_of_export(exchange_setup):
    exchange, exchange_dir, data_dir = exchange_setup

    # First export creates batches
    first_export = exchange.export_gen(
        conditions=["clean"],
        splits=["test"],
        gen_batch_size=10,
        n_test=10,
        n_dev=0,
    )
    assert len(first_export) > 0

    # Second export immediately without done files should export nothing new
    second_export = exchange.export_gen(
        conditions=["clean"],
        splits=["test"],
        gen_batch_size=10,
        n_test=10,
        n_dev=0,
    )
    assert len(second_export) == 0


def test_manifest_records_generator_and_verifier(exchange_setup):
    exchange, exchange_dir, data_dir = exchange_setup

    # Export, simulate generation & verification, ingest
    exchange.export_gen(conditions=["clean"], splits=["test"], gen_batch_size=5, n_test=5, n_dev=0)

    # Write gen done
    opening_phrases = [
        "Please read text from optical scanner",
        "Perform document optical scanning extraction",
        "Transcribe printed characters on shipping labels",
        "Inspect invoice paperwork for printed letters",
        "Process scanned forms to retrieve text",
    ]
    idx = 0
    for gf in exchange.gen_todo_dir.glob("*.jsonl"):
        done_f = exchange.gen_done_dir / gf.name
        with open(gf, "r", encoding="utf-8") as f_in, open(done_f, "w", encoding="utf-8") as f_out:
            for line in f_in:
                obj = json.loads(line)
                prefix = opening_phrases[idx % len(opening_phrases)]
                idx += 1
                f_out.write(json.dumps({
                    "item_id": obj["item_id"],
                    "text": f"{prefix} for payload item {obj['item_id']}",
                    "generator": "gemini-3.8-flash-high@agy-1.2.8",
                }) + "\n")

    exchange.ingest_gen()
    ver_files = exchange.export_ver(conditions=["clean"], splits=["test"], ver_batch_size=5)

    # Write ver done
    for jf in [f for f in ver_files if f.name.endswith(".jsonl")]:
        done_ver = exchange.ver_done_dir / jf.name
        with open(jf, "r", encoding="utf-8") as f_in, open(done_ver, "w", encoding="utf-8") as f_out:
            for line in f_in:
                obj = json.loads(line)
                it = exchange.items[obj["item_id"]]
                f_out.write(json.dumps({
                    "item_id": obj["item_id"],
                    "labels": it.tuple_labels,
                    "verifier": "claude-3-opus@subagent-ver",
                }) + "\n")

    exchange.ingest_ver()
    exchange.assemble(n_test=5, n_dev=0)

    # Generate manifest
    manifest = generate_manifest(data_dir, yields=exchange.compute_yields(), total_cost_usd=0.0, seed=20260924)

    assert "generators" in manifest
    assert "verifiers" in manifest
    assert "items" in manifest
    assert "gemini-3.8-flash-high@agy-1.2.8" in manifest["generators"]
    assert "claude-3-opus@subagent-ver" in manifest["verifiers"]

    # Check per-item recording
    assert len(manifest["items"]) == 5
    for cid, meta in manifest["items"].items():
        assert meta["generator"] == "gemini-3.8-flash-high@agy-1.2.8"
        assert meta["verifier"] == "claude-3-opus@subagent-ver"


def test_status_report_formatting(exchange_setup):
    exchange, exchange_dir, data_dir = exchange_setup
    exchange.init_targets(conditions=["clean", "negation"], splits=["test"], n_test=20, n_dev=0)
    status_str = exchange.print_status()

    assert "Condition" in status_str
    assert "Target N" in status_str
    assert "Generated" in status_str
    assert "Verified" in status_str
    assert "Accepted" in status_str
    assert "Remaining" in status_str
    assert "RQ2/clean" in status_str
    assert "RQ2/negation" in status_str
    assert "TOTAL" in status_str
