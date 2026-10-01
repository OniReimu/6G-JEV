"""RANIntent v1 corpus (EXP-2026-003 C1): tuples, telemetry invariants, schemas, prompts, lints, exchange, assembly."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.edgebench.corpus.tuples import TupleItem
from src.ranbench.corpus.assemble import assemble_corpus, case_record, schema_file_contents
from src.ranbench.corpus.cli import main as cli_main
from src.ranbench.corpus.exchange import RanExchange
from src.ranbench.corpus.lints import check_cluster_leak_lint, check_length_lint, check_literal_token_lint
from src.ranbench.corpus.prompts import build_generator_prompt, build_verifier_context
from src.ranbench.corpus.spec import (
    ACTUATABLE_ACTIONS,
    CLUSTERS,
    FIELDS,
    METRIC_COLUMNS,
    OPTIONS,
    SCOPE_METRIC,
    STATE_DEPENDENT_SCOPES,
    build_policy_type,
    conditions,
    is_unsafe_policy,
    load_config,
    validate_truth,
)
from src.ranbench.corpus.stub import STUB_GENERATOR, stub_text
from src.ranbench.corpus.telemetry import (
    build_table,
    parse_table_text,
    render_table,
    resolve_extreme,
    resolve_target_cluster,
)
from src.ranbench.corpus.tuples import sample_tuples, truth_record
from src.ranbench.schemas import POLICY_TYPE_PATH, load_policy_type, load_reading_rules, validate_policy

CFG = load_config()
QUALITIES = CFG["qualities"]
SIZES = CFG["cell_counts"]


@pytest.fixture(scope="module")
def tuples() -> dict[str, list[TupleItem]]:
    return {sp: sample_tuples(sp, CFG) for sp in ("test", "dev")}


@pytest.fixture(scope="module")
def tables(tuples) -> dict[tuple[str, int, str], tuple[dict, dict]]:
    """Every (tuple, cell count, quality) table of both splits."""
    return {
        (t.tuple_id, n, q): build_table(t, n, q, CFG)
        for items in tuples.values() for t in items for n in SIZES for q in QUALITIES
    }


def _sd(tuples):
    return [t for items in tuples.values() for t in items if t.meta["half"] == "state_dependent"]


# ---------------------------------------------------------------- tuples


def test_tuples_are_deterministic_and_seeded(tuples):
    for sp in ("test", "dev"):
        assert [t.to_dict() for t in sample_tuples(sp, CFG)] == [t.to_dict() for t in tuples[sp]]
    other = copy.deepcopy(CFG)
    other["base_seed"] += 1
    assert [t.tuple_labels for t in sample_tuples("test", other)] != [t.tuple_labels for t in tuples["test"]]


def test_split_is_150_150_plus_dev_30_30(tuples):
    assert Counter(t.meta["half"] for t in tuples["test"]) == {"state_dependent": 150, "named_scope": 150}
    assert Counter(t.meta["half"] for t in tuples["dev"]) == {"state_dependent": 30, "named_scope": 30}
    assert len({t.tuple_id for items in tuples.values() for t in items}) == 360
    for t in tuples["test"] + tuples["dev"]:
        lab = t.tuple_labels
        if t.meta["half"] == "state_dependent":
            assert lab["scope"] in STATE_DEPENDENT_SCOPES and lab["target_cluster"] in CLUSTERS
            assert lab["action"] != "unsupported"
        else:
            assert lab["action"] in ACTUATABLE_ACTIONS and lab["scope"] in CLUSTERS
            assert lab["target_cluster"] == "none"
        if lab["action"] == "revert_default":
            assert lab["priority"] == "unspecified"


def test_marginals_are_balanced(tuples):
    for half in ("state_dependent", "named_scope"):
        items = [t for t in tuples["test"] if t.meta["half"] == half]
        keys = {
            "action": lambda lab: lab["action"],
            "class": lambda lab: lab["class"],
            "scope_target": lambda lab: (lab["scope"], lab["target_cluster"]),  # joint for the state-dependent half
        }
        for name, key in keys.items():
            counts = Counter(key(t.tuple_labels) for t in items)
            assert max(counts.values()) - min(counts.values()) <= 1, (half, name, counts)


def test_truth_records_validate_against_schema_and_rights(tuples):
    for t in tuples["test"] + tuples["dev"]:
        assert validate_truth(truth_record(t), CFG) == [], t.tuple_id
    base = truth_record(next(t for t in tuples["test"] if t.meta["half"] == "named_scope"))
    base |= {"issuer": "operator", "class": "best_effort", "action": "prioritise", "priority": "high"}
    assert validate_truth(base, CFG) == []
    bad_cases = [
        base | {"action": "revert_default"},  # revert_default carries no priority
        base | {"action": "unsupported", "priority": "unspecified"},  # unsupported: scope must be unspecified
        base | {"scope": "most_loaded_cells"},  # state-dependent scope needs a target cluster
        base | {"target_cluster": "stadium"},  # named scope has none
        base | {"issuer": "lumen_media"},  # tenant does not own best_effort
        base | {"issuer": "lumen_media", "class": "xr_gaming", "priority": "critical"},  # no critical right
        base | {"extra": "x"},
    ]
    for rec in bad_cases:
        assert validate_truth(rec, CFG), rec
    assert not is_unsafe_policy(base | {"class": "emergency_video", "priority": "critical"}, "civic_safety", CFG)


def test_schema_files_are_pinned_to_the_generator():
    for path, text in schema_file_contents(CFG).items():
        assert path.read_text(encoding="utf-8") == text, f"{path.name} stale: run write-schemas"


def test_policy_type_matches_spike_envelope_and_validates():
    spike_like = {"policySchema", "statusSchema"}
    pt = load_policy_type()
    assert set(pt) == spike_like and pt == build_policy_type(CFG)
    assert pt["policySchema"]["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert pt["statusSchema"]["properties"]["enforceStatus"]["enum"] == ["ENFORCED", "NOT_ENFORCED"]
    assert set(pt["policySchema"]["required"]) == set(FIELDS)
    good = {f: next(iter(OPTIONS[f])) for f in FIELDS}
    assert validate_policy(good) == []
    assert validate_policy(good | {"priority": "urgent"})
    assert validate_policy(good | {"note": "x"})
    assert validate_policy({k: v for k, v in good.items() if k != "scope"})
    assert "20001" in POLICY_TYPE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------- telemetry


def test_tables_are_deterministic(tuples, tables):
    for t in tuples["test"][::37]:
        for n in SIZES:
            for q in QUALITIES:
                assert build_table(t, n, q, CFG) == tables[(t.tuple_id, n, q)]


def test_render_parse_round_trip(tuples, tables):
    for t in tuples["dev"]:
        for n in SIZES:
            for q in QUALITIES:
                table = tables[(t.tuple_id, n, q)][0]
                assert parse_table_text(render_table(table)) == table


def test_argmax_lies_in_target_cluster_for_every_tuple_quality_and_size(tuples, tables):
    checked = 0
    for t in _sd(tuples):
        for n in SIZES:
            for q in QUALITIES:
                table, info = tables[(t.tuple_id, n, q)]
                assert resolve_target_cluster(table, t.tuple_labels["scope"]) == t.tuple_labels["target_cluster"], (t.tuple_id, n, q)
                metric, direction = SCOPE_METRIC[t.tuple_labels["scope"]]
                assert resolve_extreme(table, metric, direction)["cell_id"] == info["extreme_cells"][metric]
                checked += 1
    assert checked == 180 * len(SIZES) * len(QUALITIES)


def test_every_table_has_well_defined_cells_and_clusters(tuples, tables):
    for t in tuples["test"] + tuples["dev"]:
        for n in SIZES:
            table = tables[(t.tuple_id, n, "fresh")][0]
            clusters = [r[1] for r in table["rows"]]
            assert len(table["rows"]) == n and len({r[0] for r in table["rows"]}) == n
            assert set(clusters) <= set(CLUSTERS)
            assert len(set(clusters)) == min(n, 4)
            if t.meta["half"] == "named_scope":
                assert t.tuple_labels["scope"] in clusters  # the named cluster is present at every size
            for q in QUALITIES:  # every quality covers the same cells
                assert {r[0] for r in tables[(t.tuple_id, n, q)][0]["rows"]} == {r[0] for r in table["rows"]}


def test_other_metric_points_to_another_cluster(tuples, tables):
    for t in _sd(tuples):
        other = next(s for s in STATE_DEPENDENT_SCOPES if s != t.tuple_labels["scope"])
        for n in SIZES:
            table = tables[(t.tuple_id, n, "fresh")][0]
            assert resolve_target_cluster(table, other) != t.tuple_labels["target_cluster"]


def test_noisy_never_changes_the_extreme_cells(tuples, tables):
    t_cfg = CFG["telemetry"]
    for t in tuples["test"] + tuples["dev"]:
        for n in SIZES:
            fresh, info = tables[(t.tuple_id, n, "fresh")]
            noisy, _ = tables[(t.tuple_id, n, "noisy")]
            assert noisy["noise_bounds"] == {m: t_cfg[m]["noise"] for m in METRIC_COLUMNS}
            for metric, direction in SCOPE_METRIC.values():
                assert resolve_extreme(noisy, metric, direction)["cell_id"] == info["extreme_cells"][metric]
                # worst case: no draw within the bound can change the extreme cell
                col = fresh["columns"].index(metric)
                vals = sorted((r[col] for r in fresh["rows"]), reverse=direction == "max")
                if len(vals) > 1:
                    assert abs(vals[0] - vals[1]) > 2 * t_cfg[metric]["noise"]
                # and every noisy value stays within its bound
                by_cell = {r[0]: r[col] for r in fresh["rows"]}
                for r in noisy["rows"]:
                    assert abs(r[col] - by_cell[r[0]]) <= t_cfg[metric]["noise"] + 1e-6


def test_stale_and_contradictory_rows_follow_the_rules(tuples, tables):
    t_cfg = CFG["telemetry"]
    from datetime import datetime

    for t in tuples["test"] + tuples["dev"]:
        for n in SIZES:
            fresh = {r[0]: r for r in tables[(t.tuple_id, n, "fresh")][0]["rows"]}
            stale = tables[(t.tuple_id, n, "stale")][0]
            snap = datetime.strptime(stale["snapshot_time"], "%Y-%m-%dT%H:%M:%SZ")
            ages: dict[str, list[float]] = {}
            for r in stale["rows"]:
                ages.setdefault(r[0], []).append((snap - datetime.strptime(r[2], "%Y-%m-%dT%H:%M:%SZ")).total_seconds())
            for cell, a in ages.items():
                fresh_rows = [x for x in a if x <= t_cfg["stale_after_s"]]
                assert len(fresh_rows) == 1 and max(fresh_rows) <= t_cfg["fresh_age_s"][1]
                assert all(x >= t_cfg["stale_age_s"][0] for x in a if x > t_cfg["stale_after_s"])
            contra = tables[(t.tuple_id, n, "contradictory")][0]
            e2 = {r[0]: r for r in contra["rows"] if r[2] == "E2_KPM"}
            assert set(e2) == set(fresh)
            for r in contra["rows"]:
                if r[2] == "O1_PM":
                    assert all(a != b for a, b in zip(r[3:], e2[r[0]][3:])), "O1 row must disagree on every metric"
                else:
                    assert r[3:] == fresh[r[0]][2:]


def test_stale_and_contradictory_are_informative(tuples, tables):
    """Ignoring the rule (pooling all rows) gives a wrong cluster in most state-dependent cases."""
    shares = {}
    for q in ("stale", "contradictory"):
        for n in SIZES:
            wrong = decoys = 0
            items = _sd(tuples)
            for t in items:
                table, info = tables[(t.tuple_id, n, q)]
                pooled = resolve_target_cluster(table, t.tuple_labels["scope"], apply_rules=False)
                is_wrong = pooled != t.tuple_labels["target_cluster"]
                assert is_wrong == (info["decoy_cell"] is not None)  # exactly the decoy tables are informative
                wrong += is_wrong
                decoys += info["decoy_cell"] is not None
            shares[f"c{n}_{q}"] = wrong / len(items)
            assert shares[f"c{n}_{q}"] >= 0.7, shares
    print("\nrules-ignored wrong share (state-dependent, test+dev):", json.dumps(shares))


def test_fresh_and_noisy_pooled_reading_is_the_truth(tuples, tables):
    for t in _sd(tuples):
        for n in SIZES:
            for q in ("fresh", "noisy"):
                table, _ = tables[(t.tuple_id, n, q)]
                assert resolve_target_cluster(table, t.tuple_labels["scope"], apply_rules=False) == t.tuple_labels["target_cluster"]


def test_case_record_carries_token_estimates(tuples):
    t = _sd(tuples)[0]
    small = case_record(t, "some text here", 3, "fresh", CFG, {})
    big = case_record(t, "some text here", 57, "stale", CFG, {})
    assert 0 < small["meta"]["tokens_cl100k"]["telemetry"] < big["meta"]["tokens_cl100k"]["telemetry"]
    assert big["meta"]["tokens_cl100k"]["telemetry"] > CFG["tokens"]["laya_context_tokens"]
    assert small["truth"]["target_cluster"] == t.tuple_labels["target_cluster"]


# ---------------------------------------------------------------- prompts and lints


def test_generator_prompt_never_depends_on_target_cluster(tuples):
    for t in _sd(tuples)[:20]:
        prompts = set()
        for c in CLUSTERS:
            alt = TupleItem(t.condition, t.split, t.tuple_id, t.tuple_labels | {"target_cluster": c}, t.meta)
            prompts.add(build_generator_prompt(alt, CFG))
        assert len(prompts) == 1
        assert "Identify the cells only by their current state" in prompts.pop()
    ns = next(t for t in tuples["test"] if t.meta["half"] == "named_scope" and t.tuple_labels["scope"] == "stadium")
    assert "the stadium cluster" in build_generator_prompt(ns, CFG)


def test_verifier_context_and_rules_are_shared():
    ctx = build_verifier_context(CFG)
    assert load_reading_rules() in ctx
    for field in FIELDS:
        assert f"### {field}" in ctx


def test_cluster_leak_lint_catches_planted_leaks():
    assert not check_cluster_leak_lint("Boost responder video where the stadium cells are busiest")[0]
    assert not check_cluster_leak_lint("Lower gaming in the northern sector's most congested cells")[0]
    assert not check_cluster_leak_lint("Ambulance feeds near the Hospital need priority in congested cells")[0]
    assert check_cluster_leak_lint("Give responder video priority wherever the cells are most congested")[0]
    assert check_cluster_leak_lint("Prioritise video in the stadium cluster", {"stadium"})[0]
    assert not check_cluster_leak_lint("Prioritise video in the stadium cluster, not the south", {"stadium"})[0]


def test_literal_and_length_lints():
    assert not check_literal_token_lint("please apply emergency_video priority")[0]
    assert not check_literal_token_lint("look at prb_util_pct")[0]
    assert not check_literal_token_lint("scope most_loaded_cells")[0]
    assert check_literal_token_lint("raise the priority of emergency video in the busiest cells")[0]
    assert not check_length_lint("too short", 6, 80)[0]
    assert not check_length_lint("word " * 81, 6, 80)[0]
    assert check_length_lint("one two three four five six", 6, 80)[0]


# ---------------------------------------------------------------- exchange and assembly


def _small_cfg() -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["splits"] = {"test": {"state_dependent": 6, "named_scope": 6}, "dev": {"state_dependent": 2, "named_scope": 2}}
    return cfg


def test_exchange_ver_files_are_blind_and_gen_lints_reject(tmp_path: Path):
    cfg = _small_cfg()
    ex = RanExchange(exchange_dir=tmp_path / "ex", data_dir=tmp_path / "data", cfg=cfg)
    gen_files = ex.export_gen(gen_batch_size=5)
    assert gen_files and all(set(json.loads(l)) == {"item_id", "split", "prompt", "max_words"}
                             for f in gen_files for l in f.read_text().splitlines())
    sd_ids = [i for i, it in ex.items.items() if it.meta["half"] == "state_dependent"]
    leak_id, dup_ids = sd_ids[0], sd_ids[1:3]
    for f in gen_files:
        lines = []
        for l in f.read_text().splitlines():
            iid = json.loads(l)["item_id"]
            it = ex.items[iid]
            text = stub_text(TupleItem(it.condition, it.split, it.tuple_id, it.tuple_labels, it.meta))
            if iid == leak_id:
                text = "Boost responder video in the busiest cells around the stadium right now please."
            if iid in dup_ids:
                text = "Demote consumer streaming wherever cells are busiest, at low level, for an hour."
            lines.append(json.dumps({"item_id": iid, "text": text, "generator": STUB_GENERATOR}))
        (ex.gen_done_dir / f.name).write_text("\n".join(lines) + "\n")
    res = ex.ingest_gen()
    assert res["rejections"]["lint_cluster_leak"] == 1 and ex.items[leak_id].status == "pending_gen"
    assert res["rejections"]["lint_duplicate"] == 1

    ver_files = ex.export_ver(ver_batch_size=4)
    for f in ver_files:
        if f.suffix == ".jsonl":
            for l in f.read_text().splitlines():
                rec = json.loads(l)
                assert set(rec) == {"item_id", "issuer", "text", "telemetry"}
                assert rec["item_id"].startswith("it_") and "ranintent" not in l
                assert "target_cluster" not in rec["telemetry"]
        else:
            assert "item_id" in f.read_text() and "it_" not in f.read_text()


def test_end_to_end_stub_pipeline_produces_frozen_manifest(tmp_path: Path):
    data_dir, ex_dir = tmp_path / "data", tmp_path / "ex"
    args = ["--data-dir", str(data_dir), "--exchange-dir", str(ex_dir)]
    cli_main(args + ["dry-run"])
    manifest = json.loads((data_dir / "manifest.json").read_text())
    assert manifest["stub"] is True and manifest["counts"]["conditions"] == 16
    assert manifest["counts"]["tuples"] == {"test": 300, "dev": 60}
    assert manifest["yields"]["test"]["first_pass"] == 1.0  # stub texts pass every lint
    for rel, sha in manifest["files"].items():
        assert hashlib.sha256((data_dir / rel).read_bytes()).hexdigest() == sha, rel
    for cond, _, _ in conditions(CFG):
        for sp, n in (("test", 300), ("dev", 60)):
            cases = [json.loads(l) for l in (data_dir / "RQ4" / cond / f"{sp}.jsonl").read_text().splitlines()]
            assert len(cases) == n and Counter(c["half"] for c in cases)["state_dependent"] == n // 2
    texts = {}
    for cond, _, _ in conditions(CFG):  # one text per tuple, reused in every condition
        for l in (data_dir / "RQ4" / cond / "test.jsonl").read_text().splitlines():
            c = json.loads(l)
            assert texts.setdefault(c["case_id"], c["text"]) == c["text"]
    pool = [json.loads(l) for l in (data_dir / "pool" / "c2c3_pool.jsonl").read_text().splitlines()]
    assert len(pool) == 150
    for p in pool:
        assert p["source_condition"] == "c21_fresh" and p["actuated"]["scope"] in CLUSTERS
        assert p["actuated"]["action"] in ACTUATABLE_ACTIONS
        assert p["c2_class"] == CFG["c2_class_map"][p["truth"]["class"]]
    assert set(CFG["c2_class_map"]) == set(OPTIONS["class"]) and set(CFG["c2_class_map"].values()) == {"video", "xr", "iot", "be"}
    # the verifier saw exactly the c21_fresh telemetry of the corpus
    ex = RanExchange(exchange_dir=ex_dir, data_dir=data_dir)
    c21 = {json.loads(l)["case_id"]: json.loads(l) for l in (data_dir / "RQ4" / "c21_fresh" / "test.jsonl").read_text().splitlines()}
    it = next(i for i in ex.items.values() if i.status == "accepted" and i.split == "test")
    assert ex.verify_line(it)["telemetry"] == c21[it.tuple_id]["telemetry"]

    # frozen: re-assembly is byte-identical; a changed text is refused; stubs are refused without allow_stub
    before = {rel: (data_dir / rel).read_bytes() for rel in manifest["files"]}
    assemble_corpus(ex, allow_stub=True)
    assert all((data_dir / rel).read_bytes() == b for rel, b in before.items())
    with pytest.raises(ValueError, match="stub"):
        assemble_corpus(ex)
    it.text = it.text + " Thanks."
    with pytest.raises(ValueError, match="frozen"):
        assemble_corpus(ex, allow_stub=True)
    assert ex.export_gen() == [] and ex.export_ver() == []  # frozen splits are never exported again
    with pytest.raises(ValueError, match="frozen"):
        ex._refuse_frozen_changes("ingest_ver", [it.item_id])


def test_ingest_ver_rejects_mismatch_and_applies_diversity_caps(tmp_path: Path):
    from src.ranbench.corpus.stub import write_stub_gen_done, write_stub_ver_done

    ex = RanExchange(exchange_dir=tmp_path / "ex", data_dir=tmp_path / "data", cfg=_small_cfg())
    ex.export_gen()
    write_stub_gen_done(ex)
    assert ex.ingest_gen()["items_rejected"] == 0
    ex.export_ver(ver_batch_size=100)
    write_stub_ver_done(ex.ver_todo_dir, ex.ver_done_dir)
    done = next(ex.ver_done_dir.glob("*.jsonl"))
    lines = [json.loads(l) for l in done.read_text().splitlines()]
    sd = next(r for r in lines if ex.items[r["item_id"]].meta["half"] == "state_dependent")
    wrong = next(c for c in CLUSTERS if c != sd["labels"]["target_cluster"])
    sd["labels"]["target_cluster"] = wrong  # a verifier that misreads the table
    done.write_text("".join(json.dumps(r) + "\n" for r in lines))
    res = ex.ingest_ver(check_diversity=True)
    assert res["mismatched_fields"] == {"target_cluster": 1}
    assert ex.items[sd["item_id"]].status == "pending_gen" and ex.items[sd["item_id"]].attempt == 2
    # templated stub texts share openings and 5-grams, so the caps reject most of them
    assert res["rejections"]["lint_diversity_opening_4gram"] + res["rejections"]["lint_diversity_5gram"] > 0
    per_gen = ex.yields()["per_generator"][STUB_GENERATOR]
    assert per_gen["texts"] == 16 and per_gen["verified"] == 16 and per_gen["accepted"] == res["items_accepted"]


# ---------------------------------------------------------------- review fixes (round 1)


def test_verifier_and_interpreter_get_byte_identical_rules():
    from src.ranbench.corpus.prompts import build_interpreter_context
    from src.ranbench.schemas import READING_RULES_PATH

    rules = READING_RULES_PATH.read_bytes()
    issuer = {"id": "operator", "kind": "operator"}
    ctx = build_interpreter_context(issuer, "Boost responder video where cells are busiest.", "KPM telemetry ...\n")
    ver = build_verifier_context(CFG)
    assert ctx["rules"].encode("utf-8") == rules
    assert ver.encode("utf-8").endswith(rules)
    assert ver[len(ver) - len(ctx["rules"]):] == ctx["rules"]
    assert ctx["case"].startswith("## Issuer\n") and "## Telemetry\nKPM telemetry" in ctx["case"]


def _stub_round(tmp: Path, shuffle_ver: bool) -> tuple[RanExchange, dict]:
    import random as _random
    from src.ranbench.corpus.stub import write_stub_gen_done, write_stub_ver_done

    ex = RanExchange(exchange_dir=tmp / "ex", data_dir=tmp / "data", cfg=_small_cfg())
    ex.export_gen()
    write_stub_gen_done(ex)
    ex.ingest_gen()
    ex.export_ver(ver_batch_size=100)
    write_stub_ver_done(ex.ver_todo_dir, ex.ver_done_dir)
    if shuffle_ver:
        for done in ex.ver_done_dir.glob("*.jsonl"):
            lines = done.read_text().splitlines()
            _random.Random(7).shuffle(lines)
            lines.reverse()
            done.write_text("\n".join(lines) + "\n")
    res = ex.ingest_ver(check_diversity=True)
    return ex, res


def test_done_file_order_does_not_change_the_accepted_set(tmp_path: Path):
    ex_a, res_a = _stub_round(tmp_path / "a", shuffle_ver=False)
    ex_b, res_b = _stub_round(tmp_path / "b", shuffle_ver=True)
    order_a = [json.loads(l)["item_id"] for f in ex_a.ver_done_dir.glob("*.jsonl") for l in f.read_text().splitlines()]
    order_b = [json.loads(l)["item_id"] for f in ex_b.ver_done_dir.glob("*.jsonl") for l in f.read_text().splitlines()]
    assert sorted(order_a) == sorted(order_b) and order_a != order_b  # same items, different line order
    accepted = lambda ex: sorted(i for i, it in ex.items.items() if it.status == "accepted")
    assert 0 < len(accepted(ex_a)) < 16  # the diversity caps bind, so order could have mattered
    assert accepted(ex_a) == accepted(ex_b)
    assert res_a["rejections"] == res_b["rejections"]
    assert {i: it.to_dict() for i, it in ex_a.items.items()} == {i: it.to_dict() for i, it in ex_b.items.items()}


def test_verifier_labels_must_be_exactly_the_policy_fields(tmp_path: Path):
    from src.ranbench.corpus.stub import write_stub_gen_done, write_stub_ver_done

    ex = RanExchange(exchange_dir=tmp_path / "ex", data_dir=tmp_path / "data", cfg=_small_cfg())
    ex.export_gen()
    write_stub_gen_done(ex)
    ex.ingest_gen()
    ex.export_ver(ver_batch_size=100)
    write_stub_ver_done(ex.ver_todo_dir, ex.ver_done_dir)
    done = next(ex.ver_done_dir.glob("*.jsonl"))
    lines = sorted((json.loads(l) for l in done.read_text().splitlines()), key=lambda r: r["item_id"])
    extra, missing, bad_value = lines[0]["item_id"], lines[1]["item_id"], lines[2]["item_id"]
    lines[0]["labels"]["confidence"] = "high"  # otherwise correct labels plus an extra field
    del lines[1]["labels"]["duration"]
    lines[2]["labels"]["priority"] = "urgent"
    done.write_text("".join(json.dumps(r) + "\n" for r in lines))
    res = ex.ingest_ver(check_diversity=False)
    assert res["rejections"]["verifier_invalid_labels"] == 3
    for iid in (extra, missing, bad_value):
        assert ex.items[iid].status == "pending_gen" and ex.items[iid].reject_reasons == ["verifier_invalid_labels"]
    assert res["items_accepted"] == len(lines) - 3


def test_frozen_manifest_is_byte_identical_and_wall_clock_free(tmp_path: Path):
    from src.ranbench.corpus.cli import dry_run

    ex = RanExchange(exchange_dir=tmp_path / "ex", data_dir=tmp_path / "data", cfg=_small_cfg())
    dry_run(ex)
    manifest = (tmp_path / "data" / "manifest.json").read_bytes()
    prov = json.loads((tmp_path / "data" / "provenance.json").read_text())
    assert not {"assembled_at", "git_head", "tree_dirty"} & set(json.loads(manifest))
    assert prov["manifest_sha256"] == hashlib.sha256(manifest).hexdigest() and "assembled_at" in prov
    import time

    time.sleep(1.1)  # a later wall clock must not matter
    assemble_corpus(RanExchange(exchange_dir=tmp_path / "ex", data_dir=tmp_path / "data", cfg=_small_cfg()), allow_stub=True)
    assert (tmp_path / "data" / "manifest.json").read_bytes() == manifest
    assert json.loads((tmp_path / "data" / "provenance.json").read_text()) == prov


# ---- independent oracle over the rendered interpreter input (does not use the production reader)

import csv as _csv
from datetime import datetime as _dt
import re as _re

_ORACLE_METRIC = {"most_loaded_cells": ("prb_util_pct", max), "worst_edge_cells": ("edge_ue_thr_mbps", min)}


def _interpreter_case(t: TupleItem, n: int, q: str) -> str:
    from src.ranbench.corpus.prompts import build_interpreter_context
    from src.ranbench.corpus.spec import issuer_block

    table, _ = build_table(t, n, q, CFG)
    return build_interpreter_context(issuer_block(t.meta["issuer"], CFG), stub_text(t), render_table(table))["case"]


def _oracle_parse(case: str) -> tuple[dict, _dt, list[str], list[dict[str, str]]]:
    """Issuer, snapshot time, header lines and CSV rows, parsed from the text an interpreter receives."""
    issuer = json.loads(case.split("## Issuer\n", 1)[1].split("\n", 1)[0])
    lines = case.split("## Telemetry\n", 1)[1].strip().splitlines()
    m = _re.fullmatch(r"KPM telemetry snapshot at (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ); (\d+) cells", lines[0])
    assert m, lines[0]
    snap = _dt.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ")
    header = [l for l in lines if not l.startswith("cell-") and not l.startswith("cell_id")]
    body = [l for l in lines[1:] if l not in header]
    return issuer, snap, header, list(_csv.DictReader(body))


def _oracle_target(case: str, scope: str) -> str:
    """design.md "RQ4 truth rules", applied from scratch."""
    _, snap, _, rows = _oracle_parse(case)
    if "measured_at" in rows[0]:
        rows = [r for r in rows if (snap - _dt.strptime(r["measured_at"], "%Y-%m-%dT%H:%M:%SZ")).total_seconds() <= 10]
        newest: dict[str, dict[str, str]] = {}
        for r in rows:
            if r["cell_id"] not in newest or r["measured_at"] > newest[r["cell_id"]]["measured_at"]:
                newest[r["cell_id"]] = r
        rows = list(newest.values())
    if "source" in rows[0]:
        e2_cells = {r["cell_id"] for r in rows if r["source"] == "E2_KPM"}
        rows = [r for r in rows if r["source"] == "E2_KPM" or r["cell_id"] not in e2_cells]
    metric, pick = _ORACLE_METRIC[scope]
    best = pick(float(r[metric]) for r in rows)
    hits = [r for r in rows if float(r[metric]) == best]
    assert len(hits) == 1, "extreme must be unique"
    return hits[0]["cluster"]


@pytest.fixture(scope="module")
def sd_cases(tuples) -> list[dict]:
    """Every state-dependent (tuple, size, quality) as the rendered interpreter case, with its truth."""
    out = []
    for t in _sd(tuples):
        for n in SIZES:
            for q in QUALITIES:
                out.append({"tuple_id": t.tuple_id, "idx": t.meta["item_index"], "split": t.split, "cond": f"c{n}_{q}",
                            "scope": t.tuple_labels["scope"], "target": t.tuple_labels["target_cluster"],
                            "case": _interpreter_case(t, n, q)})
    return out


def test_independent_oracle_reproduces_truth_everywhere(sd_cases):
    assert len(sd_cases) == 180 * len(SIZES) * len(QUALITIES)
    for c in sd_cases:
        assert _oracle_target(c["case"], c["scope"]) == c["target"], (c["tuple_id"], c["cond"])


# ---- shortcut features on the rendered interpreter input (item 6)

SHORTCUT_TOL_POOLED = 0.05  # best rule, pooled over all 16 conditions (2,880 cases), vs pooled chance
SHORTCUT_TOL_CONDITION = 0.15  # best rule within one condition (180 cases), vs that condition's chance


def _shortcut_view(c: dict) -> dict:
    issuer, snap, header, rows = _oracle_parse(c["case"])
    clusters = [r["cluster"] for r in rows]
    order = list(dict.fromkeys(clusters))
    counts = Counter(clusters)
    ids: dict[str, list[int]] = {}
    for r in rows:
        ids.setdefault(r["cluster"], []).append(int(r["cell_id"].split("-")[1]))
    per_cell = Counter(r["cell_id"] for r in rows)
    repeated = {r["cluster"] for r in rows if per_cell[r["cell_id"]] > 1}
    return {"issuer": issuer["id"], "snap": snap, "header": header, "clusters": clusters, "order": order,
            "counts": counts, "ids": ids, "min_id": min(i for v in ids.values() for i in v), "repeated": repeated}


def _direct_rules() -> dict[str, dict[str, callable]]:
    """Rules that read a cluster straight off the table layout, grouped by feature family."""
    def at(p):
        return lambda v: v["clusters"][round(p * (len(v["clusters"]) - 1))]

    def kth(k):
        return lambda v: v["order"][min(k, len(v["order"]) - 1)]

    return {
        "first/last row cluster": {"first row": at(0.0), "last row": at(1.0)},
        "row position (extreme row)": {f"row at {p:.2f}": at(p) for p in (0.0, 0.25, 0.5, 0.75, 1.0)},
        "cluster order": {
            **{f"{k}-th cluster to appear": kth(k) for k in range(4)},
            "cluster with most rows": lambda v: max(v["order"], key=lambda c: (v["counts"][c], c)),
            "cluster with fewest rows": lambda v: min(v["order"], key=lambda c: (v["counts"][c], c)),
        },
        "cell-id pattern": {
            "lowest mean cell id": lambda v: min(v["ids"], key=lambda c: sum(v["ids"][c]) / len(v["ids"][c])),
            "highest mean cell id": lambda v: max(v["ids"], key=lambda c: sum(v["ids"][c]) / len(v["ids"][c])),
        },
        "extra-row pattern": {
            "first cluster without a repeated cell": lambda v: (sorted(set(v["order"]) - v["repeated"]) or sorted(v["order"]))[0],
            "first cluster with a repeated cell": lambda v: (sorted(v["repeated"]) or sorted(v["order"]))[0],
        },
    }


def _learned_features() -> dict[str, dict[str, callable]]:
    """Features mapped to a cluster by a majority table, fitted on one half of the tuples and scored on the other."""
    return {
        "issuer": {"issuer id": lambda v: v["issuer"]},
        "header": {
            "snapshot day": lambda v: v["snap"].day,
            "snapshot hour": lambda v: v["snap"].hour,
            "header lines": lambda v: len(v["header"]),
        },
        "cluster order": {"full cluster order": lambda v: tuple(v["order"])},
        "cell-id pattern": {"lowest cell id, last digit": lambda v: v["min_id"] % 10},
    }


def _cv_predictions(cases: list[dict], views: list[dict], feat) -> list[str]:
    """2-fold (tuple index parity) majority-map predictions; unseen values fall back to the fold's majority class."""
    preds = [""] * len(cases)
    for fold in (0, 1):
        train = [i for i, c in enumerate(cases) if c["idx"] % 2 != fold]
        table: dict = {}
        for i in train:
            table.setdefault(feat(views[i]), Counter())[cases[i]["target"]] += 1
        fallback = Counter(cases[i]["target"] for i in train).most_common(1)[0][0]
        for i, c in enumerate(cases):
            if c["idx"] % 2 == fold:
                cnt = table.get(feat(views[i]))
                preds[i] = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if cnt else fallback
    return preds


def test_target_cluster_is_not_predictable_from_layout_shortcuts(sd_cases):
    views = [_shortcut_view(c) for c in sd_cases]
    conds = sorted({c["cond"] for c in sd_cases}, key=lambda s: (int(s[1:].split("_")[0]), s))
    by_cond = {k: [i for i, c in enumerate(sd_cases) if c["cond"] == k] for k in conds}
    chance = {k: sum(1 / len(views[i]["order"]) for i in idx) / len(idx) for k, idx in by_cond.items()}
    chance_pooled = sum(1 / len(v["order"]) for v in views) / len(views)

    # rule name -> per-case correctness (learned rules are cross-validated within each condition)
    families: dict[str, dict[str, list[bool]]] = {}
    for fam, rules in _direct_rules().items():
        for name, rule in rules.items():
            families.setdefault(fam, {})[name] = [rule(v) == c["target"] for c, v in zip(sd_cases, views)]
    for fam, feats in _learned_features().items():
        for name, feat in feats.items():
            ok = [False] * len(sd_cases)
            for idx in by_cond.values():
                sub = [sd_cases[i] for i in idx]
                for i, p in zip(idx, _cv_predictions(sub, [views[i] for i in idx], feat)):
                    ok[i] = p == sd_cases[i]["target"]
            families.setdefault(fam, {})[f"{name} (learned, 2-fold CV)"] = ok
    oracle_ok = [_oracle_target(c["case"], c["scope"]) == c["target"] for c in sd_cases]

    report = [f"\n{'feature family':28s} {'best rule (pooled)':42s} pooled  chance  worst-cond best  its chance"]
    for fam, rules in families.items():
        name, ok = max(rules.items(), key=lambda kv: sum(kv[1]))
        pooled = sum(ok) / len(ok)
        per_cond = {k: max(sum(r[i] for i in idx) / len(idx) for r in rules.values()) for k, idx in by_cond.items()}
        worst = max(per_cond, key=lambda k: per_cond[k] - chance[k])
        report.append(f"{fam:28s} {name:42s} {pooled:.3f}   {chance_pooled:.3f}   {worst:18s} {per_cond[worst]:.3f}  {chance[worst]:.3f}")
        assert pooled <= chance_pooled + SHORTCUT_TOL_POOLED, (fam, name, pooled, chance_pooled)
        for k in conds:
            assert per_cond[k] <= chance[k] + SHORTCUT_TOL_CONDITION, (fam, k, per_cond[k], chance[k])
    report.append(f"{'independent oracle (item 5)':28s} {'design.md truth rules':42s} {sum(oracle_ok) / len(oracle_ok):.3f}")
    print("\n".join(report))
    assert all(oracle_ok)
