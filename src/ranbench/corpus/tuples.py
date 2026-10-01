"""Deterministic sampling of RANIntent v1 truth tuples.

One tuple set per split, shared by all 16 RQ4 conditions (the intent text is generated once per tuple; only the
telemetry table changes per condition). Each split = a state-dependent half (scope most_loaded_cells /
worst_edge_cells, answer in target_cluster) and a named-scope actuatable half (action prioritise / deprioritise /
revert_default, scope a named cluster).
"""
from __future__ import annotations

import itertools
import random
from typing import Any

from src.edgebench.corpus.tuples import TupleItem, _stable_seed, sample_balanced_sequence
from src.ranbench.corpus.spec import (
    ACTION_PARAM,
    ACTUATABLE_ACTIONS,
    CLUSTERS,
    CORPUS_CONDITION,
    FIELDS,
    OPTIONS,
    STATE_DEPENDENT_SCOPES,
    issuer_block,
)


def _balanced(values: list[str], n: int, key: str, seed: int) -> list[str]:
    return sample_balanced_sequence(values, n, _stable_seed(key, seed)) if n else []


def _pick_issuer(cls: str, priority: str, rng: random.Random, cfg: dict[str, Any]) -> str:
    """Operator with probability operator_share, else a tenant that owns the class and holds the needed right."""
    tenants = [
        t for t in sorted(cfg["issuers"]["tenants"])
        if cls in issuer_block(t, cfg)["owned_classes"]
        and (priority != "critical" or issuer_block(t, cfg)["may_set_critical"])
    ]
    if not tenants or rng.random() < cfg["issuers"]["operator_share"]:
        return "operator"
    return rng.choice(tenants)


def _sample_half(half: str, split: str, n: int, cfg: dict[str, Any]) -> list[tuple[dict[str, str], str]]:
    """(labels, issuer) for n tuples of one half, with balanced marginals per field."""
    seed = cfg["base_seed"]
    key = f"{split}:{half}"
    if half == "state_dependent":
        actions = [a for a in OPTIONS["action"] if a != "unsupported"]
        combos = [f"{s}|{c}" for s, c in itertools.product(STATE_DEPENDENT_SCOPES, CLUSTERS)]
        scope_target = [c.split("|") for c in _balanced(combos, n, f"{key}:scope_target", seed)]
    else:
        actions = list(ACTUATABLE_ACTIONS)
        scope_target = [[c, "none"] for c in _balanced(CLUSTERS, n, f"{key}:scope", seed)]
    action_seq = _balanced(actions, n, f"{key}:action", seed)
    class_seq = _balanced(list(OPTIONS["class"]), n, f"{key}:class", seed)

    labels = [
        {f: "unspecified" for f in FIELDS} | {"action": a, "class": c, "scope": st[0], "target_cluster": st[1]}
        for a, c, st in zip(action_seq, class_seq, scope_target)
    ]
    # Each action carries its own parameter, balanced within the action; revert_default carries none
    for action in actions:
        idx = [i for i, lab in enumerate(labels) if lab["action"] == action]
        param = ACTION_PARAM[action]
        if param is not None:
            for i, v in zip(idx, _balanced(param[1], len(idx), f"{key}:{action}:{param[0]}", seed)):
                labels[i][param[0]] = v
    dur_idx = [i for i, lab in enumerate(labels) if lab["action"] != "revert_default"]
    for i, v in zip(dur_idx, _balanced(list(OPTIONS["duration"]), len(dur_idx), f"{key}:duration", seed)):
        labels[i]["duration"] = v

    out = []
    for i, lab in enumerate(labels):
        rng = random.Random(_stable_seed(f"{key}:{i}:issuer", seed))
        out.append((lab, _pick_issuer(lab["class"], lab["priority"], rng, cfg)))
    return out


def sample_tuples(split: str, cfg: dict[str, Any]) -> list[TupleItem]:
    """All tuples of a split (test: 150 + 150, dev: 30 + 30), halves interleaved by a seeded shuffle."""
    sizes = cfg["splits"][split]
    rows: list[tuple[str, dict[str, str], str]] = []
    for half in ("state_dependent", "named_scope"):
        rows.extend((half, lab, iss) for lab, iss in _sample_half(half, split, sizes[half], cfg))
    random.Random(_stable_seed(f"{split}:interleave", cfg["base_seed"])).shuffle(rows)

    families = cfg["text"]["wording_families"]
    items: list[TupleItem] = []
    for i, (half, lab, iss) in enumerate(rows):
        items.append(
            TupleItem(
                condition=CORPUS_CONDITION,
                split=split,
                tuple_id=f"{CORPUS_CONDITION}_{split}_{i:04d}",
                tuple_labels=lab,
                meta={
                    "half": half,
                    "issuer": iss,
                    "wording_family": families[i % len(families)],
                    "item_index": i,
                },
            )
        )
    return items


def truth_record(item: TupleItem) -> dict[str, str]:
    """The truth record validated by the truth schema: nine fields + issuer id."""
    return dict(item.tuple_labels) | {"issuer": item.meta["issuer"]}
