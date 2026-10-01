"""Sampling label tuples per condition with balanced marginals and disjoint seeds."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import random
from typing import Any

from src.edgebench.contract import DEFAULT_CRITERIA, FIELD_SETS
from src.edgebench.corpus.catalog import is_valid_unsupported_target
from src.edgebench.corpus.catalog_data import RAW_NOVEL_SERVICES, RAW_SERVICES

ALL_SERVICES_MAP: dict[str, dict[str, Any]] = {
    s["id"]: s for s in RAW_SERVICES + RAW_NOVEL_SERVICES
}

WORDING_FAMILIES: list[str] = [
    "operator ticket",
    "casual chat",
    "formal SLA clause",
    "IoT/app notification",
    "voice-assistant utterance",
    "email",
]

EDGE_SCENARIOS: list[str] = [
    "hospital ward patient monitor",
    "port container crane camera",
    "agricultural farm drone",
    "retail shelf scanner",
    "substation thermal camera",
    "school bus fleet telemetry",
    "warehouse automated guided vehicle",
    "stadium turnstile access gate",
    "wind turbine blade inspection drone",
    "parcel locker kiosk",
    "tunnel ventilation sensor",
    "ferry terminal passenger gangway",
    "mining haul truck telematics",
    "urban traffic intersection camera",
    "railway track defect scanner",
    "offshore oil platform flare monitor",
    "commercial greenhouse climate controller",
    "construction site safety helmet scanner",
    "airport baggage handling carousel",
    "aquaculture fish pen water sensor",
    "cold chain refrigerated delivery van",
    "solar farm photovoltaic inverter",
    "automotive assembly robotic arm",
    "dockside cargo container spreader",
    "mountain ski lift ticket gate",
    "dairy cattle robotic milking stall",
    "smart city street lighting pole",
    "cargo vessel engine room telemetry",
    "municipal trash compactor bin",
    "wastewater treatment pump station",
    "highway electronic tolling gantry",
    "brewery fermentation tank monitor",
    "sawmill timber log inspection camera",
    "hospital surgical suite air sensor",
    "metro platform automated screen door",
    "vineyard microclimate weather station",
    "customs border checkpoint scanner",
    "wildfire watchtower thermal camera",
    "suspension bridge structural strain gauge",
    "petrochemical pipeline pressure sensor",
    "electric vehicle charging hub",
    "airport runway debris detection camera",
    "pharmaceutical cleanroom particle counter",
    "forestry logging harvester vehicle",
    "zoo animal habitat camera",
]
RQ4_CONDITIONS: set[str] = {
    "K4",
    "K15",
    "K64",
    "K128",
    "K254",
    "churn25",
    "churn50",
}


def is_rq4_condition(condition: str) -> bool:
    """Return True if condition belongs to RQ4."""
    norm = condition.split("/")[-1]
    if norm.startswith("RQ4_"):
        norm = norm[4:]
    return norm in RQ4_CONDITIONS


CODESWITCH_LANGS: list[str] = ["zh", "es", "fr", "de"]

DEFAULT_BASE_SEED = 20260924


@dataclass
class TupleItem:
    condition: str
    split: str
    tuple_id: str
    tuple_labels: dict[str, str]  # ground truth labels
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "split": self.split,
            "tuple_id": self.tuple_id,
            "tuple_labels": dict(self.tuple_labels),
            "meta": dict(self.meta),
        }


def _stable_seed(key: str, base_seed: int) -> int:
    h = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def sample_balanced_sequence(values: list[str], n: int, seed: int) -> list[str]:
    """Sample a sequence of length n where every value appears floor(n/k) or ceil(n/k) times."""
    if not values:
        raise ValueError("Cannot sample from empty values list")
    k = len(values)
    base = n // k
    rem = n % k

    rng = random.Random(seed)
    val_indices = list(range(k))
    rng.shuffle(val_indices)
    chosen_for_rem = set(val_indices[:rem])

    seq: list[str] = []
    for i, v in enumerate(values):
        count = base + (1 if i in chosen_for_rem else 0)
        seq.extend([v] * count)

    rng.shuffle(seq)
    assert len(seq) == n, f"Expected {n} items, got {len(seq)}"
    return seq


def rq4_active_catalog_and_unsupported_pool(
    condition: str,
    catalogs: dict[str, list[str]],
    novel_service_ids: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Active catalog ids and the unsupported-target pool of an RQ4 condition.

    The pool is C_254 minus the active catalog (the novel services for K254), filtered by the
    catalog-aware is_valid_unsupported_target.
    """
    norm_cond = condition.split("/")[-1]
    if norm_cond in ("K4", "RQ4_K4"):
        active_catalog_ids = catalogs["C_4"]
        out_catalog_pool = [s for s in catalogs["C_254"] if s not in active_catalog_ids]
    elif norm_cond in ("K15", "RQ4_K15"):
        active_catalog_ids = catalogs["C_15"]
        out_catalog_pool = [s for s in catalogs["C_254"] if s not in active_catalog_ids]
    elif norm_cond in ("K64", "RQ4_K64", "churn0", "RQ4_churn0"):
        active_catalog_ids = catalogs["C_64"]
        out_catalog_pool = [s for s in catalogs["C_254"] if s not in active_catalog_ids]
    elif norm_cond in ("K128", "RQ4_K128"):
        active_catalog_ids = catalogs["C_128"]
        out_catalog_pool = [s for s in catalogs["C_254"] if s not in active_catalog_ids]
    elif norm_cond in ("K254", "RQ4_K254"):
        active_catalog_ids = catalogs["C_254"]
        if not novel_service_ids:
            raise ValueError("novel_service_ids required for K254 unsupported sampling")
        out_catalog_pool = list(novel_service_ids)
    elif norm_cond in ("churn25", "RQ4_churn25"):
        active_catalog_ids = catalogs["v25"]
        out_catalog_pool = [s for s in catalogs["C_254"] if s not in active_catalog_ids]
    elif norm_cond in ("churn50", "RQ4_churn50"):
        active_catalog_ids = catalogs["v50"]
        out_catalog_pool = [s for s in catalogs["C_254"] if s not in active_catalog_ids]
    else:
        raise ValueError(f"Unknown RQ4 condition: {condition}")

    # Enforce unsupported target pre-filter: cannot be counting, detection, or ocr, nor covered by an active service
    out_catalog_pool = [
        s for s in out_catalog_pool
        if is_valid_unsupported_target(ALL_SERVICES_MAP.get(s, {}), active_catalog_ids)
    ]
    return active_catalog_ids, out_catalog_pool


def sample_tuples_for_condition(
    condition: str,
    split: str,
    n: int,
    base_seed: int = DEFAULT_BASE_SEED,
    catalogs: dict[str, list[str]] | None = None,
    novel_service_ids: list[str] | None = None,
) -> list[TupleItem]:
    """Sample n label tuples for a given condition and split with balanced marginals."""
    split_seed_offset = 0 if split == "test" else 777779
    eff_base_seed = base_seed + split_seed_offset

    # Normalize condition name
    norm_cond = condition.split("/")[-1]

    # Determine contract fields
    if norm_cond.startswith("RQ3_F4") or norm_cond in ("F4_low", "F4_medium", "F4_high"):
        fields = list(FIELD_SETS[4])
    elif norm_cond.startswith("RQ3_F6") or norm_cond in ("F6_low", "F6_medium", "F6_high"):
        fields = list(FIELD_SETS[6])
    elif norm_cond.startswith("RQ3_F8") or norm_cond in ("F8_low", "F8_medium", "F8_high"):
        fields = list(FIELD_SETS[8])
    else:
        fields = list(FIELD_SETS[4])

    # Check if this is RQ4
    is_rq4 = is_rq4_condition(condition)

    # Check if this is RQ2
    cond_stem = norm_cond.removeprefix("RQ2_")
    is_rq2 = cond_stem in (
        "clean",
        "colloquial",
        "negation",
        "codeswitch",
        "defaultbait",
        "revised",
        "keyvalue",
        "noise",
    )

    # For RQ2, all generated conditions share "RQ2_shared"
    # For RQ3, all complexity levels (low, medium, high) share the same generator seed at a given F
    tuple_seed_key = norm_cond
    if is_rq2:
        tuple_seed_key = "RQ2_shared"
    else:
        for prefix in ("F4_", "F6_", "F8_"):
            if norm_cond.startswith(prefix):
                tuple_seed_key = f"{prefix}shared"
                break

    # Sample non-service fields
    field_sequences: dict[str, list[str]] = {}
    if is_rq2:
        # Constrain the shared RQ2 tuple set so every tuple has >= 1 non-service field 'unspecified'
        non_service_fields = [f for f in fields if f != "service_type"]
        valid_combos = [
            combo
            for combo in itertools.product(*(list(DEFAULT_CRITERIA[f].keys()) for f in non_service_fields))
            if any(v == "unspecified" for v in combo)
        ]
        combo_seed = _stable_seed(f"{tuple_seed_key}:{split}:combos", eff_base_seed)
        combo_indices = sample_balanced_sequence(list(range(len(valid_combos))), n, combo_seed)
        for idx, f in enumerate(non_service_fields):
            field_sequences[f] = [valid_combos[c_i][idx] for c_i in combo_indices]
    else:
        for f in fields:
            if f == "service_type":
                continue
            # Available options from contract
            options = list(DEFAULT_CRITERIA[f].keys())
            f_seed = _stable_seed(f"{tuple_seed_key}:{split}:{f}", eff_base_seed)
            field_sequences[f] = sample_balanced_sequence(options, n, f_seed)

    # Sample service_type
    service_truth_seq: list[str] = []
    target_service_seq: list[str] = []
    is_unsupported_seq: list[bool] = []
    seen_seq: list[bool | None] = []

    if not is_rq4:
        # Default catalog: base 3 services
        base_services = ["count", "detection", "ocr"]
        svc_seed = _stable_seed(f"{tuple_seed_key}:{split}:service_type", eff_base_seed)
        service_truth_seq = sample_balanced_sequence(base_services, n, svc_seed)
        target_service_seq = list(service_truth_seq)
        is_unsupported_seq = [False] * n
        seen_seq = [True] * n
    else:
        if catalogs is None:
            raise ValueError("catalogs required for RQ4 tuple sampling")

        active_catalog_ids, out_catalog_pool = rq4_active_catalog_and_unsupported_pool(
            condition, catalogs, novel_service_ids
        )

        # 10% unsupported targets
        n_unsupported = max(1, round(n * 0.10)) if n >= 10 else 1
        n_supported = n - n_unsupported

        # Sample supported services from active catalog with balanced marginals
        supp_seed = _stable_seed(f"{condition}:{split}:supported_services", eff_base_seed)
        supp_services = sample_balanced_sequence(active_catalog_ids, n_supported, supp_seed)

        # Sample unsupported targets from out-of-catalog pool with balanced marginals
        unsupp_seed = _stable_seed(f"{condition}:{split}:unsupported_services", eff_base_seed)
        unsupp_services = sample_balanced_sequence(out_catalog_pool, n_unsupported, unsupp_seed)

        # Interleave supported and unsupported deterministically
        interleave_seed = _stable_seed(f"{condition}:{split}:interleave", eff_base_seed)
        rng_intl = random.Random(interleave_seed)
        unsupp_positions = set(rng_intl.sample(range(n), n_unsupported))

        v0_set = set(catalogs["v0"])
        idx_supp = 0
        idx_unsupp = 0
        for i in range(n):
            if i in unsupp_positions:
                target_svc = unsupp_services[idx_unsupp]
                idx_unsupp += 1
                service_truth_seq.append("unsupported")
                target_service_seq.append(target_svc)
                is_unsupported_seq.append(True)
                seen_seq.append(None)
            else:
                target_svc = supp_services[idx_supp]
                idx_supp += 1
                service_truth_seq.append(target_svc)
                target_service_seq.append(target_svc)
                is_unsupported_seq.append(False)
                seen_seq.append(target_svc in v0_set)

    # Sample edge scenarios: balanced for non-RQ4, generator-chosen for RQ4
    if not is_rq4:
        scenario_seed = _stable_seed(f"{tuple_seed_key}:{split}:scenario", eff_base_seed)
        scenario_seq = sample_balanced_sequence(EDGE_SCENARIOS, n, scenario_seed)
    else:
        scenario_seq = ["generator-chosen"] * n

    is_high_or_revised = (
        norm_cond.endswith("_high")
        or norm_cond in ("high", "revised", "RQ2_revised", "F4_high", "F6_high", "F8_high")
    )

    # Assemble tuples
    items: list[TupleItem] = []
    for i in range(n):
        tuple_id = f"{condition}_{split}_{i:04d}"
        labels: dict[str, str] = {
            "service_type": service_truth_seq[i],
        }
        for f in fields:
            if f != "service_type":
                labels[f] = field_sequences[f][i]

        meta: dict[str, Any] = {
            "wording_family": WORDING_FAMILIES[i % len(WORDING_FAMILIES)],
            "scenario": scenario_seq[i],
            "target_service": target_service_seq[i],
            "is_unsupported": is_unsupported_seq[i],
            "seen": seen_seq[i],
            "item_index": i,
        }
        if "codeswitch" in condition:
            meta["language"] = CODESWITCH_LANGS[i % len(CODESWITCH_LANGS)]

        # For high or revised conditions, choose one specified field and an initial value != target
        if is_high_or_revised:
            specified_fields = [f for f, v in labels.items() if v != "unspecified"]
            if specified_fields:
                rev_seed = _stable_seed(f"{tuple_id}:revision", eff_base_seed)
                rng_rev = random.Random(rev_seed)
                rev_field = rng_rev.choice(specified_fields)
                target_val = labels[rev_field]
                if rev_field == "service_type":
                    if not is_rq4:
                        options_pool = ["count", "detection", "ocr"]
                    else:
                        options_pool = list(active_catalog_ids)
                    cand_inits = [v for v in options_pool if v != target_val and v != "unspecified"]
                else:
                    cand_inits = [
                        v for v in DEFAULT_CRITERIA[rev_field].keys()
                        if v != target_val and v != "unspecified"
                    ]
                if cand_inits:
                    init_val = rng_rev.choice(cand_inits)
                    meta["revision_field"] = rev_field
                    meta["revision_initial_value"] = init_val

        items.append(
            TupleItem(
                condition=condition,
                split=split,
                tuple_id=tuple_id,
                tuple_labels=labels,
                meta=meta,
            )
        )

    return items
