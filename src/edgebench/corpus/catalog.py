"""Catalog building and management for EdgeIntent v1.

Builds services.json (254 services + 40 novel services) and
catalogs.json (nested catalogs C_4, C_15, C_64, C_128, C_254 and churn v0, v25, v50).
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import re
from typing import Any

from src.edgebench.corpus.catalog_data import FAMILIES, RAW_NOVEL_SERVICES, RAW_SERVICES

CATALOG_SEED = 20260924

UNSUPPORTED_EXCLUDED_KEYWORDS: list[str] = [
    "count",
    "counting",
    "tally",
    "detect",
    "detecting",
    "detection",
    "detector",
    "localize",
    "localizing",
    "localization",
    "bounding box",
    "ocr",
    "read text",
    "reading text",
    "license plate",
    "character recognition",
    "text reading",
]


# Catalog-aware exclusions: an out-of-catalog service is not "unsupported" while an active service covers it.
# Document field extraction is text reading (ocr); these two screeners find objects in images (detection).
DETECTION_COVERED_TARGETS: frozenset[str] = frozenset({"under_vehicle_inspection_scanner", "package_xray_contraband_screener"})
# Pool id -> active near-duplicates that cover it
NEAR_DUPLICATE_COVERERS: dict[str, frozenset[str]] = {
    "handwritten_survey_digitizer": frozenset({"handwritten_form_ocr"}),
    "loitering_at_atm_detector": frozenset({"loitering_detector"}),
    "drone_acoustic_classifier": frozenset({"drone_perimeter_alert"}),
    "identity_card_extractor": frozenset({"visitor_id_scanner", "passport_mrz_ocr"}),
    "customer_dwell_time_meter": frozenset({"endcap_promotional_gaze_tracker"}),
    "speeding_vehicle_detector": frozenset({"school_zone_speed_monitor"}),
    "h265_video_transcoder": frozenset({"h264_bitrate_downscaler"}),
}


def is_valid_unsupported_target(
    service: dict[str, Any], active_catalog_ids: list[str] | set[str] | None = None
) -> bool:
    """Check if a service is valid as an unsupported target.

    Unsupported targets must be outside the scope of every active service including generic ones:
    must not be counting, generic detection/localisation, or text reading tasks.
    With active_catalog_ids, also reject a service that an active service covers: document_processing
    while ocr is active, the DETECTION_COVERED_TARGETS while detection is active, and a
    NEAR_DUPLICATE_COVERERS key while one of its near-duplicates is active.
    """
    fam = service.get("family", "")
    if fam in ("object_counting", "object_detection", "text_reading"):
        return False
    if active_catalog_ids is not None:
        active = set(active_catalog_ids)
        sid = service.get("id", "")
        if fam == "document_processing" and "ocr" in active:
            return False
        if sid in DETECTION_COVERED_TARGETS and "detection" in active:
            return False
        if active & NEAR_DUPLICATE_COVERERS.get(sid, frozenset()):
            return False
    combined_text = f"{service.get('id', '')} {service.get('name', '')} {service.get('description', '')}".lower()
    for kw in UNSUPPORTED_EXCLUDED_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", combined_text):
            return False
    return True


def compute_token_jaccard(name1: str, name2: str) -> float:
    t1 = set(name1.lower().split())
    t2 = set(name2.lower().split())
    if not t1 and not t2:
        return 1.0
    return len(t1 & t2) / len(t1 | t2)


def get_services_catalog() -> dict[str, Any]:
    return {
        "services": list(RAW_SERVICES),
        "novel_services": list(RAW_NOVEL_SERVICES),
        "families": list(FAMILIES),
    }


def build_nested_and_churn_catalogs(seed: int = CATALOG_SEED) -> dict[str, list[str]]:
    rng = random.Random(seed)
    services_by_family: dict[str, list[str]] = {f: [] for f in FAMILIES}
    for s in RAW_SERVICES:
        services_by_family[s["family"]].append(s["id"])

    # Base 3 services
    c4 = ["count", "detection", "ocr"]
    used_families = {"object_counting", "object_detection", "text_reading"}
    remaining_fams = [f for f in FAMILIES if f not in used_families]
    remaining_fams.sort()
    rng.shuffle(remaining_fams)
    f4 = remaining_fams[0]
    s4_candidates = sorted(services_by_family[f4])
    rng.shuffle(s4_candidates)
    c4.append(s4_candidates[0])

    # C_15 (15 services across 15 families)
    c15 = list(c4)
    c15_fams = set(used_families) | {f4}
    remaining_12_fams = [f for f in FAMILIES if f not in c15_fams]
    remaining_12_fams.sort()
    rng.shuffle(remaining_12_fams)
    chosen_11_fams = remaining_12_fams[:11]
    for f in chosen_11_fams:
        candidates = sorted(services_by_family[f])
        rng.shuffle(candidates)
        c15.append(candidates[0])

    # C_64 (4 per family across all 16 families)
    c64 = list(c15)
    for f in FAMILIES:
        current_in_f = [sid for sid in c15 if sid in services_by_family[f]]
        needed = 4 - len(current_in_f)
        candidates = [sid for sid in services_by_family[f] if sid not in current_in_f]
        candidates.sort()
        rng.shuffle(candidates)
        c64.extend(candidates[:needed])

    # C_128 (8 per family across all 16 families)
    c128 = list(c64)
    for f in FAMILIES:
        current_in_f = [sid for sid in c64 if sid in services_by_family[f]]
        needed = 8 - len(current_in_f)
        candidates = [sid for sid in services_by_family[f] if sid not in current_in_f]
        candidates.sort()
        rng.shuffle(candidates)
        c128.extend(candidates[:needed])

    # C_254 (all 254 services)
    c254 = [s["id"] for s in RAW_SERVICES]

    # Churn versions of C_64
    v0 = list(c64)
    churn_rng = random.Random(seed)
    fams_sorted = sorted(FAMILIES)
    churn_rng.shuffle(fams_sorted)
    v25_fams = fams_sorted[:4]

    c64_set = set(c64)
    v25 = list(v0)
    for f in v25_fams:
        in_v0 = [sid for sid in services_by_family[f] if sid in c64_set]
        out_of_c64 = [sid for sid in services_by_family[f] if sid not in c64_set]
        out_of_c64.sort()
        churn_rng.shuffle(out_of_c64)
        replacements = out_of_c64[:4]
        for old_sid, new_sid in zip(in_v0, replacements):
            idx = v25.index(old_sid)
            v25[idx] = new_sid

    v50 = list(v0)
    churn_rng50 = random.Random(seed)
    churn_rng50.shuffle(fams_sorted)
    v50_fams = fams_sorted[:8]
    for f in v50_fams:
        in_v0 = [sid for sid in services_by_family[f] if sid in c64_set]
        out_of_c64 = [sid for sid in services_by_family[f] if sid not in c64_set]
        out_of_c64.sort()
        churn_rng50.shuffle(out_of_c64)
        replacements = out_of_c64[:4]
        for old_sid, new_sid in zip(in_v0, replacements):
            idx = v50.index(old_sid)
            v50[idx] = new_sid

    catalogs = {
        "C_4": c4,
        "C_15": c15,
        "C_64": c64,
        "C_128": c128,
        "C_254": c254,
        "v0": v0,
        "v25": v25,
        "v50": v50,
    }
    return catalogs


def verify_catalog_invariants(
    services_data: dict[str, Any], catalogs_data: dict[str, list[str]]
) -> None:
    services = services_data["services"]
    novel = services_data["novel_services"]
    assert len(services) == 254, f"Expected 254 services, got {len(services)}"
    assert len(novel) == 40, f"Expected 40 novel services, got {len(novel)}"

    all_svcs = services + novel
    ids = [s["id"] for s in all_svcs]
    assert len(ids) == len(set(ids)), f"Duplicate IDs detected: {len(ids)} vs {len(set(ids))}"

    for s in all_svcs:
        words = len(s["description"].split())
        assert words <= 20, f"Description exceeds 20 words ({words}) in {s['id']}"

    # Base fallback services check
    svc_map = {s["id"]: s for s in services}
    assert (
        svc_map["count"]["description"]
        == "general-purpose object counting fallback: count objects and return only a number when no specialized counting service matches"
    )
    assert (
        svc_map["detection"]["description"]
        == "general-purpose object detection fallback: detect and localize objects with bounding boxes when no specialized detection service matches"
    )
    assert (
        svc_map["ocr"]["description"]
        == "general-purpose optical character recognition fallback: read text from an image when no specialized text reading service matches"
    )

    # Token Jaccard check
    for i in range(len(all_svcs)):
        for j in range(i + 1, len(all_svcs)):
            jacc = compute_token_jaccard(all_svcs[i]["name"], all_svcs[j]["name"])
            assert jacc < 0.8, (
                f"Near duplicate names (Jaccard={jacc:.2f}): "
                f"{all_svcs[i]['name']} and {all_svcs[j]['name']}"
            )

    # Catalogs check
    c4 = catalogs_data["C_4"]
    c15 = catalogs_data["C_15"]
    c64 = catalogs_data["C_64"]
    c128 = catalogs_data["C_128"]
    c254 = catalogs_data["C_254"]
    v0 = catalogs_data["v0"]
    v25 = catalogs_data["v25"]
    v50 = catalogs_data["v50"]

    assert len(c4) == 4
    assert len(c15) == 15
    assert len(c64) == 64
    assert len(c128) == 128
    assert len(c254) == 254
    assert len(v0) == 64
    assert len(v25) == 64
    assert len(v50) == 64

    # Subsets
    assert set(c4).issubset(set(c15))
    assert set(c15).issubset(set(c64))
    assert set(c64).issubset(set(c128))
    assert set(c128).issubset(set(c254))
    assert v0 == c64

    # Churn differences exact
    assert len(set(v0) - set(v25)) == 16
    assert len(set(v25) - set(v0)) == 16
    assert len(set(v0) - set(v50)) == 32
    assert len(set(v50) - set(v0)) == 32

    # All churn services exist in C_254
    assert set(v25).issubset(set(c254))
    assert set(v50).issubset(set(c254))


def save_catalogs(
    catalog_dir: Path | str, seed: int = CATALOG_SEED
) -> tuple[Path, Path]:
    out_dir = Path(catalog_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    services_data = get_services_catalog()
    catalogs_data = build_nested_and_churn_catalogs(seed=seed)

    verify_catalog_invariants(services_data, catalogs_data)

    services_file = out_dir / "services.json"
    catalogs_file = out_dir / "catalogs.json"

    with open(services_file, "w", encoding="utf-8") as f:
        json.dump(services_data, f, indent=2)

    with open(catalogs_file, "w", encoding="utf-8") as f:
        json.dump(catalogs_data, f, indent=2)

    return services_file, catalogs_file


def load_services(catalog_dir: Path | str) -> dict[str, Any]:
    services_file = Path(catalog_dir) / "services.json"
    with open(services_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_catalogs(catalog_dir: Path | str) -> dict[str, list[str]]:
    catalogs_file = Path(catalog_dir) / "catalogs.json"
    with open(catalogs_file, "r", encoding="utf-8") as f:
        return json.load(f)
