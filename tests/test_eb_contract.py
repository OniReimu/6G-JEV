"""Tests for Edgebench contract v2."""
from __future__ import annotations

import pytest

from src.edgebench.contract import (
    CORE_FIELDS,
    DEFAULT_CRITERIA,
    FIELD_SETS,
    MAX_SERVICE_OPTIONS,
    Case,
    case_from_dict,
    case_to_dict,
    validate_service_options,
)


def test_field_sets():
    assert len(FIELD_SETS[4]) == 4
    assert len(FIELD_SETS[6]) == 6
    assert len(FIELD_SETS[8]) == 8
    assert FIELD_SETS[4] == CORE_FIELDS
    assert set(FIELD_SETS[6]) == set(CORE_FIELDS + ["retention", "energy"])
    assert set(FIELD_SETS[8]) == set(
        CORE_FIELDS + ["retention", "energy", "redundancy", "latency_class"]
    )


def test_unspecified_always_present():
    for field, options in DEFAULT_CRITERIA.items():
        if field == "service_type":
            assert "unsupported" in options
        else:
            assert "unspecified" in options, f"Field {field} missing unspecified option"


def test_service_options_cap():
    # Valid catalog with 255 options
    valid_opts = {f"svc_{i}": f"Service description {i}" for i in range(254)}
    valid_opts["unsupported"] = "unsupported service"
    assert len(valid_opts) == 255
    validate_service_options(valid_opts)

    # Exceeding 255 options -> must raise ValueError
    invalid_too_large = {f"svc_{i}": f"Service {i}" for i in range(255)}
    invalid_too_large["unsupported"] = "unsupported service"
    assert len(invalid_too_large) == 256
    with pytest.raises(ValueError, match="cannot exceed 255 options"):
        validate_service_options(invalid_too_large)

    # Missing 'unsupported' -> must raise ValueError
    missing_unsupp = {f"svc_{i}": f"Service {i}" for i in range(10)}
    with pytest.raises(ValueError, match="must contain 'unsupported'"):
        validate_service_options(missing_unsupp)


def test_case_roundtrip():
    case = Case(
        case_id="case_001",
        text="1. Process image locally. 2. Run OCR standard.",
        fields=FIELD_SETS[4],
        bundle_size=2,
        truth=[
            {"service_type": "detection", "locality": "site_only", "quality_floor": "standard", "urgency": "normal"},
            {"service_type": "ocr", "locality": "unspecified", "quality_floor": "standard", "urgency": "unspecified"},
        ],
        meta={"seen": True, "category": "smoke"},
    )
    d = case_to_dict(case)
    loaded = case_from_dict(d)
    assert loaded.case_id == case.case_id
    assert loaded.bundle_size == 2
    assert loaded.truth == case.truth
    assert loaded.meta == case.meta
