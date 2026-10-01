"""Tests for decisions adapter (Jev and System-One APIs)."""
from __future__ import annotations

import json
from unittest.mock import patch

from src.edgebench.contract import Case, FIELD_SETS
from src.edgebench.interpreters.decisions import DecisionsClient


def test_decisions_request_k1():
    client = DecisionsClient(
        name="Jev-1.13.0",
        model="typesafe/jev-1.13",
        accepted_resolved_models={"typesafe/jev-1.13", "typesafe/jev-1.13-20260917"},
    )
    case = Case(
        case_id="c1",
        text="Count the cars in the driveway, keep image on site.",
        fields=FIELD_SETS[4],
        bundle_size=1,
        truth=[{"service_type": "count", "locality": "site_only", "quality_floor": "unspecified", "urgency": "unspecified"}],
    )
    req = client.build_request(case)
    assert req["model"] == "typesafe/jev-1.13"
    assert req["state"] == case.text
    assert req["provider"] == {"allow_fallbacks": False}
    questions = req["questions"]
    assert len(questions) == 4
    for f in FIELD_SETS[4]:
        qid = f"r1__{f}"
        assert qid in questions
        assert questions[qid]["type"] == "choice"
        # k=1 should NOT mention "for request number"
        assert "for request number" not in questions[qid]["instructions"]


def test_decisions_request_k3():
    client = DecisionsClient(name="Jev-1.13.0")
    case = Case(
        case_id="c3",
        text="1. Count cars.\n2. OCR receipt.\n3. Detect pedestrians urgently.",
        fields=FIELD_SETS[4],
        bundle_size=3,
        truth=[
            {"service_type": "count", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"},
            {"service_type": "ocr", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"},
            {"service_type": "detection", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "urgent"},
        ],
    )
    req = client.build_request(case)
    questions = req["questions"]
    assert len(questions) == 12  # 3 requests * 4 fields
    for req_idx in (1, 2, 3):
        for f in FIELD_SETS[4]:
            qid = f"r{req_idx}__{f}"
            assert qid in questions
            assert f"for request number {req_idx} in the text" in questions[qid]["instructions"]


def test_decisions_request_254_catalog():
    client = DecisionsClient(name="Jev-1.13.0")
    catalog = {f"svc_{i}": f"Custom service {i}" for i in range(254)}
    catalog["unsupported"] = "unsupported service"
    case = Case(
        case_id="c_cat",
        text="Run svc_42 with standard quality.",
        fields=FIELD_SETS[4],
        service_options=catalog,
        bundle_size=1,
        truth=[{"service_type": "svc_42", "locality": "unspecified", "quality_floor": "standard", "urgency": "unspecified"}],
    )
    req = client.build_request(case)
    q_svc = req["questions"]["r1__service_type"]
    assert len(q_svc["criteria"]) == 255
    assert "svc_42" in q_svc["criteria"]
    assert "unsupported" in q_svc["criteria"]


def test_decisions_parse_canned_jev_response():
    client = DecisionsClient(
        name="Jev-1.13.0",
        model="typesafe/jev-1.13",
        accepted_resolved_models={"typesafe/jev-1.13", "typesafe/jev-1.13-20260917"},
    )
    case = Case(
        case_id="c1",
        text="Please read text urgently.",
        fields=FIELD_SETS[4],
        bundle_size=1,
        truth=[{"service_type": "ocr", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "urgent"}],
    )

    canned_payload = {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {
            "r1__service_type": {
                "type": "choice",
                "choice": "ocr",
                "probabilities": {"ocr": 0.96, "count": 0.02, "detection": 0.01, "unsupported": 0.01},
                "confidence": 0.94,
            },
            "r1__locality": {
                "type": "choice",
                "choice": "unspecified",
                "probabilities": {"site_only": 0.05, "remote_allowed": 0.05, "unspecified": 0.90},
                "confidence": 0.90,
            },
            "r1__quality_floor": {
                "type": "choice",
                "choice": "unspecified",
                "probabilities": {"standard": 0.05, "high": 0.05, "unspecified": 0.90},
                "confidence": 0.90,
            },
            "r1__urgency": {
                "type": "choice",
                "choice": "urgent",
                "probabilities": {"normal": 0.02, "urgent": 0.97, "unspecified": 0.01},
                "confidence": 0.97,
            },
        },
        "usage": {
            "input_tokens": 342,
            "output_tokens": 45,
            "cost": 0.000014364,
        },
        "id": "gen-dec-12345",
        "provider": "TypeSafe",
    }

    with patch("src.edgebench.interpreters.decisions.make_request") as mock_make:
        mock_make.return_value = (200, json.dumps(canned_payload), 0.042, 1000.0, 1000.042, None, {})
        decision = client.decide(case)

    assert decision.valid is True
    assert decision.error_type is None
    assert decision.resolved_model == "typesafe/jev-1.13-20260917"
    assert decision.provider == "TypeSafe"
    assert decision.input_tokens == 342
    assert decision.output_tokens == 45
    assert decision.cost_usd == 0.000014364
    assert decision.latency_s == 0.042
    assert len(decision.labels) == 1
    assert decision.labels[0]["service_type"] == "ocr"
    assert decision.labels[0]["urgency"] == "urgent"


def test_decisions_model_mismatch():
    client = DecisionsClient(
        name="Jev-1.13.0",
        model="typesafe/jev-1.13",
        accepted_resolved_models={"typesafe/jev-1.13", "typesafe/jev-1.13-20260917"},
    )
    case = Case(
        case_id="c1",
        text="Please read text.",
        fields=FIELD_SETS[4],
        bundle_size=1,
        truth=[{"service_type": "ocr", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"}],
    )

    canned_payload = {
        "model": "untrusted/fallback-model",
        "answers": {},
        "usage": {"input_tokens": 100, "output_tokens": 20, "cost": 0.001},
        "id": "gen-dec-999",
        "provider": "Unknown",
    }

    with patch("src.edgebench.interpreters.decisions.make_request") as mock_make:
        mock_make.return_value = (200, json.dumps(canned_payload), 0.05, 1000.0, 1000.05, None, {})
        decision = client.decide(case)

    assert decision.valid is False
    assert decision.error_type == "model_mismatch"
    assert decision.resolved_model == "untrusted/fallback-model"
