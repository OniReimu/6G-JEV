"""Tests for ledger module and key redaction."""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.edgebench.interpreters.base import Decision
from src.edgebench.interpreters.transport import make_request
from src.edgebench.ledger import LedgerWriter, completed_keys


def test_ledger_resume_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = Path(tmpdir) / "ledger.jsonl"
        with LedgerWriter(
            path=ledger_path,
            run_id="run_test",
            git_sha="abcdef1",
            rq="RQ1",
            condition="baseline",
        ) as writer:
            writer.write_row(
                model="Jev-1.13.0",
                case_id="case_001",
                repeat=0,
                decision=Decision(valid=True),
                correct={"service_type": True},
                em=True,
            )
            writer.write_row(
                model="Jev-1.13.0",
                case_id="case_001",
                repeat=1,
                decision=Decision(valid=True),
                correct={"service_type": True},
                em=True,
            )
            writer.write_row(
                model="GLM-5.3-Flash",
                case_id="case_002",
                repeat=0,
                decision=Decision(valid=False),
                correct={"service_type": False},
                em=False,
            )

        keys = completed_keys(ledger_path)
        assert len(keys) == 3
        assert ("RQ1", "baseline", "Jev-1.13.0", "case_001", 0) in keys
        assert ("RQ1", "baseline", "Jev-1.13.0", "case_001", 1) in keys
        assert ("RQ1", "baseline", "GLM-5.3-Flash", "case_002", 0) in keys


def test_fake_key_never_appears_in_written_bytes():
    fake_key = "sk-fake-openrouter-secret-key-9999"
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = Path(tmpdir) / "ledger.jsonl"

        # Simulate a raw response from server that echoed back the secret key
        raw_echo = f'{{"error": "Invalid auth token: {fake_key}"}}'
        redacted_raw = raw_echo.replace(fake_key, "[REDACTED]")

        decision = Decision(
            labels=[{"service_type": "unsupported"}],
            valid=False,
            raw_response=redacted_raw,
        )

        with LedgerWriter(
            path=ledger_path,
            run_id="run_leak_test",
            git_sha="abcdef1",
            rq="RQ1",
            condition="test",
        ) as writer:
            writer.write_row(
                model="Jev-1.13.0",
                case_id="c_leak",
                repeat=0,
                decision=decision,
                correct={"service_type": False},
                em=False,
            )

        file_bytes = ledger_path.read_bytes()
        assert fake_key.encode("utf-8") not in file_bytes
        assert b"[REDACTED]" in file_bytes
