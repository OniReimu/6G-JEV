"""Append-only thread-safe JSONL ledger for Edgebench evaluations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any

from src.edgebench.interpreters.base import Decision
from src.edgebench.provenance import verify_provenance


def get_git_status() -> tuple[str, bool]:
    """Returns (git_sha, is_dirty).

    A working tree is considered dirty if any tracked files are modified/deleted/staged.
    Untracked files (??) do not make the working tree dirty.
    """
    try:
        sha_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_sha = sha_proc.stdout.strip()
    except Exception:
        git_sha = "unknown"

    try:
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in status_proc.stdout.splitlines() if line.strip()]
        # Any line not starting with ?? is a modified tracked file
        modified_tracked = [line for line in lines if not line.startswith("??")]
        is_dirty = len(modified_tracked) > 0
    except Exception:
        is_dirty = True

    return git_sha, is_dirty


def _git_available() -> bool:
    """True when a git binary exists and the current directory is inside a git work tree."""
    if shutil.which("git") is None:
        return False
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True, check=False
        )
    except Exception:
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def resolve_git_provenance() -> tuple[str, bool, str, str | None]:
    """Returns (git_sha, is_dirty, source, provenance_tree_sha256).

    With git available, source is "git" and the values come from get_git_status() (no tree hash).
    Without git (e.g. a cluster node with no git binary, or a synced tree without .git), the tree must
    carry PROVENANCE.json from scripts/eb_provenance.py: every listed file is re-hashed and any mismatch
    or missing file refuses the run; source is then "provenance". EB_GIT_SHA, when set, must equal the
    recorded HEAD. Raises RuntimeError otherwise.
    """
    if _git_available():
        sha, dirty = get_git_status()
        return sha, dirty, "git", None
    head, tree = verify_provenance()
    env_sha = os.environ.get("EB_GIT_SHA", "").strip().lower()
    if env_sha and env_sha != head:
        raise RuntimeError(
            f"Refusing to start: EB_GIT_SHA={env_sha} does not match PROVENANCE.json head_sha={head}."
        )
    return head, False, "provenance", tree


def completed_keys(path: str | Path) -> set[tuple[str, str, str, str, int]]:
    """Read existing ledger JSONL and return set of (rq, condition, model, case_id, repeat)."""
    p = Path(path)
    if not p.exists():
        return set()

    keys: set[tuple[str, str, str, str, int]] = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = (
                    str(row["rq"]),
                    str(row["condition"]),
                    str(row["model"]),
                    str(row["case_id"]),
                    int(row["repeat"]),
                )
                keys.add(key)
            except Exception:
                continue
    return keys


def drop_torn_tail(path: str | Path) -> bool:
    """Truncate a final line without a trailing newline (a write cut short by a crash); True if one was dropped.

    Its key is then not complete, so resume re-runs that case.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    with open(p, "rb+") as f:
        data = f.read()
        if data.endswith(b"\n"):
            return False
        f.truncate(data.rfind(b"\n") + 1)
    return True


class LedgerWriter:
    def __init__(
        self,
        path: str | Path,
        run_id: str,
        git_sha: str,
        rq: str,
        condition: str,
        allow_dirty: bool = False,
        schema_version: str = "2.0",
        git_source: str = "git",
        provenance_tree_sha256: str | None = None,
        cases_sha256: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.git_sha = git_sha
        self.rq = rq
        self.condition = condition
        self.allow_dirty = allow_dirty
        self.schema_version = schema_version
        self.git_source = git_source
        self.provenance_tree_sha256 = provenance_tree_sha256
        self.cases_sha256 = cases_sha256
        self._lock = threading.Lock()
        self.torn_line_dropped = drop_torn_tail(self.path)
        self._file = open(self.path, "a", encoding="utf-8")

    def write_row(
        self,
        model: str,
        case_id: str,
        repeat: int,
        decision: Decision,
        correct: dict[str, bool],
        em: bool,
        thread: str | None = None,
        unsafe_locality: bool = False,
        spurious_count: int = 0,
        missed_count: int = 0,
        unspecified_truth_count: int = 0,
        specified_truth_count: int = 0,
        request_em: list[bool] | None = None,
        request_field_correct: list[dict[str, bool]] | None = None,
        platform: str | None = None,
    ) -> dict[str, Any]:
        thread_name = thread if thread is not None else threading.current_thread().name
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "git_source": self.git_source,
            "provenance_tree_sha256": self.provenance_tree_sha256,
            "cases_sha256": self.cases_sha256,
            "rq": self.rq,
            "condition": self.condition,
            "model": model,
            "platform": platform,
            "case_id": str(case_id),
            "repeat": int(repeat),
            "thread": thread_name,
            "labels": decision.labels,
            "probabilities": decision.probabilities,
            "confidence": decision.confidence,
            "valid": decision.valid,
            "error_type": decision.error_type,
            "http_status": decision.http_status,
            "latency_s": decision.latency_s,
            "t_send_wall": decision.t_send_wall,
            "t_recv_wall": decision.t_recv_wall,
            "input_tokens": decision.input_tokens,
            "output_tokens": decision.output_tokens,
            "reasoning_tokens": decision.reasoning_tokens,
            "completion_tokens_raw": getattr(decision, "completion_tokens_raw", decision.output_tokens),
            "cost_usd": decision.cost_usd,
            # null usage/cost = not reported by the server/provider (unknown, not 0)
            "usage_reported": decision.input_tokens is not None or decision.output_tokens is not None,
            "reasoning_tokens_missing": bool(getattr(decision, "reasoning_tokens_missing", False)),
            "post_timeout_wait_s": getattr(decision, "post_timeout_wait_s", None),
            "resolved_model": decision.resolved_model,
            "provider": decision.provider,
            "raw_response": decision.raw_response,
            "raw_response_sha256": decision.raw_response_sha256,
            "correct": correct,
            "em": bool(em),
            "unsafe_locality": bool(unsafe_locality),
            "spurious_count": int(spurious_count),
            "missed_count": int(missed_count),
            "unspecified_truth_count": int(unspecified_truth_count),
            "specified_truth_count": int(specified_truth_count),
            "request_em": request_em if request_em is not None else [bool(em)],
            "request_field_correct": request_field_correct if request_field_correct is not None else [dict(correct)],
            "schema_version": self.schema_version,
            "allow_dirty": self.allow_dirty,
        }
        line = json.dumps(row) + "\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()
        return row

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()

    def __enter__(self) -> LedgerWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
