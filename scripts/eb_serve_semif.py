#!/usr/bin/env python3
"""SemIf localhost HTTP server exposing POST /api/alpha/decisions.

Exposes the same request/response JSON shape as OpenRouter's Jev endpoint.
Model: Qwen/Qwen3.5-4B at revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
Model ID in responses: semif-qwen3.5-4b@23cf1f39 (MLX / MPS) or semif-qwen3.5-4b@23cf1f39+cuda
(CUDA: SemIf's PyTorch reference path, BF16).
Confidence calculation: max probability − second max (difference between top-1 and top-2).
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

# One accelerator: accept every connection (thread per connection) but run one inference at a time.
# A single-threaded HTTPServer served only the first keep-alive client and starved concurrent ones (RQ5 pilot).
INFERENCE_LOCK = threading.Lock()
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

MODEL_SOURCE = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RESOLVED_MODEL_ID = "semif-qwen3.5-4b@23cf1f39"
RESOLVED_MODEL_ID_CUDA = RESOLVED_MODEL_ID + "+cuda"
DEFAULT_PORT = 8601
BACKENDS = ("mlx", "mps", "cuda")


def resolved_model_id(backend: str) -> str:
    """Model id reported by the server for a backend."""
    return RESOLVED_MODEL_ID_CUDA if backend == "cuda" else RESOLVED_MODEL_ID


def load_backend_model(backend: str) -> tuple[Any, Any, Any]:
    """Load (model, tokenizer, metadata) for a backend at the pinned revision."""
    if backend == "mlx":
        from semif_phase1.mlx_backend import load_model
        return load_model(MODEL_SOURCE, revision=MODEL_REVISION)
    if backend in ("mps", "cuda"):
        from semif_phase1.core import load_causal_model
        return load_causal_model(MODEL_SOURCE, revision=MODEL_REVISION, device=backend, dtype="bfloat16")
    raise ValueError(f"Unsupported backend '{backend}'. Must be one of {BACKENDS}")


def compute_confidence(probs: list[float]) -> float:
    """Calculate confidence as max probability minus second max probability."""
    if not probs:
        return 0.0
    if len(probs) == 1:
        return float(probs[0])
    sorted_probs = sorted(probs, reverse=True)
    return float(sorted_probs[0] - sorted_probs[1])


def translate_request_to_semif_rows(state: str, questions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Jev/OpenRouter questions dictionary into SemIf row definitions.

    Raises ValueError if any question violates SemIf constraints (e.g. not 2-16 options).
    """
    rows = []
    for qid, q_data in questions.items():
        if not isinstance(q_data, dict):
            raise ValueError(f"Question '{qid}' must be an object")
        criteria = q_data.get("criteria", {})
        if not isinstance(criteria, dict):
            raise ValueError(f"Question '{qid}' criteria must be a dictionary")
        if not (2 <= len(criteria) <= 16):
            raise ValueError(
                f"Question '{qid}' has {len(criteria)} options; SemIf supports 2-16 options"
            )
        instruction = q_data.get("instructions", "")
        options = [{"id": opt_id, "description": desc} for opt_id, desc in criteria.items()]
        rows.append({
            "id": qid,
            "state": state,
            "question": instruction,
            "options": options,
        })
    return rows


def process_decisions_request(
    body: dict[str, Any],
    score_fn: Any,
    mode: str = "shared",
    max_tokens: int = 32768,
    model_id: str = RESOLVED_MODEL_ID,
) -> dict[str, Any]:
    """Process a Jev-style decision request and produce a compliant response."""
    state = body.get("state")
    if not state or not isinstance(state, str):
        raise ValueError("Field 'state' must be a nonempty string")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Field 'questions' must be a nonempty object")

    rows = translate_request_to_semif_rows(state, questions)

    answers: dict[str, dict[str, Any]] = {}
    total_input_tokens = 0

    if mode == "shared" and len(rows) > 1:
        results, timing = score_fn(rows, max_tokens=max_tokens)
        total_input_tokens = int(timing.get("prefix_tokens", 0) + timing.get("true_suffix_tokens", 0))
        for res in results:
            qid = res["id"]
            opt_ids = res["option_ids"]
            probs = res["probabilities"]
            best_idx = max(range(len(probs)), key=lambda i: probs[i])
            conf = compute_confidence(probs)
            answers[qid] = {
                "type": "choice",
                "choice": opt_ids[best_idx],
                "probabilities": {oid: float(p) for oid, p in zip(opt_ids, probs)},
                "confidence": conf,
            }
    else:
        for row in rows:
            res = score_fn(row, max_tokens=max_tokens)
            qid = res["id"]
            opt_ids = res["option_ids"]
            probs = res["probabilities"]
            total_input_tokens += int(res.get("input_tokens", 0))
            best_idx = max(range(len(probs)), key=lambda i: probs[i])
            conf = compute_confidence(probs)
            answers[qid] = {
                "type": "choice",
                "choice": opt_ids[best_idx],
                "probabilities": {oid: float(p) for oid, p in zip(opt_ids, probs)},
                "confidence": conf,
            }

    return {
        "model": model_id,
        "provider": "local",
        "answers": answers,
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": 0,
            "cost": 0.0,
        },
    }


class SemIfRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_model: Any = None
    server_tok: Any = None
    server_meta: Any = None
    server_backend: str = "mlx"
    server_model_id: str = RESOLVED_MODEL_ID
    server_mode: str = "shared"
    server_max_tokens: int = 32768

    def _send_json(self, status: int, data: dict[str, Any], compute_ms: float = 0.0) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("x-compute-ms", f"{compute_ms:.2f}")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "pid": os.getpid(),  # the job script checks it equals the server PID
                    "model": self.server_model_id,
                    "backend": self.server_backend,
                    "mode": self.server_mode,
                },
            )
        else:
            self._send_json(404, {"error": f"Endpoint not found: {self.path}"})

    def do_POST(self) -> None:
        with INFERENCE_LOCK:
            self._do_post()

    def _do_post(self) -> None:
        if self.path != "/api/alpha/decisions":
            self._send_json(404, {"error": f"Endpoint not found: {self.path}"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        try:
            req_data = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:
            self._send_json(400, {"error": f"Invalid JSON payload: {exc}"})
            return

        t0 = time.perf_counter()
        try:
            if self.server_backend == "mlx":
                from semif_phase1.mlx_backend import score as mlx_score, score_shared as mlx_score_shared
                if self.server_mode == "shared":
                    def score_fn(rows_or_row, max_tokens):
                        if isinstance(rows_or_row, list):
                            return mlx_score_shared(
                                self.server_model, self.server_tok, rows_or_row, self.server_meta, max_tokens=max_tokens
                            )
                        return mlx_score(
                            self.server_model, self.server_tok, rows_or_row, self.server_meta, max_tokens=max_tokens
                        )
                else:
                    def score_fn(row, max_tokens):
                        return mlx_score(
                            self.server_model, self.server_tok, row, self.server_meta, max_tokens=max_tokens
                        )
            else:
                from semif_phase1.direct import score as torch_score
                from semif_phase1.shared import score_shared as torch_score_shared
                if self.server_mode == "shared":
                    def score_fn(rows_or_row, max_tokens):
                        if isinstance(rows_or_row, list):
                            return torch_score_shared(
                                self.server_model, self.server_tok, rows_or_row, self.server_meta, max_tokens=max_tokens
                            )
                        return torch_score(
                            self.server_model, self.server_tok, rows_or_row, self.server_meta, max_tokens=max_tokens
                        )
                else:
                    def score_fn(row, max_tokens):
                        return torch_score(
                            self.server_model, self.server_tok, row, self.server_meta, max_tokens=max_tokens
                        )

            resp_data = process_decisions_request(
                req_data,
                score_fn=score_fn,
                mode=self.server_mode,
                max_tokens=self.server_max_tokens,
                model_id=self.server_model_id,
            )
            compute_ms = (time.perf_counter() - t0) * 1000.0
            self._send_json(200, resp_data, compute_ms=compute_ms)
        except ValueError as val_err:
            compute_ms = (time.perf_counter() - t0) * 1000.0
            self._send_json(400, {"error": str(val_err), "model": self.server_model_id}, compute_ms=compute_ms)
        except Exception as exc:
            compute_ms = (time.perf_counter() - t0) * 1000.0
            self._send_json(500, {"error": f"Internal error: {exc}", "model": self.server_model_id}, compute_ms=compute_ms)

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy standard request log lines
        pass


def warmup(model: Any, tok: Any, meta: Any, backend: str, mode: str, count: int = 5) -> None:
    """Run 5 dummy inference calls to warm up Metal/MPS/CUDA kernels and cache."""
    print(f"Warming up SemIf ({backend}, {mode}) with {count} dummy calls...")
    dummy_row = {
        "id": "warmup-q",
        "state": "The system health check passed at 08:00 UTC.",
        "question": "Is the system healthy?",
        "options": [
            {"id": "yes", "description": "The system is healthy."},
            {"id": "no", "description": "The system is unhealthy."},
        ],
    }
    if backend == "mlx":
        from semif_phase1.mlx_backend import score, score_shared
        for _ in range(count):
            if mode == "shared":
                score_shared(model, tok, [dummy_row], meta, max_tokens=512)
            else:
                score(model, tok, dummy_row, meta, max_tokens=512)
    else:
        from semif_phase1.direct import score
        from semif_phase1.shared import score_shared
        for _ in range(count):
            if mode == "shared":
                score_shared(model, tok, [dummy_row], meta, max_tokens=512)
            else:
                score(model, tok, dummy_row, meta, max_tokens=512)
    print("Warmup complete.")


def run_server(port: int = DEFAULT_PORT, backend: str = "mlx", mode: str = "shared", max_tokens: int = 32768) -> None:
    print(f"Loading SemIf model ({MODEL_SOURCE} @ {MODEL_REVISION[:8]}, backend={backend})...")
    model, tok, meta = load_backend_model(backend)

    warmup(model, tok, meta, backend=backend, mode=mode, count=5)

    SemIfRequestHandler.server_model = model
    SemIfRequestHandler.server_tok = tok
    SemIfRequestHandler.server_meta = meta
    SemIfRequestHandler.server_backend = backend
    SemIfRequestHandler.server_model_id = resolved_model_id(backend)
    SemIfRequestHandler.server_mode = mode
    SemIfRequestHandler.server_max_tokens = max_tokens

    server = ThreadingHTTPServer(("127.0.0.1", port), SemIfRequestHandler)
    print(f"SemIf server listening on http://127.0.0.1:{port} (model={resolved_model_id(backend)}, single worker thread)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down SemIf server.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SemIf localhost HTTP server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind (default: 8601)")
    parser.add_argument("--backend", choices=list(BACKENDS), default="mlx", help="Backend (default: mlx)")
    parser.add_argument("--mode", choices=["shared", "direct"], default="shared", help="Decision mode (default: shared)")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Max prompt tokens limit (default: 32768)")
    args = parser.parse_args()
    run_server(port=args.port, backend=args.backend, mode=args.mode, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
