"""RQ1a programmatic padding with stop-word verification and balanced positioning."""
from __future__ import annotations

import hashlib
from pathlib import Path
import random
import re
from typing import Any

import tiktoken

from src.edgebench.contract import Case

PAD_SEED = 20260924
TARGET_LENGTHS: list[int] = [512, 2048, 8192, 16384]

STOP_PATTERNS = re.compile(
    r"\b(site|sites|local|locally|remote|remotely|cloud|clouds|transfer|transferred|transferring|"
    r"leave|leaves|leaving|quality|qualities|high|higher|highest|standard|standards|"
    r"urgent|urgently|urgency|priority|priorities|asap|rush|rushes|rushed|"
    r"ocr|text|texts|read|reading|reads|count|counts|counting|counted|"
    r"detect|detects|detection|detecting|box|boxes|retain|retains|retaining|retained|retention|"
    r"energy|eco|replica|replicas|replicated|replication|realtime|real-time|latency|latencies|deadline|deadlines)\b",
    re.IGNORECASE,
)

_enc: tiktoken.Encoding | None = None


def get_tokenizer() -> tiktoken.Encoding:
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def assert_padding_clean(text: str) -> None:
    match = STOP_PATTERNS.search(text)
    if match:
        raise AssertionError(
            f"Stop-word cue '{match.group(0)}' found in generated padding context at pos {match.start()}!"
        )


def generate_irrelevant_line_pool(
    min_lines: int = 50000, seed: int = PAD_SEED
) -> list[str]:
    lines: list[str] = []
    i = 0
    while len(lines) < min_lines:
        node = (i % 32) + 1
        cpu = (i * 7) % 95 + 4
        mem = (i * 123) % 15000 + 1024
        tx = round((i * 13.7) % 900 + 10, 1)
        rx = round((i * 11.3) % 900 + 10, 1)
        drop = round(((i * 3) % 10) * 0.0001, 5)

        t_line = (
            f"[2026-09-24T06:{(i//60)%60:02d}:{(i)%60:02d}.{i%1000:03d}Z] "
            f"host=worker-{node:02d} cpu_pct={cpu} mem_mb={mem} tx_mbps={tx} rx_mbps={rx} drop_ratio={drop}"
        )
        lines.append(t_line)

        idx = i % 5
        if idx == 0:
            c_line = "sysctl.net.ipv4.tcp_window_scaling=1 sysctl.net.core.somaxconn=1024"
        elif idx == 1:
            c_line = "kernel.pid_max=65536 fs.file-max=2097152 vm.swappiness=10"
        elif idx == 2:
            c_line = "net.ipv4.ip_forward=1 net.ipv4.conf.all.rp_filter=2"
        elif idx == 3:
            c_line = "fs.inotify.max_user_watches=524288 vm.max_map_count=262144"
        else:
            c_line = "net.core.rmem_max=16777216 net.core.wmem_max=16777216"
        lines.append(c_line)

        h_line = (
            f"[daemon:tick] queue_depth={i%4} worker_total=16 io_wait=0.01 fs_status=nominal epoch={189000+i}"
        )
        lines.append(h_line)
        i += 1

    return lines


def _extract_padding_slice(
    line_pool: list[str],
    line_offset: int,
    pad_needed: int,
    enc: tiktoken.Encoding,
) -> str:
    collected: list[str] = []
    curr = line_offset
    n = len(line_pool)
    needed_lines = (pad_needed // 15) + 30
    for _ in range(needed_lines):
        collected.append(line_pool[curr % n])
        curr += 1
    chunk = "\n".join(collected)
    toks = enc.encode(chunk)
    while len(toks) < pad_needed + 50:
        for _ in range(50):
            collected.append(line_pool[curr % n])
            curr += 1
        chunk = "\n".join(collected)
        toks = enc.encode(chunk)
    pad_text = enc.decode(toks[:pad_needed])
    assert_padding_clean(pad_text)
    return pad_text


def create_padded_case(
    case: Case,
    target_length: int,
    position: str,
    token_pool: list[int] | list[str],
    enc: tiktoken.Encoding,
    seed: int = PAD_SEED,
) -> Case:
    req_text = case.text
    req_tokens = enc.encode(req_text)
    sep_tokens = len(enc.encode("\n\n"))

    h = int(hashlib.sha256(f"{seed}:{case.case_id}:{target_length}".encode("utf-8")).hexdigest()[:8], 16)

    if token_pool and isinstance(token_pool[0], str):
        line_pool = token_pool  # type: ignore
    else:
        line_pool = generate_irrelevant_line_pool(min_lines=max(50000, target_length * 2), seed=seed)

    line_offset = h % len(line_pool)

    if position == "start":
        pad_needed = target_length - len(req_tokens) - sep_tokens
        pad_text = _extract_padding_slice(line_pool, line_offset, pad_needed, enc)
        padded_text = f"{req_text}\n\n{pad_text}"
    elif position == "end":
        pad_needed = target_length - len(req_tokens) - sep_tokens
        pad_text = _extract_padding_slice(line_pool, line_offset, pad_needed, enc)
        padded_text = f"{pad_text}\n\n{req_text}"
    else:  # middle
        pad_needed = target_length - len(req_tokens) - (2 * sep_tokens)
        p1 = pad_needed // 2
        p2 = pad_needed - p1
        pad1 = _extract_padding_slice(line_pool, line_offset, p1, enc)
        line_offset_2 = (line_offset + 500) % len(line_pool)
        pad2 = _extract_padding_slice(line_pool, line_offset_2, p2, enc)
        padded_text = f"{pad1}\n\n{req_text}\n\n{pad2}"

    actual_tokens = len(enc.encode(padded_text))
    error = abs(actual_tokens - target_length) / target_length
    if error > 0.03:
        raise ValueError(
            f"Padded length {actual_tokens} deviates > 3% from target {target_length} (error: {error:.4f})"
        )

    meta = dict(case.meta)
    meta["position"] = position
    meta["target_length"] = target_length
    meta["actual_length"] = actual_tokens
    meta["derived_from"] = case.case_id
    meta["derivation"] = f"pad_{target_length}"
    meta["generator"] = case.meta.get("generator", "")
    meta["verifier"] = case.meta.get("verifier", "")
    meta["generator_model"] = case.meta.get("generator_model", "")
    meta["verifier_model"] = case.meta.get("verifier_model", "")

    return Case(
        case_id=f"{case.case_id}_pad_{target_length}",
        text=padded_text,
        fields=list(case.fields),
        service_options=case.service_options,
        bundle_size=case.bundle_size,
        truth=[dict(t) for t in case.truth],
        meta=meta,
    )


def create_padded_dataset(
    clean_cases: list[Case],
    target_length: int,
    seed: int = PAD_SEED,
) -> list[Case]:
    sorted_clean = sorted(clean_cases, key=lambda c: c.case_id)
    enc = get_tokenizer()
    line_pool = generate_irrelevant_line_pool(min_lines=max(50000, target_length * 2), seed=seed)

    positions = ["start", "middle", "end"]
    rng = random.Random(seed + target_length)

    # Balance positions across cases
    n = len(sorted_clean)
    base_pos = n // 3
    rem_pos = n % 3
    pos_list = (
        ["start"] * (base_pos + (1 if rem_pos > 0 else 0))
        + ["middle"] * (base_pos + (1 if rem_pos > 1 else 0))
        + ["end"] * base_pos
    )
    rng.shuffle(pos_list)

    padded_cases: list[Case] = []
    for i, case in enumerate(sorted_clean):
        pos = pos_list[i]
        p_case = create_padded_case(case, target_length, pos, line_pool, enc, seed=seed)
        padded_cases.append(p_case)

    return padded_cases
