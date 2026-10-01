#!/usr/bin/env python3
"""CLI shim for the EXP-2026-003 C2 local queue runner."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `src.` imports

from src.ranbench.c2.queue import main

if __name__ == "__main__":
    raise SystemExit(main())
