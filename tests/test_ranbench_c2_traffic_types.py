"""C2 traffic `type` conformance: the config follows the frozen design's class mapping and uses only
the types the scenario source implements (EXP-2026-003 design.md, C2 component)."""
from __future__ import annotations

import json
import re
from pathlib import Path

NS3 = Path(__file__).resolve().parents[1] / "src" / "ranbench" / "ns3"
DESIGN_TYPES = {"video": "cbr", "xr": "cbr", "iot": "periodic", "be": "backlogged"}


def _source_types() -> set[str]:
    src = (NS3 / "ran-closed-loop.cc").read_text(encoding="utf-8")
    guard = re.search(r'NS_ABORT_MSG_IF\((t\.type != "\w+"(?:\s*&&\s*t\.type != "\w+")*)', src)
    assert guard, "scenario source has no traffic-type guard"
    return set(re.findall(r'"(\w+)"', guard.group(1)))


def test_source_implements_exactly_the_design_types():
    assert _source_types() == set(DESIGN_TYPES.values())
    src = (NS3 / "ran-closed-loop.cc").read_text(encoding="utf-8")
    assert 't.type == "backlogged"' in src  # the type selects the generator, not only parsed


def test_default_config_class_types_follow_design():
    cfg = json.loads((NS3 / "c2-default.json").read_text(encoding="utf-8"))
    types = {cls: spec["type"] for cls, spec in cfg["traffic"]["classes"].items()}
    assert types == DESIGN_TYPES
