"""The single, fixed RANIntent -> E2SM-RC mapping used by C3.

Spike B2 verified only slice index 0 and the 5%, 30%, and 100% maximum-PRB
quota settings.  C3 deliberately stays inside that measured envelope:

* every RANIntent traffic class maps to the stack's one configured slice, 0;
* low -> 5%, normal -> 30%, high/critical -> 100%;
* unspecified -> 100%, the default restored by ``revert_default``.

The benchmark's clean C2/C3 pool pairs ``deprioritise`` with low/normal,
``prioritise`` with high/critical, and ``revert_default`` with unspecified.
Action is checked as a guard but priority is the sole quota lookup.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

CLASS_TO_SLICE: dict[str, int] = {
    "emergency_video": 0,
    "mission_voice": 0,
    "factory_control": 0,
    "xr_gaming": 0,
    "video_streaming": 0,
    "iot_metering": 0,
    "analytics_offload": 0,
    "best_effort": 0,
}

PRIORITY_TO_QUOTA: dict[str, int] = {
    "low": 5,
    "normal": 30,
    "high": 100,
    "critical": 100,
    "unspecified": 100,
}

VALID_ACTIONS = frozenset({"prioritise", "deprioritise", "revert_default"})


@dataclass(frozen=True)
class SlicePrbControl:
    slice_id: int
    min_prb_ratio: int
    max_prb_ratio: int
    dedicated_prb_ratio: int = 100

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def control_for_policy(policy: Mapping[str, str]) -> SlicePrbControl:
    """Translate one valid named-scope RANIntent policy to the fixed C3 control."""
    action = policy.get("action")
    traffic_class = policy.get("class")
    priority = policy.get("priority")
    if action not in VALID_ACTIONS:
        raise ValueError(f"C3 cannot actuate action {action!r}")
    if traffic_class not in CLASS_TO_SLICE:
        raise ValueError(f"C3 cannot map traffic class {traffic_class!r}")
    if priority not in PRIORITY_TO_QUOTA:
        raise ValueError(f"C3 cannot map priority {priority!r}")
    if action == "revert_default" and priority != "unspecified":
        raise ValueError("revert_default must have unspecified priority")
    if action != "revert_default" and priority == "unspecified":
        raise ValueError(f"{action} requires a stated priority")
    return SlicePrbControl(
        slice_id=CLASS_TO_SLICE[traffic_class],
        min_prb_ratio=0,
        max_prb_ratio=PRIORITY_TO_QUOTA[priority],
    )
