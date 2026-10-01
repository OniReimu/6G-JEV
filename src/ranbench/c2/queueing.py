"""RQ3 utilisation (numpy only, so the self-hosted load runner imports it without the analysis stack)."""
from __future__ import annotations

from typing import Any

import numpy as np


def utilisation(rate: float, service_s: np.ndarray, slots: int = 4) -> dict[str, Any]:
    """rho = rate x mean service time / slots; rho >= 1 -> non-stationary, excluded from block inference."""
    rho = rate * float(np.nanmean(service_s)) / slots
    return {"rho": rho, "non_stationary": rho >= 1.0}
