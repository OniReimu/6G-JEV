"""Skip, rather than fail, the tests that need the self-hosted model libraries when they are not installed.

The lightweight environment (requirements/edgebench.txt + requirements/analysis.txt) runs every test that needs no
model library. Install requirements/edgebench-cuda.txt (Linux + CUDA) to run the rest.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repository root, for `src.` imports
sys.path.insert(0, str(Path(__file__).resolve().parent))       # shared test helpers (c2_fixture, test_* imports)

MODEL_LIBRARIES = ("torch", "transformers", "sentence_transformers", "semif_phase1", "vllm")
MISSING = {name for name in MODEL_LIBRARIES if importlib.util.find_spec(name) is None}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    exc = call.excinfo
    if exc is not None and exc.errisinstance(ModuleNotFoundError) and exc.value.name in MISSING:
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1], f"requires the model library '{exc.value.name}'")
