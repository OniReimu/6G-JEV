"""tab_c1_cost: RQ4 tokens, fees and energy per decision from quality_and_cost.csv."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import c2_fixture as fx


@pytest.fixture(scope="module")
def pa() -> Any:
    return fx.load_paper_assets()


def quality_frame(pa: Any) -> pd.DataFrame:
    """Hosted: fee 0.2 USD, no energy. Self-hosted: fee 0, energy 100 J (c3) and 150 J (c57)."""
    rows = []
    for model in pa.MODELS:
        for count in (3, 57):
            hosted = model in pa.HOSTED
            rows.append({
                "model": model, "condition": f"c{count}_fresh",
                "mean_input_tokens": 1234.6 * count, "mean_output_tokens": 57.0,
                "cost_per_1000_correct_usd": 0.2 if hosted else 0.0,
                "energy_j_per_decision": float("nan") if hosted else (100.0 if count == 3 else 150.0),
            })
    return pd.DataFrame(rows)


def render(pa: Any, frame: pd.DataFrame, tmp: Path) -> list[tuple[str, list[str]]]:
    out = pa.Out(tmp / "paper", tmp / "analysis", png=False)
    pa.tab_c1_cost(SimpleNamespace(quality=frame, one=pa.Data.one), out)
    text = (tmp / "paper" / "tables" / "tab_c1_cost.tex").read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if line.endswith(r"\\ ") and not line.startswith(r"\rowcolor{ebBlue}"):
            cells = line.removeprefix(r"\rowcolor{ebCoral!10}").removesuffix(r" \\ ").split(" & ")
            assert len(cells) == 9, line
            rows.append((cells[0], cells[1:]))
    assert [artifact.artifact_id for artifact in out.artifacts] == ["tab_c1_cost"]
    return rows


def plain(cell: str) -> str:
    return cell.removeprefix(r"\cellcolor{ebCoral!35}").removeprefix(r"\textbf{").removesuffix("}")


def test_hosted_energy_is_na_and_self_hosted_fee_is_zero(pa: Any, tmp_path: Path) -> None:
    rows = render(pa, quality_frame(pa), tmp_path)
    assert len(rows) == len(pa.MODELS)
    by_model = {model: cells for model, (_, cells) in zip(pa.MODELS, rows)}
    for model, cells in by_model.items():
        values = [plain(cell) for cell in cells]
        assert values[0] == "3{,}704" and values[4] == "70{,}372"
        if model in pa.HOSTED:
            assert values[2] == values[6] == "0.200"
            assert values[3] == values[7] == "--"
        else:
            assert values[2] == values[6] == "0"
            assert (values[3], values[7]) == ("100.0", "150.0")


@pytest.mark.parametrize("model, column, value", [
    ("Jev-1.13.0", "energy_j_per_decision", 150.0),        # hosted with an energy value
    ("AnyJev-L0", "cost_per_1000_correct_usd", float("nan")),  # self-hosted without the 0 fee
    ("SemIf-Qwen3.5-4B", "energy_j_per_decision", float("nan")),  # self-hosted without its trace
])
def test_table_refuses_a_row_that_breaks_the_deployment_rules(
    pa: Any, tmp_path: Path, model: str, column: str, value: float,
) -> None:
    frame = quality_frame(pa)
    frame.loc[(frame["model"] == model) & (frame["condition"] == "c57_fresh"), column] = value
    with pytest.raises(ValueError, match=model):
        render(pa, frame, tmp_path)
