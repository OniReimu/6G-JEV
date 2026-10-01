"""paper_assets.py C2 rendering: strict output is pinned; a partial render marks every pending unit.

Run: python -m pytest tests
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

import c2_fixture as fx

GOLDEN = Path(__file__).with_name("golden_c2_strict.json")
PENDING = r"\pending{pending}"
PENDING_MARK = r"\pending{$^{?}$}"


@pytest.fixture(scope="module")
def pa() -> Any:
    return fx.load_paper_assets()


def table_rows(text: str, ncols: int) -> list[tuple[str, str, list[str]]]:
    """(block title, row label, cells) for every body row of a render_big_table table."""
    rows, title = [], ""
    for line in text.splitlines():
        if line.startswith(r"\rowcolor{ebAmber!45}"):
            title = line
            continue
        if not line.endswith(r"\\ ") or line.startswith(r"\rowcolor{ebBlue}"):
            continue
        cells = line.removeprefix(r"\rowcolor{ebCoral!10}").removesuffix(r" \\ ").split(" & ")
        assert len(cells) == ncols, line
        rows.append((title, cells[0], cells[1:]))
    return rows


def model_of(label: str, pa: Any) -> str:
    return next(model for model, tex in pa.TEX_NAME.items() if tex == label)


def render(pa: Any, directory: Path, tmp: Path, allow_partial: bool, monkeypatch: pytest.MonkeyPatch
           ) -> tuple[dict[str, bytes], dict[tuple[float, str], list[float]]]:
    """Render; also capture the y values of every fig_c2_grid line (NaN = point omitted)."""
    lines: dict[tuple[float, str], list[float]] = {}
    panel = iter(fx.RATES * 100)
    original = pa.model_line

    def spy(ax: Any, model: str, x: Any, y: Any, **kwargs: Any) -> None:
        if model == pa.MODELS[0]:
            spy.rate = next(panel)
        lines[(spy.rate, model)] = [float(value) for value in y]
        original(ax, model, x, y, **kwargs)

    monkeypatch.setattr(pa, "model_line", spy)
    files = fx.render(pa, directory, tmp / "out", allow_partial=allow_partial)
    return files, lines


def text(files: dict[str, bytes], name: str) -> str:
    return files[name].decode("utf-8")


# ---------------------------------------------------------------- strict mode


def test_strict_render_of_complete_fixture_is_byte_identical_to_pre_change(pa: Any, tmp_path: Path) -> None:
    files = fx.render(pa, fx.write_complete(tmp_path / "c2"), tmp_path / "out")
    got = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    assert got == json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_strict_render_refuses_a_partial_analysis(pa: Any, tmp_path: Path) -> None:
    c2 = fx.write_complete(tmp_path / "c2")
    fx.drop_l_arm(c2, fx.RQ2_POINTS[3], "AnyJev-L0")
    with pytest.raises(SystemExit, match="partial analysis"):
        pa.C2Data(c2, allow_partial=False)


def test_complete_fixture_with_partial_flag_renders_no_placeholder(pa: Any, tmp_path: Path) -> None:
    files = fx.render(pa, fx.write_complete(tmp_path / "c2"), tmp_path / "out", allow_partial=True)
    assert not any(b"\\pending" in data for data in files.values())
    assert text(files, "analysis/c2_pending.csv") == "unit,reason,missing_run_ids\n"


def test_partial_render_refuses_a_gap_skipped_json_does_not_explain(pa: Any, tmp_path: Path) -> None:
    c2 = fx.write_complete(tmp_path / "c2")
    fx._filter(c2, "an_decomposition.csv", lambda row: row["design_point"] != fx.RQ2_POINTS[0])
    with pytest.raises(SystemExit, match="does not name pending"):
        pa.C2Data(c2, allow_partial=True)


# ---------------------------------------------------------------- one absent control


def test_absent_control_pends_every_cell_of_its_design_point(
        pa: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    point = fx.point(0.3, 60.0)  # RQ2 grid, rate 0.3, second speed column
    c2 = fx.write_complete(tmp_path / "c2")
    missing = fx.drop_control(c2, point)
    files, lines = render(pa, c2, tmp_path, True, monkeypatch)

    grid = table_rows(text(files, "paper/tables/tab_c2_grid.tex"), 7)
    column = fx.SPEEDS.index(60.0)
    for title, label, cells in grid:
        at_point = cells[2 * column: 2 * column + 2]
        if "0.3 intents/s" in title:  # AnyJev has no comparator: its gap is "--" even with complete data
            assert at_point == [PENDING, "--" if model_of(label, pa) == "AnyJev-L0" else PENDING]
            assert f"60 km/h {PENDING}" in title
        else:
            assert PENDING not in cells
    assert sum(cell == PENDING for _, _, cells in grid for cell in cells) == 2 * (2 * len(fx.MODELS) - 1)

    controls = table_rows(text(files, "paper/tables/tab_c2_controls.tex"), 12)
    for label, cells in ((label, cells) for _, label, cells in controls):
        assert (cells == [PENDING] * 11) == (label == "0.3/s, 60 km/h"), label

    radio = table_rows(text(files, "paper/tables/tab_c2_radio.tex"), 11)
    radio_column = [(0.1, 3), (0.1, 60), (0.1, 120), (0.3, 3), (0.3, 30), (0.3, 60), (0.3, 120)].index((0.3, 60))
    for title, _, cells in radio:
        stale = "Stale-policy" in title
        assert (cells[radio_column] == PENDING) == stale
        assert sum(cell == PENDING for cell in cells) == (1 if stale else 0)
        if stale:  # no best mark over a column with a pending cell
            assert r"\cellcolor" not in cells[radio_column]

    # Nothing is plotted there: every line in the 0.3/s panel has NaN at 60 km/h, other points are plotted.
    for model in fx.MODELS:
        assert math.isnan(lines[(0.3, model)][column])
        assert all(math.isfinite(value) for rate in fx.RATES for i, value in enumerate(lines[(rate, model)])
                   if (rate, i) != (0.3, column))

    # The H2 family waits for the control: tab_c2_grid resolution marks on eligible affected gaps are pending.
    for title, label, cells in grid:
        model = model_of(label, pa)
        if model in pa.H1_ANCHOR and "affected class" in title:
            for index, speed in enumerate(fx.SPEEDS):
                rate = float(re.search(r"\{([\d.]+) intents?/s", title).group(1))
                eligible = fx.point(rate, speed) != fx.INELIGIBLE
                if cells[2 * index + 1] != PENDING:
                    assert cells[2 * index + 1].endswith(PENDING_MARK) == eligible

    pending = list(csv.DictReader(io.StringIO(text(files, "analysis/c2_pending.csv"))))
    assert {row["missing_run_ids"] for row in pending} == {missing}
    units = {row["unit"] for row in pending}
    assert f"controls {point}" in units and "holm H2 primary_d10" in units
    assert {f"l {point} {model}" for model in fx.MODELS} <= units
    h2 = list(csv.DictReader(io.StringIO(text(files, "analysis/c2_h2_summary.csv"))))
    assert all(row["share_resolved"] == "" for row in h2)


# ---------------------------------------------------------------- one absent L arm


def test_absent_l_arm_outside_every_holm_family_pends_only_that_contrast(
        pa: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    point, model = fx.point(1.0, 3.0), "AnyJev-L0"
    c2 = fx.write_complete(tmp_path / "c2")
    fx.drop_l_arm(c2, point, model)
    complete = fx.render(pa, fx.write_complete(tmp_path / "ref"), tmp_path / "ref-out" / "out")
    files, lines = render(pa, c2, tmp_path, True, monkeypatch)

    grid = table_rows(text(files, "paper/tables/tab_c2_grid.tex"), 7)
    reference = table_rows(text(complete, "paper/tables/tab_c2_grid.tex"), 7)
    for (title, label, cells), (_, _, want) in zip(grid, reference, strict=True):
        if model_of(label, pa) == model and title.find("{1 intent/s") >= 0:
            assert cells[:2] == [PENDING, "--"]  # level pending; AnyJev has no gap even with complete data
            assert cells[2:] == want[2:]
        else:
            assert cells == [re.sub(r"\\cellcolor\{ebCoral!35\}\\textbf\{([^}]*)\}", r"\1", cell)
                             if "1 intent/s" in title and index == 0 else cell
                             for index, cell in enumerate(want)]
    assert math.isnan(lines[(1.0, model)][0])
    assert sum(not math.isfinite(v) for values in lines.values() for v in values) == 1

    radio = table_rows(text(files, "paper/tables/tab_c2_radio.tex"), 11)
    column = [(0.1, 3), (0.1, 60), (0.1, 120), (0.3, 3), (0.3, 30), (0.3, 60), (0.3, 120),
              (1.0, 3), (1.0, 60), (1.0, 120)].index((1.0, 3))
    for _, label, cells in radio:
        assert (cells[column] == PENDING) == (model_of(label, pa) == model)
        assert sum(cell == PENDING for cell in cells) == (model_of(label, pa) == model)
    for name in ("paper/tables/tab_c2_controls.tex", "paper/tables/tab_c2_h1.tex", "analysis/c2_h2_summary.csv"):
        assert files[name] == complete[name], name


def test_absent_challenger_l_arm_pends_its_contrast_and_the_h2_marks(
        pa: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    point, model = fx.point(0.1, 3.0), "GLM-5.3-Flash"
    c2 = fx.write_complete(tmp_path / "c2")
    missing = fx.drop_l_arm(c2, point, model)
    files, _ = render(pa, c2, tmp_path, True, monkeypatch)
    grid = table_rows(text(files, "paper/tables/tab_c2_grid.tex"), 7)
    for title, label, cells in grid:
        if model_of(label, pa) == model and "0.1 intents/s" in title:
            assert cells[:2] == [PENDING, PENDING]
        else:
            assert PENDING not in cells
    marks = [cell for title, label, cells in grid if "affected class" in title for cell in cells[1::2]
             if cell.endswith(PENDING_MARK)]
    assert len(marks) == 4 * (len(fx.RQ2_POINTS) - 1) - 1  # four challengers, eight eligible points, one pending
    pending = list(csv.DictReader(io.StringIO(text(files, "analysis/c2_pending.csv"))))
    assert {row["missing_run_ids"] for row in pending} == {missing}
    assert f"l {point} {model}" in {row["unit"] for row in pending}


# ---------------------------------------------------------------- H2 below the eligibility minimum


def test_h2_below_min_eligible_renders_not_testable(pa: Any, tmp_path: Path) -> None:
    c2 = fx.write_complete(tmp_path / "c2")
    path = c2 / "hypotheses.json"
    hypotheses = json.loads(path.read_text(encoding="utf-8"))
    family = hypotheses["primary_d10"]["H2"]
    family["n_prespecified_points"] = 9
    family["result"]["n_eligible"] = pa.H2_MIN_ELIGIBLE - 2
    path.write_text(json.dumps(hypotheses, indent=2) + "\n", encoding="utf-8")
    files = fx.render(pa, c2, tmp_path / "out")
    note = (f"H2 is not testable, because only {pa.H2_MIN_ELIGIBLE - 2} of the 9 design points are eligible and "
            f"it needs {pa.H2_MIN_ELIGIBLE}.")
    assert note in text(files, "paper/figures/fig_c2_grid.tex")
    assert note in text(files, "paper/tables/tab_c2_grid.tex")
    h2 = list(csv.DictReader(io.StringIO(text(files, "analysis/c2_h2_summary.csv"))))
    primary = [row for row in h2 if row["rule"] == "primary_d10"]
    assert primary and all(row["holds"] == "not testable" for row in primary)
    assert all(row["source_key"] == "primary_d10.H2.result.n_eligible" for row in primary)
    assert all(row["holds"] in ("True", "False") for row in h2 if row["rule"] != "primary_d10")


def test_h2_at_min_eligible_renders_the_verdict(pa: Any, tmp_path: Path) -> None:
    c2 = fx.write_complete(tmp_path / "c2")
    path = c2 / "hypotheses.json"
    hypotheses = json.loads(path.read_text(encoding="utf-8"))
    hypotheses["primary_d10"]["H2"]["result"]["n_eligible"] = pa.H2_MIN_ELIGIBLE
    path.write_text(json.dumps(hypotheses, indent=2) + "\n", encoding="utf-8")
    files = fx.render(pa, c2, tmp_path / "out")
    assert not any(b"not testable" in data for data in files.values())


# ---------------------------------------------------------------- RQ5 and RQ3 radio tables


def test_rq5_table_strict_values_and_contrasts(pa: Any, tmp_path: Path) -> None:
    c2 = fx.write_complete(tmp_path / "c2")
    files = fx.render(pa, c2, tmp_path / "out")
    rows = table_rows(text(files, "paper/tables/tab_c2_rq5.tex"), 6)
    assert len(rows) == 7 * len(fx.MODELS)
    csv_rows = {row["interpreter"]: row for row in csv.DictReader(io.StringIO((c2 / "rq5.csv").read_text()))}
    for title, label, cells in rows:
        assert PENDING not in "".join(cells)
        row = csv_rows[model_of(label, pa)]
        if "Network-wide" in title:
            assert re.fullmatch(r"(\\cellcolor\{ebCoral!35\}\\textbf\{)?" + f"{100 * float(row['network_b_point']):.2f}"
                                + r"\}?", cells[1]), cells[1]
            low = 100 * float(row["network_b_minus_a_ci_low"])
            assert f"[{low:.2f}, ".replace("-", "$-$") in cells[3]
        if "best effort" in title:  # per-run aggregates: signed point difference, no interval
            diff = float(row["jain_be_b"]) - float(row["jain_be_c"])
            assert cells[4] == f"{diff:+.3f}".replace("-", "$-$")
    assert r"$B=" in text(files, "paper/tables/tab_c2_rq5.tex")


def test_rq3_table_marks_non_stationary_cells_and_drops_their_intervals(pa: Any, tmp_path: Path) -> None:
    files = fx.render(pa, fx.write_complete(tmp_path / "c2"), tmp_path / "out")
    rows = table_rows(text(files, "paper/tables/tab_c2_rq3.tex"), 1 + len(fx.RQ3_CELLS))
    assert len(rows) == 3 * len(fx.MODELS)
    for title, label, cells in rows:
        model = model_of(label, pa)
        for (rate, ues), cell in zip(fx.RQ3_CELLS, cells):
            flagged = (model, rate, ues) in fx.NON_STATIONARY
            assert cell.endswith(r"$^{\dagger}$") is flagged, (title, model, rate, ues)
            assert not (flagged and r"\textbf" in cell), "a non-stationary cell is never ranked best"
            if "SLA" in title:
                assert ("[" in cell) is not flagged


def test_absent_rq5_run_pends_only_its_interpreter(pa: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c2 = fx.write_complete(tmp_path / "c2")
    missing = fx.drop_rq5(c2, "GLM-5.3-Flash")
    with pytest.raises(SystemExit, match="partial analysis"):
        pa.C2Data(c2, allow_partial=False)
    files, _ = render(pa, c2, tmp_path, True, monkeypatch)
    for _, label, cells in table_rows(text(files, "paper/tables/tab_c2_rq5.tex"), 6):
        assert (cells == [PENDING] * 5) is (model_of(label, pa) == "GLM-5.3-Flash")
    assert not any(PENDING in text(files, f"paper/tables/{name}.tex")
                   for name in ("tab_c2_rq3", "tab_c2_h1", "tab_c2_grid", "tab_c2_radio", "tab_c2_controls"))
    pending = list(csv.DictReader(io.StringIO(text(files, "analysis/c2_pending.csv"))))
    assert [(row["unit"], row["missing_run_ids"]) for row in pending] == [
        (f"rq5 {fx.BASE} GLM-5.3-Flash", missing), (f"radio {fx.BASE} GLM-5.3-Flash E b-requery-1s", missing)]


def test_absent_rq3_replay_pends_only_its_cell(pa: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c2 = fx.write_complete(tmp_path / "c2")
    fx.drop_rq3(c2, "DeepSeek-V4.1-Flash", 1.0, 20)
    files, _ = render(pa, c2, tmp_path, True, monkeypatch)
    for _, label, cells in table_rows(text(files, "paper/tables/tab_c2_rq3.tex"), 1 + len(fx.RQ3_CELLS)):
        want = model_of(label, pa) == "DeepSeek-V4.1-Flash"
        assert [cell == PENDING for cell in cells] == [False] * 4 + [want]
    assert PENDING not in text(files, "paper/tables/tab_c2_rq5.tex")


def test_non_stationary_cell_with_an_interval_is_refused(pa: Any, tmp_path: Path) -> None:
    c2 = fx.write_complete(tmp_path / "c2")
    rows = fx._read_csv(c2 / "rq3_radio.csv")
    for row in rows:
        if row["non_stationary"] == "True":
            row["network_ci_low"], row["network_ci_high"] = "0.1", "0.3"
    fx._write_csv(c2 / "rq3_radio.csv", rows)
    with pytest.raises(ValueError, match="non-stationary cell carries a network interval"):
        fx.render(pa, c2, tmp_path / "out")


@pytest.mark.parametrize("unit", ["rq5", "rq3"])
def test_pending_unit_whose_row_is_present_is_refused(pa: Any, tmp_path: Path, unit: str) -> None:
    c2 = fx.write_complete(tmp_path / "c2")
    if unit == "rq5":
        fx._pend(c2, [fx._unit("RQ5 division of labour", fx.BASE, ["x"], interpreter="GLM-5.3-Flash")], [])
    else:
        point = fx.rq3_point("GLM-5.3-Flash", 0.5, 5)
        fx._pend(c2, [fx._unit("RQ3 radio replay", point, ["x"], interpreter="GLM-5.3-Flash")], [])
    with pytest.raises(SystemExit, match="rows are present"):
        pa.C2Data(c2, allow_partial=True)
# ---------------------------------------------------------------- D-17 platform contract
# A level from another platform than its design point's (or rate's) reference keeps its value, carries no marker and
# is left out of the best-value ranking; a contrast whose own runs are on different platforms renders "--".

MIXED = r"$^{\S}$"
OLD_NOTES = ("another platform", "platform binary", "level shift")
BEST = re.compile(r"\\cellcolor\{ebCoral!35\}\\textbf\{([^{}]*)\}")
GRID, RADIO, CONTROLS = (f"paper/tables/{name}.tex" for name in ("tab_c2_grid", "tab_c2_radio", "tab_c2_controls"))
RQ3 = "paper/tables/tab_c2_rq3.tex"
NCOLS = {GRID: 7, RADIO: 11, CONTROLS: 12, RQ3: 1 + len(fx.RQ3_CELLS)}
RADIO_COLUMNS = [(0.1, 3.0), (0.1, 60.0), (0.1, 120.0), (0.3, 3.0), (0.3, 30.0), (0.3, 60.0), (0.3, 120.0),
                 (1.0, 3.0), (1.0, 60.0), (1.0, 120.0)]
ROW = "0.1/s, 120 km/h"  # the controls row of fx.point(0.1, 120.0)


@pytest.fixture(scope="module")
def baseline(pa: Any, tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    """Strict render of the complete fixture, every C2 run on cluster-A."""
    root = tmp_path_factory.mktemp("baseline")
    return fx.render(pa, fx.write_complete(root / "c2"), root / "out")


def plain(cell: str) -> str:
    return BEST.sub(r"\1", cell)


def is_best(cell: str) -> bool:
    return r"\textbf" in cell


def blocks(files: dict[str, bytes], name: str) -> dict[str, list[tuple[str, list[str]]]]:
    """{block title: [(row label, cells)]} of one rendered table."""
    out: dict[str, list[tuple[str, list[str]]]] = {}
    for title, label, cells in table_rows(text(files, name), NCOLS[name]):
        out.setdefault(title, []).append((label, cells))
    return out


def changed(base: dict[str, bytes], got: dict[str, bytes], name: str) -> dict[tuple[str, str, int], str]:
    """Cells whose text differs from the baseline once the best-value highlight is removed."""
    before, after = table_rows(text(base, name), NCOLS[name]), table_rows(text(got, name), NCOLS[name])
    assert [row[:2] for row in before] == [row[:2] for row in after]
    return {(title, label, index): new
            for (title, label, old_cells), (_, _, new_cells) in zip(before, after)
            for index, (old, new) in enumerate(zip(old_cells, new_cells)) if plain(old) != plain(new)}


def bold_labels(rows: list[tuple[str, list[str]]], column: int) -> set[str]:
    return {label for label, cells in rows if is_best(cells[column])}


def lowest(rows: list[tuple[str, list[str]]], column: int, excluded: set[str]) -> set[str]:
    """Labels a min-ranked column should bold: lowest level among ranked cells (pending and dagger cells unranked)."""
    levels = {label: float(re.match(r"[0-9.]+", plain(cells[column])).group())
              for label, cells in rows
              if label not in excluded and cells[column] != PENDING and r"\dagger" not in cells[column]}
    return {label for label, value in levels.items() if value == min(levels.values())}


def labels(pa: Any, *models: str) -> set[str]:
    return {pa.TEX_NAME[model] for model in models}


def assert_no_platform_marker(files: dict[str, bytes]) -> None:
    for name, data in files.items():
        if name.endswith(".tex"):
            body = data.decode("utf-8")
            assert MIXED not in body, name
            assert not any(note in body for note in OLD_NOTES), name


def test_all_same_platform_renders_no_marker(pa: Any, tmp_path: Path) -> None:
    files = fx.render(pa, fx.write_complete(tmp_path / "c2"), tmp_path / "out")
    assert not any(MIXED.encode() in data for name, data in files.items() if name.endswith(".tex"))


def test_unknown_binary_fails_closed(pa: Any, tmp_path: Path) -> None:
    c2 = fx.write_complete(tmp_path / "c2")
    fx.set_binary(c2, fx.run_id(fx.BASE, "control", "oracle"), "deadbeef" + "0" * 56)
    with pytest.raises(ValueError, match="binary_sha256 without a platform"):
        pa.C2Data(c2, allow_partial=False)


def test_foreign_level_is_unranked_and_split_contrasts_render_a_dash(
        pa: Any, baseline: dict[str, bytes], tmp_path: Path) -> None:
    point = fx.point(0.1, 120.0)
    c2 = fx.write_complete(tmp_path / "c2")
    # the anchor's L arm and the 1 s fixed-latency control rerun on the M4 D-11 build
    fx.set_binary(c2, fx.run_id(point, "Jev-1.13.0", "L"), fx.SHA["M4"])
    fx.set_binary(c2, fx.run_id(point, "control", "fixed-1"), fx.SHA["M4"])
    files = fx.render(pa, c2, tmp_path / "out")
    assert_no_platform_marker(files)

    # grid: only the gaps of the moved anchor's challengers at 120 km/h split; every value else is unchanged
    jev, challengers = labels(pa, "Jev-1.13.0"), labels(pa, *(m for m, a in fx.ANCHOR.items() if a == "Jev-1.13.0"))
    rate_blocks = [title for title in blocks(baseline, GRID) if "0.1 intents/s" in title]
    assert len(rate_blocks) == 2 and len(challengers) == 3
    assert changed(baseline, files, GRID) == {(title, label, 5): "--" for title in rate_blocks for label in challengers}
    for title in rate_blocks:
        rows = blocks(files, GRID)[title]
        assert not bold_labels(rows, 4) & jev
        assert bold_labels(rows, 4) == lowest(rows, 4, excluded=jev)  # the rest of the column is still ranked
    assert bold_labels(blocks(baseline, GRID)[rate_blocks[0]], 4) == jev  # the moved level was the best before

    # radio: same values, the moved level is never bold although it was before
    column = RADIO_COLUMNS.index((0.1, 120.0))
    assert changed(baseline, files, RADIO) == {}
    assert not any(bold_labels(rows, column) & jev for rows in blocks(files, RADIO).values())
    assert any(bold_labels(rows, column) & jev for rows in blocks(baseline, RADIO).values())

    # controls: C-2 and Monotone use the three fixed arms (split); C-1 (no-update vs oracle) stays shown
    assert changed(baseline, files, CONTROLS) == {("", ROW, 1): "--", ("", ROW, 5): "--"}
    for name in ("tab_c2_h1", "tab_c2_rq5", "tab_c2_rq3"):
        assert text(files, f"paper/tables/{name}.tex") == text(baseline, f"paper/tables/{name}.tex")
    for name in (RADIO, RQ3):
        assert "platform" not in text(files, name)


def test_rq3_cell_off_its_rate_platform_keeps_its_value_unranked(
        pa: Any, baseline: dict[str, bytes], tmp_path: Path) -> None:
    c2 = fx.write_complete(tmp_path / "c2")
    fx.set_binary(c2, "rq3__Qwen3.8-Flash__0.1__5", fx.SHA["cluster-B"])
    fx.set_binary(c2, "rq3__GLM-5.3-Flash__2__5", fx.SHA["cluster-B"])
    fx.set_binary(c2, "rq3__Jev-1.13.0__2__5", fx.SHA["cluster-B"])  # best in both 2/s SLA columns of the baseline
    files = fx.render(pa, c2, tmp_path / "out")
    assert_no_platform_marker(files)
    assert "platform" not in text(files, RQ3)
    assert changed(baseline, files, RQ3) == {}
    moved = {0: labels(pa, "Qwen3.8-Flash"), 3: labels(pa, "GLM-5.3-Flash", "Jev-1.13.0")}
    sla = [title for title in blocks(files, RQ3) if "SLA" in title]
    assert len(sla) == 2
    for title in sla:
        rows = blocks(files, RQ3)[title]
        for column in range(len(fx.RQ3_CELLS)):
            excluded = moved.get(column, set())
            assert not bold_labels(rows, column) & excluded
            assert bold_labels(rows, column) == lowest(rows, column, excluded)
        assert bold_labels(blocks(baseline, RQ3)[title], 3) == labels(pa, "Jev-1.13.0")


@pytest.mark.parametrize("mode", fx.MODES)
def test_h1_base_point_arm_on_another_binary_is_refused(pa: Any, tmp_path: Path, mode: str) -> None:
    c2 = fx.write_complete(tmp_path / "c2")
    fx.set_binary(c2, fx.run_id(fx.BASE, "GLM-5.3-Flash", "L", mode), fx.SHA["M4"])
    with pytest.raises(ValueError, match="tab_c2_h1"):
        fx.render(pa, c2, tmp_path / "out")


def test_foreign_level_in_a_partial_render_is_unranked(
        pa: Any, baseline: dict[str, bytes], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    point = fx.point(0.1, 60.0)
    c2 = fx.write_complete(tmp_path / "c2")
    fx.set_binary(c2, fx.run_id(point, "AnyJev-L0", "L"), fx.SHA["M4"])
    fx.drop_l_arm(c2, point, "GLM-5.3-Flash")
    files, _ = render(pa, c2, tmp_path, True, monkeypatch)
    assert_no_platform_marker(files)
    assert "--" not in changed(baseline, files, GRID).values()  # AnyJev-L0 is no challenger, so nothing splits
    (anyjev,), (glm,) = labels(pa, "AnyJev-L0"), labels(pa, "GLM-5.3-Flash")
    rate_blocks = [title for title in blocks(files, GRID) if "0.1 intents/s" in title]
    assert len(rate_blocks) == 2
    for title in rate_blocks:
        rows = blocks(files, GRID)[title]
        assert dict(rows)[glm][2] == PENDING
        assert plain(dict(rows)[anyjev][2]) == plain(dict(blocks(baseline, GRID)[title])[anyjev][2])
        assert bold_labels(rows, 2) == set()  # the pending GLM cell suspends the ranking of the whole column
        for column in (0, 4):  # the other speeds of the rate rank as in the baseline
            assert bold_labels(rows, column) == bold_labels(blocks(baseline, GRID)[title], column)
    assert bold_labels(blocks(baseline, GRID)[rate_blocks[0]], 2) == {anyjev}  # the moved level was the best before


def test_same_platform_patched_build_is_not_marked(pa: Any, tmp_path: Path) -> None:
    point = fx.point(0.3, 120.0)
    c2 = fx.write_complete(tmp_path / "c2")
    fx.set_binary(c2, fx.run_id(point, "GLM-5.3-Flash", "L"), "025d3dc4" + "1" * 56)  # cluster-A D-7 build
    fx.set_binary(c2, fx.run_id(point, "control", "fixed-5"), "74d32e46" + "2" * 56)  # cluster-A D-7b build
    files = fx.render(pa, c2, tmp_path / "out")
    assert not any(MIXED.encode() in data for name, data in files.items() if name.endswith(".tex"))


def test_absent_oracle_takes_the_ranking_reference_from_no_update(
        pa: Any, baseline: dict[str, bytes], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    point = fx.point(0.3, 60.0)
    column = RADIO_COLUMNS.index((0.3, 60.0))
    c2 = fx.write_complete(tmp_path / "c2")
    fx.set_binary(c2, fx.run_id(point, "AnyJev-L0", "L"), fx.SHA["M4"])
    fx.drop_control(c2, point, arm="oracle")
    assert pa.C2Data(c2, allow_partial=True).reference_platform(point) == "cluster-A"
    files, _ = render(pa, c2, tmp_path, True, monkeypatch)
    assert_no_platform_marker(files)
    stale = next(title for title in blocks(files, RADIO) if "Stale" in title)
    diff = changed(baseline, files, RADIO)
    assert set(diff.values()) == {PENDING}  # stale exposure pends with the controls; every other value is unchanged
    assert {(title, index) for title, _, index in diff} == {(stale, column)} and len(diff) == len(fx.MODELS)
    anyjev = labels(pa, "AnyJev-L0")
    ranked = 0
    for title, rows in blocks(files, RADIO).items():
        assert not bold_labels(rows, column) & anyjev
        if title != stale:  # the cluster-A models stay ranked against the no-update reference, as in the baseline
            assert bold_labels(rows, column) == bold_labels(blocks(baseline, RADIO)[title], column)
            ranked += bool(bold_labels(rows, column))
    assert ranked == 4

    # with no-update also on the M4, the reference follows it: AnyJev-L0 is the only ranked level in the column
    fx.set_binary(c2, fx.run_id(point, "control", "no-update"), fx.SHA["M4"])
    assert pa.C2Data(c2, allow_partial=True).reference_platform(point) == "M4"
    moved, _ = render(pa, c2, tmp_path / "moved", True, monkeypatch)
    assert_no_platform_marker(moved)
    for title, rows in blocks(moved, RADIO).items():
        if title != stale and bold_labels(blocks(baseline, RADIO)[title], column):
            assert bold_labels(rows, column) == anyjev


@pytest.mark.parametrize("mode", fx.MODES)
def test_h1_check_uses_no_update_when_the_oracle_run_is_absent(pa: Any, tmp_path: Path, mode: str) -> None:
    c2 = fx.write_complete(tmp_path / "c2")
    fx.set_binary(c2, fx.run_id(fx.BASE, "GLM-5.3-Flash", "L", mode), fx.SHA["M4"])
    oracle = fx.run_id(fx.BASE, "control", "oracle")
    fx._filter(c2, "radio_kpis.csv", lambda row: row["run_id"] != oracle)
    with pytest.raises(ValueError, match="tab_c2_h1"):
        fx.render(pa, c2, tmp_path / "out")


def resolved_c2(directory: Path) -> Path:
    c2 = fx.write_complete(directory)
    rows = fx._read_csv(c2 / "controls.csv")
    for row in rows:  # every C-2 resolved, so each shown C-2 carries the marker
        row["c2_resolved_both"] = "True"
    fx._write_csv(c2 / "controls.csv", rows)
    return c2


@pytest.mark.parametrize("arm", ["fixed-0.1", "fixed-1", "fixed-5"])
def test_c2_split_across_platforms_renders_a_dash(pa: Any, tmp_path: Path, arm: str) -> None:
    base = fx.render(pa, resolved_c2(tmp_path / "base"), tmp_path / "base_out")
    c2 = resolved_c2(tmp_path / "c2")
    fx.set_binary(c2, fx.run_id(fx.point(0.1, 120.0), "control", arm), fx.SHA["M4"])
    files = fx.render(pa, c2, tmp_path / "out")
    assert_no_platform_marker(files)
    # C-2 and Monotone split; the moved arm's own d column keeps its value; C-1 and every other row are unchanged
    assert changed(base, files, CONTROLS) == {("", ROW, 1): "--", ("", ROW, 5): "--"}
    for _, label, cells in table_rows(text(files, CONTROLS), 12):
        assert (r"$^{\ast}$" in cells[1]) is (label != ROW)


def test_c1_split_renders_a_dash_while_c2_and_monotone_stay_shown(
        pa: Any, baseline: dict[str, bytes], tmp_path: Path) -> None:
    c2 = fx.write_complete(tmp_path / "c2")
    fx.set_binary(c2, fx.run_id(fx.point(0.1, 120.0), "control", "no-update"), fx.SHA["M4"])
    files = fx.render(pa, c2, tmp_path / "out")
    assert_no_platform_marker(files)
    assert changed(baseline, files, CONTROLS) == {("", ROW, 0): "--"}
    cells = dict(blocks(files, CONTROLS)[""])[ROW]
    assert r"$^{\ast}$" in cells[1] and cells[5] == "yes"


def test_contrast_whose_runs_share_a_foreign_platform_is_shown(
        pa: Any, baseline: dict[str, bytes], tmp_path: Path) -> None:
    point = fx.point(0.1, 120.0)
    c2 = fx.write_complete(tmp_path / "c2")
    for arm in fx.CONTROL_ARMS:  # oracle included, so the point's reference platform is the M4
        fx.set_binary(c2, fx.run_id(point, "control", arm), fx.SHA["M4"])
    for model in ("Jev-1.13.0", "DeepSeek-V4.1-Flash"):  # both arms of the DeepSeek gap on the M4
        fx.set_binary(c2, fx.run_id(point, model, "L"), fx.SHA["M4"])
    assert pa.C2Data(c2, allow_partial=False).reference_platform(point) == "M4"
    files = fx.render(pa, c2, tmp_path / "out")
    assert_no_platform_marker(files)

    # C-1, C-2 and Monotone each share the M4: the whole controls table is unchanged, markers included
    assert changed(baseline, files, CONTROLS) == {}
    assert dict(blocks(files, CONTROLS)[""])[ROW] == dict(blocks(baseline, CONTROLS)[""])[ROW]

    # grid: the DeepSeek gap (M4 vs M4) and the Qwen3.5-4B-JSON gap (cluster-A vs cluster-A, off the M4 reference) are shown;
    # GLM and Qwen3.8 (cluster-A vs the M4 anchor) split
    rate_blocks = [title for title in blocks(baseline, GRID) if "0.1 intents/s" in title]
    split = labels(pa, "GLM-5.3-Flash", "Qwen3.8-Flash")
    assert changed(baseline, files, GRID) == {(title, label, 5): "--" for title in rate_blocks for label in split}
    on_reference = labels(pa, "Jev-1.13.0", "DeepSeek-V4.1-Flash")
    for title in rate_blocks:
        rows = blocks(files, GRID)[title]
        for label in labels(pa, "DeepSeek-V4.1-Flash", "Qwen3.5-4B-JSON"):
            assert dict(rows)[label][5] not in ("--", PENDING)
        others = {label for label, _ in rows} - on_reference
        assert bold_labels(rows, 4) == lowest(rows, 4, excluded=others)  # only the two M4 levels are ranked
    assert [bold_labels(blocks(files, GRID)[title], 4) for title in rate_blocks] == [
        labels(pa, "Jev-1.13.0"), labels(pa, "DeepSeek-V4.1-Flash")]

    column = RADIO_COLUMNS.index((0.1, 120.0))
    assert changed(baseline, files, RADIO) == {}
    assert all(bold_labels(rows, column) <= on_reference for rows in blocks(files, RADIO).values())


def test_point_without_any_reference_control_pends_its_radio_cells(
        pa: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    point = fx.point(0.3, 60.0)
    c2 = fx.write_complete(tmp_path / "c2")
    fx.drop_control(c2, point, arm="oracle")
    no_update = fx.run_id(point, "control", "no-update")
    fx._filter(c2, "radio_kpis.csv", lambda row: row["run_id"] != no_update)  # both reference runs absent
    files, _ = render(pa, c2, tmp_path, True, monkeypatch)
    column = [(0.1, 3.0), (0.1, 60.0), (0.1, 120.0), (0.3, 3.0), (0.3, 30.0), (0.3, 60.0)].index((0.3, 60.0))
    for _, _, cells in table_rows(text(files, "paper/tables/tab_c2_radio.tex"), 11):
        assert [cell == PENDING for cell in cells] == [index == column for index in range(10)]
