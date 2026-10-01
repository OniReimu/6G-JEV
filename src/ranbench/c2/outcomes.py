"""Per-intent outcomes and per-run radio KPIs from C2 traces (EXP-2026-003 analysis-plan "Definitions").

A (UE, slot) pair misses when the UE's class has an SLA in SLA_CLASSES and its slot throughput is below the class
floor, or (delay class) its slot delay statistic exceeds the target or no packet arrived. Targets come from the run
config (the truth), never from an interpreter. A window [t_j, t_j + W] covers the slots whose end lies in
(t_j, t_j + W].
"""
from __future__ import annotations

import csv
from typing import Any

import numpy as np

from src.ranbench.c2.loader import SLOT_S, RunTraces

SLA_CLASSES = ("video", "xr", "iot")  # analysis plan: floors for video and IoT, delay target for XR; BE has none
WINDOWS = (1.0, 2.0, 5.0, 10.0)
W_PRIMARY = 2.0
OUTAGE_MBPS = 0.1
EPS = 1e-9


class SlotTable:
    """ue_slots sorted by slot end, with the per-pair miss flag."""

    def __init__(self, tr: RunTraces):
        df = tr.ue_slots.sort_values(["t", "imsi"], kind="stable")
        self.t = df["t"].to_numpy(float)
        self.imsi = df["imsi"].to_numpy()
        self.cell = df["cell"].to_numpy()
        self.cls = df["cls"].astype(str).to_numpy()
        self.n_rx = df["n_rx"].to_numpy(float)
        self.mean_delay = df["mean_delay_ms"].to_numpy(float)
        self.thr_kbps = df["rx_bytes"].to_numpy(float) * 8.0 / SLOT_S / 1000.0
        self.miss = np.zeros(len(df), dtype=bool)
        for cls in SLA_CLASSES:
            spec = tr.class_targets.get(cls)
            if spec is None:
                continue
            m = self.cls == cls
            kind, v = spec["target_type"], float(spec["target_value"])
            if kind == "throughput_floor_kbps":
                self.miss[m] = self.thr_kbps[m] < v
            elif kind in ("delay_p95_ms", "delay_mean_ms"):
                col = "p95_delay_ms" if kind == "delay_p95_ms" else "mean_delay_ms"
                d = df[col].to_numpy(float)[m]
                self.miss[m] = (self.n_rx[m] == 0) | (d > v)
            else:
                raise ValueError(f"unknown target_type {kind} for {cls}")
        self.clusters = {k: np.array(v) for k, v in tr.clusters.items()}

    def span(self, a: float, b: float) -> slice:
        """Rows whose slot end lies in (a, b]."""
        return slice(int(np.searchsorted(self.t, a + EPS, "right")), int(np.searchsorted(self.t, b + EPS, "right")))

    def member(self, sl: slice, scope: str, cls: str) -> np.ndarray:
        return np.isin(self.cell[sl], self.clusters[scope]) & (self.cls[sl] == cls)

    def share(self, a: float, b: float, scope: str | None = None, cls: str | None = None) -> float | None:
        sl = self.span(a, b)
        miss = self.miss[sl]
        if scope is not None:
            miss = miss[self.member(sl, scope, cls)]
        return float(miss.mean()) if len(miss) else None

    def exposure(self, a: float, b: float, scope: str, cls: str,
                 exclude: tuple[str, str] | None = None) -> float:
        """UE-seconds of class `cls` UEs served in `scope` over (a, b] (optionally minus another (scope, cls) set)."""
        if not b > a:
            return 0.0
        sl = self.span(a, b + SLOT_S)
        m = self.member(sl, scope, cls)
        if exclude is not None:
            m &= ~self.member(sl, *exclude)
        t = self.t[sl][m]
        return float(np.clip(np.minimum(t, b) - np.maximum(t - SLOT_S, a), 0.0, None).sum())


def _next_same_issue(events: list[dict[str, Any]]) -> list[float]:
    """Issue time of the next intent with the same (c2 class, scope) truth; inf if none."""
    out, last = [np.inf] * len(events), {}
    for k in sorted(range(len(events)), key=lambda k: events[k]["t_issue_s"], reverse=True):
        key = tuple(events[k]["true_install"][:2])
        out[k] = last.get(key, np.inf)
        last[key] = events[k]["t_issue_s"]
    return out


def _next_same_install(events: list[dict[str, Any]]) -> list[float]:
    """Time of this arm's next install on the same (scope, class) as event k's install; inf if none."""
    out = [np.inf] * len(events)
    inst = sorted((k for k, e in enumerate(events) if e["install"] and e["e_s"] is not None),
                  key=lambda k: (events[k]["e_s"], events[k]["j"]))
    last: dict[tuple, float] = {}
    for k in reversed(inst):
        key = tuple(events[k]["install"][:2])
        out[k] = last.get(key, np.inf)
        last[key] = events[k]["e_s"]
    return out


def intent_outcomes(tr: RunTraces, events: list[dict[str, Any]], slots: SlotTable | None = None,
                    t_end: float | None = None, windows: tuple[float, ...] = WINDOWS) -> list[dict[str, Any]]:
    """One row per intent event of one arm run."""
    st = slots or SlotTable(tr)
    t_end = tr.sim_time_s if t_end is None else t_end
    nxt_issue, nxt_inst = _next_same_issue(events), _next_same_install(events)
    rows = []
    for k, ev in enumerate(events):
        t = ev["t_issue_s"]
        scope, cls = ev["true_install"][:2]
        e = ev["e_s"] if ev["e_s"] is not None else np.inf
        row: dict[str, Any] = {"j": ev["j"], "intent_id": ev["intent_id"], "t_issue_s": t, "cls": cls,
                               "scope": scope, "correct": ev["correct"],
                               "E_s": e - t if ev["correct"] and np.isfinite(e) else np.inf}
        for w in windows:
            row[f"aff_W{w:g}"] = st.share(t, t + w, scope, cls) if cls in SLA_CLASSES else None
            row[f"net_W{w:g}"] = st.share(t, t + w)
        installed = ev["install"] is not None and e <= t_end
        life_end = min(nxt_inst[k], t_end)
        row["aff_life"] = st.share(e, life_end, scope, cls) if installed and cls in SLA_CLASSES else None
        row["net_life"] = st.share(e, life_end) if installed else None
        s_end = min(e if ev["correct"] else np.inf, nxt_issue[k], t_end)
        row["S_ue_s"] = st.exposure(t, s_end, scope, cls)
        wrong = installed and not ev["correct"]
        row["collateral_ue_s"] = (
            st.exposure(e, life_end, ev["install"][0], ev["install"][1], exclude=(scope, cls)) if wrong else 0.0
        )
        rows.append(row)
    return rows


def actuation(events: list[dict[str, Any]], clusters: dict[str, list[int]],
              priorities: dict[str, Any]) -> list[dict[str, Any]]:
    """Deviation D-5: per intent, whether its true policy changes the installed base level of an SLA class in its
    scope, and the intended direction (+1 raises the class's priority, -1 lowers it, 0 non-actuating).

    `events` are the ORACLE arm's events: its installs (in schedule order, (e_s rounded as in schedule.csv, j)) are
    replayed on a per-(cell, class) base level that starts at `default_policy` in every cell, as the scenario does.
    The level active just before intent j's enforcement is compared with its true install level. Levels are
    m_priority values; the scheduler weight is proportional to (100 - m_priority), so a LOWER value is a HIGHER
    priority and direction = sign(sum over scope cells of (prior - new))."""
    levels = {k: int(v) for k, v in priorities["levels"].items()}
    default = levels[priorities["default_policy"]]
    value = lambda lvl: default if lvl == "default" else levels[lvl]
    base: dict[tuple[int, str], int] = {}
    order = sorted((k for k, e in enumerate(events) if e["install"] and e.get("in_schedule", True)
                    and e["e_s"] is not None), key=lambda k: (round(events[k]["e_s"], 6), events[k]["j"]))
    prior: dict[int, list[int]] = {}
    for k in order:
        scope, cls, lvl = events[k]["install"]
        prior[k] = [base.get((c, cls), default) for c in clusters[scope]]
        for c in clusters[scope]:
            base[(c, cls)] = value(lvl)
    rows = []
    for k, ev in enumerate(events):
        scope, cls, lvl = ev["true_install"]
        new = value(lvl)
        before = prior.get(k)
        d = 0
        if before is not None and cls in SLA_CLASSES:
            d = int(np.sign(sum(b - new for b in before)))
        rows.append({"j": ev["j"], "cls": cls, "scope": scope, "level": lvl, "new": new,
                     "prior": sorted(set(before)) if before is not None else None,
                     "actuating": d != 0, "direction": d})
    return rows


def stale_policy_exposure(tr: RunTraces, events: list[dict[str, Any]], actuating: set[int],
                          slots: SlotTable | None = None) -> dict[str, Any]:
    """Stale-policy exposure of one arm run (descriptive KPI; no Holm family, no CI).

    For each ACTUATING intent j (`actuating`: the D-5 actuation set of the C-1 control, i.e. SLA-class true installs
    that change the oracle-arm base level in scope, as `actuation` computes it on the oracle arm's events):
      t_enf  = the time at which this arm's schedule installs j's policy, as recorded in the run's schedule.csv. If
               the arm never installs it before the next intent affecting the same (class, scope) or before run end,
               t_enf is capped at that time.
      n_UEs  = the UEs of j's class served in j's scope cells at t_issue (the ue_slots slot that contains t_issue).
      exposure_j = n_UEs * (t_enf - t_issue), in UE-seconds.
    Returns the mean exposure over actuating intents (UE s per intent) and the median of t_enf - t_issue (s).
    """
    st = slots or SlotTable(tr)
    t_end = tr.sim_time_s
    with open(tr.run_dir / "schedule.csv", newline="", encoding="utf-8") as handle:
        schedule = {(round(float(r["t_s"]), 6), r["scope"], r["class"], r["priority"])
                    for r in csv.DictReader(handle)}
    nxt_issue = _next_same_issue(events)
    exposure, delay = [], []
    for k, ev in enumerate(events):
        if ev["j"] not in actuating:
            continue
        t = ev["t_issue_s"]
        scope, cls = ev["true_install"][:2]
        cap = min(nxt_issue[k], t_end)
        t_enf = cap
        if ev["install"] == ev["true_install"] and ev.get("in_schedule", True) and ev["e_s"] is not None:
            row = (round(ev["e_s"], 6), *ev["install"])
            if row not in schedule:
                raise RuntimeError(f"{tr.run_dir}: install of intent {ev['j']} {row} missing from schedule.csv")
            t_enf = min(row[0], cap)
        i = int(np.searchsorted(st.t, t - EPS, "left"))  # first slot ending at or after t_issue
        n_ue = int(st.member(st.span(st.t[i] - SLOT_S / 2, st.t[i]), scope, cls).sum()) if i < len(st.t) else 0
        exposure.append(n_ue * (t_enf - t))
        delay.append(t_enf - t)
    return {
        "mean_ue_s": float(np.mean(exposure)) if exposure else None,
        "median_delay_s": float(np.median(delay)) if delay else None,
        "n_actuating": len(exposure),
    }


def policy_rows_check(enforcement, events: list[dict[str, Any]], clusters: dict[str, list[int]],
                      priorities: dict[str, Any]) -> dict[str, int]:
    """Every `policy` row of the oracle run's enforcement.csv must carry the level of the oracle install at that time
    on that (cell, class). (The scenario logs a row only when the level changes, so installs without rows are fine.)"""
    levels = {k: int(v) for k, v in priorities["levels"].items()}
    default = levels[priorities["default_policy"]]
    want: dict[tuple[float, int, str], int] = {}
    for e in sorted((e for e in events if e["install"] and e.get("in_schedule", True) and e["e_s"] is not None),
                    key=lambda e: (round(e["e_s"], 6), e["j"])):
        scope, cls, lvl = e["install"]
        for c in clusters[scope]:
            want[(round(e["e_s"], 6), int(c), cls)] = default if lvl == "default" else levels[lvl]
    rows = enforcement[enforcement["source"].astype(str) == "policy"]
    bad = sum(want.get((round(float(t), 6), int(c), str(k))) != int(n)
              for t, c, k, n in zip(rows["t"], rows["cell"], rows["cls"], rows["new"]))
    return {"n_policy_rows": int(len(rows)), "n_mismatch": int(bad)}


def network_series(tr: RunTraces, slots: SlotTable | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(slot end, network-wide miss share) per slot: the series whose autocorrelation sets the block length."""
    st = slots or SlotTable(tr)
    ts, inv = np.unique(st.t, return_inverse=True)
    return ts, np.bincount(inv, weights=st.miss.astype(float)) / np.bincount(inv)


def _wpct(values: np.ndarray, weights: np.ndarray, q: float) -> float | None:
    """Weighted percentile (inverted CDF)."""
    ok = np.isfinite(values) & (weights > 0)
    if not ok.any():
        return None
    v, w = values[ok], weights[ok]
    o = np.argsort(v, kind="stable")
    cw = np.cumsum(w[o])
    return float(v[o][min(int(np.searchsorted(cw, q / 100.0 * cw[-1], "left")), len(v) - 1)])


def jain_index(values: np.ndarray) -> float | None:
    """Jain's fairness index (sum x)^2 / (n sum x^2); None when there is no value or every value is 0."""
    x = np.asarray(values, float)
    s2 = float((x * x).sum())
    return float(x.sum() ** 2 / (len(x) * s2)) if len(x) and s2 > 0 else None


def class_jain(st: SlotTable, t_from: float, t_to: float) -> dict[str, float | None]:
    """Per class, Jain's index of the per-UE mean slot throughput over slots ending in (t_from, t_to]."""
    sl = st.span(t_from, t_to)
    thr, cls, imsi = st.thr_kbps[sl], st.cls[sl], st.imsi[sl]
    out: dict[str, float | None] = {}
    for c in sorted(set(cls)):
        m = cls == c
        _, inv = np.unique(imsi[m], return_inverse=True)
        out[c] = jain_index(np.bincount(inv, weights=thr[m]) / np.bincount(inv))
    return out


def radio_kpis(tr: RunTraces, t_from: float, t_to: float | None = None,
               slots: SlotTable | None = None) -> dict[str, Any]:
    """Per-run radio KPIs over slots ending in (t_from, t_to]. Delay percentiles are packet-weighted over the
    per-(UE, slot) mean delays (the scenario does not export per-packet delays)."""
    st = slots or SlotTable(tr)
    t_to = tr.sim_time_s if t_to is None else t_to
    sl = st.span(t_from, t_to)
    thr = st.thr_kbps[sl] / 1000.0
    cls, nrx, dly, imsi = st.cls[sl], st.n_rx[sl], st.mean_delay[sl], st.imsi[sl]
    out: dict[str, Any] = {"t_from_s": t_from, "t_to_s": t_to, "n_ue_slots": int(len(thr))}
    groups = {"all": np.ones(len(thr), dtype=bool)} | {c: cls == c for c in sorted(set(cls))}
    for g, m in groups.items():
        sfx = "" if g == "all" else f".{g}"
        out[f"thr_mean_mbps{sfx}"] = float(thr[m].mean()) if m.any() else None
        # 5th percentile across UEs of each UE's mean slot throughput over the window (pre-registered definition)
        if m.any():
            _, inv = np.unique(imsi[m], return_inverse=True)
            out[f"thr_p5_mbps{sfx}"] = float(np.percentile(np.bincount(inv, weights=thr[m]) / np.bincount(inv), 5))
        else:
            out[f"thr_p5_mbps{sfx}"] = None
        out[f"outage_share{sfx}"] = float((thr[m] < OUTAGE_MBPS).mean()) if m.any() else None
        for q in (25, 50, 75, 95, 99):
            out[f"delay_p{q}_ms{sfx}"] = _wpct(dly[m], nrx[m], q)
        p25, p75 = out.pop(f"delay_p25_ms{sfx}"), out.pop(f"delay_p75_ms{sfx}")
        out[f"jitter_iqr_ms{sfx}"] = None if p25 is None else p75 - p25
    prb = tr.cell_prb[(tr.cell_prb["t"] > t_from + EPS) & (tr.cell_prb["t"] <= t_to + EPS)]
    out["prb_util_mean"] = float(prb["prb_share"].mean()) if len(prb) else None
    out["prb_util_per_cell"] = {int(c): float(v) for c, v in prb.groupby("cell")["prb_share"].mean().items()}
    ho = tr.handover.copy()
    t_ho = ho["t_start"].astype(float).fillna(ho["t_end"].astype(float)) if len(ho) else ho["t_start"]
    ho = ho[(t_ho > t_from) & (t_ho <= t_to)] if len(ho) else ho
    ok = ho[ho["outcome"] == "ok"] if "outcome" in ho.columns else ho
    n_ue = len(np.unique(st.imsi))
    out["ho_count"] = int(len(ok))
    out["ho_fail_count"] = int((ho["outcome"] == "end_error").sum()) if "outcome" in ho.columns else None
    out["ho_rate_per_ue_s"] = len(ok) / (n_ue * (t_to - t_from)) if n_ue and t_to > t_from else None
    tot = ok["total_ms"].astype(float).dropna().to_numpy() if len(ok) else np.array([])
    out["ho_total_time_p50_ms"] = float(np.percentile(tot, 50)) if len(tot) else None
    out["ho_total_time_p95_ms"] = float(np.percentile(tot, 95)) if len(tot) else None
    rlf_t = tr.rlf["t"].astype(float)
    out["rlf_count"] = int(((rlf_t > t_from) & (rlf_t <= t_to)).sum())
    out["rlf_count_total"] = int(len(tr.rlf))
    return out
