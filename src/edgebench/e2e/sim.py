"""Discrete-event service simulation generalised to N edge nodes + 1 cloud node.

Supports offline callback mode and replaying recorded external controller timelines.
"""
from __future__ import annotations

from collections import Counter, deque
import heapq
import itertools
import math
from typing import Any, Callable

from src.edgebench.contract import Case
from src.edgebench.e2e.traces import absolute_deadline
from src.edgebench.interpreters.base import Decision
from src.edgebench.scoring import score_decision
from src.intent_cache import IntentCache
from src.schema import FIELDS


def get_speed_factors(n_edges: int) -> list[float]:
    """Service-speed factors: evenly spread over [1.0, 1.3] for N edge nodes, 0.65 for cloud."""
    if n_edges < 1:
        raise ValueError(f"n_edges must be >= 1, got {n_edges}")
    if n_edges == 1:
        edge_factors = [1.0]
    else:
        edge_factors = [1.0 + i * (0.3 / (n_edges - 1)) for i in range(n_edges)]
    return edge_factors + [0.65]  # Index n_edges is cloud


def valid_labels(pred: Any) -> bool:
    return (
        isinstance(pred, dict)
        and set(pred) == set(FIELDS)
        and all(pred[f] in v for f, v in FIELDS.items())
    )


def simulate(
    arrivals: list[dict[str, Any]],
    decide: Callable[[dict[str, Any]], tuple[dict[str, str] | None, float, dict[str, Any]]] | None = None,
    deadline: float = 2.0,
    queue_limit: int = 32,
    timeout: float = 3.0,
    concurrency: int = 1,
    cache: bool = False,
    controller_trace: dict[int, dict[str, Any]] | None = None,
    n_edges: int = 4,
) -> list[dict[str, Any]]:
    """Simulate N edge nodes + 1 cloud node.

    Args:
        arrivals: list of arrival dicts with id, arrival, deadline (or row['deadline']), origin, size_mb, text, truth.
        decide: offline decision callback (ignored if controller_trace is provided).
        deadline: default deadline if not in row.
        queue_limit: max waiting queue in admission.
        timeout: decision timeout.
        concurrency: admission slots.
        cache: enable semantic intent cache.
        controller_trace: mapping arrival id -> admission trace dict.
        n_edges: number of edge nodes (0..n_edges-1 are edges, n_edges is cloud).

    Returns:
        List of arrival dicts with terminal status, latency breakdown, node, violations.
    """
    if not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("invalid concurrency")
    if deadline <= 0 or queue_limit < 0:
        raise ValueError("invalid deadline/queue limit")
    if len({a["id"] for a in arrivals}) != len(arrivals):
        raise ValueError("duplicate arrival ids")
    if controller_trace is not None and set(controller_trace) != {a["id"] for a in arrivals}:
        raise ValueError("incomplete external controller trace")

    total_nodes = n_edges + 1
    cloud_node = n_edges
    speed_factors = get_speed_factors(n_edges)

    events: list[tuple[float, int, str, int]] = []
    counter = itertools.count()
    rows: dict[int, dict[str, Any]] = {}
    waiting: deque[int] = deque()
    active: set[int] = set()
    semantic_cache = IntentCache() if cache else None
    nodes = [dict(running=None, finish=0.0, pending=[]) for _ in range(total_nodes)]

    def event(t: float, kind: str, rid: int) -> None:
        heapq.heappush(events, (t, next(counter), kind, rid))

    for a in arrivals:
        eff_deadline = absolute_deadline(a, deadline)
        rows[a["id"]] = dict(
            a,
            deadline=eff_deadline,
            status="pending",
            locality_violation=False,
            transfer_s=0.0,
            service_s=0.0,
            service_queue_s=0.0,
            node=None,
            predicted=None,
            priority=None,
            joint_correct=False,
            locality_ok=False,
            quality_ok=False,
            decision_start=None,
            decision_end=None,
            decision_timed_out=False,
            network_ready=None,
            service_start=None,
        )
        event(a["arrival"], "arrival", a["id"])

    def prune(now: float) -> None:
        keep: deque[int] = deque()
        while waiting:
            i = waiting.popleft()
            if rows[i]["deadline"] <= now:
                rows[i].update(status="queue_expired", terminal=rows[i]["deadline"])
            else:
                keep.append(i)
        waiting.extend(keep)

    def cached_start(r: dict[str, Any], now: float) -> bool:
        labels = semantic_cache.lookup(r["text"], now) if semantic_cache else None
        if labels is None:
            return False
        r.update(
            decision_start=now,
            queue_wait_s=now - r["arrival"],
            decision_elapsed_s=0.0,
            predicted=labels,
            decision_metadata={"cache_hit": True, "api_calls": 0, "cost_usd": 0.0},
            decision_timed_out=False,
        )
        event(now, "decision_done", r["id"])
        return True

    def start_controller(now: float) -> None:
        if controller_trace is not None:
            return
        prune(now)
        while waiting and len(active) < concurrency:
            rid = waiting.popleft()
            r = rows[rid]
            if cached_start(r, now):
                continue
            active.add(rid)
            r["decision_start"] = now
            if decide is None:
                raise ValueError("No decision callback or controller trace provided")
            labels, elapsed, extra = decide(r)
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("invalid decision time")
            r.update(
                queue_wait_s=now - r["arrival"],
                decision_elapsed_s=elapsed,
                predicted=labels,
                decision_metadata=extra,
                decision_timed_out=elapsed > timeout,
            )
            event(now + min(elapsed, timeout), "decision_done", rid)

    def start_service(node: int, now: float) -> None:
        state = nodes[node]
        if state["running"] is not None:
            return
        ready = [i for i in state["pending"] if rows[i]["network_ready"] <= now]
        if not ready:
            return
        i = min(ready, key=lambda idx: (rows[idx]["priority"], rows[idx]["network_ready"], idx))
        state["pending"].remove(i)
        state["running"] = i
        r = rows[i]
        r["service_start"] = now
        r["service_queue_s"] = now - r["network_ready"]
        state["finish"] = now + r["service_s"]
        event(state["finish"], "service_done", i)

    while events:
        now, _, kind, i = heapq.heappop(events)
        r = rows[i]

        if kind == "arrival":
            if controller_trace is not None:
                trace = controller_trace[i]
                if "decision_start" not in trace:
                    r.update(status=trace["status"], terminal=trace["terminal"])
                    continue
                for key in [
                    "decision_start",
                    "decision_elapsed_s",
                    "predicted",
                    "decision_metadata",
                    "decision_timed_out",
                ]:
                    r[key] = trace[key]
                if r["decision_start"] < r["arrival"] or trace["decision_end"] < r["decision_start"]:
                    raise ValueError("invalid external decision timeline")
                r["queue_wait_s"] = r["decision_start"] - r["arrival"]
                event(trace["decision_end"], "decision_done", i)
                continue

            prune(now)
            if cached_start(r, now):
                continue
            if len(active) >= concurrency and len(waiting) >= queue_limit:
                r.update(status="queue_overflow", terminal=now)
            else:
                waiting.append(i)
                start_controller(now)

        elif kind == "decision_done":
            active.discard(i)
            r["decision_end"] = now
            pred = r["predicted"]
            valid = valid_labels(pred)
            if semantic_cache and valid and not r["decision_timed_out"]:
                semantic_cache.store(r["text"], pred, now)

            if r["decision_timed_out"]:
                r.update(status="decision_timeout", terminal=now)
            elif not valid:
                r.update(status="invalid_decision", terminal=now)
            elif now >= r["deadline"]:
                r.update(status="decision_late", terminal=now)
            elif pred["service_type"] == "unsupported":
                r.update(status="unsupported", terminal=now)
            else:
                candidates = (
                    range(total_nodes) if pred["locality"] == "remote_allowed" else [r["origin"]]
                )
                priority = 0 if pred["urgency"] == "urgent" else 1
                options = []
                for node in candidates:
                    # Link parameters
                    if node == r["origin"]:
                        prop = 0.002
                        bw = 1000.0
                    elif node < n_edges:
                        prop = 0.020
                        bw = 100.0
                    else:  # Cloud node
                        prop = 0.060
                        bw = 50.0

                    transfer = prop + (8.0 * r["size_mb"]) / bw

                    duration = {"count": 0.040, "detection": 0.080, "ocr": 0.060}[
                        pred["service_type"]
                    ]
                    if pred["quality_floor"] == "high":
                        duration *= 1.8
                    duration *= speed_factors[node]

                    state = nodes[node]
                    work = sum(
                        rows[j]["service_s"]
                        for j in state["pending"]
                        if rows[j]["priority"] <= priority
                    )
                    estimate = max(now + transfer, state["finish"]) + work + duration

                    # Scheduler deadline rule: min predicted finish <= deadline, ties -> local
                    if estimate <= r["deadline"]:
                        is_remote = int(node != r["origin"])
                        options.append((estimate, is_remote, node, transfer, duration))

                if not options:
                    r.update(status="no_feasible_node", terminal=now)
                else:
                    # min() chooses lowest estimate; ties broken by is_remote (0=local, 1=remote), then node index
                    _, _, node, transfer, duration = min(options)
                    
                    # Truth locality check for locality violation flag
                    truth = r["truth"]
                    truth_dict = truth[0] if isinstance(truth, list) else truth
                    truth_locality = truth_dict.get("locality", "unspecified")
                    
                    locality_violation = (node != r["origin"]) and (truth_locality != "remote_allowed")

                    r.update(
                        node=node,
                        network_ready=now + transfer,
                        transfer_s=transfer,
                        service_s=duration,
                        priority=priority,
                        locality_violation=locality_violation,
                    )
                    nodes[node]["pending"].append(i)
                    event(now + transfer, "network_ready", i)

            start_controller(now)

        elif kind == "network_ready":
            start_service(r["node"], now)

        elif kind == "service_done":
            node = r["node"]
            nodes[node]["running"] = None
            truth = r["truth"]
            truth_dict = truth[0] if isinstance(truth, list) else truth

            joint = r["predicted"] == truth_dict
            locality_ok = truth_dict.get("locality") == "remote_allowed" or node == r["origin"]
            quality_ok = truth_dict.get("quality_floor") != "high" or r["predicted"].get("quality_floor") == "high"
            good = joint and locality_ok and quality_ok and now <= r["deadline"]

            r.update(
                status="success" if good else "wrong_or_late",
                terminal=now,
                joint_correct=joint,
                locality_ok=locality_ok,
                quality_ok=quality_ok,
            )
            start_service(node, now)

    result = list(rows.values())
    assert all(r["status"] != "pending" for r in result)

    # Compute full latency T = f - a and cost for each row
    for r in result:
        r["T"] = r["terminal"] - r["arrival"]
        meta = r.get("decision_metadata", {})
        r["cost_usd"] = meta.get("cost_usd", 0.0) if meta else 0.0

    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(r["status"] for r in rows)
    waits = [r["queue_wait_s"] for r in rows if "queue_wait_s" in r]
    costs = [r.get("decision_metadata", {}).get("cost_usd", 0.0) for r in rows]
    known = sum(c for c in costs if c is not None)
    
    def is_supp(r: dict[str, Any]) -> bool:
        t = r["truth"]
        t_dict = t[0] if isinstance(t, list) else t
        return t_dict.get("service_type") != "unsupported"

    supported = sum(is_supp(r) for r in rows)
    durations = [r["T"] for r in rows if r["status"] == "success"]

    return dict(
        n=len(rows),
        success_rate=counts["success"] / len(rows) if rows else 0.0,
        counts=dict(counts),
        mean_decision_queue_s=sum(waits) / len(waits) if waits else 0.0,
        useful_throughput_s=(
            counts["success"] / max(1e-12, max(r["terminal"] for r in rows) - min(r["arrival"] for r in rows))
            if rows else 0.0
        ),
        known_decision_cost_usd=known,
        unknown_cost_requests=sum(c is None for c in costs),
        decision_cost_per_success_usd=known / counts["success"] if counts["success"] and None not in costs else None,
        api_calls=sum(r.get("decision_metadata", {}).get("api_calls", 0) for r in rows),
        cache_hits=sum(bool(r.get("decision_metadata", {}).get("cache_hit")) for r in rows),
        locality_violations=sum(bool(r.get("locality_violation")) for r in rows),
        supported_arrivals=supported,
        success_rate_among_supported=counts["success"] / supported if supported else None,
    )
