#!/usr/bin/env python3
"""C3 polling xApp for the O-RAN SC i-release Python xApp runner.

This module intentionally uses the same E2 setup and
``control_slice_level_prb_quota`` call as spike B2.  The A1 simulator is
polled every 10 ms because the i-release compose has no A1 mediator.
"""
from __future__ import annotations

import datetime
import json
import os
import signal
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from src.ranbench.c3.mapping import control_for_policy
except ImportError:  # copied next to mapping.py in /opt/xApps on the real stack
    from mapping import control_for_policy

POLL_PERIOD_S = 0.010
DEFAULT_A1_BASE_URL = "http://a1simulator:8085"
DEFAULT_POLICY_TYPE_ID = 20008


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def emit(event: dict) -> None:
    print(json.dumps(event, separators=(",", ":"), sort_keys=True), flush=True)


def policy_belongs_to_run(policy_id: str, run_id: str) -> bool:
    return policy_id.startswith(f"c3-{run_id}-")


def submit_control(
    e2sm_rc,
    e2_node_id: str,
    control,
    event: dict,
    *,
    now=utc_now,
    emit_event=emit,
) -> None:
    """Submit one E2 control and publish evidence only after the call succeeds."""
    t_control_sent = now()  # pre-call boundary required by the C3 latency definition
    requestor_id = (int(e2sm_rc.requestorID) + 1) % 255
    e2sm_rc.control_slice_level_prb_quota(
        e2_node_id,
        control.slice_id,
        min_prb_ratio=control.min_prb_ratio,
        max_prb_ratio=control.max_prb_ratio,
        dedicated_prb_ratio=control.dedicated_prb_ratio,
        ack_request=1,
    )
    emit_event(dict(event, t_control_sent=t_control_sent, e2_requestor_id=requestor_id))


class A1PolicyReader:
    def __init__(self, base_url: str, policy_type_id: int, timeout_s: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.policy_type_id = policy_type_id
        self.timeout_s = timeout_s

    @property
    def policies_url(self) -> str:
        return f"{self.base_url}/A1-P/v2/policytypes/{self.policy_type_id}/policies"

    def _get_json(self, url: str):
        request = Request(url, method="GET", headers={"Accept": "application/json"})
        with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 - fixed operator A1 endpoint
            return json.loads(response.read().decode("utf-8"))

    def policy_ids(self) -> list[str]:
        body = self._get_json(self.policies_url)
        if isinstance(body, list):
            return [str(value) for value in body]
        if isinstance(body, dict):
            values = body.get("policy_ids", body.get("policies", []))
            return [str(value) for value in values]
        raise ValueError(f"unexpected A1 policy-list response: {type(body).__name__}")

    def policy(self, policy_id: str) -> dict:
        body = self._get_json(f"{self.policies_url}/{policy_id}")
        if not isinstance(body, dict):
            raise ValueError("A1 policy instance is not a JSON object")
        return body


def make_xapp_class():
    """Import the stack-only xApp library lazily so package tests need no RIC image."""
    from lib.xAppBase import xAppBase

    class C3Xapp(xAppBase):
        def __init__(
            self,
            reader: A1PolicyReader,
            run_id: str,
            http_server_port: int = 8090,
            rmr_port: int = 4560,
        ) -> None:
            super().__init__("", http_server_port, rmr_port)
            self.reader = reader
            self.run_id = run_id
            self.seen_policy_ids: set[str] = set()

        def kpm_callback(self, e2_agent_id, subscription_id, indication_hdr, indication_msg) -> None:
            meas_data = self.e2sm_kpm.extract_meas_data(indication_msg)
            throughput = meas_data["measData"].get("DRB.UEThpDl", [0.0])[0]
            prbs = meas_data["measData"].get("RRU.PrbTotDl", [0])[0]
            emit({
                "event": "kpm",
                "t_utc": utc_now(),
                "DRB.UEThpDl": round(float(throughput), 2),
                "RRU.PrbTotDl": int(prbs),
            })

        def _poll_a1(self, e2_node_id: str) -> None:
            while True:
                poll_started = time.monotonic()
                try:
                    for policy_id in self.reader.policy_ids():
                        if policy_id in self.seen_policy_ids or not policy_belongs_to_run(policy_id, self.run_id):
                            continue
                        body = self.reader.policy(policy_id)
                        t_seen = utc_now()
                        policy = body.get("ran_intent", body)
                        intent_id = str(body.get("intent_id", policy_id))
                        control = control_for_policy(policy)
                        event = {
                            "event": "control_sent",
                            "run_id": self.run_id,
                            "policy_id": policy_id,
                            "intent_id": intent_id,
                            "t_seen": t_seen,
                            "control": control.to_dict(),
                        }
                        # Exact E2SM-RC call exercised by spike_b2_xapp.py.
                        # The evidence event is emitted only if submission succeeds.
                        submit_control(
                            self.e2sm_rc,
                            e2_node_id,
                            control,
                            event,
                        )
                        self.seen_policy_ids.add(policy_id)
                except (HTTPError, URLError, TimeoutError, ValueError, KeyError, RuntimeError, OSError) as exc:
                    emit({"event": "poll_error", "t_utc": utc_now(), "error": type(exc).__name__, "detail": str(exc)})
                delay = POLL_PERIOD_S - (time.monotonic() - poll_started)
                if delay > 0:
                    time.sleep(delay)

        @xAppBase.start_function
        def start(self, e2_node_id: str) -> None:
            self.e2sm_kpm.set_ran_func_id(2)
            self.e2sm_rc.set_ran_func_id(3)
            self.e2sm_kpm.subscribe_report_service_style_1(
                e2_node_id,
                100,
                ["DRB.UEThpDl", "RRU.PrbTotDl"],
                100,
                self.kpm_callback,
            )
            emit({"event": "xapp_ready", "t_utc": utc_now(), "poll_period_ms": 10, "e2_node_id": e2_node_id})
            worker = threading.Thread(target=self._poll_a1, args=(e2_node_id,), daemon=True)
            worker.start()

    return C3Xapp


def main() -> None:
    base_url = os.environ.get("C3_A1_BASE_URL", DEFAULT_A1_BASE_URL)
    policy_type_id = int(os.environ.get("C3_POLICY_TYPE_ID", str(DEFAULT_POLICY_TYPE_ID)))
    e2_node_id = os.environ.get("C3_E2_NODE_ID", "gnbd_001_001_00019b_0")
    run_id = os.environ.get("C3_RUN_ID")
    if not run_id:
        raise RuntimeError("C3_RUN_ID is required so the xApp ignores policies from other runs")
    cls = make_xapp_class()
    xapp = cls(A1PolicyReader(base_url, policy_type_id), run_id)
    signal.signal(signal.SIGQUIT, xapp.signal_handler)
    signal.signal(signal.SIGTERM, xapp.signal_handler)
    signal.signal(signal.SIGINT, xapp.signal_handler)
    xapp.start(e2_node_id)


if __name__ == "__main__":
    main()
