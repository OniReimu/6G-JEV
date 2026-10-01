"""Small A1 simulator client with client-side PUT acknowledgement timing."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import time
from typing import Any, Callable, Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.ranbench.schemas import load_policy_schema

POLICY_TYPE_ID = 20008
POLICY_TYPE = {
    "policySchema": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "EXP-2026-003 C3 RANIntent policy instance",
        "type": "object",
        "required": ["intent_id", "interpreter", "ran_intent", "expected_control"],
        "properties": {
            "intent_id": {"type": "string"},
            "interpreter": {"type": "string"},
            "ran_intent": load_policy_schema(),
            "expected_control": {
                "type": "object",
                "required": ["slice_id", "min_prb_ratio", "max_prb_ratio", "dedicated_prb_ratio"],
                "properties": {
                    "slice_id": {"type": "integer", "const": 0},
                    "min_prb_ratio": {"type": "integer", "const": 0},
                    "max_prb_ratio": {"type": "integer", "enum": [5, 30, 100]},
                    "dedicated_prb_ratio": {"type": "integer", "const": 100},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    },
    "statusSchema": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"enforceStatus": {"type": "string", "enum": ["ENFORCED", "NOT_ENFORCED"]}},
    },
}


def iso_utc(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: bytes


class Transport(Protocol):
    def request(self, method: str, url: str, body: bytes | None, timeout_s: float) -> HttpResult: ...


class UrllibTransport:
    def request(self, method: str, url: str, body: bytes | None, timeout_s: float) -> HttpResult:
        request = Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - operator supplies the A1 URL
                return HttpResult(status=response.status, body=response.read())
        except HTTPError as exc:
            return HttpResult(status=exc.code, body=exc.read())


@dataclass(frozen=True)
class PutAcknowledgement:
    policy_id: str
    status: int
    t_request_s: float
    t_ack_s: float
    elapsed_s: float
    response_body: str


class A1Client:
    def __init__(
        self,
        base_url: str,
        *,
        policy_type_id: int = POLICY_TYPE_ID,
        timeout_s: float = 5.0,
        transport: Transport | None = None,
        clock: Callable[[], float] = time.time,
        log: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.policy_type_id = policy_type_id
        self.timeout_s = timeout_s
        self.transport = transport or UrllibTransport()
        self.clock = clock
        self.log = log

    @property
    def policy_type_url(self) -> str:
        return f"{self.base_url}/A1-P/v2/policytypes/{self.policy_type_id}"

    def ensure_policy_type(self) -> int:
        existing = self.transport.request("GET", self.policy_type_url, None, self.timeout_s)
        if existing.status == 200:
            return existing.status
        if existing.status != 404:
            raise RuntimeError(
                f"A1 policy-type GET returned HTTP {existing.status}: {existing.body.decode(errors='replace')}"
            )
        result = self.transport.request(
            "PUT",
            f"{self.base_url}/policytype?id={self.policy_type_id}",
            json.dumps(POLICY_TYPE).encode("utf-8"),
            self.timeout_s,
        )
        if result.status not in (200, 201, 204):
            raise RuntimeError(f"A1 policy-type PUT returned HTTP {result.status}: {result.body.decode(errors='replace')}")
        return result.status

    def put_policy(self, policy_id: str, policy: dict[str, Any]) -> PutAcknowledgement:
        url = f"{self.policy_type_url}/policies/{policy_id}"
        body = json.dumps(policy, separators=(",", ":")).encode("utf-8")
        t_request = self.clock()
        result = self.transport.request("PUT", url, body, self.timeout_s)
        t_ack = self.clock()  # first userspace instant after the HTTP response is received
        if result.status not in (200, 201, 202, 204):
            raise RuntimeError(f"A1 policy PUT returned HTTP {result.status}: {result.body.decode(errors='replace')}")
        ack = PutAcknowledgement(
            policy_id=policy_id,
            status=result.status,
            t_request_s=t_request,
            t_ack_s=t_ack,
            elapsed_s=t_ack - t_request,
            response_body=result.body.decode(errors="replace"),
        )
        if self.log is not None:
            self.log({
                "event": "a1_put_ack",
                "policy_id": policy_id,
                "status": result.status,
                "t_request": iso_utc(t_request),
                "t_utc": iso_utc(t_ack),
                "elapsed_s": ack.elapsed_s,
            })
        return ack
