from __future__ import annotations

import json

from scripts.rb_c3_tunnel_rtt import inclusive_p95, measure


class Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps({"status": "ok"}).encode()


class EmptyResponse(Response):
    def read(self) -> bytes:
        return b""


def test_measure_and_inclusive_p95() -> None:
    times = iter([1.0, 1.001, 2.0, 2.003, 3.0, 3.002])
    values = measure("http://127.0.0.1:18000/health", 3, opener=lambda *args, **kwargs: Response(),
                     clock=times.__next__)
    assert values == [0.0009999999999998899, 0.0030000000000001137, 0.0019999999999997797]
    assert inclusive_p95([1.0, 2.0, 3.0]) == 2.9
    empty_times = iter([4.0, 4.001])
    assert measure("http://127.0.0.1:18000/health", 1, opener=lambda *args, **kwargs: EmptyResponse(),
                   clock=empty_times.__next__) == [0.001000000000000334]
