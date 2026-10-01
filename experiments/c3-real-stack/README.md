# C3: the real A1 and E2 control path (RQ6)

Each interpreter decides 30 clean intents from the C2/C3 pool live. The driver PUTs each policy to an O-RAN SC A1
simulator, an xApp in the O-RAN SC near-RT RIC polls the A1 policy every 10 ms and maps it to an E2SM-RC slice-level
PRB quota, and the srsRAN gNB acknowledges the RIC control. KPM reports (about 0.2 s period), the gNB E2 log, and an
iperf3 stream through srsUE record the change. Per intent the records give the decision latency, the A1 delay (PUT
acknowledgement plus poll wait), the E2 delay (control sent to gNB acknowledgement), and an upper bound on the first
KPM-observed change. The median A1 PUT-acknowledgement time of the hosted runs (4.3 ms) plus half the poll period sets
the A1 delay of 9.3 ms used in C2.

The analysis is descriptive: median and percentiles per component over the 30 intents of each interpreter.

## Results in this directory

| File | Content |
|---|---|
| `results/c3_path_decomposition.csv` | Per interpreter: decision latency, A1 delay, E2 delay, and KPM bound (median and p95), counts, and the tunnel round-trip time of the self-hosted servers |
| `results/c3_path_samples.csv` | The per-intent samples behind the figure, written by `scripts/paper_assets.py`, which checks every percentile against the table above |
| `results/runs/<run>/records.jsonl`, `integrity.json` | Per-intent records and the integrity report of each main run |
| `results/runs/selfhosted-rtt/` | Round-trip time through the SSH tunnel to the self-hosted servers, before and after each run |

## Re-running

The stack runs on one Docker host (the paper used Docker Desktop on an Apple M4 Max). Prepare it once, from the
repository root:

```bash
REPO=$(pwd)
cd configs/ranbench/realstack
cp configs/open5gs.env.example configs/open5gs.env          # Open5GS settings and the srsUE test subscriber
git clone https://github.com/srsran/oran-sc-ric.git && git -C oran-sc-ric checkout 621ade2
cp "$REPO/src/ranbench/c3/xapp.py" oran-sc-ric/xApps/python/c3_xapp.py
cp "$REPO/src/ranbench/c3/mapping.py" oran-sc-ric/xApps/python/mapping.py
docker build -t rtmgr_sim:i-release -f oran-sc-ric/ric/images/rtmgr_sim/Dockerfile oran-sc-ric/ric/images/rtmgr_sim
docker build -t python_xapp_runner:i-release -f oran-sc-ric/ric/images/ric-plt-xapp-frame-py/Dockerfile \
    oran-sc-ric/ric/images/ric-plt-xapp-frame-py
docker build -f Dockerfile.srsue -t srsue:local .
cd "$REPO"
```

The other images are pulled at the tags named in `docker-compose.yml`. Then, per interpreter, from the repository root.
Every log goes into the run directory that the analysis reads:

```bash
REPO=$(pwd)
INTERPRETER=Jev-1.13.0                     # Jev-1.13.0, DeepSeek-V4.1-Flash, GLM-5.3-Flash, Qwen3.8-Flash, or a self-hosted one
RUN_ID=c3main-$(echo "$INTERPRETER" | tr 'A-Z' 'a-z')-$(date +%Y%m%d%H%M)
RUN_DIR=$REPO/experiments/c3-real-stack/results/runs/$RUN_ID
mkdir -p "$RUN_DIR"
cd configs/ranbench/realstack
docker compose up -d                                   # Open5GS, gNB, srsUE, near-RT RIC
until docker logs open5gs_5gc 2>&1 | grep -q "AMF initialize...done"; do sleep 1; done
docker compose restart gnb srsue                       # once, after Open5GS is ready
docker logs --since 0s -f ocudu_gnb > "$RUN_DIR/gnb.log" 2>&1 &
until docker exec srsue sh -c "ip addr show tun_srsue 2>/dev/null | grep -q 10.45.1.2"; do sleep 1; done
docker run -d --name a1simulator --network realstack_ric_network -p 127.0.0.1:8085:8085 \
    -e A1_VERSION=STD_2.0.0 -e ALLOW_HTTP=true --platform linux/amd64 \
    nexus3.o-ran-sc.org:10002/o-ran-sc/a1-simulator:2.8.1
until curl -fsS http://127.0.0.1:8085/A1-P/v2/policytypes > /dev/null; do sleep 1; done
docker exec -e C3_RUN_ID="$RUN_ID" python_xapp_runner python3 /opt/xApps/c3_xapp.py >> "$RUN_DIR/xapp.jsonl" 2>&1 &
until grep -q '"event":"xapp_ready"' "$RUN_DIR/xapp.jsonl"; do sleep 1; done
docker exec -d open5gs_5gc iperf3 -s -B 10.45.1.1
docker exec srsue iperf3 -c 10.45.1.1 -t 3600 -J > "$RUN_DIR/iperf3.json" 2> "$RUN_DIR/iperf3.stderr" &
cd "$REPO"
uv run python -m src.ranbench.c3.driver --interpreter "$INTERPRETER" --out "$RUN_DIR/records.jsonl" \
    --iperf3-log "$RUN_DIR/iperf3.json" --xapp-log "$RUN_DIR/xapp.jsonl" --gnb-log "$RUN_DIR/gnb.log" \
    --run-id "$RUN_ID"                                 # also writes records.a1.jsonl next to records.jsonl
docker exec srsue pkill -INT -x iperf3; sleep 5
uv run python scripts/rb_c3_check.py --records "$RUN_DIR/records.jsonl" --a1 "$RUN_DIR/records.a1.jsonl" \
    --xapp "$RUN_DIR/xapp.jsonl" --gnb "$RUN_DIR/gnb.log" --iperf3 "$RUN_DIR/iperf3.json" --expected 30 \
    --output "$RUN_DIR/integrity.json"
kill $(jobs -p)
(cd configs/ranbench/realstack && docker rm -f a1simulator && docker compose down -v)
```

Self-hosted interpreters were served from the GPU server through an SSH tunnel, with the driver given
`--base-url http://127.0.0.1:8700` (SemIf-Qwen3.5-4B, 8000 for AnyJev-L0, 8900 for Qwen3.5-4B-JSON).
`scripts/rb_c3_tunnel_rtt.py` measured the tunnel round-trip time before and after each run:

```bash
uv run python scripts/rb_c3_tunnel_rtt.py --url http://127.0.0.1:8700/health --phase before \
    --output experiments/c3-real-stack/results/runs/selfhosted-rtt/semif.jsonl
```

**Analysis.** With the raw run records placed under `results/runs/` and the RQ3 traces under
`experiments/c1-interpretation/results/rq3-load/traces/`:

```bash
uv run python scripts/c3_rq3_analysis.py
```

## Recorded deviations

The design used the A1 simulator with a polling xApp because the near-RT RIC compose of the i-release has no A1
mediator. The SemIf-Qwen3.5-4B tunnel round-trip time was measured on the same node and path before a rerun of that
interpreter, as noted in `c3_path_decomposition.csv`. No other departure from the protocol applies to C3.
