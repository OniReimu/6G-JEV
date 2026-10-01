# C1: the interpretation layer on RANIntent v1, and live load traces

Every interpreter reads the same intent text, the same KPM telemetry table, the same reading rules, and the same
policy fields, and returns an A1 policy that is validated against the policy type's JSON Schema. The runs record
decision latency, correctness per field, billed API fees, and, for the self-hosted interpreters, GPU energy. A second
set of live runs sends Poisson intent streams through four interpretation slots to measure queueing under load.

| RQ | What is measured | Conditions |
|---|---|---|
| RQ1 | Near-RT feasibility: the share of decisions with decision latency + 5 ms E2 control time within 1 s | All 16 RANIntent conditions |
| RQ3 | Queue wait, slot utilization, and the share of intents enforced within 1 s, at 0.1, 0.5, 1, and 2 intents/s | 300 clean actuatable intents per interpreter and rate |
| RQ4 | Accuracy on the telemetry-dependent field `target_cluster`, full-policy match, schema validity, unsafe-policy rate, latency, tokens, fees | Telemetry of 3, 7, 21, or 57 cells, each fresh, stale, noisy, or contradictory |

Each RANIntent condition has 300 test cases (150 state-dependent, 150 named-scope) and 60 development cases. The
development cases served only prompt checks.

## Pre-registered hypothesis

- **H3 (RQ4, fresh telemetry).** For each of the six interpreters in the Holm family (Jev-1.13.0, SemIf-Qwen3.5-4B,
  AnyJev-L0, DeepSeek-V4.1-Flash, GLM-5.3-Flash, Qwen3.8-Flash), accuracy on `target_cluster` at 3 cells minus
  accuracy at 57 cells is above zero and resolved: the paired case-bootstrap 95% interval excludes zero and the
  Holm-adjusted p-value over these six is below 0.05 (150 state-dependent cases per condition, 10,000 resamples, seed
  20260925). Qwen3.5-4B-JSON is reported as a reference outside the family, with no Holm p-value. At 57 cells,
  Cochran's Q across the six and paired McNemar tests of each against Jev-1.13.0 are reported without a predicted
  direction.
- Near-RT feasibility, latency, fees, energy, the stale, noisy, and contradictory contrasts, and the RQ3 load curves are
  descriptive or exploratory.

## Results in this directory

| File | Content |
|---|---|
| `results/h3_confirmatory.csv` | H3 per interpreter: accuracy at 3 and 57 cells, difference with 95% CI, raw and Holm p, verdict (`family` is `reference` for Qwen3.5-4B-JSON) |
| `results/h3_c57_comparisons.csv` | Cochran's Q and McNemar tests at 57 cells |
| `results/cell_count_curves.csv` | Accuracy on `target_cluster`, actuated-vector match, and full-policy match per interpreter, quality, and cell count |
| `results/exploratory_contrasts.csv` | Stale, noisy, and contradictory contrasts |
| `results/quality_and_cost.csv` | Per interpreter and condition: validity, unsafe-policy rate, accuracy, latency percentiles, tokens, fees per 1,000 correct policies, energy per decision |
| `results/near_rt_feasibility.csv` | Near-RT feasibility per interpreter and condition with 95% CI |
| `results/availability.csv`, `results/sensitivity.csv` | Transport errors and HTTP 429 shares per block, and the non-primary blocks |
| `results/results.md` | Human-readable report of the C1 analysis |
| `results/ledger-latency/` | The per-call fields of the C1 ledgers that the latency ECDF reads (model, condition, case, HTTP status, error, latency, send time), in the layout of the run directories |
| `results/c1_latency_cdf_summary.csv` | Pooled latency summary behind the ECDF, written by `scripts/paper_assets.py`, which also checks it against `quality_and_cost.csv` |
| `results/rq3-load/rq3_load.csv` | RQ3 per interpreter and rate: utilization, queue-wait percentiles, share enforced within 1 s, stationarity flag |
| `results/rq3-load/rq3_availability.csv` | Every RQ3 block with its session, HTTP 429 share, and whether it is the primary block |
| `results/rq3-load/traces/` | The primary per-intent trace and integrity report of each RQ3 cell |
| `results/rq3-load/rq3_trace_latency_summary.csv` | Decision-latency percentiles per RQ3 trace, written by `scripts/paper_assets.py` |

## Re-running

Set `OPENROUTER_API_KEY` (see the root README). All commands run from the repository root.

**Hosted interpreters**, all 16 conditions, interleaved per case in a seeded order:

```bash
CLIENT_LOCATION="Sydney AU, Apple M4 Max"     # recorded with every hosted latency; set it to your client
for n in 3 7 21 57; do for q in fresh stale noisy contradictory; do
  uv run python -m src.ranbench.interpreters.cli run --corpus-dir data/ranbench/ranintent-v1 \
      --condition c${n}_${q} --split test \
      --interpreters Jev-1.13.0,DeepSeek-V4.1-Flash,GLM-5.3-Flash,Qwen3.8-Flash --out runs/EXP-2026-003/c1_hosted \
      --workers 4 --seed 20260925 --spend-cap 20 --probe-every 10 --client-location "$CLIENT_LOCATION"
done; done
```

Qwen3.8-Flash gap-fill rounds (see the deviations below) list the cases that still lack a transport-error-free
attempt and re-issue only those, one worker at a time:

```bash
ROUND=1                                         # then 2, 3, and so on until rb_c1_gapfill.py writes no list
uv run python scripts/rb_c1_gapfill.py --runs-root runs/EXP-2026-003 --corpus-dir data/ranbench/ranintent-v1 \
    --out runs/EXP-2026-003/gapfill_lists/round$ROUND
for list in runs/EXP-2026-003/gapfill_lists/round$ROUND/*.txt; do
  uv run python -m src.ranbench.interpreters.cli run --corpus-dir data/ranbench/ranintent-v1 \
      --condition "$(basename "$list" .txt)" --split test --interpreters Qwen3.8-Flash \
      --out runs/EXP-2026-003/c1_hosted_qwen_gapfill$ROUND --only-cases "$list" \
      --workers 1 --seed 20260925 --spend-cap 20 --probe-every 10 --client-location "$CLIENT_LOCATION"
done
```

**Self-hosted interpreters.** Start one server on the GPU machine, run the 16 conditions against it, stop it, and
continue with the next model. A GPU power trace next to the runs gives the energy per decision.

```bash
# SemIf-Qwen3.5-4B (environment: requirements/edgebench-cuda-fla.txt)
python scripts/eb_serve_semif.py --backend cuda --port 8700
# AnyJev-L0 and Qwen3.5-4B-JSON (environment: requirements/vllm.txt)
vllm serve Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --host 127.0.0.1 --port 8000 \
    --max-model-len 16384 --gpu-memory-utilization 0.85 --runner generate --enable-prefix-caching \
    --logprobs-mode processed_logprobs
vllm serve Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --host 127.0.0.1 --port 8900 \
    --max-model-len 16384 --gpu-memory-utilization 0.85 --runner generate --enable-prefix-caching

# the model whose server is up: SemIf-Qwen3.5-4B semif 8700, AnyJev-L0 anyjev 8000, Qwen3.5-4B-JSON qwen_json 8900
MODEL=SemIf-Qwen3.5-4B; SLUG=semif; PORT=8700
mkdir -p runs/EXP-2026-003/c1_selfhosted/$SLUG
trace=runs/EXP-2026-003/c1_selfhosted/$SLUG/power.csv          # GPU power trace while the server is up
echo "# gpu=0 tz=$(date +%z) fields=timestamp,power.draw[W],utilization.gpu[%],memory.used[MiB]" > "$trace"
nvidia-smi -i 0 --query-gpu=timestamp,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -lms 200 >> "$trace" &
# AnyJev-L0 is never resumed: a condition that stopped early reruns with --fresh
for n in 3 7 21 57; do for q in fresh stale noisy contradictory; do
  uv run python -m src.ranbench.interpreters.cli run --corpus-dir data/ranbench/ranintent-v1 --condition c${n}_${q} \
      --split test --interpreters "$MODEL" --base-url "http://127.0.0.1:$PORT" \
      --out runs/EXP-2026-003/c1_selfhosted/$SLUG/c${n}_${q}/test
done; done
kill %1                                          # stop the power trace
```

AnyJev-L0 reads the tokenizer from the Hugging Face cache (`HF_HOME`) at the pinned snapshot.

**Analysis.** This rebuilds the C1 files in `results/`:

```bash
uv run python scripts/rb_c1_analyze.py --corpus data/ranbench/ranintent-v1 \
    --out experiments/c1-interpretation/results \
    --runs runs/EXP-2026-003/c1_hosted runs/EXP-2026-003/c1_selfhosted runs/EXP-2026-003/c1_rerun \
           runs/EXP-2026-003/c1_hosted_qwen_block2 runs/EXP-2026-003/c1_hosted_qwen_block3 \
           runs/EXP-2026-003/c1_hosted_qwen_gapfill1 runs/EXP-2026-003/c1_hosted_qwen_gapfill2 \
           runs/EXP-2026-003/c1_hosted_qwen_gapfill3 runs/EXP-2026-003/c1_hosted_qwen_gapfill4 \
           runs/EXP-2026-003/c1_hosted_qwen_gapfill5 runs/EXP-2026-003/c1_hosted_qwen_gapfill6
```

**RQ3 load traces.** Hosted interpreters run from the client machine. Each rate gets up to three blocks.

```bash
uv run python scripts/rb_rq3_hosted.py --out-root runs/EXP-2026-003/RQ3/main-hosted-v1 \
    --client-location "$CLIENT_LOCATION"
# self-hosted, with the model's server running on the same machine (here AnyJev-L0 on port 8000)
for rate in 0.1 0.5 1 2; do
  uv run python -m src.ranbench.rq3_load run --corpus-dir data/ranbench/ranintent-v1 --interpreter AnyJev-L0 \
      --rate $rate --base-url http://127.0.0.1:8000 \
      --out runs/EXP-2026-003/RQ3/selfhosted-d12-idle/anyjev_l0/rate_$(echo $rate | tr . p)__block1
done
```

`scripts/c3_rq3_analysis.py` rebuilds `results/rq3-load/rq3_load.csv` and `rq3_availability.csv` (together with the C3
path table) from the raw traces placed under `results/rq3-load/traces/`, with every block of every session.

## Recorded deviations from the pre-registered protocol

1. **AnyJev-L0 replaced Laya (before any run).** Laya's 1,024-token context cannot hold the 57-cell telemetry tables.
   AnyJev 0.0.2 at level L0 over Qwen3.5-4B, served with vLLM 0.30.0, took its place in every experiment as a reported
   interpreter outside the confirmatory pairs. Its label prior is estimated online, so every condition runs in the
   seeded case order after the same two warm-up intents.
2. **Qwen3.8-Flash availability in C1.** Its single provider rejected between 2.7% and 53% of calls per condition
   and block with HTTP 429.
   Gap-fill blocks re-issued only the cases without a transport-error-free attempt, until each case had one. The
   primary row of a case is its first transport-error-free attempt across blocks in block order, and every attempt
   stays in `availability.csv`.
3. **Qwen3.8-Flash availability in RQ3.** The three-block rule was applied again in later sessions with the same
   runner, corpus, and seed. The primary block of each cell is its first complete block in session order.
   `rq3_availability.csv` reports every block with its time.
4. **AnyJev-L0 RQ3 cells rerun.** The first AnyJev load runs counted prompt tokens for the ledger inside the timed
   slot, which added about 0.08 s of slot occupancy per decision. After the fix, the cells were rerun twice, and the
   run on an otherwise idle node is the primary one (session `selfhosted-d12-idle`). The earlier sessions stay in the
   availability table.
5. GLM-5.3-Flash cannot disable reasoning and runs at the minimal reasoning setting. Its reasoning tokens count toward
   its latency and fees.

The deviation identifiers in code comments (`D-2`, `D-3`, `D-8`, `D-12`) refer to items 1 to 4 above.
