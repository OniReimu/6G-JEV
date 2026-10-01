# C2: closed-loop multi-cell NR simulation (RQ1, RQ2, RQ3 radio, RQ5)

The recorded interpretations of C1 drive a closed-loop radio network in ns-3.48 with 5G-LENA v5.1. The layout is the
TR 38.901 urban macro scenario with 7 sites of 3 sectors (21 radio cells, 500 m inter-site distance) on one 10 MHz
carrier at 4 GHz with a common DDDSU TDD pattern. Each cell starts with 5 UEs (20 at the RQ3 scale level) that move in
random directions at the design speed, hand over on the A3 event, and run with the handover-failure and radio link
failure models enabled. Four downlink traffic classes emulate slices as QoS classes through scheduler weights: video
(CBR, throughput floor), XR (CBR, 95th-percentile delay target), IoT (periodic, throughput floor), and best effort
(backlogged). At enforcement one E2 control installs the policy's class priorities in every cell of its scope, and a
numerical xApp then adjusts them every 1 s within the bounds of the active policy.

An intent is enforced at its issue time plus the loop wait (periodic modes), the interpreter's recorded decision
latency, the A1 delay (9.3 ms, from C3), and the E2 delay (5 ms). Every interpreter has three arms on the same intent
stream and radio realization:

- **L (latency only):** the true policy, applied after the interpreter's latency. The latency hypotheses use L arms only.
- **A (accuracy only):** the interpreter's own policy, applied after the E2 delay only.
- **N (net):** the interpreter's own policy after its latency, the operator-facing outcome.

Interpreter-independent controls: the oracle (true policy after the E2 delay), fixed latencies of 0.1, 1, and 5 s, and
no update. The primary outcome is SLA violation in the 2 s after an intent: the share of 100 ms (UE, slot) pairs that
miss their class target, for the intent's class and scope (affected class) and for the whole network (network-wide).

| RQ | Design points | Outcomes |
|---|---|---|
| RQ1 | Base point (0.3 intents/s, 30 km/h) in modes E (event-triggered rApp), P-1 and P-10 (periodic rApp, 1 s and 10 s cycles) | SLA violation per mode, L arms |
| RQ2 | Intent rate 0.1, 0.3, 1 per s × UE speed 3, 60, 120 km/h, mode E | SLA violation, handover rate and interruption, RLF, mean and 5th-percentile UE throughput |
| RQ3 | The RQ3 load traces of C1 replayed in mode E at 0.1, 0.5, 1, 2 intents/s, and at 1 intent/s with 20 UEs per cell | SLA violation, PRB utilization |
| RQ5 | (a) policy plus numerical xApp, (b) the interpreter re-queried every 1 s on a KPM snapshot exported from the run, choosing per-cell class priorities, (c) oracle policy plus xApp | SLA violation, throughput, fairness |

Each run is one long simulation with a single seed and component random streams pinned across arms. Intervals come
from a paired time-block bootstrap within the run (10,000 resamples, seed 20260925).

## Pre-registered hypotheses

"Resolved" means that the 95% bootstrap interval excludes zero and the Holm-adjusted p-value is below 0.05.

- **H1 (RQ1, base point, mode E, L arms).** Affected-class and network-wide SLA violation are both higher for each
  hosted LLM than for Jev-1.13.0, resolved, and higher for Qwen3.5-4B-JSON than for SemIf-Qwen3.5-4B, resolved.
- **H2 (RQ2, mode E, L arms).** For each H1 pair, the affected-class gap is resolved at no fewer than 80% of the eligible
  design points and reversed-and-resolved at none, with at least 5 of the 9 design points eligible.
- **Controls.** A design point is eligible when the oracle beats no update (C-1) and SLA violation grows with a fixed
  latency from 0.1 s to 5 s (C-2), both resolved. Eligibility never depends on interpreter arms. C-3 (exposure grows
  linearly with latency) is a code test, and C-4 (identical UE positions and traffic arrivals across arms) is checked
  per run.
- The periodic-versus-event comparison, speed effects, the N, A, and L decomposition, RQ3, and RQ5 are exploratory.

## Results in this directory

| File | Content |
|---|---|
| `results/analysis/controls.csv` | C-1 and C-2 per design point with intervals, block length, number of blocks, and eligibility under both rules |
| `results/analysis/l_arm_contrasts.csv` | L-arm SLA violation per interpreter and design point, gaps to the anchor, Holm p, resolution under both block lengths |
| `results/analysis/an_decomposition.csv` | N minus L and N minus A per interpreter and design point |
| `results/analysis/rq1_modes.csv` | RQ1 SLA violation per interpreter and loop mode |
| `results/analysis/radio_kpis.csv` | Radio KPIs per run: throughput, delay, handover, RLF, PRB utilization, fairness, stale-policy exposure |
| `results/analysis/rq3_radio.csv`, `rq5.csv` | RQ3 replay cells and RQ5 arms |
| `results/analysis/hypotheses.json` | H1 and H2 families under the primary rule and the original 5 pp rule |
| `results/analysis/runs.csv`, `skipped.json` | The run behind every analysis row with its simulator binary, and the analysis units waiting for an absent run |
| `results/selection.csv` | Which copy of each run was used (original or rerun, platform root, binary, completion time) and why |
| `results/run-matrix.csv` | The run matrix: one row per simulation with its inputs, seed, simulated time, and arm |
| `results/run-matrix-rq5-stream.txt`, `run-matrix-d16-d17.txt` | How the RQ5 rows and the AnyJev-L0 RQ3 replay rows enter the run matrix |
| `results/layout_robustness.json` | The base-point check of how far C-1 and the H1 contrasts move when only the memory layout of the simulator changes |
| `results/c2_h2_summary.csv`, `c2_pending.csv` | H2 verdicts in one table, and the cells that wait for an absent run (both written by `scripts/paper_assets.py`) |

Paths in these files name the raw run roots as they appear in the Hugging Face dataset: `c2-runs/<platform root>/<run id>`
and `c2-inputs/<input tree>/`. The platform roots are `cluster-a`, `cluster-b` (two x86-64 Linux clusters with gcc), `l40`
(an x86-64 server), and `m4` (an Apple M4 Max with clang). Suffixes name the patched rerun builds (`-d7`, `-d7b`) and
the RQ3 and RQ5 roots.

## Re-running

Build the scenario first ([`src/ranbench/ns3/README.md`](../../src/ranbench/ns3/README.md)). All commands run from the
repository root.

**Inputs.** The run matrix in `results/run-matrix.csv` names every input by its path in the `c2-inputs/` folder of
the Hugging Face dataset, so the released inputs can be used as they are. To rebuild them, export the C1 decisions
of the 150 pool intents, then build the run matrix and materialize the intent streams, the per-arm enforcement
schedules, and the radio configurations:

```bash
uv run python scripts/rb_c2_export_c1_records.py --runs-root runs/EXP-2026-003 \
    --pool data/ranbench/ranintent-v1/pool/c2c3_pool.jsonl --out-dir c2-records
RECORDS=()                                      # one --records MODEL=FILE per interpreter
for model in Jev-1.13.0 SemIf-Qwen3.5-4B AnyJev-L0 DeepSeek-V4.1-Flash GLM-5.3-Flash Qwen3.8-Flash Qwen3.5-4B-JSON; do
  slug=$(echo "$model" | tr 'A-Z' 'a-z' | sed -E 's/[^a-z0-9]+/_/g')
  RECORDS+=(--records "$model=c2-records/$slug.jsonl")
done
LOAD_TRACES=()                                  # one --load-trace MODEL@RATE=TRACE per primary RQ3 cell
while IFS= read -r arg; do LOAD_TRACES+=(--load-trace "$arg"); done < <(uv run python -c '
import csv
for r in csv.DictReader(open("experiments/c1-interpretation/results/rq3-load/rq3_load.csv")):
    print(r["model"] + "@" + format(float(r["rate_per_s"]), "g") + "=" + r["source_trace"])')
INPUTS=c2-inputs/rebuilt
uv run python -m src.ranbench.c2.cli matrix --input-root $INPUTS/inputs --block 10 --out $INPUTS/matrix.csv \
    "${LOAD_TRACES[@]}"
uv run python -m src.ranbench.c2.cli materialize --matrix $INPUTS/matrix.csv --input-root $INPUTS/inputs \
    --base-config configs/ranbench/c2-base-config.json --pool data/ranbench/ranintent-v1/pool/c2c3_pool.jsonl \
    --block 10 --a1-put-ack-median-s 0.004300475120544434 "${RECORDS[@]}" "${LOAD_TRACES[@]}"
```

The RQ3 traces named in `rq3_load.csv` are the primary traces in `experiments/c1-interpretation/results/rq3-load/traces/`.
`configs/ranbench/c2-base-config.json` is the traffic profile the pilot selected (video 1,200 kbit/s with a
900 kbit/s floor, XR 400 kbit/s with a 35 ms delay target, IoT 300 kbit/s with a 240 kbit/s floor).

**Simulation runs.** Every arm and control of one design point runs on one simulator binary. The queue runner skips
verified runs, re-queues incomplete ones, and checks the binary hash before every run. The RQ5 (b) rows are run by
the RQ5 controller below, so they are left out of the list:

```bash
MATRIX=$INPUTS/matrix.csv
BINARY=ns-3.48/build/scratch/ns3.48-ran-closed-loop-optimized
BINARY_SHA=$(shasum -a 256 "$BINARY" | cut -d' ' -f1)
awk -F, 'NR > 1 && $2 != "RQ5" {print $1}' "$MATRIX" > c2-run-list.txt
uv run python scripts/rb_c2_queue.py --matrix "$MATRIX" --run-list c2-run-list.txt --output-root c2-runs/local \
    --binary "$BINARY" --expect-binary-sha256 "$BINARY_SHA" --jobs 8
uv run python scripts/rb_c2_verify_runs.py --run-root c2-runs/local --run-list c2-run-list.txt --matrix "$MATRIX" \
    --input-root $INPUTS/inputs --binary-sha256 "$BINARY_SHA"
```

**RQ5.** Arm (b) and its references (arm (a) and arm (c)) run on the build of the scenario with the RQ5 hook. They use
the released run matrix, whose RQ5 rows point at the base-point stream (`results/run-matrix-rq5-stream.txt`), with
the `c2-inputs/` folder of the dataset. Hosted interpreters need `OPENROUTER_API_KEY`, and the self-hosted ones need
their server (`--base-url`):

```bash
RQ5_MATRIX=experiments/c2-closed-loop/results/run-matrix.csv
RQ5_BINARY=ns-3.48/build/scratch/ns3.48-ran-closed-loop-rq5-optimized
RQ5_SHA=$(shasum -a 256 "$RQ5_BINARY" | cut -d' ' -f1)
printf '%s\n' rq1_e_r0p3_s30_oracle rq2_e_r0p3_s30_n_jev_1_13_0 rq2_e_r0p3_s30_n_deepseek_v4_1_flash \
    rq2_e_r0p3_s30_n_glm_5_3_flash rq2_e_r0p3_s30_n_qwen3_8_flash rq2_e_r0p3_s30_n_semif_qwen3_5_4b \
    rq2_e_r0p3_s30_n_anyjev_l0 rq2_e_r0p3_s30_n_qwen3_5_4b_json > c2-rq5-refs.txt
uv run python scripts/rb_c2_queue.py --matrix "$RQ5_MATRIX" --run-list c2-rq5-refs.txt --output-root c2-runs/rq5-refs-m4 \
    --binary "$RQ5_BINARY" --expect-binary-sha256 "$RQ5_SHA" --jobs 4
for model in jev_1_13_0 deepseek_v4_1_flash glm_5_3_flash qwen3_8_flash; do
  uv run python -m src.ranbench.rq5.controller queue --matrix "$RQ5_MATRIX" --output-root c2-runs/rq5-b-m4 \
      --binary "$RQ5_BINARY" --expect-binary-sha256 "$RQ5_SHA" --jobs 1 --run-id rq5_e_r0p3_s30_b_requery_1s_$model
done
```

**Selection and analysis.** One copy per run is selected across the original and rerun roots (the original when it
completed, otherwise the rerun that completed first), then analysed. With the raw run records of the Hugging Face
dataset in `c2-runs/`, the roots and restrictions of the paper's selection are read from `results/selection.csv`:

```bash
MATRIX=experiments/c2-closed-loop/results/run-matrix.csv
SELECT_ARGS=()
while IFS= read -r arg; do SELECT_ARGS+=("$arg"); done < <(uv run python -c '
import csv
rows = list(csv.DictReader(open("experiments/c2-closed-loop/results/selection.csv")))
for kind, flag in (("original", "--original-root"), ("rerun", "--rerun-root")):
    for root in sorted({r["chosen_root"] for r in rows if r["kind"] == kind}):
        print(flag); print(root)
for r in rows:
    if r["chosen_root"] and "EXCLUDED" in r["other_copies"]:
        print("--restrict"); print(r["run_id"] + "=" + r["chosen_root"])')
uv run python -m src.ranbench.c2.select_runs --matrix "$MATRIX" "${SELECT_ARGS[@]}" --out c2-selected --force
uv run python -m src.ranbench.c2.analysis --matrix "$MATRIX" --runs-root c2-selected \
    --out experiments/c2-closed-loop/results/analysis --rq5-refs-root c2-runs/rq5-refs-m4 --allow-incomplete
LAYOUT_ARGS=(); for root in c2-runs/cluster-a-layout/p*; do LAYOUT_ARGS+=(--layout-root "$root"); done
uv run python -m src.ranbench.c2.layout_robustness --matrix "$MATRIX" --design-point dp_r0p3_s30_u5 \
    --production-root c2-runs/cluster-a "${LAYOUT_ARGS[@]}" --out experiments/c2-closed-loop/results/layout_robustness.json
```

For runs of your own, use `MATRIX=$INPUTS/matrix.csv` and pass `--original-root c2-runs/local` instead of
`"${SELECT_ARGS[@]}"`.
`--allow-incomplete` lets an absent run leave its analysis units pending (listed in `skipped.json`). Drop it once
every run is present.

## Recorded deviations from the pre-registered protocol

Code comments refer to these by their identifiers.

- **D-4. Full fidelity on several platforms.** No configuration-level cost cuts were made. The 24 h walltime and the
  core-hour cap were replaced by runs on four platforms. Builds of the same source on different compilers or CPU types
  give different trajectories, so every arm and control of a design point runs on one binary of one platform. The
  exceptions are reruns of aborted arms (D-7, D-11): at 0.1 intents/s and 60 or 120 km/h, three runs each finished
  on the M4 while the rest of the point ran on cluster-B.
- **D-5. Control C-1 on the affected class.** One intent changes one class in one cluster, so a network-wide C-1 of
  5 pp is out of reach by construction. C-1 and C-2 are measured on the affected class of the intents that actually
  change an SLA class's priority. Network-wide controls stay in the report.
- **D-6. Traffic model fix before any main run.** The scenario ignored the per-class traffic type, so best effort ran as
  CBR. It was fixed (best effort is backlogged) and the pilot was repeated.
- **Pilot calibration.** The traffic profile in `configs/ranbench/c2-base-config.json` was chosen from four candidates
  on control arms only. The block length was set to 10 s, the A1 delay to 9.3 ms, and the E2 delay to 5 ms.
- **D-7 and D-7b. Simulator aborts.** Two 5G-LENA v5.1 defects aborted some long runs (an RRC event in state
  IDLE_CONNECTING, and a buffer status report for a removed logical channel). Minimal guards were added as patches
  (`src/ranbench/ns3/patches/`), and only the aborted arms were rerun, from the start, on the patched build of the same
  platform. A test to 70 s showed the patched and unpatched builds identical until a guard fires.
- **D-9. Control floor without a magnitude threshold.** The 5 pp minimum for C-1 was removed before any base-point
  control or interpreter outcome was seen: C-1 must be above zero and resolved. H1 and H2 are also reported under the
  original 5 pp rule (`sensitivity_original_5pp` in `hypotheses.json`).
- **D-10. Block length and memory layout.** The block length is the larger of 10 s and the measured autocorrelation time
  of the SLA series, and a contrast counts as resolved only if it is resolved with that block and with 10 s blocks.
  Runs with the same binary and inputs but different memory layouts follow different trajectories after about a
  minute. Rerunning the base-point arms under three padded output paths (`layout_robustness.json`) showed C-1 to be
  robust, and the three base-point H1 contrasts of the hosted LLMs not robust, so those are not claimed as confirmatory
  findings.
- **D-11. Three rate-0.1 arms on a second build.** Three arms that aborted on one cluster were also run on the patched
  M4 build, and the first completed copy is used. The order of pre-emptive backup runs was fixed before any output was
  seen.
- **D-12 and D-16. AnyJev-L0 RQ3 replays.** The AnyJev-L0 replays were rebuilt from the RQ3 traces of its corrected
  idle-node load run (see the C1 README), so the radio cells and the load table use the same traces.
- **D-13. A third simulator defect.** The AnyJev-L0 arm of the RQ3 scale point aborted with a different 5G-LENA error
  and was rerun.
- **D-14. No cutoff.** Every rerun runs to completion. Cells that depend on a run still in progress show as pending.
- **D-15. Cell-edge throughput and H2.** The 5th-percentile UE throughput is computed across UEs of each UE's mean
  throughput, as pre-registered. H2 needs at least 5 eligible design points and is reported as not testable when fewer
  are eligible.
- **D-17. Platform level shift.** For identical inputs, the base-point oracle's network-wide SLA violation differs by
  2.65 pp between the x86-64 gcc and the arm64 clang builds. RQ3 cells of one rate therefore all run on one platform,
  and RQ5 compares arm (b) with references on the same build. The paper's tables report every completed run,
  including the cells and contrasts at the two design points above whose runs come from two platforms.
- **D-18. RQ3 rows on a second platform.** The 0.1 intents/s row and the 20-UE scale row also run in full on the M4
  build, and each row comes from the platform that completes all of its cells first. Rows never mix platforms.
- The RQ5 (b) rows were run by the RQ5 controller on the base-point stream, as `results/run-matrix-rq5-stream.txt`
  records.
