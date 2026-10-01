# 6G-JEV

Code, benchmark, and results for the paper

> **Intent Interpretation at RIC Timescales: Jev Decision Models versus Large Language Models in 6G Open RAN**
> Delong Li, Xu Wang, Haochen Gong, Rui Lang, and Guangsheng Yu. University of Technology Sydney.

An operator, or a tenant through an exposure API, states a radio-network intent in natural language, for example
"give the emergency-video class priority in the stadium cluster until 21:00". An interpreter turns the intent into a
schema-valid A1 policy, and the RAN Intelligent Controller (RIC) enforces it on the gNBs. The paper asks which O-RAN
control loop can host which interpreter: the non-real-time RIC (rApps, A1, loops above 1 s) or the near-real-time
RIC (xApps, E2, 10 ms to 1 s for report, decision, and control). It also asks what interpretation latency does to the
radio network once a policy changes, measured as the share of UE time slots that miss their class SLA (SLA
violation) after an intent arrives. Three decision models (Jev-1.13.0, SemIf-Qwen3.5-4B, AnyJev-L0) are compared with
three hosted LLMs (DeepSeek-V4.1-Flash, GLM-5.3-Flash, Qwen3.8-Flash) and a self-hosted JSON-generating reference on
the same weights as SemIf and AnyJev (Qwen3.5-4B-JSON).

| Experiment | Research questions | Where |
|---|---|---|
| C1. Interpretation layer on the RANIntent v1 benchmark, and live load traces through four interpretation slots | RQ1 near-RT feasibility, RQ3 queueing under intent load, RQ4 telemetry-grounded interpretation (H3), decision latency, API fees, and energy | [`experiments/c1-interpretation/`](experiments/c1-interpretation/README.md) |
| C2. Closed-loop multi-cell NR simulation in ns-3.48 with 5G-LENA v5.1 | RQ1 loop placement (H1), RQ2 radio conditions (H2), RQ3 radio replay and UE scale, RQ5 division of labor | [`experiments/c2-closed-loop/`](experiments/c2-closed-loop/README.md) |
| C3. Real A1 and E2 control path (srsRAN gNB, O-RAN SC near-RT RIC, A1 simulator) | RQ6 control-path decomposition, and the A1 delay used in C2 | [`experiments/c3-real-stack/`](experiments/c3-real-stack/README.md) |

The experiments were pre-registered with hypotheses H1 and H2 (C2) and H3 (C1). Their decision rules are described in
Section 4 of the paper (Evaluation Methodology). Each experiment README lists its hypotheses and the recorded
deviations from the protocol that apply to it.

## Raw run records

The benchmark, the results, and the raw run records are on Hugging Face as
[datasets/OniReimu/6G-JEV](https://huggingface.co/datasets/OniReimu/6G-JEV).
`runs/` holds 8,943 record files (19.8 GB), the raw per-call and per-run records that `experiments/*/results/` is
computed from. In a clone of the GitHub repository with the Quick start environment, the commands below rerun the C1,
RQ3, C3, and C2 analyses without API keys, GPUs, or a simulator, from the repository root. The latency summaries and
path samples next to them are rebuilt by `scripts/paper_assets.py`, as in the Quick start:

| Folder | Content | Files | Size |
|---|---|---|---|
| `runs/c1/` | The 11 C1 run directories the C1 analysis reads (35,331 ledger rows, RTT probes, GPU power traces, AnyJev-L0 case orders) and the gap-fill case lists | 199 | 158 MB |
| `runs/rq3-load-traces/` | Every RQ3 load-trace block of every session (55 blocks: trace, ledger, integrity report), including the earlier AnyJev-L0 sessions kept as audit records | 184 | 86 MB |
| `runs/c3-real-stack/` | The 7 C3 main runs (records, A1 log, xApp log, iperf3 trace, integrity report, driver log, gzip-compressed gNB log) and the tunnel round-trip times | 52 | 374 MB |
| `runs/c2-runs/` | The 316 C2 runs selected in `selection.csv`, the 8 RQ5 references, and the 18 layout replicates of the robustness check, each under its platform root | 6,519 | 19.2 GB |
| `runs/c2-inputs/` | The run matrices, input manifests, intent streams, enforcement schedules, and radio configurations of the three C2 input trees | 1,981 | 41 MB |
| `runs/c2-records/` | The C1 decisions of the pool intents that the C2 schedules were built from | 8 | 402 kB |

```bash
uv pip install huggingface_hub
uv run hf download OniReimu/6G-JEV --repo-type dataset --include "runs/*" --local-dir .

# C1: rebuild the interpretation-layer analysis and compare it with the released files
uv run python scripts/rb_c1_analyze.py --corpus data/ranbench/ranintent-v1 --out out/c1 \
    --runs runs/c1/c1_hosted runs/c1/c1_selfhosted runs/c1/c1_rerun \
           runs/c1/c1_hosted_qwen_block2 runs/c1/c1_hosted_qwen_block3 runs/c1/c1_hosted_qwen_gapfill1 \
           runs/c1/c1_hosted_qwen_gapfill2 runs/c1/c1_hosted_qwen_gapfill3 runs/c1/c1_hosted_qwen_gapfill4 \
           runs/c1/c1_hosted_qwen_gapfill5 runs/c1/c1_hosted_qwen_gapfill6
for f in out/c1/*; do cmp "$f" "experiments/c1-interpretation/results/$(basename "$f")"; done

# RQ3 and C3: the raw traces and records go where the release keeps the primary ones, and the tables are rebuilt
rsync -a runs/rq3-load-traces/ experiments/c1-interpretation/results/rq3-load/traces/
rsync -a runs/c3-real-stack/ experiments/c3-real-stack/results/runs/
uv run python scripts/c3_rq3_analysis.py
git diff --stat                    # prints nothing when the rebuilt tables are byte-identical

# C2: select one copy per run as the paper did, then rerun the analysis (it reads the RQ3 traces placed above)
ln -s runs/c2-runs c2-runs && ln -s runs/c2-inputs c2-inputs && ln -s runs/c2-records c2-records
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
uv run python -m src.ranbench.c2.analysis --matrix "$MATRIX" --runs-root c2-selected --out out/c2 \
    --rq5-refs-root c2-runs/rq5-refs-m4
sed "s#$(pwd -P)/runs/##" out/c2/runs.csv > out/c2/runs.rel.csv && mv out/c2/runs.rel.csv out/c2/runs.csv
diff -r out/c2 experiments/c2-closed-loop/results/analysis      # prints nothing
```

Every C1 file, the three RQ3 and C3 tables, and every file of the C2 analysis directory come out byte-identical to the
released ones. The one adjustment is the `platform_root` column of `runs.csv`, which the analysis writes as the
absolute path of each run directory on the machine that runs it, so the `sed` line removes the local prefix first.
`c2-selected/selection.csv` names the same copy, binary, and completion time for every run as
`experiments/c2-closed-loop/results/selection.csv`, but its `reason` and `other_copies` columns only see the copies
shipped here.

## Repository layout

```
data/ranbench/ranintent-v1/   RANIntent v1 benchmark: 16 telemetry conditions, test and dev splits, tuples, the C2/C3 intent pool
src/ranbench/                 Library: corpus builder, interpreter adapters and runner, C1 analysis, RQ3 load runner,
                              C2 matrix, schedules, outcomes and analysis, C3 driver and xApp, RQ5 controller
src/ranbench/ns3/             ns-3 scenario (ran-closed-loop.cc), run wrapper, default configuration, 5G-LENA patches
src/edgebench/, src/*.py      Interpreter, ledger, scoring, and statistics code shared with the sister edge study
scripts/                      Command-line entry points (rb_*.py), the SemIf server, and the figure/table generator
tests/                        Offline test suite (no API key, no GPU, no ns-3 build)
configs/ranbench/             Interpreter manifest, benchmark configuration, real-stack Docker configuration
experiments/*/results/        The analysis outputs every number, figure, and table in the paper is computed from
experiments/paper_numbers.md  Every number in the paper's figures and tables, with the file and row it comes from
paper_assets/                 Figure/table manifest and SHA-256 hashes of the paper's figure and table files
requirements/                 Pinned dependencies (lightweight analysis environment and the GPU server environments)
```

## Quick start (no API key, no GPU, no ns-3)

Requirements: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/OniReimu/6G-JEV.git
cd 6G-JEV
uv venv --python 3.13
uv pip install -r requirements/edgebench.txt -r requirements/analysis.txt

# 1. Offline tests. Tests that need ns-3 outputs or model libraries are skipped in this environment.
uv run python -m pytest tests -q

# 2. Rebuild the paper's figures and tables from the released results (written to paper_assets/)
uv run python scripts/paper_assets.py

# 3. Compare them byte for byte with the paper's files (SHA-256 in paper_assets/expected_sha256.json)
uv run python scripts/paper_assets.py --check
```

The runners record the commit of the code in every run for provenance, so use a `git clone`. If you downloaded a ZIP
archive, run `git init && git add -A && git commit -m snapshot` once before running tests or experiments.

Byte equality of the PDF figures needs the matplotlib, NumPy, and pandas versions pinned in
`requirements/edgebench.txt` and the same serif fonts (Times New Roman, then Nimbus Roman or DejaVu Serif as fallbacks).

## Reproducing the experiments from scratch

Re-running the experiments makes live calls to hosted models, needs a GPU server for the self-hosted interpreters, an
ns-3 build for C2, and a Docker host for C3. Each experiment README gives the exact commands, and the analysis
commands rebuild `experiments/*/results/` from new runs. All commands run from the repository root.

### Credentials

Hosted interpreters (Jev-1.13.0, DeepSeek-V4.1-Flash, GLM-5.3-Flash, Qwen3.8-Flash) are called through
[OpenRouter](https://openrouter.ai) with provider fallback disabled and data collection denied. Set the key in the
environment and never commit it:

```bash
read -rs OPENROUTER_API_KEY && export OPENROUTER_API_KEY     # paste the key; it is not echoed
```

`src/credentials.py` can also read the key from `~/.config/jev/openrouter.env` (mode 600).

### Self-hosted interpreters

All three ran on one NVIDIA H100 NVL, one model at a time, with the client on the same machine.

| Display name | Model (pinned revision) | Server | Environment | Port |
|---|---|---|---|---|
| SemIf-Qwen3.5-4B | [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) @ `23cf1f39` on [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) @ `851bf6e8` | `scripts/eb_serve_semif.py --backend cuda` | `requirements/edgebench-cuda-fla.txt` | 8700 |
| AnyJev-L0 | [anyjev](https://pypi.org/project/anyjev/) 0.0.2 (level L0) over Qwen/Qwen3.5-4B @ `851bf6e8` | `vllm serve` with prefix caching and processed log-probabilities | `requirements/vllm.txt` | 8000 |
| Qwen3.5-4B-JSON (reference) | Qwen/Qwen3.5-4B @ `851bf6e8`, JSON-schema-constrained decoding, thinking off | `vllm serve` with prefix caching | `requirements/vllm.txt` | 8900 |

The server commands are in [`experiments/c1-interpretation/README.md`](experiments/c1-interpretation/README.md).
Model weights are downloaded from Hugging Face at the pinned revisions.

### ns-3 and 5G-LENA

C2 uses ns-3.48 with 5G-LENA v5.1 (commit `cedceadda17392c90587fb9400eb9b1f8c236713`) and Eigen 3.4.0.
[`src/ranbench/ns3/README.md`](src/ranbench/ns3/README.md) describes the build, the scenario sources, and the two
5G-LENA patches the reruns used.

## Mapping from the paper to this repository

| Paper element | Generator (`scripts/paper_assets.py`) | Source data |
|---|---|---|
| RQ4 accuracy over telemetry size and quality, decision-latency ECDF, interpretation table, tokens, fees and energy | `fig_c1`, `tab_c1`, `tab_c1_cost` | `experiments/c1-interpretation/results/*.csv` (built by `scripts/rb_c1_analyze.py`), `ledger-latency/` |
| RQ3 queue wait, slot utilization, enforcement within 1 s | `fig_rq3`, `tab_rq3` | `experiments/c1-interpretation/results/rq3-load/` (built by `scripts/c3_rq3_analysis.py`) |
| RQ1 loop placement and H1 | `fig_c2_h1`, `tab_c2_h1` | `experiments/c2-closed-loop/results/analysis/` (built by `src.ranbench.c2.analysis`) |
| RQ2 radio conditions and H2, radio KPIs, controls | `fig_c2_grid`, `tab_c2_grid`, `tab_c2_radio`, `tab_c2_controls` | same |
| Latency, accuracy, and net decomposition | `fig_c2_an` | same |
| RQ3 radio replay and UE scale | `tab_c2_rq3` | same |
| RQ5 division of labor | `tab_c2_rq5` | same |
| RQ6 control-path decomposition | `fig_c3`, `tab_c3` | `experiments/c3-real-stack/results/` (built by `scripts/c3_rq3_analysis.py`) |

`paper_assets/figure-manifest.yml` records, for every generated file, its generator function, source files, columns,
and filters. `experiments/paper_numbers.md` lists every value in the figures and tables with the file and row it comes
from. Cells marked `\pending{pending}` in a generated table wait for a simulation run that had not completed when the
results were frozen. `experiments/c2-closed-loop/results/c2_pending.csv` lists them, and they disappear when the
analysis is complete.

## Model names

Interpreters are identified by versioned display names throughout the code, the results, and the paper:
Jev-1.13.0, SemIf-Qwen3.5-4B, AnyJev-L0, DeepSeek-V4.1-Flash, GLM-5.3-Flash, Qwen3.8-Flash, and Qwen3.5-4B-JSON
(reference). RANIntent conditions are named `c<cells>_<quality>`, so `c57_stale` means a KPM telemetry table of 57 cells
with stale rows. C2 design points are named `dp_r<rate>_s<speed>_u<UEs per cell>`, so `dp_r0p3_s30_u5` is the base
point at 0.3 intents/s, 30 km/h, and 5 UEs per cell. C2 run identifiers name the research question, the loop mode
(`e` event-triggered, `p_1` and `p_10` periodic), the design point, and the arm (`l` latency-only, `a` accuracy-only,
`n` net, `oracle`, `no_update`, `fixed_<d>`).

## Third-party components

- ns-3.48 (GPL-2.0), 5G-LENA v5.1 (GPL-2.0), and Eigen 3.4.0 are downloaded from their original sources. Only the
  two 5G-LENA patch files are distributed here, under GPL-2.0-only.
- The real stack uses public images: srsRAN Project gNB `aetherproject/srsran-gnb:rel-1.0.0`, srsUE, Open5GS,
  the O-RAN SC near-RT RIC (i-release, via [srsran/oran-sc-ric](https://github.com/srsran/oran-sc-ric)), and the
  O-RAN SC A1 simulator 2.8.1.
- Model weights are downloaded from Hugging Face at the pinned revisions above.

Each third-party component keeps its own license.

## License

Code: [MIT](LICENSE), except the simulator patches in `src/ranbench/ns3/patches/`, which modify 5G-LENA and are
GPL-2.0-only like the code they modify ([LICENSE](src/ranbench/ns3/patches/LICENSE)). The ns-3 scenario sources in
`src/ranbench/ns3/` are original code written against the ns-3 and 5G-LENA APIs and are MIT. A binary built from them
links the GPL-2.0 ns-3 and 5G-LENA libraries. RANIntent v1 and the experiment results: [CC BY 4.0](LICENSE-DATA). If
you use the benchmark or the code, please cite the paper above.
