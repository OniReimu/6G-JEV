# RANIntent v1

RANIntent v1 is a benchmark of natural-language intents for an O-RAN network, each paired with a typed policy truth and
a KPM telemetry table that holds the network state. One set of 300 test and 60 development label tuples, with one
intent text per tuple, serves 16 conditions that differ only in the attached telemetry: 3, 7, 21, or 57 cells, each
fresh, stale, noisy, or contradictory. Half of the test cases are state-dependent: the correct `target_cluster` follows
from the telemetry and the text never names it. The other half name their cluster and carry an action that the C2
simulation can actuate. The operator issues 175 of the 300 test intents and 40 of the 60 development intents. Four
tenants issue the rest (test: civic_safety 38, lumen_media 38, axis_robotics 32, datacrest 17. Dev: civic_safety 9,
lumen_media 4, datacrest 4, axis_robotics 3).

## How the cases were made

1. Label tuples were sampled first, with balanced marginals per field (`python -m src.ranbench.corpus.cli`).
2. A generator model (Gemini-3.8-Flash) wrote the intent text of each tuple.
3. A blind verifier (Claude-Opus-5.5, in a separate session) labelled each text with its telemetry table, without seeing
   the tuple, using the same reading rules the interpreters get. A case was kept only when its labels matched the tuple
   on every field. Lint checks rejected texts that name a cluster in a state-dependent case, contain field names or
   option identifiers, are too short or long, or repeat the openings or 5-grams of other texts.
4. The telemetry tables of all 16 conditions were built programmatically around two extreme cells per tuple, so the
   target cluster is the same at every telemetry size.

The generator and verifier of every case are recorded in `meta.generator` and `meta.verifier`, and in `manifest.json`.
Neither model is among the evaluated interpreters.

## Files

| Path | Content |
|---|---|
| `RQ4/<condition>/test.jsonl`, `dev.jsonl` | Evaluation and development cases (`c3_fresh` to `c57_contradictory`) |
| `tuples/`, `_tuples/` | The label tuples of both splits, and the sampled tuples before text generation |
| `pool/c2c3_pool.jsonl` | The 150 named-scope test intents of `c21_fresh` from which C2 and C3 draw |
| `schemas/` | The A1 policy type, the truth schema, and the reading rules given to interpreters and verifier |
| `config.json` | Seeds, splits, cell counts, telemetry qualities, issuers, and the mapping of the eight intent classes to the four simulated traffic classes |
| `manifest.json`, `provenance.json` | Generators, verifier, yields, and SHA-256 of every file |
| `stats.json` | Telemetry token counts and state-dependent shares per condition |

## Case format

```json
{"case_id": "ranintent_v1_test_0000", "condition": "c7_stale", "n_cells": 7, "quality": "stale", "split": "test",
 "half": "state_dependent",
 "issuer": {"id": "operator", "kind": "operator", "...": "..."},
 "text": "Trouble Ticket: Configure an end-to-end latency threshold of 10 ms for sensor and meter reporting data on cells where boundary user throughput is minimal.",
 "telemetry": "KPM telemetry snapshot at 2026-10-03T18:36:39Z; 7 cells\ncell_id,cluster,measured_at,prb_util_pct,...",
 "truth": {"action": "set_latency_target", "class": "iot_metering", "scope": "worst_edge_cells",
           "target_cluster": "hospital_zone", "priority": "unspecified", "prb_share": "unspecified",
           "edge_site": "unspecified", "latency_target": "10", "duration": "unspecified"},
 "meta": {"wording_family": "operator ticket", "...": "..."}}
```
