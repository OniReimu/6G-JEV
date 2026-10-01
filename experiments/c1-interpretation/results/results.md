# EXP-2026-003 C1 Analysis Results (Frozen v0.2)

This report summarizes intent interpretation quality, near-RT feasibility ($\phi$),
confirmatory hypothesis H3, telemetry exploratory contrasts, and sensitivity blocks.
All numbers are strictly deterministic and traceable to generated CSV files.

Bootstrap: seed 20260925, 10,000 bootstrap resamples (pre-registered).

---

## 1. Roster Coverage & Availability

Traceable to `availability.csv` (per-block attempts and HTTP 429 rates in `block_attempts`).
Qwen3.8-Flash (D-3): primary row per case = first transport-error-free attempt across its blocks.

| Interpreter | Condition | Status | Primary Block | Primary Calls | Primary Errors | Error Types | Primary Avail | Error >5% | Blocks >5% Error | Sensitivity Blocks | Total Calls | Total Avail | HTTP 429 (all attempts) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev-1.13.0 | c3_fresh | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Jev-1.13.0 | c3_stale | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Jev-1.13.0 | c3_noisy | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Jev-1.13.0 | c3_contradictory | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Jev-1.13.0 | c7_fresh | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Jev-1.13.0 | c7_stale | complete | c1_hosted | 300 | 2 | TimeoutError: 2 | 0.9933 | no | None | None | 300 | 0.9933 | 0 (0.0%) |
| Jev-1.13.0 | c7_noisy | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Jev-1.13.0 | c7_contradictory | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Jev-1.13.0 | c21_fresh | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Jev-1.13.0 | c21_stale | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Jev-1.13.0 | c21_noisy | complete | c1_hosted | 300 | 1 | TimeoutError: 1 | 0.9967 | no | None | None | 300 | 0.9967 | 0 (0.0%) |
| Jev-1.13.0 | c21_contradictory | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Jev-1.13.0 | c57_fresh | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Jev-1.13.0 | c57_stale | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Jev-1.13.0 | c57_noisy | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Jev-1.13.0 | c57_contradictory | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c3_fresh | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c3_stale | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c3_noisy | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c3_contradictory | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c7_fresh | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c7_stale | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c7_noisy | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c7_contradictory | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c21_fresh | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c21_stale | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c21_noisy | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c21_contradictory | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c57_fresh | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c57_stale | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c57_noisy | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| DeepSeek-V4.1-Flash | c57_contradictory | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| GLM-5.3-Flash | c3_fresh | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| GLM-5.3-Flash | c3_stale | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| GLM-5.3-Flash | c3_noisy | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| GLM-5.3-Flash | c3_contradictory | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| GLM-5.3-Flash | c7_fresh | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| GLM-5.3-Flash | c7_stale | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| GLM-5.3-Flash | c7_noisy | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| GLM-5.3-Flash | c7_contradictory | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| GLM-5.3-Flash | c21_fresh | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| GLM-5.3-Flash | c21_stale | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| GLM-5.3-Flash | c21_noisy | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| GLM-5.3-Flash | c21_contradictory | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| GLM-5.3-Flash | c57_fresh | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| GLM-5.3-Flash | c57_stale | complete | c1_hosted | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| GLM-5.3-Flash | c57_noisy | complete | c1_hosted | 300 | 6 | HTTP_429: 6 | 0.9800 | no | None | None | 300 | 0.9800 | 6 (2.0%) |
| GLM-5.3-Flash | c57_contradictory | complete | c1_hosted | 300 | 24 | HTTP_429: 24 | 0.9200 | **YES** | c1_hosted (8.0%) | c1_rerun/c57_contradictory__GLM-5.3-Flash__rerun1 | 600 | 0.9600 | 24 (4.0%) |
| Qwen3.8-Flash | c3_fresh | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | c1_hosted (50.3%); c1_hosted_qwen_block2 (53.3%) | c1_hosted; c1_hosted_qwen_block2; c1_hosted_qwen_block3; c1_hosted_qwen_gapfill1 | 495 | 0.7899 | 104 (21.0%) |
| Qwen3.8-Flash | c3_stale | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | c1_hosted_qwen_gapfill1 (8.3%); c1_hosted_qwen_gapfill2 (100.0%); c1_hosted_qwen_gapfill3 (100.0%); c1_hosted_qwen_gapfill4 (100.0%); c1_hosted_qwen_gapfill5 (100.0%) | c1_hosted_qwen_block3; c1_hosted_qwen_gapfill1; c1_hosted_qwen_gapfill2; c1_hosted_qwen_gapfill3; c1_hosted_qwen_gapfill4; c1_hosted_qwen_gapfill5; c1_hosted_qwen_gapfill6 | 317 | 0.9464 | 17 (5.4%) |
| Qwen3.8-Flash | c3_noisy | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | c1_hosted_qwen_block3 (14.3%); c1_hosted_qwen_gapfill1 (20.9%); c1_hosted_qwen_gapfill2 (88.9%); c1_hosted_qwen_gapfill3 (87.5%); c1_hosted_qwen_gapfill4 (100.0%); c1_hosted_qwen_gapfill5 (85.7%) | c1_hosted_qwen_block3; c1_hosted_qwen_gapfill1; c1_hosted_qwen_gapfill2; c1_hosted_qwen_gapfill3; c1_hosted_qwen_gapfill4; c1_hosted_qwen_gapfill5; c1_hosted_qwen_gapfill6 | 380 | 0.7895 | 80 (21.1%) |
| Qwen3.8-Flash | c3_contradictory | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | c1_hosted_qwen_block3 (14.3%); c1_hosted_qwen_gapfill1 (16.3%); c1_hosted_qwen_gapfill2 (100.0%); c1_hosted_qwen_gapfill3 (100.0%); c1_hosted_qwen_gapfill4 (100.0%); c1_hosted_qwen_gapfill5 (85.7%) | c1_hosted_qwen_block3; c1_hosted_qwen_gapfill1; c1_hosted_qwen_gapfill2; c1_hosted_qwen_gapfill3; c1_hosted_qwen_gapfill4; c1_hosted_qwen_gapfill5; c1_hosted_qwen_gapfill6 | 377 | 0.7958 | 77 (20.4%) |
| Qwen3.8-Flash | c7_fresh | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | c1_hosted_qwen_block3 (11.3%) | c1_hosted_qwen_block3; c1_hosted_qwen_gapfill1 | 334 | 0.8982 | 34 (10.2%) |
| Qwen3.8-Flash | c7_stale | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | c1_hosted_qwen_block3 (36.3%) | c1_hosted_qwen_block3; c1_hosted_qwen_gapfill1 | 409 | 0.7335 | 109 (26.7%) |
| Qwen3.8-Flash | c7_noisy | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | c1_hosted_qwen_block3 (44.0%) | c1_hosted_qwen_block3; c1_hosted_qwen_gapfill1 | 432 | 0.6944 | 132 (30.6%) |
| Qwen3.8-Flash | c7_contradictory | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | c1_hosted_qwen_block3 (19.3%) | c1_hosted_qwen_block3; c1_hosted_qwen_gapfill1 | 358 | 0.8380 | 58 (16.2%) |
| Qwen3.8-Flash | c21_fresh | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | c1_hosted_qwen_block3 (6.3%); c1_hosted_qwen_gapfill1 (57.9%); c1_hosted_qwen_gapfill2 (90.9%); c1_hosted_qwen_gapfill3 (80.0%); c1_hosted_qwen_gapfill4 (87.5%); c1_hosted_qwen_gapfill5 (85.7%) | c1_hosted_qwen_block3; c1_hosted_qwen_gapfill1; c1_hosted_qwen_gapfill2; c1_hosted_qwen_gapfill3; c1_hosted_qwen_gapfill4; c1_hosted_qwen_gapfill5; c1_hosted_qwen_gapfill6 | 361 | 0.8310 | 61 (16.9%) |
| Qwen3.8-Flash | c21_stale | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | c1_hosted_qwen_block3 (9.3%); c1_hosted_qwen_gapfill1 (25.0%); c1_hosted_qwen_gapfill2 (85.7%); c1_hosted_qwen_gapfill3 (100.0%); c1_hosted_qwen_gapfill4 (66.7%); c1_hosted_qwen_gapfill5 (75.0%) | c1_hosted_qwen_block3; c1_hosted_qwen_gapfill1; c1_hosted_qwen_gapfill2; c1_hosted_qwen_gapfill3; c1_hosted_qwen_gapfill4; c1_hosted_qwen_gapfill5; c1_hosted_qwen_gapfill6 | 354 | 0.8475 | 54 (15.3%) |
| Qwen3.8-Flash | c21_noisy | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | c1_hosted_qwen_block3 (7.7%); c1_hosted_qwen_gapfill1 (56.5%); c1_hosted_qwen_gapfill2 (92.3%); c1_hosted_qwen_gapfill3 (91.7%); c1_hosted_qwen_gapfill4 (100.0%); c1_hosted_qwen_gapfill5 (81.8%) | c1_hosted_qwen_block3; c1_hosted_qwen_gapfill1; c1_hosted_qwen_gapfill2; c1_hosted_qwen_gapfill3; c1_hosted_qwen_gapfill4; c1_hosted_qwen_gapfill5; c1_hosted_qwen_gapfill6 | 379 | 0.7916 | 79 (20.8%) |
| Qwen3.8-Flash | c21_contradictory | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | None | c1_hosted_qwen_block3 | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.8-Flash | c57_fresh | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | None | c1_hosted_qwen_block3 | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.8-Flash | c57_stale | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | c1_hosted_qwen_block3 (27.7%) | c1_hosted_qwen_block3; c1_hosted_qwen_gapfill1 | 383 | 0.7833 | 83 (21.7%) |
| Qwen3.8-Flash | c57_noisy | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | c1_hosted_qwen_block3 (81.0%); c1_hosted_qwen_gapfill2 (90.0%); c1_hosted_qwen_gapfill3 (77.8%); c1_hosted_qwen_gapfill4 (85.7%); c1_hosted_qwen_gapfill5 (83.3%) | c1_hosted_qwen_block3; c1_hosted_qwen_gapfill1; c1_hosted_qwen_gapfill2; c1_hosted_qwen_gapfill3; c1_hosted_qwen_gapfill4; c1_hosted_qwen_gapfill5; c1_hosted_qwen_gapfill6 | 580 | 0.5172 | 280 (48.3%) |
| Qwen3.8-Flash | c57_contradictory | complete | per-case (D-3) | 300 | 0 | None | 1.0000 | no | c1_hosted_qwen_block3 (55.7%); c1_hosted_qwen_gapfill2 (100.0%); c1_hosted_qwen_gapfill3 (100.0%); c1_hosted_qwen_gapfill4 (100.0%); c1_hosted_qwen_gapfill5 (100.0%) | c1_hosted_qwen_block3; c1_hosted_qwen_gapfill1; c1_hosted_qwen_gapfill2; c1_hosted_qwen_gapfill3; c1_hosted_qwen_gapfill4; c1_hosted_qwen_gapfill5; c1_hosted_qwen_gapfill6 | 472 | 0.6356 | 172 (36.4%) |
| SemIf-Qwen3.5-4B | c3_fresh | complete | c1_selfhosted/semif/c3_fresh/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| SemIf-Qwen3.5-4B | c3_stale | complete | c1_selfhosted/semif/c3_stale/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| SemIf-Qwen3.5-4B | c3_noisy | complete | c1_selfhosted/semif/c3_noisy/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| SemIf-Qwen3.5-4B | c3_contradictory | complete | c1_selfhosted/semif/c3_contradictory/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| SemIf-Qwen3.5-4B | c7_fresh | complete | c1_selfhosted/semif/c7_fresh/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| SemIf-Qwen3.5-4B | c7_stale | complete | c1_selfhosted/semif/c7_stale/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| SemIf-Qwen3.5-4B | c7_noisy | complete | c1_selfhosted/semif/c7_noisy/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| SemIf-Qwen3.5-4B | c7_contradictory | complete | c1_selfhosted/semif/c7_contradictory/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| SemIf-Qwen3.5-4B | c21_fresh | complete | c1_selfhosted/semif/c21_fresh/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| SemIf-Qwen3.5-4B | c21_stale | complete | c1_selfhosted/semif/c21_stale/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| SemIf-Qwen3.5-4B | c21_noisy | complete | c1_selfhosted/semif/c21_noisy/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| SemIf-Qwen3.5-4B | c21_contradictory | complete | c1_selfhosted/semif/c21_contradictory/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| SemIf-Qwen3.5-4B | c57_fresh | complete | c1_selfhosted/semif/c57_fresh/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| SemIf-Qwen3.5-4B | c57_stale | complete | c1_selfhosted/semif/c57_stale/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| SemIf-Qwen3.5-4B | c57_noisy | complete | c1_selfhosted/semif/c57_noisy/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| SemIf-Qwen3.5-4B | c57_contradictory | complete | c1_selfhosted/semif/c57_contradictory/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c3_fresh | complete | c1_selfhosted/anyjev/c3_fresh/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c3_stale | complete | c1_selfhosted/anyjev/c3_stale/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c3_noisy | complete | c1_selfhosted/anyjev/c3_noisy/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c3_contradictory | complete | c1_selfhosted/anyjev/c3_contradictory/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c7_fresh | complete | c1_selfhosted/anyjev/c7_fresh/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c7_stale | complete | c1_selfhosted/anyjev/c7_stale/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c7_noisy | complete | c1_selfhosted/anyjev/c7_noisy/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c7_contradictory | complete | c1_selfhosted/anyjev/c7_contradictory/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c21_fresh | complete | c1_selfhosted/anyjev/c21_fresh/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c21_stale | complete | c1_selfhosted/anyjev/c21_stale/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c21_noisy | complete | c1_selfhosted/anyjev/c21_noisy/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c21_contradictory | complete | c1_selfhosted/anyjev/c21_contradictory/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c57_fresh | complete | c1_selfhosted/anyjev/c57_fresh/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c57_stale | complete | c1_selfhosted/anyjev/c57_stale/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c57_noisy | complete | c1_selfhosted/anyjev/c57_noisy/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| AnyJev-L0 | c57_contradictory | complete | c1_selfhosted/anyjev/c57_contradictory/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c3_fresh | complete | c1_selfhosted/qwen_json/c3_fresh/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c3_stale | complete | c1_selfhosted/qwen_json/c3_stale/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c3_noisy | complete | c1_selfhosted/qwen_json/c3_noisy/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c3_contradictory | complete | c1_selfhosted/qwen_json/c3_contradictory/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c7_fresh | complete | c1_selfhosted/qwen_json/c7_fresh/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c7_stale | complete | c1_selfhosted/qwen_json/c7_stale/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c7_noisy | complete | c1_selfhosted/qwen_json/c7_noisy/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c7_contradictory | complete | c1_selfhosted/qwen_json/c7_contradictory/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c21_fresh | complete | c1_selfhosted/qwen_json/c21_fresh/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c21_stale | complete | c1_selfhosted/qwen_json/c21_stale/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c21_noisy | complete | c1_selfhosted/qwen_json/c21_noisy/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c21_contradictory | complete | c1_selfhosted/qwen_json/c21_contradictory/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c57_fresh | complete | c1_selfhosted/qwen_json/c57_fresh/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c57_stale | complete | c1_selfhosted/qwen_json/c57_stale/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c57_noisy | complete | c1_selfhosted/qwen_json/c57_noisy/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |
| Qwen3.5-4B-JSON | c57_contradictory | complete | c1_selfhosted/qwen_json/c57_contradictory/test | 300 | 0 | None | 1.0000 | no | None | None | 300 | 1.0000 | 0 (0.0%) |

---

## 2. Near-RT Feasibility ($\phi = P(\text{latency} + \delta_{E2} \le 1\text{ s})$)

Traceable to `near_rt_feasibility.csv` ($\delta_{E2} = 0.005\text{ s}$, seed 20260925, 10,000 bootstrap resamples).

| Interpreter | Condition | Total (n) | Feasible | $\phi$ | 95% Bootstrap CI |
|---|---|---|---|---|---|
| Jev-1.13.0 | c3_fresh | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Jev-1.13.0 | c3_stale | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Jev-1.13.0 | c3_noisy | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Jev-1.13.0 | c3_contradictory | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Jev-1.13.0 | c7_fresh | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| Jev-1.13.0 | c7_stale | 300 | 297 | 0.9900 | [0.9767, 1.0000] |
| Jev-1.13.0 | c7_noisy | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Jev-1.13.0 | c7_contradictory | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| Jev-1.13.0 | c21_fresh | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Jev-1.13.0 | c21_stale | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| Jev-1.13.0 | c21_noisy | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| Jev-1.13.0 | c21_contradictory | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| Jev-1.13.0 | c57_fresh | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| Jev-1.13.0 | c57_stale | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Jev-1.13.0 | c57_noisy | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| Jev-1.13.0 | c57_contradictory | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| DeepSeek-V4.1-Flash | c3_fresh | 300 | 297 | 0.9900 | [0.9767, 1.0000] |
| DeepSeek-V4.1-Flash | c3_stale | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| DeepSeek-V4.1-Flash | c3_noisy | 300 | 294 | 0.9800 | [0.9633, 0.9933] |
| DeepSeek-V4.1-Flash | c3_contradictory | 300 | 291 | 0.9700 | [0.9500, 0.9867] |
| DeepSeek-V4.1-Flash | c7_fresh | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| DeepSeek-V4.1-Flash | c7_stale | 300 | 297 | 0.9900 | [0.9767, 1.0000] |
| DeepSeek-V4.1-Flash | c7_noisy | 300 | 288 | 0.9600 | [0.9367, 0.9800] |
| DeepSeek-V4.1-Flash | c7_contradictory | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| DeepSeek-V4.1-Flash | c21_fresh | 300 | 296 | 0.9867 | [0.9733, 0.9967] |
| DeepSeek-V4.1-Flash | c21_stale | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| DeepSeek-V4.1-Flash | c21_noisy | 300 | 294 | 0.9800 | [0.9633, 0.9933] |
| DeepSeek-V4.1-Flash | c21_contradictory | 300 | 291 | 0.9700 | [0.9500, 0.9867] |
| DeepSeek-V4.1-Flash | c57_fresh | 300 | 297 | 0.9900 | [0.9767, 1.0000] |
| DeepSeek-V4.1-Flash | c57_stale | 300 | 273 | 0.9100 | [0.8767, 0.9400] |
| DeepSeek-V4.1-Flash | c57_noisy | 300 | 278 | 0.9267 | [0.8967, 0.9533] |
| DeepSeek-V4.1-Flash | c57_contradictory | 300 | 287 | 0.9567 | [0.9333, 0.9767] |
| GLM-5.3-Flash | c3_fresh | 300 | 91 | 0.3033 | [0.2500, 0.3567] |
| GLM-5.3-Flash | c3_stale | 300 | 53 | 0.1767 | [0.1333, 0.2200] |
| GLM-5.3-Flash | c3_noisy | 300 | 60 | 0.2000 | [0.1567, 0.2467] |
| GLM-5.3-Flash | c3_contradictory | 300 | 56 | 0.1867 | [0.1433, 0.2333] |
| GLM-5.3-Flash | c7_fresh | 300 | 52 | 0.1733 | [0.1300, 0.2167] |
| GLM-5.3-Flash | c7_stale | 300 | 70 | 0.2333 | [0.1867, 0.2833] |
| GLM-5.3-Flash | c7_noisy | 300 | 60 | 0.2000 | [0.1567, 0.2467] |
| GLM-5.3-Flash | c7_contradictory | 300 | 46 | 0.1533 | [0.1133, 0.1933] |
| GLM-5.3-Flash | c21_fresh | 300 | 50 | 0.1667 | [0.1267, 0.2100] |
| GLM-5.3-Flash | c21_stale | 300 | 49 | 0.1633 | [0.1233, 0.2067] |
| GLM-5.3-Flash | c21_noisy | 300 | 63 | 0.2100 | [0.1633, 0.2567] |
| GLM-5.3-Flash | c21_contradictory | 300 | 36 | 0.1200 | [0.0833, 0.1600] |
| GLM-5.3-Flash | c57_fresh | 300 | 71 | 0.2367 | [0.1900, 0.2867] |
| GLM-5.3-Flash | c57_stale | 300 | 19 | 0.0633 | [0.0367, 0.0900] |
| GLM-5.3-Flash | c57_noisy | 300 | 58 | 0.1933 | [0.1500, 0.2400] |
| GLM-5.3-Flash | c57_contradictory | 300 | 24 | 0.0800 | [0.0500, 0.1133] |
| Qwen3.8-Flash | c3_fresh | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| Qwen3.8-Flash | c3_stale | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| Qwen3.8-Flash | c3_noisy | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| Qwen3.8-Flash | c3_contradictory | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| Qwen3.8-Flash | c7_fresh | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| Qwen3.8-Flash | c7_stale | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| Qwen3.8-Flash | c7_noisy | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| Qwen3.8-Flash | c7_contradictory | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| Qwen3.8-Flash | c21_fresh | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| Qwen3.8-Flash | c21_stale | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| Qwen3.8-Flash | c21_noisy | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| Qwen3.8-Flash | c21_contradictory | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| Qwen3.8-Flash | c57_fresh | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| Qwen3.8-Flash | c57_stale | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| Qwen3.8-Flash | c57_noisy | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| Qwen3.8-Flash | c57_contradictory | 300 | 0 | 0.0000 | [0.0000, 0.0000] |
| SemIf-Qwen3.5-4B | c3_fresh | 300 | 298 | 0.9933 | [0.9833, 1.0000] |
| SemIf-Qwen3.5-4B | c3_stale | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| SemIf-Qwen3.5-4B | c3_noisy | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| SemIf-Qwen3.5-4B | c3_contradictory | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| SemIf-Qwen3.5-4B | c7_fresh | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| SemIf-Qwen3.5-4B | c7_stale | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| SemIf-Qwen3.5-4B | c7_noisy | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| SemIf-Qwen3.5-4B | c7_contradictory | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| SemIf-Qwen3.5-4B | c21_fresh | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| SemIf-Qwen3.5-4B | c21_stale | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| SemIf-Qwen3.5-4B | c21_noisy | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| SemIf-Qwen3.5-4B | c21_contradictory | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| SemIf-Qwen3.5-4B | c57_fresh | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| SemIf-Qwen3.5-4B | c57_stale | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| SemIf-Qwen3.5-4B | c57_noisy | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| SemIf-Qwen3.5-4B | c57_contradictory | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| AnyJev-L0 | c3_fresh | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| AnyJev-L0 | c3_stale | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| AnyJev-L0 | c3_noisy | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| AnyJev-L0 | c3_contradictory | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| AnyJev-L0 | c7_fresh | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| AnyJev-L0 | c7_stale | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| AnyJev-L0 | c7_noisy | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| AnyJev-L0 | c7_contradictory | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| AnyJev-L0 | c21_fresh | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| AnyJev-L0 | c21_stale | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| AnyJev-L0 | c21_noisy | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| AnyJev-L0 | c21_contradictory | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| AnyJev-L0 | c57_fresh | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| AnyJev-L0 | c57_stale | 300 | 178 | 0.5933 | [0.5367, 0.6467] |
| AnyJev-L0 | c57_noisy | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| AnyJev-L0 | c57_contradictory | 300 | 299 | 0.9967 | [0.9900, 1.0000] |
| Qwen3.5-4B-JSON | c3_fresh | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Qwen3.5-4B-JSON | c3_stale | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Qwen3.5-4B-JSON | c3_noisy | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Qwen3.5-4B-JSON | c3_contradictory | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Qwen3.5-4B-JSON | c7_fresh | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Qwen3.5-4B-JSON | c7_stale | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Qwen3.5-4B-JSON | c7_noisy | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Qwen3.5-4B-JSON | c7_contradictory | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Qwen3.5-4B-JSON | c21_fresh | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Qwen3.5-4B-JSON | c21_stale | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Qwen3.5-4B-JSON | c21_noisy | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Qwen3.5-4B-JSON | c21_contradictory | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Qwen3.5-4B-JSON | c57_fresh | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Qwen3.5-4B-JSON | c57_stale | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Qwen3.5-4B-JSON | c57_noisy | 300 | 300 | 1.0000 | [1.0000, 1.0000] |
| Qwen3.5-4B-JSON | c57_contradictory | 300 | 300 | 1.0000 | [1.0000, 1.0000] |

---

## 3. Confirmatory Hypothesis H3 (Telemetry Grounding Contrast)

Traceable to `h3_confirmatory.csv` and `h3_c57_comparisons.csv`.
H3 tests target_cluster accuracy at `c3_fresh` minus `c57_fresh` > 0 over the 150 state-dependent cases.
"Resolved" requires 95% CI to exclude 0 and Holm-adjusted p < 0.05.
Holm family: the six interpreters (Jev-1.13.0, DeepSeek-V4.1-Flash, GLM-5.3-Flash, Qwen3.8-Flash, SemIf-Qwen3.5-4B, AnyJev-L0); an untested member stays in the
family (p = 1 in the adjustment) and is not resolved. Reference models are reported outside the family.
seed 20260925, 10,000 bootstrap resamples.

### H3 Paired Contrast

| Interpreter | Cases | Acc (c3_fresh) | Acc (c57_fresh) | $\Delta$ (c3 - c57) | 95% Bootstrap CI | Raw p | Holm p | Resolved |
|---|---|---|---|---|---|---|---|---|
| Jev-1.13.0 | 150 | 1.0000 | 0.9067 | +0.0933 | [+0.0467, +0.1400] | 0.0000e+00 | 0.0000e+00 | **TRUE** |
| DeepSeek-V4.1-Flash | 150 | 0.9800 | 0.9600 | +0.0200 | [+0.0000, +0.0467] | 9.2200e-02 | 1.8440e-01 | FALSE |
| GLM-5.3-Flash | 150 | 0.9933 | 0.9667 | +0.0267 | [+0.0000, +0.0600] | 1.2220e-01 | 1.8440e-01 | FALSE |
| Qwen3.8-Flash | 150 | 0.9933 | 0.8800 | +0.1133 | [+0.0667, +0.1667] | 0.0000e+00 | 0.0000e+00 | **TRUE** |
| SemIf-Qwen3.5-4B | 150 | 0.2467 | 0.1400 | +0.1067 | [+0.0333, +0.1800] | 5.0000e-03 | 1.5000e-02 | **TRUE** |
| AnyJev-L0 | 150 | 0.5933 | 0.2800 | +0.3133 | [+0.2200, +0.4067] | 0.0000e+00 | 0.0000e+00 | **TRUE** |
| Qwen3.5-4B-JSON | 150 | 0.6333 | 0.2867 | +0.3467 | [+0.2600, +0.4333] | 0.0000e+00 | N/A (outside family) | reference |

### Multi-Interpreter Comparisons at `c57_fresh`

| Test Type | Condition | Model 1 | Model 2 | Statistic | Value | df | p-value | Status | Details |
|---|---|---|---|---|---|---|---|---|---|
| Cochran_Q | c57_fresh | All six (Jev-1.13.0, DeepSeek-V4.1-Flash, GLM-5.3-Flash, Qwen3.8-Flash, SemIf-Qwen3.5-4B, AnyJev-L0) | N/A | Q | 458.0888 | 5 | 8.8382e-97 | tested | k=6 interpreters |
| McNemar | c57_fresh | DeepSeek-V4.1-Flash | Jev-1.13.0 | b/c | 11/3 | 1 | 5.7373e-02 | tested | DeepSeek-V4.1-Flash correct only: 11, Jev correct only: 3 |
| McNemar | c57_fresh | GLM-5.3-Flash | Jev-1.13.0 | b/c | 14/5 | 1 | 6.3568e-02 | tested | GLM-5.3-Flash correct only: 14, Jev correct only: 5 |
| McNemar | c57_fresh | Qwen3.8-Flash | Jev-1.13.0 | b/c | 9/13 | 1 | 5.2347e-01 | tested | Qwen3.8-Flash correct only: 9, Jev correct only: 13 |
| McNemar | c57_fresh | SemIf-Qwen3.5-4B | Jev-1.13.0 | b/c | 1/116 | 1 | 1.4204e-33 | tested | SemIf-Qwen3.5-4B correct only: 1, Jev correct only: 116 |
| McNemar | c57_fresh | AnyJev-L0 | Jev-1.13.0 | b/c | 4/98 | 1 | 1.7460e-24 | tested | AnyJev-L0 correct only: 4, Jev correct only: 98 |

---

## 4. Quality, Accuracy & Cost Summary (Selected Key Metrics)

Traceable to `quality_and_cost.csv`. Correctness is over all primary attempts (schema-invalid and
transport-failed attempts score false); latency and tokens over every row that reports them; fees per
1,000 correct over the cost-reporting rows (numerator and denominator alike).

| Interpreter | Condition | Attempts | Valid n | Schema Valid Rate | Actuated Vector Match | Full Policy Match | Target Cluster (State-Dep) | Latency p50 (s) | Net p50 (s) | Cost / 1k Correct ($) |
|---|---|---|---|---|---|---|---|---|---|---|
| Jev-1.13.0 | c3_fresh | 300 | 300 | 1.0000 | 0.9933 | 0.9933 | 1.0000 | 0.264 | 0.002 | $0.1359 |
| Jev-1.13.0 | c3_stale | 300 | 300 | 1.0000 | 0.9933 | 0.9100 | 0.8333 | 0.265 | 0.013 | $0.1453 |
| Jev-1.13.0 | c3_noisy | 300 | 300 | 1.0000 | 0.9933 | 0.9900 | 0.9933 | 0.266 | 0.009 | $0.1377 |
| Jev-1.13.0 | c3_contradictory | 300 | 300 | 1.0000 | 0.9933 | 0.8567 | 0.7267 | 0.264 | 0.019 | $0.1408 |
| Jev-1.13.0 | c7_fresh | 300 | 300 | 1.0000 | 0.9933 | 0.9767 | 0.9667 | 0.263 | 0.002 | $0.1411 |
| Jev-1.13.0 | c7_stale | 300 | 298 | 0.9933 | 0.9867 | 0.8867 | 0.7933 | 0.263 | 0.014 | $0.1629 |
| Jev-1.13.0 | c7_noisy | 300 | 300 | 1.0000 | 0.9933 | 0.9833 | 0.9800 | 0.268 | 0.027 | $0.1429 |
| Jev-1.13.0 | c7_contradictory | 300 | 300 | 1.0000 | 0.9933 | 0.8667 | 0.7467 | 0.267 | 0.013 | $0.1525 |
| Jev-1.13.0 | c21_fresh | 300 | 300 | 1.0000 | 0.9933 | 0.9433 | 0.9000 | 0.274 | 0.003 | $0.1595 |
| Jev-1.13.0 | c21_stale | 300 | 300 | 1.0000 | 0.9933 | 0.8333 | 0.6800 | 0.277 | 0.031 | $0.2245 |
| Jev-1.13.0 | c21_noisy | 300 | 299 | 0.9967 | 0.9900 | 0.9567 | 0.9267 | 0.271 | 0.002 | $0.1613 |
| Jev-1.13.0 | c21_contradictory | 300 | 300 | 1.0000 | 0.9933 | 0.8500 | 0.7133 | 0.279 | 0.018 | $0.1933 |
| Jev-1.13.0 | c57_fresh | 300 | 300 | 1.0000 | 0.9933 | 0.9467 | 0.9067 | 0.275 | 0.005 | $0.2066 |
| Jev-1.13.0 | c57_stale | 300 | 300 | 1.0000 | 0.9900 | 0.8433 | 0.7000 | 0.296 | 0.042 | $0.3840 |
| Jev-1.13.0 | c57_noisy | 300 | 300 | 1.0000 | 0.9933 | 0.9367 | 0.8867 | 0.275 | 0.009 | $0.2084 |
| Jev-1.13.0 | c57_contradictory | 300 | 300 | 1.0000 | 0.9900 | 0.8100 | 0.6333 | 0.286 | 0.032 | $0.2993 |
| DeepSeek-V4.1-Flash | c3_fresh | 300 | 300 | 1.0000 | 0.9767 | 0.9700 | 0.9800 | 0.430 | 0.136 | $0.1861 |
| DeepSeek-V4.1-Flash | c3_stale | 300 | 300 | 1.0000 | 0.9733 | 0.8433 | 0.7467 | 0.459 | 0.157 | $0.2122 |
| DeepSeek-V4.1-Flash | c3_noisy | 300 | 300 | 1.0000 | 0.9833 | 0.9733 | 0.9867 | 0.464 | 0.159 | $0.1761 |
| DeepSeek-V4.1-Flash | c3_contradictory | 300 | 300 | 1.0000 | 0.9767 | 0.8167 | 0.6800 | 0.465 | 0.166 | $0.1928 |
| DeepSeek-V4.1-Flash | c7_fresh | 300 | 300 | 1.0000 | 0.9800 | 0.9667 | 0.9867 | 0.450 | 0.146 | $0.1914 |
| DeepSeek-V4.1-Flash | c7_stale | 300 | 300 | 1.0000 | 0.9767 | 0.8867 | 0.8333 | 0.460 | 0.156 | $0.3042 |
| DeepSeek-V4.1-Flash | c7_noisy | 300 | 300 | 1.0000 | 0.9767 | 0.9667 | 0.9867 | 0.464 | 0.159 | $0.2065 |
| DeepSeek-V4.1-Flash | c7_contradictory | 300 | 300 | 1.0000 | 0.9767 | 0.8400 | 0.7267 | 0.449 | 0.144 | $0.2609 |
| DeepSeek-V4.1-Flash | c21_fresh | 300 | 300 | 1.0000 | 0.9767 | 0.9633 | 0.9667 | 0.501 | 0.189 | $0.2941 |
| DeepSeek-V4.1-Flash | c21_stale | 300 | 300 | 1.0000 | 0.9767 | 0.8600 | 0.7600 | 0.500 | 0.192 | $0.6288 |
| DeepSeek-V4.1-Flash | c21_noisy | 300 | 300 | 1.0000 | 0.9733 | 0.9633 | 0.9800 | 0.495 | 0.189 | $0.3096 |
| DeepSeek-V4.1-Flash | c21_contradictory | 300 | 300 | 1.0000 | 0.9800 | 0.8400 | 0.7267 | 0.508 | 0.206 | $0.4979 |
| DeepSeek-V4.1-Flash | c57_fresh | 300 | 300 | 1.0000 | 0.9733 | 0.9500 | 0.9600 | 0.520 | 0.202 | $0.5587 |
| DeepSeek-V4.1-Flash | c57_stale | 300 | 300 | 1.0000 | 0.9733 | 0.8300 | 0.6933 | 0.607 | 0.272 | $1.4687 |
| DeepSeek-V4.1-Flash | c57_noisy | 300 | 300 | 1.0000 | 0.9767 | 0.9633 | 0.9733 | 0.572 | 0.249 | $0.5712 |
| DeepSeek-V4.1-Flash | c57_contradictory | 300 | 300 | 1.0000 | 0.9767 | 0.8167 | 0.6667 | 0.668 | 0.340 | $1.1132 |
| GLM-5.3-Flash | c3_fresh | 300 | 300 | 1.0000 | 0.9900 | 0.9400 | 0.9933 | 1.129 | 0.695 | $0.2703 |
| GLM-5.3-Flash | c3_stale | 300 | 300 | 1.0000 | 0.9900 | 0.9600 | 0.9933 | 1.291 | 0.861 | $0.2995 |
| GLM-5.3-Flash | c3_noisy | 300 | 300 | 1.0000 | 0.9900 | 0.9600 | 0.9933 | 1.253 | 0.799 | $0.2787 |
| GLM-5.3-Flash | c3_contradictory | 300 | 300 | 1.0000 | 0.9900 | 0.9133 | 0.9267 | 1.315 | 0.841 | $0.2925 |
| GLM-5.3-Flash | c7_fresh | 300 | 300 | 1.0000 | 0.9933 | 0.9500 | 0.9933 | 1.225 | 0.743 | $0.2857 |
| GLM-5.3-Flash | c7_stale | 300 | 300 | 1.0000 | 0.9933 | 0.9333 | 0.9667 | 1.215 | 0.765 | $0.3445 |
| GLM-5.3-Flash | c7_noisy | 300 | 300 | 1.0000 | 0.9867 | 0.9567 | 0.9933 | 1.167 | 0.745 | $0.2962 |
| GLM-5.3-Flash | c7_contradictory | 300 | 300 | 1.0000 | 0.9933 | 0.9233 | 0.9600 | 1.359 | 0.902 | $0.3295 |
| GLM-5.3-Flash | c21_fresh | 300 | 300 | 1.0000 | 0.9900 | 0.9267 | 0.9800 | 1.266 | 0.761 | $0.3367 |
| GLM-5.3-Flash | c21_stale | 300 | 300 | 1.0000 | 0.9900 | 0.9633 | 0.9867 | 1.261 | 0.802 | $0.5156 |
| GLM-5.3-Flash | c21_noisy | 300 | 300 | 1.0000 | 0.9900 | 0.9533 | 1.0000 | 1.203 | 0.718 | $0.3435 |
| GLM-5.3-Flash | c21_contradictory | 300 | 300 | 1.0000 | 0.9933 | 0.9533 | 0.9800 | 1.379 | 0.936 | $0.4450 |
| GLM-5.3-Flash | c57_fresh | 300 | 300 | 1.0000 | 0.9933 | 0.9400 | 0.9667 | 1.128 | 0.680 | $0.4670 |
| GLM-5.3-Flash | c57_stale | 300 | 300 | 1.0000 | 0.9933 | 0.9467 | 0.9533 | 1.460 | 0.931 | $0.9237 |
| GLM-5.3-Flash | c57_noisy | 300 | 294 | 0.9800 | 0.9700 | 0.9400 | 0.9667 | 1.222 | 0.745 | $0.4762 |
| GLM-5.3-Flash | c57_contradictory | 300 | 276 | 0.9200 | 0.9133 | 0.8133 | 0.8533 | 1.644 | 0.685 | $0.7437 |
| Qwen3.8-Flash | c3_fresh | 300 | 300 | 1.0000 | 0.9867 | 0.9633 | 0.9933 | 1.934 | 1.059 | $0.1080 |
| Qwen3.8-Flash | c3_stale | 300 | 300 | 1.0000 | 0.9867 | 0.7967 | 0.7200 | 2.051 | 1.217 | $0.1409 |
| Qwen3.8-Flash | c3_noisy | 300 | 300 | 1.0000 | 0.9933 | 0.9633 | 1.0000 | 2.096 | unavailable (no probe) | $0.1130 |
| Qwen3.8-Flash | c3_contradictory | 300 | 300 | 1.0000 | 0.9867 | 0.7700 | 0.6267 | 2.084 | unavailable (no probe) | $0.1248 |
| Qwen3.8-Flash | c7_fresh | 300 | 300 | 1.0000 | 0.9900 | 0.9533 | 0.9733 | 2.046 | 1.216 | $0.1255 |
| Qwen3.8-Flash | c7_stale | 300 | 300 | 1.0000 | 0.9900 | 0.8267 | 0.7467 | 2.347 | 1.371 | $0.2029 |
| Qwen3.8-Flash | c7_noisy | 300 | 300 | 1.0000 | 0.9900 | 0.9433 | 0.9867 | 2.025 | 1.140 | $0.1318 |
| Qwen3.8-Flash | c7_contradictory | 300 | 300 | 1.0000 | 0.9867 | 0.8233 | 0.7400 | 2.446 | 1.565 | $0.1660 |
| Qwen3.8-Flash | c21_fresh | 300 | 300 | 1.0000 | 0.9900 | 0.9367 | 0.9267 | 2.354 | unavailable (no probe) | $0.1921 |
| Qwen3.8-Flash | c21_stale | 300 | 300 | 1.0000 | 0.9867 | 0.8000 | 0.6867 | 2.216 | unavailable (no probe) | $0.4229 |
| Qwen3.8-Flash | c21_noisy | 300 | 300 | 1.0000 | 0.9867 | 0.9400 | 0.9400 | 2.184 | unavailable (no probe) | $0.1973 |
| Qwen3.8-Flash | c21_contradictory | 300 | 300 | 1.0000 | 0.9867 | 0.8267 | 0.7467 | 2.213 | 1.391 | $0.3108 |
| Qwen3.8-Flash | c57_fresh | 300 | 300 | 1.0000 | 0.9833 | 0.9067 | 0.8800 | 2.220 | 1.359 | $0.3588 |
| Qwen3.8-Flash | c57_stale | 300 | 300 | 1.0000 | 0.9900 | 0.8500 | 0.7333 | 2.470 | 1.623 | $0.9828 |
| Qwen3.8-Flash | c57_noisy | 300 | 300 | 1.0000 | 0.9867 | 0.9133 | 0.9067 | 2.273 | unavailable (no probe) | $0.3640 |
| Qwen3.8-Flash | c57_contradictory | 300 | 300 | 1.0000 | 0.9867 | 0.7667 | 0.6200 | 2.388 | 1.532 | $0.6826 |
| SemIf-Qwen3.5-4B | c3_fresh | 300 | 300 | 1.0000 | 0.9633 | 0.3000 | 0.2467 | 0.295 | n/a (self-hosted) | $0.0000 |
| SemIf-Qwen3.5-4B | c3_stale | 300 | 300 | 1.0000 | 0.9633 | 0.1900 | 0.1000 | 0.315 | n/a (self-hosted) | $0.0000 |
| SemIf-Qwen3.5-4B | c3_noisy | 300 | 300 | 1.0000 | 0.9633 | 0.3433 | 0.2333 | 0.304 | n/a (self-hosted) | $0.0000 |
| SemIf-Qwen3.5-4B | c3_contradictory | 300 | 300 | 1.0000 | 0.9667 | 0.2067 | 0.1400 | 0.307 | n/a (self-hosted) | $0.0000 |
| SemIf-Qwen3.5-4B | c7_fresh | 300 | 300 | 1.0000 | 0.9667 | 0.2433 | 0.1267 | 0.307 | n/a (self-hosted) | $0.0000 |
| SemIf-Qwen3.5-4B | c7_stale | 300 | 300 | 1.0000 | 0.9633 | 0.1500 | 0.1133 | 0.353 | n/a (self-hosted) | $0.0000 |
| SemIf-Qwen3.5-4B | c7_noisy | 300 | 300 | 1.0000 | 0.9600 | 0.2700 | 0.1533 | 0.312 | n/a (self-hosted) | $0.0000 |
| SemIf-Qwen3.5-4B | c7_contradictory | 300 | 300 | 1.0000 | 0.9600 | 0.2167 | 0.1267 | 0.332 | n/a (self-hosted) | $0.0000 |
| SemIf-Qwen3.5-4B | c21_fresh | 300 | 300 | 1.0000 | 0.9600 | 0.1167 | 0.1333 | 0.346 | n/a (self-hosted) | $0.0000 |
| SemIf-Qwen3.5-4B | c21_stale | 300 | 300 | 1.0000 | 0.9667 | 0.1567 | 0.1000 | 0.463 | n/a (self-hosted) | $0.0000 |
| SemIf-Qwen3.5-4B | c21_noisy | 300 | 300 | 1.0000 | 0.9533 | 0.1567 | 0.1667 | 0.350 | n/a (self-hosted) | $0.0000 |
| SemIf-Qwen3.5-4B | c21_contradictory | 300 | 300 | 1.0000 | 0.9633 | 0.1033 | 0.1333 | 0.409 | n/a (self-hosted) | $0.0000 |
| SemIf-Qwen3.5-4B | c57_fresh | 300 | 300 | 1.0000 | 0.9533 | 0.1000 | 0.1400 | 0.431 | n/a (self-hosted) | $0.0000 |
| SemIf-Qwen3.5-4B | c57_stale | 300 | 300 | 1.0000 | 0.9633 | 0.1000 | 0.0533 | 0.752 | n/a (self-hosted) | $0.0000 |
| SemIf-Qwen3.5-4B | c57_noisy | 300 | 300 | 1.0000 | 0.9533 | 0.1067 | 0.1467 | 0.433 | n/a (self-hosted) | $0.0000 |
| SemIf-Qwen3.5-4B | c57_contradictory | 300 | 300 | 1.0000 | 0.9600 | 0.1033 | 0.1667 | 0.601 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c3_fresh | 300 | 300 | 1.0000 | 0.8433 | 0.2567 | 0.5933 | 0.357 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c3_stale | 300 | 300 | 1.0000 | 0.8333 | 0.1367 | 0.3533 | 0.516 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c3_noisy | 300 | 300 | 1.0000 | 0.8400 | 0.2067 | 0.5400 | 0.374 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c3_contradictory | 300 | 300 | 1.0000 | 0.8567 | 0.1200 | 0.2933 | 0.415 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c7_fresh | 300 | 300 | 1.0000 | 0.8400 | 0.1367 | 0.3667 | 0.420 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c7_stale | 300 | 300 | 1.0000 | 0.8467 | 0.1100 | 0.2933 | 0.408 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c7_noisy | 300 | 300 | 1.0000 | 0.8333 | 0.1500 | 0.4067 | 0.456 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c7_contradictory | 300 | 300 | 1.0000 | 0.8033 | 0.0700 | 0.2200 | 0.709 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c21_fresh | 300 | 300 | 1.0000 | 0.8433 | 0.0967 | 0.2800 | 0.380 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c21_stale | 300 | 300 | 1.0000 | 0.8500 | 0.0633 | 0.2400 | 0.740 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c21_noisy | 300 | 300 | 1.0000 | 0.8400 | 0.1300 | 0.3333 | 0.396 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c21_contradictory | 300 | 300 | 1.0000 | 0.8233 | 0.0900 | 0.2267 | 0.568 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c57_fresh | 300 | 300 | 1.0000 | 0.8167 | 0.0967 | 0.2800 | 0.403 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c57_stale | 300 | 300 | 1.0000 | 0.8600 | 0.0767 | 0.2133 | 0.716 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c57_noisy | 300 | 300 | 1.0000 | 0.8533 | 0.0933 | 0.2400 | 0.421 | n/a (self-hosted) | $0.0000 |
| AnyJev-L0 | c57_contradictory | 300 | 300 | 1.0000 | 0.7967 | 0.0667 | 0.2467 | 0.491 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c3_fresh | 300 | 300 | 1.0000 | 0.7233 | 0.1633 | 0.6333 | 0.528 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c3_stale | 300 | 300 | 1.0000 | 0.6767 | 0.0733 | 0.4000 | 0.550 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c3_noisy | 300 | 300 | 1.0000 | 0.7233 | 0.1700 | 0.6400 | 0.408 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c3_contradictory | 300 | 300 | 1.0000 | 0.6833 | 0.0933 | 0.3467 | 0.412 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c7_fresh | 300 | 300 | 1.0000 | 0.7067 | 0.1000 | 0.4733 | 0.412 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c7_stale | 300 | 300 | 1.0000 | 0.6800 | 0.0767 | 0.3867 | 0.413 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c7_noisy | 300 | 300 | 1.0000 | 0.7067 | 0.1133 | 0.5333 | 0.411 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c7_contradictory | 300 | 300 | 1.0000 | 0.6767 | 0.0900 | 0.3400 | 0.411 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c21_fresh | 300 | 300 | 1.0000 | 0.7000 | 0.0900 | 0.3667 | 0.413 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c21_stale | 300 | 300 | 1.0000 | 0.6967 | 0.0667 | 0.3000 | 0.438 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c21_noisy | 300 | 300 | 1.0000 | 0.7067 | 0.0833 | 0.3867 | 0.413 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c21_contradictory | 300 | 300 | 1.0000 | 0.6800 | 0.0733 | 0.2867 | 0.417 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c57_fresh | 300 | 300 | 1.0000 | 0.6933 | 0.0667 | 0.2867 | 0.429 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c57_stale | 300 | 300 | 1.0000 | 0.7100 | 0.0800 | 0.2733 | 0.518 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c57_noisy | 300 | 300 | 1.0000 | 0.7033 | 0.0700 | 0.2733 | 0.429 | n/a (self-hosted) | $0.0000 |
| Qwen3.5-4B-JSON | c57_contradictory | 300 | 300 | 1.0000 | 0.6667 | 0.0800 | 0.3000 | 0.475 | n/a (self-hosted) | $0.0000 |

---

## 5. Exploratory Quality Contrasts (c3 minus c57)

Traceable to `exploratory_contrasts.csv` (seed 20260925, 10,000 bootstrap resamples).

| Quality | Interpreter | Cases | Acc (c3) | Acc (c57) | $\Delta$ (c3 - c57) | 95% Bootstrap CI | p-value | Status |
|---|---|---|---|---|---|---|---|---|
| stale | Jev-1.13.0 | 150 | 0.8333 | 0.7000 | +0.1333 | [+0.0333, +0.2333] | 8.8000e-03 | tested |
| stale | DeepSeek-V4.1-Flash | 150 | 0.7467 | 0.6933 | +0.0533 | [-0.0400, +0.1533] | 3.2160e-01 | tested |
| stale | GLM-5.3-Flash | 150 | 0.9933 | 0.9533 | +0.0400 | [+0.0067, +0.0800] | 3.2400e-02 | tested |
| stale | Qwen3.8-Flash | 150 | 0.7200 | 0.7333 | -0.0133 | [-0.1000, +0.0733] | 8.3900e-01 | tested |
| stale | SemIf-Qwen3.5-4B | 150 | 0.1000 | 0.0533 | +0.0467 | [+0.0000, +0.1000] | 8.0200e-02 | tested |
| stale | AnyJev-L0 | 150 | 0.3533 | 0.2133 | +0.1400 | [+0.0467, +0.2333] | 2.6000e-03 | tested |
| stale | Qwen3.5-4B-JSON | 150 | 0.4000 | 0.2733 | +0.1267 | [+0.0467, +0.2067] | 2.0000e-03 | tested |
| noisy | Jev-1.13.0 | 150 | 0.9933 | 0.8867 | +0.1067 | [+0.0533, +0.1600] | 0.0000e+00 | tested |
| noisy | DeepSeek-V4.1-Flash | 150 | 0.9867 | 0.9733 | +0.0133 | [+0.0000, +0.0333] | 2.6440e-01 | tested |
| noisy | GLM-5.3-Flash | 150 | 0.9933 | 0.9667 | +0.0267 | [+0.0000, +0.0600] | 1.3120e-01 | tested |
| noisy | Qwen3.8-Flash | 150 | 1.0000 | 0.9067 | +0.0933 | [+0.0533, +0.1400] | 0.0000e+00 | tested |
| noisy | SemIf-Qwen3.5-4B | 150 | 0.2333 | 0.1467 | +0.0867 | [+0.0133, +0.1600] | 1.9800e-02 | tested |
| noisy | AnyJev-L0 | 150 | 0.5400 | 0.2400 | +0.3000 | [+0.2067, +0.3933] | 0.0000e+00 | tested |
| noisy | Qwen3.5-4B-JSON | 150 | 0.6400 | 0.2733 | +0.3667 | [+0.2733, +0.4600] | 0.0000e+00 | tested |
| contradictory | Jev-1.13.0 | 150 | 0.7267 | 0.6333 | +0.0933 | [-0.0133, +0.1933] | 8.2400e-02 | tested |
| contradictory | DeepSeek-V4.1-Flash | 150 | 0.6800 | 0.6667 | +0.0133 | [-0.0800, +0.1133] | 8.3880e-01 | tested |
| contradictory | GLM-5.3-Flash | 150 | 0.9267 | 0.8533 | +0.0733 | [+0.0000, +0.1467] | 5.5600e-02 | tested |
| contradictory | Qwen3.8-Flash | 150 | 0.6267 | 0.6200 | +0.0067 | [-0.0933, +0.1067] | 9.3780e-01 | tested |
| contradictory | SemIf-Qwen3.5-4B | 150 | 0.1400 | 0.1667 | -0.0267 | [-0.0933, +0.0400] | 5.0660e-01 | tested |
| contradictory | AnyJev-L0 | 150 | 0.2933 | 0.2467 | +0.0467 | [-0.0400, +0.1333] | 3.3960e-01 | tested |
| contradictory | Qwen3.5-4B-JSON | 150 | 0.3467 | 0.3000 | +0.0467 | [-0.0333, +0.1267] | 2.8920e-01 | tested |

---

## 6. Sensitivity Re-Run Comparisons

Traceable to `sensitivity.csv`.

| Interpreter | Condition | Primary Source | Sensitivity Source | Metric | Primary | Sensitivity | $\Delta$ |
|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash | c57_contradictory | c1_hosted | c1_rerun/c57_contradictory__GLM-5.3-Flash__rerun1 | Total Cases (n) | 300 | 300 | +0.0000 |
| GLM-5.3-Flash | c57_contradictory | c1_hosted | c1_rerun/c57_contradictory__GLM-5.3-Flash__rerun1 | Valid Cases | 276 | 300 | +24.0000 |
| GLM-5.3-Flash | c57_contradictory | c1_hosted | c1_rerun/c57_contradictory__GLM-5.3-Flash__rerun1 | Transport Errors | 24 | 0 | -24.0000 |
| GLM-5.3-Flash | c57_contradictory | c1_hosted | c1_rerun/c57_contradictory__GLM-5.3-Flash__rerun1 | Availability Rate | 0.9200 | 1.0000 | +0.0800 |
| GLM-5.3-Flash | c57_contradictory | c1_hosted | c1_rerun/c57_contradictory__GLM-5.3-Flash__rerun1 | Actuated Vector Match | 0.9133 | 0.9867 | +0.0733 |
| GLM-5.3-Flash | c57_contradictory | c1_hosted | c1_rerun/c57_contradictory__GLM-5.3-Flash__rerun1 | Full Policy Match (EM) | 0.8133 | 0.8767 | +0.0633 |
| GLM-5.3-Flash | c57_contradictory | c1_hosted | c1_rerun/c57_contradictory__GLM-5.3-Flash__rerun1 | Latency p50 (s) | 1.6438 | 1.3549 | -0.2889 |
| GLM-5.3-Flash | c57_contradictory | c1_hosted | c1_rerun/c57_contradictory__GLM-5.3-Flash__rerun1 | Latency p95 (s) | 2.9510 | 2.8664 | -0.0846 |
