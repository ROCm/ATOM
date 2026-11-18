# 2026-11-18 MI455 Workload Weekly Status Report

## Highlights

| Model | What happened |
|---|---|
| ✅ DeepSeek-V4.1-Flash | recovered from the previous failure |
| ✅ DeepSeek-V4-Pro | recovered from the previous failure |

## Results

| Model | 2026-11-14 (GSM8K) | 2026-11-18 (GSM8K) | Delta |
|---|---|---|---|
| DeepSeek R1 | 95.7% | 95.7% | 0.0 pts |
| DeepSeek-V4.1-Flash | — | 91.2% | first |
| DeepSeek-V4-Pro | — | 95.1% | first |

## Environment

- Image: `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261118`
- ATOM: `main` @ `9b2e67b7f6f5841b0d32db9589e989ff52673761` (2026-11-18)
- aiter: `main` @ `b1d849ff36907402d7ea6a3483e6f0dafa995aca` (2026-11-18)

## Findings

- Setup: the three DeepSeek models ran on g13-01, g13-03 and g13-11 (one model each) instead of e15-06/e15-07; the other five models of the session are in the unsuffixed files. The image tag was pushed twice during the session, and all three models ran on the second digest (`sha256:d17a28d3…`, aiter `b1d849ff`).
- 8K/1K · DeepSeek-V4-Pro recovered: the server was ready in 601 s and GSM8K is 0.9507 (2026-11-14 revert check: 0.9477), with no MegaMoE S1F GEMM1 `assert slot_width <= 128`. The image has https://github.com/ROCm/aiter/pull/7973 (merged 2026-11-16), which targets that assert; https://github.com/ROCm/aiter/pull/7921, still in `pending_fixes.csv` for V4-Pro, was not needed.
- MLPerf Endpoint · DeepSeek-V4.1-Flash: TP4 gives 0.9189 (same as 2026-11-14), and DEP4 gives 0.9121, its first GSM8K on main since https://github.com/ROCm/ATOM/pull/2989 (DEP4 runs FP4 KV and index caches). https://github.com/ROCm/ATOM/pull/2724 (merged 2026-11-18) makes the recipe's `ATOM_DSPARK_PROB_DRAFT` and `ATOM_DSPARK_BLOCK_VERIFY` take effect on main; TP4's score did not move.
- MLPerf Endpoint · DeepSeek-V4.1-Flash DEP4: the first start exited on `Found 12 duplicate shape entries during merge of 'bf16_tuned_gemm'`, because the recipe's `$R/configs/dsv41_bf16_all_tuned.csv` repeats 6 shapes of aiter main's `model_configs/dsv41_bf16_tuned_gemm.csv`. The rerun drops that table, as the recipe says for duplicate tables, in a fresh container (aiter rewrites the source CSVs when it finds duplicates). The recipe's DEP4 command should drop it too.
- Process: `wait_infer_drain.sh` ran with `WAIT_CLIENT_PID` set to the lm_eval process. Its own argv holds `lm_eval.log`, which its client pattern matches (the false hang of 2026-11-14); with the PID every GSM8K drained with rc=0 as soon as lm_eval exited.

## Details

### Environment details

| Item | Value |
|---|---|
| Test image | `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261118` (digest `sha256:d17a28d3ea228c67c25fd1d15ae7138162287a7002ba13e2bae3ee33ed14cb69`) |
| ATOM | `main` @ [`9b2e67b7f6f5841b0d32db9589e989ff52673761`](https://github.com/ROCm/ATOM/commit/9b2e67b7f6f5841b0d32db9589e989ff52673761), 2026-11-18 fix(engine): honor ATOM_TORCH_PROFILER_DIR (#2925) |
| aiter | `main` @ [`b1d849ff36907402d7ea6a3483e6f0dafa995aca`](https://github.com/ROCm/aiter/commit/b1d849ff36907402d7ea6a3483e6f0dafa995aca), 2026-11-18 [Config] [Perf] Add DeepSeek-V4.1-Flash EP8 MXFP4 (a4w4) fused-MoE rows for gfx950 (#7582) |
| Test date | 2026-11-18 |

### Accuracy (GSM8K, 3-shot) · MLPerf Endpoint

| Model | flexible-extract | strict-match | 2026-11-14 | vs 2026-11-14 |
|---|---|---|---|---|
| DeepSeek R1 | 95.7% | 95.6% | 95.7% | 0.0 pts |
| DeepSeek-V4.1-Flash | 91.2% | 91.1% | — | first |

### Trends · MLPerf Endpoint

![MLPerf Endpoint GSM8K](../455-CI/runs/20261118/charts/accuracy-endpoint-zejun.png)

### Accuracy (GSM8K, 3-shot) · 8K/1K

| Model | flexible-extract | strict-match | — | vs — |
|---|---|---|---|---|
| DeepSeek-V4-Pro | 95.1% | 95.2% | — | first |

### Trends · 8K/1K

![8K/1K GSM8K](../455-CI/runs/20261118/charts/accuracy-8k1k-zejun.png)

### Failures

No failures.
