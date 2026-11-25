# Target runs on the owner perf images (2026-11-25, LLM-Booster#743)

Each run takes the model's 455-CI recipe as-is (server, bench) on the image its owner uses for performance work, so the gap to main comes only from what that image carries beyond main. No GSM8K. Machine e15-07 (4 x MI455X). Times are Beijing time.

| Model | Image | Scenario | Result | Used in `perf_target.csv` |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash | `rocm/fw-bringup-mlperf:gfx1250-atom-dsv41-pd-20261119` | MLPerf Endpoint, TP4, c4 | 227.9 tok/s/GPU | yes, c4 = 228 |
| DeepSeek-V4-Pro | same | 8K/1K, dp4ep4, c4 / c64 | 243.9 / 6,898.7 tok/s/GPU | no |
| DeepSeek R1 | `rocm/fw-bringup-mlperf:gfx1250-atom-20261123-dsr1` | MLPerf Endpoint, dp4ep4 | failed at the first requests | no |

The images are the ones Jiaoliang Yu named for the DeepSeek models (2026-11-24): R1 on the dsr1 image, V4.1-Flash and V4-Pro on the dsv41-pd image.

## DeepSeek-V4.1-Flash, c4

16:24-16:45. The recipe's TP4 server ([server_cmd_tp4.sh](deepseek-v4.1-flash/server_cmd_tp4.sh)) with aiter#8267's `DEFAULT.json` ([`../../20261121/aiter8267-check/DEFAULT.json`](../../20261121/aiter8267-check/DEFAULT.json)) mounted at `/app/aiter/aiter/ops/triton/configs/gfx1250/triton/fusions/fused_mixed_sample/DEFAULT.json`: this image samples MTP verification rows with `fused_mixed_sample` and has no gfx1250 config for it, as main on 2026-11-21.

- `tps` 911.76 over a 599.8 s window, 1141/1141 samples, 0 failed ([result_summary.json](deepseek-v4.1-flash/c4/performance/result_summary.json)).
- tok/s/GPU = 911.76 / 4 = 227.9 (guide §4.1). E2E avg interactivity 541.7 tok/s/user.
- main on 2026-11-14: 211.74 tok/s/GPU, 92.9% of it. The 2026-11-21 point failed.

The c64 target (3,043) is not from this run: it is the owner's daily report of 2026-11-20 (same image, 1.01x Vera Rubin).

## DeepSeek-V4-Pro, c4 and c64

17:41-17:59. The recipe as-is ([server_cmd.sh](deepseek-v4-pro/server_cmd.sh)); the weights are a copy of e15-06's `/mnt/data/xiaobizh/models/DeepSeek-V4-Pro` (275 of its 276 files, byte sizes equal; the one left out is an unreadable `.cache/huggingface/trees/*.json`).

| Concurrency | total_token_throughput | tok/s/GPU | ITL P50 (ms) | main 2026-11-21 | main / this |
|---|---|---|---|---|---|
| 4 | 975.56 | 243.9 | 31.72 | 267.51, ITL P50 30.63 | 109.7% |
| 64 | 27,594.62 | 6,898.7 | 13.21 | 6,879.81, ITL P50 13.69 | 99.7% |

[c4](deepseek-v4-pro/c4/result.json), [c64](deepseek-v4-pro/c64/result.json); 40/40 and 640/640 requests completed. Not used as a target: what the dsv41-pd image adds to main (ATOM#2877, #3144; aiter#7332, #7668) is for V4.1-Flash, so V4-Pro on it is at main's level. The V4-Pro PRs on the list (ATOM#3753, aiter#8505) are not in this image.

## DeepSeek R1

16:13-16:24. The recipe as-is ([server_cmd.sh](deepseek-r1/server_cmd.sh)). The server came up in 321 s; at the first c4 requests (16:19) ModelRunner0 aborted on an LLVM assertion in a JIT compile, right after aiter skipped the OPUS row of the LM head shape (1, 129280, 7168): `PHINode::setIncomingValue: All operands to PHI node must be the same type as the PHI node!` ([error.log](deepseek-r1/error.log)).

The image's own R1 config, `/lab-mlperf-inference/code/deepseek-r1/offline_mi455x_atom_team_test_perf.yaml`, is MLPerf Offline with greedy sampling (temperature 0) and turns on its optimizations by switches the CI recipe does not set (`ATOM_MLA_NVFP4`, `ATOM_DS_QKV_A_NORM_FP4`, `ATOM_MLA_FUSED_KUP_QPE`, `ATOM_DP_MAIN_ARGMAX`, the MoE tables under `/opt/dsr1_best_20261117`). The endpoint bench samples at temperature 1.0. An R1 endpoint target needs an endpoint recipe from the owner.
