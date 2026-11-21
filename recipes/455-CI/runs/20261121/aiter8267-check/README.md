# aiter#8267 check (2026-11-21, 20:10-20:45 BJ)

Not a full bench: each server took MLPerf endpoint bench traffic (temperature 1.0) until requests went through MTP verification sampling, then it was stopped.

- Image `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261121`, with [`DEFAULT.json`](DEFAULT.json) from https://github.com/ROCm/aiter/pull/8267 (head `df49d27c`) mounted at `/app/aiter/aiter/ops/triton/configs/gfx1250/triton/fusions/fused_mixed_sample/DEFAULT.json`.
- Servers: the recipe commands. DEP4 leaves `$R/configs/dsv41_bf16_all_tuned.csv` out of `AITER_CONFIG_GEMM_BF16` ([command](deepseek-v4.1-flash_dep4_server_cmd.sh)).

| Model | Server | Requests | Errors | MTP draft tokens (accepted) |
|---|---|---|---|---|
| Qwen-35B-A3B | TP1, c4 | 174 responses 200 OK | 0 | 82,998 (56.2%) |
| DeepSeek-V4.1-Flash | TP4, c4 | warmup 464/464 successful | 0 | 326,885 (56.3%) |
| DeepSeek-V4.1-Flash | DEP4, c64 | server up in 601 s; 295 requests, 30 s of decode on 4 DP engines | 0 | 96,215 (43.9%), includes server warmup |

Logs: [e15-06](log_06.txt), [e15-07](log_07.txt).
