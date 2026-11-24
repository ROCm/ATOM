# Sliding-window decode attention, fp8 KV, 16 query heads per KV head, head dim 128

## What it does

Decode attention of a sliding-window layer over an e4m3 KV cache (per-tensor K / V scales), 1-8 query rows per
request. Caller: the MiniMax-M3.1 DSpark draft (5 sliding layers, window 4096): TP4 16 q / 1 KV heads, 128-token
pages, 3 or 7 rows per request, causal or not (the upstream draft is non-causal), up to 256 requests.
`q [rows, 16 Hkv, 128]` bf16; cache `[pages, Hkv, page, 256]` e4m3, 128 K then 128 V bytes per token, page a multiple
of 32. A request's rows are the last positions of its sequence; row p attends `p - window < k < seq_len` (causal:
`k <= p`); requests with `seq_len < rows` (graph padding) write nothing. Contract:
`reference.py`: exact bf16 K / V, bf16 Q and P, fp32 softmax and sums, bf16 output, splits combined in split order
(deterministic, but the split count follows the request count); the MXFP8 output equals the o_proj quantizer's bytes.

## Entry points

- `swa_decode_fp8kv`: the attention of a step's draft query blocks (int32 metadata) into bf16, or into o_proj's MXFP8
  input (row-major or 16-row tiles, e8m0 scales AITER-shuffled). Not a `warm_up` entry: launch each split count once
  before capture (72 MiB scratch).
- `supports_swa_decode_fp8kv`: whether a call fits (gfx950, <= 8 rows, `window <= max_window(rows)`).
- `layer_blocker`: why a layer does not fit at 8 rows (None: it fits).
- `swa_decode_fp8kv_oproj_input`: the attention as its o_proj's input: MXFP8 codes and scales in the o_proj table
  row's layout (`oproj_codes`) from 8 rows of 3- or 7-row blocks, else the bf16 output. An `amd_kernels.warm_up`
  entry: the first call of a mode (rows, window, causal, heads, page, o_proj N) compiles 1-256 requests.

`host.py` picks the kernel and the splits (`plan_splits`: one round of CUs, a request's splits on one XCD);
`oproj_input.mxfp8_out` picks the output.

## Performance

Graph replay of 100 launches with their own queries, block tables and lengths, K / V cold (full windows), us per call.

| Case | Requests -> us |
|---|---|
| TP4 7 rows non-causal, bf16 | 1 / 2 / 4 / 8 / 11 / 16 / 24 / 32 / 48 / 64 -> 9.84 / 10.17 / 11.83 / 15.71 / 19.69 / 23.22 / 30.64 / 33.44 / 39.63 / 47.15 (the request-wide `kernel.py`: 34.32 / 34.70 / 35.33 / 35.65 / 36.67 / 37.28 / 40.35 / 41.70 / 43.54 / 47.18). vs the nightly vLLM Triton draft attention (40 GB pool, 8K-200K contexts): 1 / 8 / 16 / 32 -> 10.3 / 15.7 / 23.1 / 33.2 vs 171.7 / 180.2 / 181.8 / 181.9 |
| TP4 7 rows causal, bf16 | 1 / 2 / 4 / 16 / 32 / 64 -> 9.83 / 10.22 / 11.85 / 23.35 / 33.65 / 46.78 |
| TP4 3 rows non-causal | 1 / 4 / 16 / 32 / 48 / 64 -> bf16 15.90 / 16.80 / 17.54 / 20.22 / 21.91 / 24.25, MXFP8 16.30 / 17.20 / 17.93 / 20.59 / 22.22 / 24.66 |
| TP4 4 rows causal, bf16 | 1 / 16 / 64 / 128 / 256 -> 20.55 / 22.83 / 28.86 / 39.75 / 65.70 |

## Tests

```bash
python3 -m pytest tests/attention/swa_decode_fp8kv_gqa16_d128 -q
python3 benchmarks/attention/swa_decode_fp8kv_gqa16_d128/benchmark_decode.py --case tp4_dspark7_noncausal_p128_bf16
```
