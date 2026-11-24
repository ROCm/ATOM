# QKV MXFP8 GEMM + per-head norm / RoPE + NVFP4 page inserts, N 2560, K 6144, rotary 64, pages 128

## What it does

The QKV chain of a MiniMax-M3.1-Flash TP4 sparse attention layer in one entry: the MXFP8 QKV GEMM (N 2560 = 16 q + k +
v + index-q + index-k heads of 128, K 6144), then per head the Gemma RMS norm and partial NeoX RoPE (rotary 64), bf16
q `[m, 2048]`, e4m3 index-q `[m, 128]` (row stride divisible by 4), and the NVFP4 K / V / index-K inserts into
`[blocks, 2, 128, 72]` main and `[blocks, 128, 72]` index pages (bytes of skipped and unwritten tokens stay). Input:
the producer's e4m3 codes in the (16, 16) tile (`prequant_code_tile_rows(m, 2560, 6144)` rows), shuffled E8M0 scales
`[ceil(rows / 32) * 32, 192]` and the `preshuffle_mxfp8_weight` weight; heads 16 / 1 / 1, `eps >= 2^-126`. Bytes:
bit for bit the deferred chain at every row count (`gemm.mxfp8_common.mxfp8_gemm_shuffled(defer_reduce=True)` +
`splitk_reduce_norm_rope_nvfp4_insert`, or the bf16 GEMM +
[`norm_rope_nvfp4_insert`](../norm_rope_nvfp4_insert_d128_rot64_p128/README.md)). Contract: `reference.py` (the GEMM
reference, then the insert's); index-q holds `INDEX_Q_REL_L2` (5e-3): its e4m3 turns a moved bf16 QKV bit into a code
flip. Usual caller: `mxfp8_gemm_qkv_norm_rope_nvfp4_insert` of the insert package.

## Entry points

- `mxfp8_qkv_norm_rope_nvfp4_insert`: the whole chain into the caller's q / index-q outputs and pages (slot -1
  skips); M 0 is a no-op, rows outside the shuffled-code tile's 1-1536 raise `ValueError`. An `amd_kernels.warm_up`
  entry: the first call compiles rows 1-1536 and must be eager (capture raises).

Counters: one set per eager stream or per (capture ID, stream), from a pool of 512 sets (3.75 MiB) per device; a
capture keeps its set for the process lifetime, so graphs captured on one stream may replay concurrently. `host.py`
picks the kernel by row count.

## Performance

Graph replay, us per call, layer 39 rank 0, 26 cold weight copies, 100 inputs, the entry vs the deferred chain
in one process, one MI355X. Each row is its own run.

| Range | Rows -> entry vs deferred chain (us) |
|---|---|
| 8-256 | 8 / 16 / 32 / 64 / 128 / 256 -> 5.879 / 6.337 / 7.155 / 8.676 / 11.147 / 14.365 vs 7.657 / 7.910 / 8.558 / 9.942 / 12.083 / 15.343; 145 -> 11.838 |
| 65-96 (two launches) | 65 / 80 / 96 -> 9.425 / 9.702 / 10.107 vs 10.508 / 10.885 / 11.229 |
| 193-288 (two launches) | 193 / 224 / 288 -> 14.205 / 14.416 / 14.968 vs 15.162 / 15.310 / 16.016 |
| 512-1536 (prefill) | 512 / 1024 / 1536 -> 20.629 / 27.450 / 36.503 vs 20.626 / 27.459 / 36.511 (no gain) |

## Tests

```bash
python3 -m pytest tests/attention/mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128 -q
python3 benchmarks/attention/mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128/benchmark_chain.py \
    --checkpoint <M3.1-Flash checkpoint> --out /tmp/qkv_chain
```
