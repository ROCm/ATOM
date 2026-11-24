# Norm + RoPE + NVFP4 KV insert, head dim 128, rotary 64, 128-token pages

## What it does

The attention input of a MiniMax-M3.1 sparse attention layer with an NVFP4 KV cache, in one launch: per token the
Gemma RMS norm and partial NeoX RoPE of the q, k, index-q and index-k heads in fp32, q to bf16, index-q to e4m3, and
the NVFP4 inserts of K, V (not normed) and index-K. Input `qkv [n, (nq + 2 nkv + niq + 1) * 128]` bf16 (q, k, v,
index-q, index-k heads) or the QKV GEMM's fp32 split-K partials; heads q16 kv1 iq1, rotary 64 (TP4).
128-token pages: main `[blocks, 2 nkv, 128, 72]` u8, index `[blocks, 128, 72]` u8 (64 code + 8 e4m3 scale bytes per
128 values, V token-paired, free block stride); unwritten tokens keep their bytes. Contract: `reference.py`; bitwise
the HIP norm / RoPE + Triton insert chain. Needs `eps >= 2^-126`, index-q row stride % 4 == 0, padding at the
batch tail only (a -1 slot inside a run of mapped tokens in one V octet races: strict xfail).

## Entry points

- `insert_blocker`: why a layer (this rank's heads) does not fit every entry below (None: it fits); a caller keeps
  its stock path on a reason, and the entries assert it.
- `norm_rope_nvfp4_insert`: norm / RoPE / inserts from bf16 QKV rows into the caller's q and index-q outputs; slot -1
  skips a token; a prefill call takes the prefill kernel at large row counts.
- `qkv_norm_rope_nvfp4_insert`: the same, allocating q and index-q.
- `splitk_reduce_norm_rope_nvfp4_insert`: the same from the QKV GEMM's fp32 split-K slab (1-8 splits), summed
  `((0 + s0) + s1) + ...` and rounded once to bf16 as the split-K reduce stores it.
- `mxfp8_gemm_qkv_norm_rope_nvfp4_insert`: from the QKV GEMM's MXFP8 input: the one-launch entry
  `mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128` where it applies, else the GEMM and the insert (bitwise
  either way). A caller that captures each layer segment on its own (breakable PIECEWISE) turns the one launch off.
- `warmup_prefill_kernels`: none of these is a `warm_up` entry. Call it at init: graph capture reaches only the main
  kernel, so a >= 1537-token step would compile on the forward path. On a layer `insert_blocker` rejects it compiles
  nothing and returns the reason.

`host.py` picks the kernel (main, streaming q stores, prefill) and the token pairs per wave by row count.

## Performance

Graph replay, us per call; the chain is the QKV GEMM + this insert (layer 39 rank 0), the insert alone HBM-cold.

| Shape | Rows -> us |
|---|---|
| TP4 chain | 289 / 384 / 512 / 1024 / 1536 / 2048 / 4096 / 8192 / 16384 / 32768 -> 18.19 / 19.28 / 21.35 / 28.76 / 37.66 / 44.24 / 66.68 / 124.94 / 227.37 / 415.28 |
| Insert alone | 1 / 4 / 64 / 172 / 256 -> 2.21 / 2.42 / 2.75 / 3.03 / 3.31; 32768 -> 61.76 |

## Tests

```bash
python3 -m pytest tests/attention/norm_rope_nvfp4_insert_d128_rot64_p128 -q
python3 benchmarks/attention/norm_rope_nvfp4_insert_d128_rot64_p128/benchmark_norm_rope_insert.py  # also --reduce
python3 benchmarks/gemm/mxfp8_n2560_k6144/benchmark_qkv_insert_stores.py --checkpoint <ckpt> --baseline <baseline package>
```
