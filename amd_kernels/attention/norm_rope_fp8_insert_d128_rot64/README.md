# Gemma norm + partial RoPE + fp8 KV insert, head dim 128, rotary 64

## What it does

Turns a projection's bf16 output into an attention layer's inputs in one launch: per-head Gemma RMS norm, NeoX RoPE
over dims 0-63 of 128 (halves of 32), and the e4m3 K / V insert into the cache pages (V is not normed or rotated).
Caller: the MiniMax-M3.1 DSpark draft (5 layers): the query block's fused qkv row and the context rows' fused K / V
row, which feeds all 5 layers' caches. Cache `[pages, kv_heads, page, 256]` e4m3 (128 K bytes, then 128 V bytes per
token), e4m3 of `value / scale` saturated at +-448, per-tensor fp32 scales; `cos_sin_cache [rows, 64]` bf16 (cos |
sin). Served: TP4 16 q / 1 KV heads, page 128. Contract: `reference.py`. The qkv norm takes `rstd` and the products
in fp64 (as Inductor compiles the Triton norm); the multi-layer norm stays in fp32 (the eager Triton norm), so the two
K paths can differ by one bf16 ulp. Every output is within one bf16 ulp / e4m3 step of the reference or `ROW_ATOL`
(2^-6) of its row maximum, at least `EXACT_MIN` (99.9%) bit-equal; cache bytes outside the written slots stay
unchanged.

## Entry points

- `qkv_norm_rope_insert.qkv_norm_rope_insert`: q / k norm + RoPE of one layer's fused bf16 qkv rows into the caller's
  bf16 q, and the K / V insert of the tokens that have slots (slot < 0 skips); without a cache it writes q only (the
  profile run).
- `qkv_norm_rope_insert.supports_qkv_norm_rope_insert`: whether the inputs fit (gfx950, bf16, the cache layout).
- `rope_blocker`: why a layer's RoPE does not fit these kernels (None: it fits).
- `kv_norm_rope_insert_multilayer.kv_norm_rope_insert_multilayer`: K norm + RoPE and the K / V inserts of the context
  rows' fused K / V row into every layer's cache, from bf16 rows or the context GEMM's deferred split-K partials
  (`gemm.mxfp8_common.SkPartials`) summed in the reduce's order (`0 + slab 0 + slab 1 + ...`, one bf16 rounding): the
  bytes of reduce-then-insert, one launch fewer.
- `kv_norm_rope_insert_multilayer.supports_multilayer`: whether it takes these caches and scales.

Both are entries (`amd_kernels.warm_up`): the first call of each mode (heads, insert or not, page size; layers, KV
heads, page size, split) compiles its kernel with a slot -1 call that writes nothing. The caller lists no warm-up. No
row-count dispatch: rows, positions and slots are runtime arguments.

## Performance

One MI355X; one graph of up to 100 launches with their own inputs, median of 5 x 10 replays, us per call.
Bitwise the Triton / C++ chain of the same ops.

| Kernel | Heads | Page | Rows -> us |
|---|---|---|---|
| qkv | 16 q / 1 KV | 128 | 4 / 64 / 256 / 1024 -> 2.00 / 2.65 / 3.37 / 7.42 |
| multi-layer, 5 layers | 1 KV | 128 | 4 / 384 / 4096 -> 1.95 / 2.91 / 12.93 |

## Tests

```bash
python3 -m pytest tests/attention/norm_rope_fp8_insert_d128_rot64 -q
python3 benchmarks/attention/norm_rope_fp8_insert_d128_rot64/benchmark_norm_rope_insert.py
```
