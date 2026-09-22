# Ulysses Sequence Parallelism — Handoff

Work-in-progress branch. Native ATOM only; no vLLM plugin path.

## Where this stands

Ulysses SP is implemented, accuracy-validated at SP4 on MiniMax-M3, and one
collective-level optimization has landed on top of it. **On M3 it is still a
net loss against TP4, and the reason is structural rather than a missing
optimization** — see "Why M3 loses" below. It is a win on dense models
(Llama-3.1-8B measured earlier in the same harness).

Nothing here is ready for upstream review. The open decision is described in
"Open decision" at the end.

## What the feature does

Ulysses trades sequence for heads around attention. Non-attention layers run on
`T/W` tokens with replicated weights; an all-to-all before attention swaps that
for all `T` tokens carrying `1/W` of the heads, and a second all-to-all swaps
back. SP rides the existing PCP process group (`get_sp_group()` returns
`get_pcp_group()`), so it composes with TP as an independent axis: attention
heads shard by `TP x SP` (`attn_head_shard_size()`).

Entry points, all in `atom/distributed/ulysses_sp.py`:

| Function | Role |
|---|---|
| `sp_split_tokens` / `sp_gather_tokens` | contiguous token shard in/out of the model |
| `ulysses_attention` | the q/k/v exchange, called from inside the attention custom op |
| `ulysses_gather_heads` | the reverse exchange on the attention output |
| `sp_moe_gather` / `sp_moe_reduce_scatter` | full sequence in/out of the MoE |

The split is contiguous rather than round-robin (unlike PCP): attention is
head-sharded and still sees every token, so contiguous chunks carry no
causal-mask imbalance and make the all-to-all a plain equal-split exchange.

## Measured results

MiniMax-M3 MXFP4, 4x MI300X. Prefill numbers are rank-0 GPU kernel time over
3 requests of 30720 tokens; `max_model_len=max_num_batched_tokens=32768`.

| | TP4 | SP4 | SP4 + QR reduce |
|---|---|---|---|
| GPU busy | 1557 ms | 1914 ms | **1818 ms** |
| collective | 473.0 | 881.2 | **775.0** |
| attention | 307.6 | 302.3 | 307.3 |
| moe expert | 244.4 | 237.4 | 239.6 |
| dense gemm | 182.8 | 156.4 | 157.6 |
| norm+rope | 139.8 | 79.6 | 80.2 |
| quant | 43.7 | 26.8 | 26.4 |
| reshape | 0.2 | 62.8 | 63.9 |

gsm8k, full dataset, 64 concurrent: SP4 **0.9431**, SP4+QR **0.9454**.

Decode and EP results from earlier sessions: SP4 decode GPU busy went 1033 ms ->
755 ms over the course of the kernel-level cleanups; SP4 is still ~1.74x TP4 at
decode. Expert parallelism was tried and is slightly worse than TP-style MoE
sharding under SP (decode 806.7 vs 754.9, prefill 1977 vs 1924) — EP gets better
GEMM shapes but falls off aiter's fused MXFP4 kernels onto generic ones.

## Why M3 loses (established, do not re-derive)

**1. Attention cannot benefit from Ulysses, by construction.** At a fixed GPU
count every parallel axis gives each rank the same 1/W of the attention FLOPs:
TP and Ulysses both give `T` queries x `H/W` heads; ATOM's native PCP gives
`T/W` queries x `H` heads; DCP gives `T` x `H` x `KV/W` plus an LSE merge. The
measured attention row above is flat across all three configs, as it must be.
An earlier note in this work attributed the flatness to M3's sparse O(T)
attention — that is wrong; it holds for dense attention too.

**2. The MoE communication is what sinks it.** M3 is 57 MoE layers out of 60.
Per request at 30k:

| | calls | us/call | ms/req |
|---|---|---|---|
| MoE reduce-scatter (before fix) | 57 | 1921 | 109.5 |
| MoE all-gather | 114 | 822 | 93.7 |
| attention all-to-all | 120 | 711 | 87.0 |

against TP4's 121 all-reduces at 1291 us = 156.2 ms/req covering everything.
SP's MoE gather + reduce-scatter alone exceeds TP's entire collective budget.
The replication savings SP does earn (norm+rope -60, dense gemm -26, quant -17)
are an order of magnitude smaller.

**3. aiter's fast paths are not symmetric across collectives.** This is the
single most useful thing to know for any further work here:

| collective | quantized kernel | registered-buffer kernel | fallback |
|---|---|---|---|
| `all_reduce` | **QuickAllReduce, 2 GB** | CustomAllreduce, 64 MiB | RCCL |
| `all_gather` | none | CustomAllreduce, 64 MiB | RCCL |
| `reduce_scatter` | none | CustomAllreduce, 64 MiB | RCCL |
| `all_to_all` | none | **none** | RCCL |

The servers set `AITER_QUICK_REDUCE_QUANTIZATION=INT4`, so TP's all-reduce is a
*transport codec*: the tensor is bf16 on both ends, but `CodecQ4` in
`csrc/include/quick_all_reduce.cuh` puts block-scaled INT4 on the wire. Ulysses
moves communication off the one primitive with a quantized kernel and onto the
one with no fast path at all. Attention illustrates it exactly — SP's two
all-to-alls move 208 MB against TP's 566 MB of algorithmic traffic, a genuine
2.7x reduction, but TP's 566 MB becomes ~142 MB after INT4 and SP's 208 MB
stays 208 MB, so SP ends up moving 1.5x *more* over the wire.

## What landed on top

`sp_moe_reduce_scatter` now routes large payloads through all-reduce + row
slice instead of reduce-scatter. Reduce-scatter has no quantized kernel and
exceeds CustomAllreduce's 64 MiB ceiling at prefill, so it was landing on RCCL
at 1921 us; the all-reduce over the same tensor hits QuickAllReduce at 1327 us
even though it moves the rows we then discard. Collective time 881 -> 775 ms,
GPU busy 1914 -> 1818 ms, accuracy unchanged. INT4 on this sum is safe for the
same reason it is safe under TP: it is the identical partial-sum reduction.

Decode is untouched — the gate only fires when CustomAllreduce declines the
payload, which at decode sizes it does not.

## Ruled out (measured, do not retry)

- **A second all-gather/all-to-all crossover at prefill scale.** The 2 MiB
  threshold in `_ALLGATHER_MAX_PAYLOAD_BYTES` was calibrated below 5 MiB, so the
  ~148 MiB prefill regime was re-measured: all-to-all wins by a widening margin
  (at 78 MiB, 536 us vs 1534 us) and the custom gather is unavailable past 64
  MiB anyway. The existing threshold is correct.
- **Head-sharding the qkv projection and all-gathering `x` instead.** Replaces a
  113 MB all-to-all with a 283 MB all-gather: 822 + 711 = 1533 us against the
  current 1422 us. All-to-all is the cheaper primitive here, which is the point
  of Ulysses.
- **Splitting the M3 fused kernel to quantize before the exchange ("route A").**
  Ceiling is 0.6% of prefill and it needs an aiter change. The exchange payload
  is 83% q (8192 of 9856 columns), and q is bf16 by the sparse attention
  kernel's contract — `q_out` is allocated `dtype=qkv.dtype` and aiter has only
  `pa_bf16_*` sparse kernels, no fp8-q variant. Only k/v/index_q/index_k (16.9%)
  have an fp8 form in the model's own pipeline. Worse, K and V never materialize
  as tensors on this path: `fused_qknorm_idxrqknorm` writes them straight into
  the paged cache with head-indexed addressing and per-(token,head) scales, so
  extracting them pre-exchange needs a new aiter entry point.

## Open decision

The only remaining lever on attention communication is a **transport codec on
the all-to-all** — quantize the payload before the exchange, dequantize after,
leaving every kernel on both sides untouched. This is the same technique
QuickAllReduce already applies to TP's all-reduce, so it is not a new class of
risk; it would run at fp8 against TP's INT4, on a tensor (qkv projection output,
immediately followed by RMSNorm) that is less sensitive than TP's (o_proj
partial sums feeding the residual stream).

The payload is conveniently 77 rows of 128 dims (64 q + 4 k + 4 v + 4 index_q +
1 index_k), so per-row fp8 scaling aligns with field boundaries and matches the
granularity the fused kernel already uses for its own K/V scales.

Estimated, not yet measured: 19712 -> 10164 bytes/token (51.6%), all-to-all
711 -> ~370 us, codec ~116 us, net ~27 ms/req or ~4.5% of prefill. Attention
comm would drop to ~974 us/layer, below TP's 1291 us for the first time.
aiter's regime enum also has FP8 and INT4, so if fp8 holds there is room to go
further.

The user was asked whether to do fp8 across the whole payload at once, stage it
(k/v/index first, then q), or microbenchmark first — **the question was not
answered before this handoff.** Ask before implementing.

## Reproducing

Harness lives outside the repo at `/app/test_scripts` (not committed):

- `run_server.sh <m3|llama8b|qwen30b> <tp4|sp4|sp4ep|tp4ep|tp1>` — launch
- `profile_prefill.sh m3 sp4 <tag>` — 30k prefill trace; needs
  `MAX_LEN=32768 MAX_BATCHED=32768 GPU_UTIL=0.8`
- `profile_decode.sh`, `accuracy.sh <tag> [limit]` — gsm8k at 64 concurrent
- `bench_sp_exchange.py` — all-to-all vs all-gather on the real exchange shape
- `bench_sp_moe_combine.py` — reduce-scatter vs all-reduce+slice
- `bench_qr_group.py` — whether QuickAllReduce is live on a given group

The benches need `AITER_QUICK_REDUCE_QUANTIZATION=INT4` in the environment or
QuickAllReduce stays disabled and the results invert — this cost a false
negative once.

Clear `~/.cache/atom` and `/root/.cache/{inductor,vllm,atom}` between runs, and
kill stray `openai_server` processes before each GPU experiment.

## Suggested skills

- `systematic-debugging` before proposing any fix from a profile
- `verification-before-completion` — every performance claim here is backed by a
  profile or a bench; keep it that way
- `requesting-code-review` before upstreaming, per the repo's review-then-push
  rule
