# DeepSeek-V4.1 cache format and graph execution

V4.1 reuses ATOM's V4 attention, MoE and graph execution infrastructure.
Model arithmetic, cache representation and execution policy have separate
owners. See [the runtime guide](deepseek_v41_runtime.md) for configuration
admission and [the model recipe](../recipes/DeepSeek-V4.1-Flash.md) for launch
commands.

## Shared V4 primitives

Target and draft use `atom.model_ops.layernorm.RMSNorm` for attention and FFN
inputs, Q/KV, index keys and compressor output. Forward RoPE uses AITER's
`rope_cached_positions_fwd_inplace`; the V4.1 adapter supplies its positions,
rotary lanes and YaRN cache. Inverse RoPE is fused into the `wo_a` input quant,
as on V4. V4.1 does not apply
V4's weightless per-head Q normalization.

Dense projections use native FP8 microscaling with FP32 accumulation. On
gfx950 their 32x32 FP8 weights are (16, 16)-shuffled at load for AITER's
preshuffled group32 GEMM, a FlyDSL kernel with its own tuned table
(`ATOM_GROUP32_WEIGHT_PRESHUFFLE=0` keeps checkpoint bytes). The
grouped `wo_a` output projection keeps its FP8 weight and 32x32 e8m0 scale:
AITER's `inverse_rope_group_quant` un-rotates the attention output against
the BF16 RoPE table and quantizes it to MXFP8 (1x32) in one launch, and
`batched_gemm_a8w8_mxscale_bpreshuffle` runs the grouped GEMM. Delayed mHC
uses AITER stages.

On AITER builds without the native group32 interface, the compatibility FP8
GEMM keeps two pipeline stages, including its four-way packed-K layout.
Triton 3.7's async LDS load can lose the neutral scale for masked K/N tails,
including the TP2 shared-expert down projection at K=1152. The packed kernel
loads weight scales from bounded addresses and explicitly selects E8M0 code
127 for padding, preserving the scale value without disabling the pipeline.
This handling also applies inside FULL graph execution.

V4/FusedMoE owns expert activation formats, routing-weight placement, GEMM
dispatch, shared-expert overlap and expert-parallel exchange. V4.1 supplies
its image routing bias through the shared `mm_topk` operator; see
[image routing](deepseek_v41_vision.md#routing-and-engram).

## Request metadata

Request metadata reuses V4's persistent `CpuGpuBuffer` storage, token-layout
helpers and state-slot publisher. V4.1 consumes `v4_meta_state_slot_out`.
Pool-slot conversion belongs to the cache geometry.

Actual request and token counts are `scheduled_bs` / `scheduled_tokens`;
execution capacities are `running_bs` / `running_tokens`. Block-table stride
and buffer addresses remain fixed across steps. Padding tokens carry V4's
`-1` request-ID sentinel; padding requests have zero query length.

## Cache format and attention boundary

The main pool accepts `kv_cache_dtype="bf16"` or `"fp4"`. The index plane is
independent: `index_cache_dtype="fp8"` (default) or `"fp4"`.

| Region | BF16 main-pool configuration | FP4 main-pool configuration |
|---|---|---|
| Main KV, 512 dimensions | BF16 | E2M1 FP4 with E4M3 scales, group 16 |
| SWA, 512 dimensions | BF16 | E4M3 FP8 with E8M0 scales, group 32 |
| Index keys, 128 dimensions | E4M3 FP8 with a per-key power-of-two scale | Same FP8 index format |

The index writer stores scales as FP32 in preshuffled tiles. Global owner
regions share block IDs while keeping separate index planes. FP32 compressor
tails and committed Engram histories retain their precision and checkpoint
semantics. `V41PoolGeometry` declares allocation sizes, alignment and image
layout identity; memory planning and checkpoint copies use those declarations.

`packed_rows.py` owns packed row encoding and decoding.
`packed_attention.py` decodes selected history into bounded BF16 scratch and
passes ordinary int32 row indices to V4 attention. Prefill keeps the current
chunk in BF16; decode preserves V4's batch size and split-K dispatch. Neither
path materializes the entire historical main KV pool.

On the BF16 pool, decode attention reads two sources
(`sparse_attn_v4_paged_2src`): the selection from the pool, and the window
through the layer's own ring (`ring_view`), whose rows are one list shared by
every layer. An index group's layers share one selection plane. The OPUS kernel
splits each token's keys across blocks when the step has too few tokens to
fill the GPU.

Packed PAGE writes reuse V4's sentinel-aware `swa_scatter_rows`. A zero-copy
view preserves gaps between PAGE fields, and compression-plan offsets select
the destination rows. Padding rows are skipped on the device, including during
graph replay.

## Graph ownership

Use `CompilationConfig(level=0, cudagraph_mode=CUDAGraphMode.FULL)` with
`enforce_eager=False` for whole-forward decode capture. The existing runner
owns one target graph per `(batch size, query bucket)`. Non-speculative decode
replays one target graph; DSpark additionally replays its request-count-keyed
draft graph. Prefill and host Engram lookup stay outside these graphs.

Capture and replay use the forward's declared execution capacity. Compression
plans have fixed capacity and sentinel-filled tails. Host cursor publication
occurs during input preparation, since replay does not execute Python.
`BatchStep.begin_forward` clears derived per-forward tensors before capture.

Startup capture uses the serving allocation and block 0 for synthetic
requests. The runtime validator checks that capture preserves live cache pages.
`PIECEWISE` is also accepted by the runtime; development and validation use
level 0, and broader compiler support remains outside the supported scope.

## Memory and numerical limits

The FP4 index plane packs the scorer's FP32 logits: each row as long as its own
visibility, aligned, in one buffer sized from the PAGE pool the memory budget
can hold rather than from the model length. A step whose rows overflow that
buffer -- requests sharing a prefix the pool stores once -- runs eagerly,
scored in bands, instead of from a decode graph. The FP8 plane
keeps a plane band at the int32 addressing limit, which can require nearly
8 GiB, plus its tile tables; reserve scratch headroom for it when sizing
long-context workloads. Quantized queries and selection workspace occupy memory
beside either.

A Reindex layer scores `candidate_topk_blocks` blocks rather than the whole
context: the index plane is paged at `candidate_block_size`, so the candidate
source's list is the scorer's block table. Compaction clamps the last kept
block's visible span to one block, so its context stays within the actual
kept count even if selection omits the newest block. Candidate ranking still
pins the newest block independently.

Cache packing and expert activation quantization affect numerical behavior.
Native FP8/FP4 execution does not promise bitwise equality with an all-BF16
implementation or the mathematical reference. Validate quality for the chosen
cache format, batch shape and workload. Image quality limitations are described
in [the vision guide](deepseek_v41_vision.md#current-limitations); speculative
execution has [separate constraints](deepseek_v41_dspark.md#current-limitations).

For throughput and latency, use the standard benchmark in
[the serving guide](serving_benchmarking_guide.md) with matched request lengths,
concurrency, cache configuration and graph mode. Include warmup and repeated,
alternating runs. Cache capacity savings and isolated kernel timings do not
establish end-to-end serving speedups.
