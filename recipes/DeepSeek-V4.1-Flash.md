# DeepSeek-V4.1-Flash

The native text backbone and the paged ModelRunner/Scheduler lifecycle are
usable. The original V4 BF16 attention and inverse RoPE kernels are reused.
Chat, tool and reasoning-effort protocol integration is complete, and native
image requests support chunked prefill and request-owned embedding lifetime.

The reference revision is `dba1be0a40aa45a94ad051997016db3960a90277`, AITER is
pinned to `2039d2b96cd547ebc52f8d55f5f29ec1b8290796`, and Engram builds on
ROCm/ATOM PR #2185. The primary deployment scope is TP4 with whole-expert EP
on MI355X GPUs.

| Capability | Status |
|---|---|
| Nested configuration, CSA2 topology, format schema | Complete |
| Native FP8/FP4 weights and kernel interfaces | Complete |
| Single-Pass mHC, MoE arithmetic, Engram math and history | Complete |
| Full-layer text execution | Implemented |
| Paging, batching and request-state lifecycle | Complete |
| Chat, tools and reasoning-effort protocol | Complete |
| Vision and image requests | Implemented; [visual quality remains unstable](../docs/deepseek_v41_vision.md#current-limitations) in eager and FULL graph modes |
| Multimodal chunking and embedding lifetime | Supported; output can vary with chunk size |
| Packed cache, native quantized experts and graph execution | Implemented; see [memory and numerical limits](../docs/deepseek_v41_performance.md#memory-and-numerical-limits) |
| FP8 index plane, paged scorer, whole-forward target graph | Complete; a decode step is the draft's replay plus the target's, and prefill scores in the plane too |
| DSpark speculation and accepted-prefix state commit | Fixed-width lifecycle supported at level 0; [quality and throughput acceptance remain incomplete](../docs/deepseek_v41_dspark.md#current-limitations) |
| Candidate-only indexing, Engram residency and fusion | Partial: Engram UVA/overlap and fusions exist. Visibility-sized scratch, candidate-only Reindex and HBM residency remain |
| Optional CED decoder replay | Not started |
| Optional encoder replay with a persistent global cache | Not started |
| TP4 with routed experts sharded across TP instead of whole-expert EP | Supported; relative throughput is workload-dependent |
| Distributed and deployment combinations beyond TP4 | Not validated |

The full-layer path remains the numerical and performance baseline. CED and
bounded replay are approximate modes with separate quality gates and cache
provenance. Model math, cache ownership, Engram preparation and graph execution
have separate implementations.

## Run and validate

Development and acceptance currently use `--level 0` (or
`CompilationConfig(level=0)`). Add `--enforce-eager` when isolating a failure;
level 0 also supports the existing FULL decode CUDA graphs. Level-3 compiler
correctness is deferred and is not part of the current acceptance claim.

See the [runtime guide](../docs/deepseek_v41_runtime.md) for paged execution and
supported configuration. The [protocol guide](../docs/deepseek_v41_protocol.md)
covers numeric reasoning effort, tool calls and multi-turn history. The runtime
validates unsupported combinations before loading weights. Arithmetic is judged
against the published model through the `ATOM_DSV41_REFERENCE` unit tests, and
quality end to end through `lm_eval`.

For a bounded real-checkpoint TP4 scheduler regression:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 \
AITER_REUSE_IDENTICAL_COMM_GROUPS=1 OMP_NUM_THREADS=4 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
torchrun --standalone --nproc_per_node=4 \
  -m tests.attentions.deepseek_v41.validate_runtime \
  --model /mnt/DeepSeek-V4.1-Flash --cache-dtype fp4 --graph \
  --output /tmp/v41-runtime.json
```

Omit `--graph` and use `--cache-dtype bf16` for the uncaptured BF16 baseline.
[Cache format and graph execution](../docs/deepseek_v41_performance.md)
describes what is packed and what is captured.

Weights remain native FP8 32x32 or FP4 group32. Dense projections use native
FP8 microscaling MFMA (`tl.dot_scaled`) with FP32 accumulation; the grouped
`wo_a` path remains BF16. Routed experts reuse V4/FusedMoE's configured kernels,
with numerical behavior described in the cache and graph guide. They
run under whole-expert EP or shard across TP, whichever the launch selects.
V4.1 initializes the same distributed environment as every other model and has no
collective policy of its own.

## Equal-score index selection

Equal scores go to the smaller position, for index top-k, candidate-block ties
and Reindex alike. The published implementation's `torch.topk` defines no tie
rule at all; ATOM needs one because every tensor-parallel rank has to select
the same KV set. The newest visible candidate block remains mandatory, and
selected indices are returned in ascending position order.
