# DeepSeek-V4.1-Flash

The native text backbone and paged ModelRunner/Scheduler lifecycle are usable
on the user-accepted P04 arithmetic baseline. The original V4 BF16 attention
and inverse RoPE kernels are reused. Chat/tool/reasoning protocol integration is
complete, and native image requests support chunked prefill and request-owned
embedding lifetime. See the [DSpark guide](../docs/deepseek_v41_dspark.md) for speculative execution
and its open zero-shot raw quality limitation.

The reference revision is `dba1be0a40aa45a94ad051997016db3960a90277`.
Real-checkpoint acceptance uses `ljin_dev`, four MI355X GPUs, TP4 and whole-expert
EP, with weights at `/mnt/DeepSeek-V4.1-Flash`. AITER is pinned to
`2039d2b96cd547ebc52f8d55f5f29ec1b8290796`. Engram builds on ROCm/ATOM PR #2185.

| Stage | Deliverable | Status |
|---|---|---|
| P00 | Pinned checkpoint, numerical oracle and input fixtures | Complete |
| P01 | Nested configuration, CSA2 topology and format schema | Complete |
| P02 | Native FP8/FP4 weights and kernel interfaces | Complete |
| P03 | Single-Pass mHC, MoE arithmetic and Engram math/history | Complete |
| P04 | Full-layer text baseline | Accepted; the eager comparison path has since been removed |
| P05 | Paging, batching and request-state lifecycle | Complete |
| P06 | Chat, tools and reasoning-effort protocol | Complete |
| P07 | Vision and image requests | Complete |
| P08 | Multimodal chunking and embedding lifetime | Complete; small-chunk quality tradeoff accepted |
| P09 | Packed cache, AITER W4A8 experts and PIECEWISE graphs | Complete; MoE precision tradeoff accepted |
| P10 | V4.1 DSpark and accepted-prefix state commit | Implemented and tested; zero-shot raw quality decision pending |
| P10b | FP8 index plane, paged scorer and the whole-forward target graph | Implemented; a decode step is the draft's replay plus the target's, and prefill scores in the plane too |
| P11 | Candidate-only indexing, Engram residency and fusion | Pending |
| P12 | Optional CED decoder replay | Pending |
| P13 | Optional encoder replay and persistent global cache | Pending |
| P14 | Validated distributed/deployment combinations | Pending |

The full-layer path remains the numerical and performance baseline. CED and
bounded replay are approximate modes with separate quality gates and cache
provenance. Model math, cache ownership, Engram preparation and graph execution
have separate implementations.

## Run and validate

See the [runtime guide](../docs/deepseek_v41_runtime.md) for paged execution and
supported configuration. The [protocol guide](../docs/deepseek_v41_protocol.md)
covers numeric reasoning effort, tool calls and multi-turn history. The runtime
validates unsupported combinations before loading weights. Arithmetic is judged
against the published model through the `ATOM_DSV41_REFERENCE` unit tests, and
quality end to end through `lm_eval`.

For a bounded real-checkpoint TP4 regression in the development container:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=/tmp/atom-dsv41-flash:/tmp/aiter-dsv41-2039d2b96c \
AITER_META_DIR=/tmp/aiter-dsv41-2039d2b96c-meta \
AITER_JIT_DIR=/tmp/aiter-dsv41-2039d2b96c-jit \
AITER_REUSE_IDENTICAL_COMM_GROUPS=1 OMP_NUM_THREADS=4 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
torchrun --standalone --nproc_per_node=4 \
  -m tests.attentions.deepseek_v41.validate_runtime \
  --model /mnt/DeepSeek-V4.1-Flash --cache-dtype fp4 \
  --graph --expert-backend aiter --output /tmp/v41-runtime.json
```

Omit `--graph` and `--expert-backend aiter` and use `--cache-dtype bf16` for
the P05 runtime baseline. The [P09 report](../docs/deepseek_v41_performance.md)
describes what is packed/captured and the performance comparison.

Weights remain native FP8 32x32 or FP4 group32. Dense projections and eager
experts use BF16 MFMA for group32 dot products and FP64 scaled accumulation.
The optional AITER MoE backend uses existing native A8W4 GEMMs with the
user-accepted numerical change documented in the P09 report. A native FP8 MFMA
replacement has not passed the precision gate; packed storage does not imply
that this replacement has been enabled. Routed experts use whole-expert EP;
shared experts reduce FP32 partials before rounding. Production collectives
use RCCL.

## Equal-score index selection

Equal scores go to the smaller position, for index top-k, candidate-block ties
and Reindex alike. The published implementation's `torch.topk` defines no tie
rule at all; ATOM needs one because every tensor-parallel rank has to select
the same KV set. The newest visible candidate block remains mandatory, and
selected indices are returned in ascending position order.
