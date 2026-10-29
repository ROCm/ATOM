# DeepSeek-V4.1-Flash

The native text backbone and the paged ModelRunner/Scheduler lifecycle are
usable. The KV cache is FP8 (`--kv_cache_dtype fp8`, MXFP8 rows attended by
AITER's `mla_sparse_prefill_v41`); on the BF16 or FP4 pool the original V4
BF16 attention and inverse RoPE kernels are reused.
Chat, tool and reasoning-effort protocol integration is complete, and native
image requests support chunked prefill and request-owned embedding lifetime.

The reference revision is `dba1be0a40aa45a94ad051997016db3960a90277`, AITER is
pinned to `2039d2b96cd547ebc52f8d55f5f29ec1b8290796`, and Engram builds on
ROCm/ATOM PR #2185. The primary deployment scope is TP4 with whole-expert EP
on MI355X GPUs. For measured TP2/TP4 no-EP, level-3 FULL AgentX performance,
see the [AgentX benchmark recipe](DeepSeek-V4.1-Flash-Agentic.md), using
fixed acceptance length 3.51. For MI455X, see [gfx1250 (MI455X)](#gfx1250-mi455x).

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
| FP8 (MXFP8) KV cache (`--kv_cache_dtype fp8`; BF16 and FP4 also accepted) | Complete on the original path and in mono; 1.84x the rows of the BF16 pool, GSM8K within the BF16 baseline's spread |
| FP8 (default) or FP4 (`--index_cache_dtype fp4`) index plane, paged scorer, whole-forward target graph | Complete; a decode step is the draft's replay plus the target's, and prefill scores in the plane too |
| DSpark speculation and accepted-prefix state commit | Fixed-width lifecycle supported at level 0; [quality and throughput acceptance remain incomplete](../docs/deepseek_v41_dspark.md#current-limitations) |
| Candidate-only indexing, Engram residency and fusion | Partial: Engram UVA/overlap and fusions exist. Visibility-sized scratch, candidate-only Reindex and HBM residency remain |
| Fused decode (mono) for DSpark verify steps of up to 12 rows and their draft backbone | On by default (`ATOM_MONO_ENABLE`) but currently refused: its MoE kernel still runs MXFP4 activations; see [Fused decode](#fused-decode-mono) |
| Decoder SWA bounded replay (the decoder on each prompt's last window rows) | On by default (`--no-decoder-swa-bounded-replay`); see [the runtime guide](../docs/deepseek_v41_runtime.md#decoder-swa-bounded-replay) |
| Encoder SWA bounded replay (no state checkpoints; a hit rebuilds one PAGE) | On by default (`--no-encoder-swa-bounded-replay`); see [the runtime guide](../docs/deepseek_v41_runtime.md#encoder-swa-bounded-replay) |
| P/D disaggregation (Mooncake) | 1P1D TP2+TP2 validated with DSpark (gsm8k and AgentX c32); MoRI-IO not yet admitted. See [the PD guide](pd_disaggregation_guide.md#deepseek-v41-flash) |
| TP4 with routed experts sharded across TP instead of whole-expert EP | Supported; relative throughput is workload-dependent |
| TP2 no EP | GPU benchmark completed at level 3 FULL; see the [AgentX recipe](DeepSeek-V4.1-Flash-Agentic.md) for commands and measured points |
| Other TP sizes | Configuration tests cover TP1/2/4/8 and dimension divisibility; this does not establish GPU or quality validation for TP1/8. TP8 does not start: AITER's FP8 blockscale GEMM runs no K = 288 (the shared expert's down projection) |

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
`wo_a` keeps its FP8 weight in AITER's batched MXFP8 GEMM, its input quantized
by the fused inverse RoPE. Routed experts reuse V4/FusedMoE's configured kernels,
with numerical behavior described in the cache and graph guide. They
run under whole-expert EP or shard across TP, whichever the launch selects.
V4.1 initializes the same distributed environment as every other model and has no
collective policy of its own.

## gfx1250 (MI455X)

V4.1 runs on gfx1250 at TP4 (`--level 0`, BF16 KV, FP8 index), with or without
DSpark, and at DEP4 (EP + DP attention). On gfx1250 its FP8 linears, the grouped
`wo_a` and the FP8 index scoring run on AITER's FlyDSL kernels:

- FP8 linears and `wo_a`: the batched mxfp8_32 GEMM, which reads the 32x32
  weight scale as stored. The 32x32 linears, including DSpark's `main_proj`
  and Engram's `wkv`, are held preshuffled on gfx1250 as on gfx950. A K that
  128 does not divide (the shared expert's down projection at TP4, K = 576)
  runs as a ragged last k-tile.
- FP8 index scoring: `flydsl_fp8_paged_mqa_logits`, over the same packed,
  preshuffled 8-row-page cache.

What differs from gfx950:

- **Mono fused decode is gfx950-only** (wave64, `permlane32`). On gfx1250 the
  deployment check refuses it and logs `V4.1 mono decode off: arch gfx1250`, so
  `ATOM_MONO_ENABLE` needs no setting. DSpark still runs: its draft passes and
  verify steps take the original, unfused path.
- **The router GEMV** (bf16 MFMA) is gfx950-only; gfx1250 uses the regular gate.
- **FP4 index interface (hardware validation pending).** With
  `--index-cache-dtype fp4`, gfx1250 uses natural query/cache rows with AITER's
  H32 OPUS MQA API, including full-layer and candidate tables and packed logits.
  Requires AITER [#7128](https://github.com/ROCm/aiter/pull/7128) and
  [#6952](https://github.com/ROCm/aiter/pull/6952). gfx1250 correctness, model E2E
  and performance remain unvalidated. gfx950 keeps the FlyDSL FP4 scorer.
- **`--level 0`.** A first `--level 3` start can stall while ranks compile and
  autotune at different times.

TP4 (add `--method dspark --num-speculative-tokens 5` for DSpark):

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 ATOM_DISABLE_MMAP=true \
AITER_REUSE_IDENTICAL_COMM_GROUPS=1 HF_HUB_OFFLINE=1 \
python3 -m atom.entrypoints.openai_server \
  --model /models/deepseek-ai/DeepSeek-V4.1-Flash --trust-remote-code \
  --server-port 8000 -tp 4 \
  --kv_cache_dtype bf16 --index-cache-dtype fp8 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --block-size 16 --level 0 --max-num-seqs 64 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,12,16,24,32,40,48,56,64]'
```

DEP4 (EP + DP attention, MegaMoE over mori v2):

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 ATOM_DISABLE_MMAP=true \
AITER_REUSE_IDENTICAL_COMM_GROUPS=1 HF_HUB_OFFLINE=1 \
MORI_GPU_ARCHS=gfx1250 FLYDSL_GPU_ARCH=gfx1250 \
ATOM_MORI_V2=1 MEGA_DISPATCH_WIRE=fp4 ATOM_MEGA_COMBINE_WIRE=fp4 \
ATOM_MOE_GU_ITLV=1 ATOM_DP_LM_HEAD_MODE=allgather MORI_SHMEM_HEAP_SIZE=16G \
python3 -m atom.entrypoints.openai_server \
  --model /models/deepseek-ai/DeepSeek-V4.1-Flash --trust-remote-code \
  --server-port 8000 -tp 4 --enable-expert-parallel --enable-dp-attention \
  --kv_cache_dtype bf16 --index-cache-dtype fp8 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --block-size 16 --level 0 --max-num-seqs 512 \
  --cudagraph-capture-sizes '[1,2,4,8,16,32,48,64,96,128,192,256,320,384,448,512]'
```

`--max-num-seqs` counts per DP rank. If the image's mori loads only
`libamd_smi.so.<N>`, point `MORI_AMD_SMI_PATH` at it.

Validation on MI455X, GSM8K 3-shot on the full test set (`lm_eval
local-completions`, `num_concurrent=32`):

| Launch | GSM8K |
|---|---|
| TP4 | 0.926 |
| TP4 + DSpark (5 speculative tokens) | 0.926; acceptance about 50% on these prompts |
| DEP4 | 0.916 |

The DEP4 gap to TP4 has not been investigated; its fp4 dispatch / combine wire
is the first suspect.

## Fused decode (mono)

A DSpark verify step of up to 12 rows (two requests' verifies; it is routed by
its rows alone) runs as two persistent kernels a layer
(`atom/models/deepseek_v41/mono/`): K1 (mHC seam, attention norm, q / kv
projections, the indexer query) and K2, which runs K2a (the indexer's scores
and top-k at any context length, the sparse decode attention, the output
projections and their all-reduce, the FFN seam) and then K2b (the MoE and its
all-reduce) in one launch. Each step width has its own kernel build. The
compressor and Engram stay on the original modules. The DSpark draft's block
backbone of up to two requests (its three window-only stages, 5 rows each)
runs on the same kernels, after a prefill as after a decode; its LM head and Markov sampler are unchanged. It is on by
default; `ATOM_MONO_ENABLE=0` turns it off. The deployment it serves and the
conditions it checks before taking over are listed under `ATOM_MONO_ENABLE` in
the [environment variable guide](../docs/environment_variables.md); any other
deployment, and every other target step (prefill, more rows), keeps the
original path.

The routed experts now take MXFP8 activations on interleaved gate/up rows
(the checkpoint reference's precision; MXFP4 activations cost ~1.4pp of
GSM8K at TP4). K2b still reads separated rows with MXFP4 activations, so
mono refuses every layer until K2b follows, and decode runs the original path.

Against the original path on the same inputs (`ATOM_MONO_CHECK=1` with
`--enforce-eager`), every target layer's outputs and the indexer's selection
are bit-identical against the original run 6 rows at a time: the original
decode attention picks its splits from a call's rows, while the mono attention
keeps a 6-row call's order, so a row's result never depends on the other rows
of its step. The draft's attention sums its keys in another order than the
original draft's Triton kernel, so the draft is not; its acceptance is
unchanged. Measured on MI355X, TP4, `--level 0` with FULL graphs (2026-09-29;
a step is the draft's pass plus the target's verify, GSM8K run at the step's
request count):

| Requests a step | Step time, mono (graph trace) | Step time, original | GSM8K 3-shot, mono / original | Acceptance, mono / original |
|---|---|---|---|---|
| 1 | 3.7 ms | 10.9 ms | 0.9219 / - | 50.3% / - |
| 2 | 5.1-6.3 ms | 11.3 ms | 0.9158 / 0.9265 | 50.2% / 50.2% |

The kernels are built for the TP size of the deployment (TP2, TP4 or TP8;
`kernels/dims.py` holds every per-rank width). TP4's kernels are unchanged
instruction for instruction. At TP2 (2026-09-30, MI355X, `--level 0`,
concurrency 2) the check mode holds as at TP4, and GSM8K 3-shot on the first
200 questions is 0.940 / 0.945 (mono / original) with acceptance 49.5% /
49.2%. At TP8 the kernels compile and fit, but the server does not start on
the original path: the shared expert's down projection has K = 288 there, and
AITER's gfx950 FP8 blockscale GEMM needs K to be a multiple of its K tile.

## Equal-score index selection

Equal scores go to the smaller position, for index top-k, candidate-block ties
and Reindex alike. The published implementation's `torch.topk` defines no tie
rule at all; ATOM needs one because every tensor-parallel rank has to select
the same KV set. The newest visible candidate block remains mandatory, and
selected indices are returned in ascending position order.
