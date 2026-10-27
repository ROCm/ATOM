# Qwen3.8-Flash-Next Usage Guide

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is a
multimodal MoE model combining Gated DeltaNet (GDN), sparse attention (QSA),
and n-gram memory. ATOM supports text and image inference with BF16 or FP8
weights, BF16 activations, and a BF16 or FP8 KV cache.

## Preparing the environment

Follow the [ATOM installation instructions](../README.md). Use the
dependencies pinned by ATOM, including `transformers==5.16.1`, which provides
the model configuration and multimodal processor.

## Serving FP8 on one GPU

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --trust-remote-code \
  -tp 1
```

The API listens on port 8000 by default. Use
`Qwen/Qwen3.8-Flash-Next-FP8` as the `model` in API requests.

The KV cache is BF16 by default. The default block size already satisfies
`indexer_compress_ratio` (4); no block-size override is needed.

## FP8 KV cache

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --trust-remote-code \
  -tp 1 \
  --kv_cache_dtype fp8 --index-cache-dtype bf16 --block-size 64
```

`--kv_cache_dtype fp8` stores the QSA K/V as E4M3 with one FP32 scale per
token and KV head, which fits about 1.65x the tokens of a BF16 cache. The
compressed index keys follow the KV dtype unless `--index-cache-dtype` is
given: `bf16`, `fp8` (one FP32 scale a row) or `fp4` (MXFP4). The raw index
keys stay BF16. With `--block-size 64` and a BF16 index cache, prefill index
scoring runs on the FlyDSL kernel, which takes BF16 keys only; this is the
recommended combination. FP4 index keys work but score slower than FP8 or
BF16 on MI355X. The MTP draft layer keeps BF16 caches unless
`ATOM_QSA_DRAFT_KV_BF16=0`.

Choose a GPU with enough memory for the weights, recurrent state, and KV
cache. Adjust `--max-num-seqs`, `--max-model-len`, and
`--max-num-batched-tokens` for the available memory and workload.
In particular, `--max-num-seqs` defaults to 512; lowering it reduces the
preallocated recurrent-state memory. These are resource controls, not
model-specific requirements.

## Native MTP

Use the draft weights included in the same checkpoint:

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --trust-remote-code \
  -tp 1 \
  --method mtp --num-speculative-tokens 2 \
  --no-enable_prefix_caching
```

Draft depths 1, 2, and 3 are supported. The checkpoint contains one draft
layer, reused at each step; no separate draft model is needed. The target
and draft share the embedding and output head.

## Usage notes and limitations

- For image requests, use `--no-enable_prefix_caching` and sufficient cache
  capacity to avoid preemption until the image caching/recompute limitations
  are resolved. Image prompts must fit within `--max-num-batched-tokens`.
- Preserve the checkpoint's quantization exclusions and `ple_embedding_dtype`
  when present. GDN inputs support unquantized weights or per-channel FP8 QKV/Z
  with unquantized B/A. GDN input online quantization is not supported.
- PTPC (`compressed-tensors`) checkpoint loading is supported, but end-to-end
  execution on gfx950 is blocked by the default small-batch MoE dispatch:
  its second-stage kernel requires a 256-aligned intermediate size, whereas
  this model uses 640. The FP8 example above uses block-wise quantization.
- Chunked text prefill and full decode CUDA graphs are supported. Piecewise
  compilation/CUDA graphs are not supported.
- Only native MTP speculation is supported. Pipeline parallelism, context
  parallelism, DP attention, TBO, and external KV transfer/offload are not
  supported.
- This example covers FP8 with TP1. Pure TP2 is not currently validated;
  it can fail MoE warmup on gfx950 due to an MoE dispatch limitation.
- Video inference and long-context accuracy have not been validated.

## Mono decode

On MI355X (`gfx950`), the optional Mono path supports the official FP8
checkpoint with BF16 activations and KV cache, TP1 or TP8 with expert
parallelism at TP8. Both TP1 and TP8 serve non-speculative decode batches
of one, two or four independent requests. Mono is enabled by default for
qualifying deployments through the shared `ATOM_MONO_ENABLE` switch. Set
`ATOM_MONO_ENABLE=0` to use the original model throughout.

```bash
ATOM_MONO_ENABLE=1 \
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-Flash-Next-FP8 --trust-remote-code \
  -tp 8 --enable-expert-parallel \
  --max-model-len 8192 --max-num-seqs 4 \
  --kv-cache-dtype bf16 --no-enable_prefix_caching \
  --cudagraph-mode FULL --cudagraph-capture-sizes '[1,2,4]'
```

For TP1, replace `-tp 8 --enable-expert-parallel` with `-tp 1`.
Prefill and unsupported decode widths use the original model. Ordinary batches
use weight-reusing GDN projections and FP8 MoE; TP8 owns experts by rank. Each routed
G1 output tile runs independently. Shared output projection and routed
intermediate quantization share one launch; routed/shared merging preserves
the BF16 endpoints and expert order. B4 uses MFMA HC, and TP1 B2 plus both B4
layouts fuse MLP HC up with the MoE front. TP8 B2 keeps separate HC and front
launches. Batch GDN combines recurrence and norm without a global core buffer.
QSA retains native projections, selection/index-cache preparation and sparse
attention. Batch Q/K norm, mRoPE and K/V publication share one kernel; TP1
uses the native attention reduction and output gate, while TP8 stages the gate
into output projection LDS. Collectives retain their native implementation.
MTP and plugin serving are outside this Mono deployment contract.
Use prefix caching disabled for the validated configuration. Mono sizes its
QSA scores and candidate lists from `--max-model-len`, within
the checkpoint's context capacity. TP1 uses a fixed-LDS exact selection tree
when that capacity exceeds 8,192 tokens; 8,192 remains a selection segment,
not a deployment limit. Resource admission still applies to each build.
TP8 retains native sparse attention when the metadata context bound exceeds
256 tokens. Native FULL capture sets this bound to the deployment capacity,
so its captured QSA seam remains safe across short and long request contexts.

For single-request decode, TP1 uses one shared GDN launcher for ordinary and PLE layers and two-launch
QSA. FULL graphs use the native batch/query-length keys and the deployment
context capacity; no context-dependent capture policy is added to the engine. TP8 uses rank-local attention, in-kernel reductions and expert-owned
MoE. For QSA contexts above 256 tokens, TP8 keeps native sparse attention and
its attention all-reduce between Mono projection and finish kernels.

The integration follows ATOM's architecture-based model registration. A decode
wrapper installs after the native model is loaded; the GPU runner is imported
only on the first eligible step. One runner owns model bindings and the layer
loop; `execution.py` shares the GDN/QSA layer
lifecycle, PLE handling and diagnostics. `tp1.py` and `tp8.py` own their build
sets and the TP8 pull endpoint. `stages/` contains reusable GDN, QSA, HC and MoE
computations; `kernels/` composes those stages into complete launches and owns
TP-specific mailbox and collective schedules. Both QSA K1 variants share
QKV writeout, norm/mRoPE and index-cache math. QSA output projections share
contiguous-K accumulation while keeping their ready-group waits and peer
publication schedules. Both K2 variants consume the same
paged selection/attention stages. GDN shares convolution and the AITER-derived
recurrence views, norm and MFMA projection math while retaining its different
TP1/TP8 output schedules. TP1 scheduling lives in `kernels/gdn_tp1.py`;
typed field groups live in `kernels/arguments.py`, and `abi.py` validates
and flattens complete launch contracts into the original scalar GPU ABI. Raw views and cache loads live in `stages/memory.py`; task
geometry and scratch regions live in the device-independent `layout.py`.
QSA K1/K2 share composition factories, and GDN K2 shares the TP8 finish
factory. S1 and batch launchers share static weight bindings, while native
metadata and state views bind per step. Immutable layer records retain module
and weight ownership. `MonoState` owns persistent S1 buffers; per-width builds
own named batch workspaces. `WidthBuilds` compiles each build once, checks private
scratch and the Qwen resource contract, then performs TP agreement and caching.
It remembers a refusal for that width only.
`PERSISTENT_ROWS=(1,)` and `BATCH_ROWS=(2,4)` distinguish the two schedules;
both serve ordinary one-token-per-request decode.
Kernel dimensions live in `shapes.py`/`dims.py`; deployment checks live in
`config.py`. Zero-copy weight contracts live in `weights.py`, graph-stable
buffers in `state.py`, and per-step native metadata views in `step_meta.py`.
`ATOM_MONO_CHECK=1` enables eager comparisons with the original layers for
FP8/BF16 S1 diagnosis. It does not cover B2/B4 or Quark. QSA checks run the
reference attention on its own cache values and restore the Mono cache before
Mono K2; use checks separately from performance runs.

Validate the actual Mono route before reporting a speedup: a build refusal
falls back to the native model. Task accuracy uses the full GSM8K protocol
below and the corresponding model's dashboard `exact_match,flexible-extract`
threshold (currently 0.96 for Flash-Next). Retain sample outputs and truncation
records; paired lost-question counts are diagnostic and are not a fixed
acceptance allowance.

## Quark MXFP4 Mono decode

The Quark checkpoint `amd/Qwen3.8-Flash-Next-Quark-MXFP4` supports the
same ordinary-decode widths at TP1 and TP8+EP8, with FP8 KV and a BF16
index cache. Expert weights stay packed MXFP4 with E8M0 scales. Other
weights retain the checkpoint's quantization exclusions, and layer
activations remain BF16. This is not an all-linear W4A8 conversion.

The Quark MoE path uses AITER's BF16-output G1 and separate native MXFP4
intermediate quantization. It requires no local AITER source patch. The
dependency revision used for this integration is `0.1.1.dev1+g508736740`.

```bash
ATOM_MONO_ENABLE=1 AITER_LOG_LEVEL=WARNING \
python -m atom.entrypoints.openai_server \
  --model amd/Qwen3.8-Flash-Next-Quark-MXFP4 --trust-remote-code \
  -tp 8 --enable-expert-parallel \
  --kv-cache-dtype fp8 --index-cache-dtype bf16 --block-size 64 \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --no-enable_prefix_caching \
  --cudagraph-mode FULL --cudagraph-capture-sizes '[1,2,4]'
```

For TP1, replace `-tp 8 --enable-expert-parallel` with `-tp 1`.
Verify that the log reports `Qwen Quark seam kernels built for width`
for each served width. A missing dependency contract or unsupported
deployment produces a refusal and runs the native model instead.

The path fuses HC/router, route sorting and MXFP4 input preparation, and
QSA norm/RoPE/FP8 cache publication. TP8 also combines QSA projections,
stable sparse selection and index expansion. Native AITER expert GEMMs,
sparse attention and tensor collectives remain part of the execution.
MTP, other checkpoints, and other widths are outside this validation.

The CPU unit test checks deployment/step admission, native cache views,
workspace bounds and build setup without GPU kernel dependencies:

```bash
python -m pytest tests/test_qwen4_exp_mono.py
```

Performance comparisons must use the same checkpoint, dependencies and
service parameters on both paths, using stock AITER. Disable
Mono with `ATOM_MONO_ENABLE=0` for the native baseline. Repeat both paths
and retain full GSM8K results and actual per-rank route evidence.

## GSM8K evaluation

Install the evaluation client:

```bash
python -m pip install 'lm-eval[api]'
```

With the server running, evaluate all 1,319 test questions using 5-shot chat:

```bash
lm_eval --model local-chat-completions \
  --model_args 'model=Qwen/Qwen3.8-Flash-Next-FP8,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=64,max_retries=3,timeout=900,tokenized_requests=False,trust_remote_code=True' \
  --tasks gsm8k --num_fewshot 5 \
  --apply_chat_template --fewshot_as_multiturn \
  --gen_kwargs 'max_gen_toks=4096,until=<|im_end|>' \
  --log_samples --output_path ./results/gsm8k
```

The model emits reasoning before its final answer, so allow enough output
tokens to avoid truncation. The explicit stop string replaces GSM8K's default
`Question:` stop, which can prematurely end a reasoning response.
The harness scores the final `content`, not `reasoning_content`.
The timeout accommodates long requests under load; it does not change the
generation limit or scoring. Set evaluation concurrency to suit the server's
capacity. For Mono acceptance, match `num_concurrent` and `--max-num-seqs`
to the intended C1, C2 or C4 configuration, use `temperature=0` and the fixed
4096-token budget, and record actual Mono routing alongside the score.
GSM8K tests task accuracy; long-context attention requests require separate
functional validation.
