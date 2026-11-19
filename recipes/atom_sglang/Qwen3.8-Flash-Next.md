# Qwen3.8-Flash-Next on SGLang-ATOM

Flash uses `qwen4_exp` / `Qwen4ExpForConditionalGeneration`. Target and MTP
compute run through Native ATOM `Qwen4Exp` and `Qwen4ExpMTP`; the plugin
adapts SGLang scheduling, QSA KV pools, PLE state and HC hidden storage.
It requires the Native Flash implementation in this tree. The original
TP2/EP2 example uses SGLang 0.5.17, with Flash config registration supplied
by the recognition patch. The gfx942 pure-TP MTP configuration below uses
SGLang 0.5.22.

`Qwen4ExpForConditionalGeneration` is 48 layers: 36 Gated-DeltaNet and 12 QSA
(query-sparse, 2048-token budget) attention layers, 4-stream
hyper-connections, 512 routed experts (top-10) plus a sigmoid-gated shared
expert, and n-gram PLE on layer 1. Weights: per-channel FP8 (PTPC), KV cache
BF16.

## Text serving with MTP

The following configuration is tested on MI308X with TP2/EP2 and the
Qwen3.8-Flash-Next-PTPC-FP8 checkpoint. Set `MODEL_PATH` to the local checkpoint.
Target verification uses CUDA graphs and overlap scheduling. Draft graphs
remain disabled because Native Flash MTP uses mRoPE.

```bash
export MODEL_PATH=/models/Qwen3.8-Flash-Next-PTPC-FP8
export CUDA_VISIBLE_DEVICES=0,1
export HIP_VISIBLE_DEVICES=0,1
export SGLANG_PLUGINS=atom_sglang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=atom.plugin.sglang.models
export SGLANG_USE_AITER=1
export SGLANG_AITER_KV_CACHE_LAYOUT=nhd
export AITER_MOE_PADDING_SIZE=128

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" --trust-remote-code \
  --host 127.0.0.1 --port 8000 \
  --tp 2 --ep-size 2 --attention-backend aiter \
  --kv-cache-dtype bf16 --page-size 64 \
  --context-length 8192 --max-running-requests 8 \
  --mem-fraction-static 0.70 --disable-radix-cache \
  --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 8 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 2 --speculative-eagle-topk 1 \
  --sampling-defaults openai
```

SGLang and ATOM must both be installed or available on `PYTHONPATH`.
The target and draft share the same checkpoint, including its `mtp.*` weights.
Flash MTP supports one QSA draft layer and top-k 1; separate draft checkpoints
and other speculative algorithms are rejected. For eager comparison, replace
the two CUDA graph flags with `--disable-cuda-graph` and add
`--disable-overlap-schedule`. To serve without MTP, omit the speculative flags.

A deterministic smoke request:

```bash
curl http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The result of 17 + 25 is", "sampling_params":{"temperature":0,"top_k":1,"top_p":1,"max_new_tokens":64}}'
```

## MI308X pure TP2 / TP4 with BF16 MTP

For the `Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16` checkpoint on gfx942, use
`--ep-size 1`. ATOM keeps the logical expert shard at 320 (TP2) or 160
(TP4), and stores it at 320 or 192 respectively. Both the FP8 target and
BF16 MTP fuse the shared expert into slot 512, so AITER sees 513 experts
and top-k 11. The target and draft retain their checkpoint dtypes.

The following settings use SGLang 0.5.22. Set `AITER_MODEL_CONFIGS` to the
`aiter/configs/model_configs` directory in the AITER installation:

```bash
export MODEL_PATH=/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16
export CUDA_VISIBLE_DEVICES=0,1
export HIP_VISIBLE_DEVICES=0,1
export GPU_ARCHS=gfx942
export SGLANG_PLUGINS=atom_sglang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=atom.plugin.sglang.models
export SGLANG_USE_AITER=1
export AITER_BYPASS_TUNE_CONFIG=0
export AITER_ONLINE_TUNE=0

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" --trust-remote-code \
  --host 127.0.0.1 --port 8000 --tp-size 2 --ep-size 1 \
  --attention-backend aiter --kv-cache-dtype bf16 --page-size 64 \
  --context-length 32768 --max-running-requests 8 \
  --mem-fraction-static 0.95 --disable-radix-cache \
  --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 8 \
  --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 80 \
  --mamba-ssm-dtype float32 \
  --speculative-algorithm EAGLE --speculative-num-steps 2 \
  --speculative-num-draft-tokens 3 --speculative-eagle-topk 1 \
  --speculative-use-rejection-sampling
```

For TP4, expose four GPUs in both visibility variables and use
`--tp-size 4 --ep-size 1`. TP4 expert padding adds about 5.64 GiB per rank
for the 48 FP8 target layers, plus about 240 MiB for the BF16 draft layer;
include that storage in the memory budget.

Load both MoE tables before importing AITER. They contain FlyDSL, ASM and
CK entries, so a successful dispatch means the selected row's actual
kernel runs. Check that the logs show `gfx942`, 80 CU, physical intermediate
320/192, 513 experts and top-k 11, with FP8/per-token for target and BF16/no
quantization for draft. A `ck kernel not found` warning is a failed tuned
dispatch even if inference continues with a fallback.

CK instances must also have been **generated for gfx942**. A mixed-arch
prebuild can embed gfx942 code objects while its Python code generator chose
the gfx950 instance list. In that case, setting `GPU_ARCHS` at runtime does
not replace the installed module: rebuild the affected BF16 and FP8 CK MoE
modules with `GPU_ARCHS=gfx942` and use a separate `AITER_JIT_DIR` to keep
the original installation intact. Do not change the CSV kernel names to
match the incorrect prebuild.

ATOM applies this weight alignment in its model/method layer;
`AITER_MOE_PADDING_SIZE` alone does not enable it. Graph token padding is
handled separately from the expert intermediate padding. For correctness
comparisons, keep the speculative horizon, sampling parameters, SSM dtype,
cache settings and prompt template the same across runs.

## Long-context high-throughput serving (no MTP)

For throughput on a two-GPU host, trading the MTP draft for a much larger
context and KV pool:

```bash
export MODEL_PATH=/models/Qwen3.8-Flash-Next-PTPC-FP8
export SGLANG_PLUGINS=atom_sglang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 --port 30080 \
  --tp-size 2 --ep-size 1 \
  --kv-cache-dtype bf16 \
  --page-size 64 \
  --context-length 132096 \
  --max-running-requests 64 \
  --mem-fraction-static 0.88 \
  --disable-radix-cache \
  --trust-remote-code
```

Notes:

* `--mem-fraction-static`: SGLang multiplies it by 0.85 whenever the
  attention backend is `aiter` and the context exceeds 8k, to leave room for
  AITER attention workspaces. The QSA layers run ATOM's own kernels, so that
  room is never used; 0.88 lands at an effective 0.75 and nearly doubles the
  KV cache (measured on `rocm/atom-dev:sglang-latest`: 1.91M tokens at 0.75,
  3.57M at 0.88), which lets 128k-token requests run about twice as many at a
  time.
* `--context-length` must cover prompt + output (129024 + 2048 here).

The configuration above is the one the MI308X-specific pieces below were
validated with.

## MI308X-specific pieces in ATOM

* AITER tuned GEMM tables for this model's FP8 (a8w8 bpreshuffle) and BF16
  shapes ship with AITER (`aiter/configs/model_configs/qwen38_flash_next_*`,
  [ROCm/aiter#5974](https://github.com/ROCm/aiter/pull/5974)); older AITER
  builds run these shapes on default kernels.
* The SGLang plugin advertises the native GPU arch to AITER. The images export
  `GPU_ARCH_LIST=gfx942;gfx950`; forwarded unchanged, AITER's `get_gfx()`
  reported gfx950 on MI308X and every gfx-keyed tuned table missed.

## Fused paths (no switches)

The fused kernels are the only path the served model takes; they are not
gated behind environment variables:

* hyper-connections: `process_weights_after_loading` builds the combined
  `[down | inject]` weight, and the decoder layer defers each sub-layer's
  combine into the next mix.
* shared expert: routed through the fused MoE as expert 512 (top-k + 1),
  scoring with sigmoid.
* batch-1 MoE: gfx942 PTPC uses the AITER tuned table; other supported
  layouts retain the routing/expert Triton specialization.
* QSA indexer scoring: matrix cores for prefill, CUDA cores for decode-sized
  batches.

The MTP drafter's single layer keeps separate HC `mix`/`combine` operations.
On gfx942 its BF16 shared expert is fused into the MoE; other MTP layouts
retain the standalone shared expert.

## Optional: quantized prefill all-reduce (lossy)

Prefill all-reduces of 16k-token chunks (84 MB at TP2) take about 1.8 ms on
RCCL. AITER's quick all-reduce with INT6 codes halves that
(`AITER_QUICK_REDUCE_QUANTIZATION=INT6`); it only engages above its minimum
message size, so decode all-reduces stay exact. It perturbs prefill
activations (max ~3% of a tensor's range per reduction), so validate accuracy
on the target workload before enabling it.

## Benchmark

```bash
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30080 \
  --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" --dataset-name random \
  --random-input-len 4096 --random-output-len 2048 --random-range-ratio 1.0 \
  --num-prompts 64 --max-concurrency 32 --request-rate inf
```

## Boundaries

- Use BF16 KV and page size 64. Pure TP2/TP4 with BF16 MTP uses the gfx942
  configuration above. Other platforms, expert quantization formats and TP
  sizes are outside that configuration's validation scope.
- Leave PLE embeddings in Native ATOM; do not enable PLE embedding offload.
  The memory fraction above leaves room for plugin-owned QSA indexer caches.
- Validation covers text requests, concurrent mixed lengths, long prompts,
  slot reuse, and eager fallback above the largest captured batch followed by
  graph replay. It does not establish
  multimodal MTP, radix caching, performance gains, or correctness at other
  parallel sizes and speculative horizons.
- On a SGLang upgrade, review recognition and MTP EntryClass mapping separately.
  Upstream Flash recognition does not replace Native ATOM compute or the
  plugin's graph padding and post-draft metadata refresh contracts.
