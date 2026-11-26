# Qwen3.8-Flash-Next on SGLang-ATOM

Flash uses `qwen4_exp` / `Qwen4ExpForConditionalGeneration`. Target and MTP
compute run through Native ATOM `Qwen4Exp` and `Qwen4ExpMTP`; the plugin
adapts SGLang scheduling, QSA KV pools, PLE state and HC hidden storage.
It requires the Native Flash implementation in this tree and SGLang 0.5.20 or
newer (CI and the release image use v0.5.22); upstream recognizes
`qwen4_exp`, and the recognition patch adds only the MTP draft-arch and
HC-width adapters.

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
* batch-1 MoE: routing and experts in two Triton kernels.
* QSA indexer scoring: matrix cores for prefill, CUDA cores for decode-sized
  batches.

The MTP drafter's single layer runs unfused (separate `mix`/`combine`) and
keeps a standalone shared expert.

## Optional: quantized prefill all-reduce (lossy)

Prefill all-reduces of 16k-token chunks (84 MB at TP2) take about 1.8 ms on
RCCL. AITER's quick all-reduce with INT6 codes halves that
(`AITER_QUICK_REDUCE_QUANTIZATION=INT6`); it only engages above its minimum
message size, so decode all-reduces stay exact. It perturbs prefill
activations (max ~3% of a tensor's range per reduction), so validate accuracy
on the target workload before enabling it.

## Optional: sequence-parallel collectives over SDMA

TP prefills above 1024 tokens keep the HC residual token-sharded
(`ATOM_QWEN4_EXP_SP_MIN_TOKENS`); their reduce-scatters and all-gathers block
the compute stream on RCCL. `ATOM_QWEN4_EXP_SP_SDMA=1` runs them on the copy
engines (mori CCO SDMA) instead, overlapped with the hyper-connections around
them (MI308X, 12k-token prefill: TP2 -12% step time, TP4/EP4 -7%). The
transport is bf16 (no quantization).

```bash
# mori ships in the ATOM images; elsewhere add it from the ATOM checkout
# (amd-mori-nightly >= 1.2.6.dev20261116, CCO SDMA with MI308X engine balancing):
pip install -e ".[qwen38-tp-sdma]"
export ATOM_QWEN4_EXP_SP_SDMA=1
# Only when torch loads ROCm's private libdrm (rocm_sysdeps), as TheRock wheels do:
export LD_PRELOAD=/opt/rocm/lib/rocm_sysdeps/lib/librocm_sysdeps_drm.so.2:/opt/rocm/lib/rocm_sysdeps/lib/librocm_sysdeps_drm_amdgpu.so.1
```

It applies to TP serving on gfx942 only (validated on MI308X) and needs mori
(above), FlyDSL and aiter's `SdmaComm` (ROCm/aiter#6514). Each rank holds
(4 x TP - 2) x ceil(max_num_batched_tokens / TP) rows x hidden x 2 B of receive
buffers (16k tokens: ~252 MB at TP2, ~294 MB at TP4). If any TP rank cannot use
it (a dependency or the preload missing, TP across nodes), every rank logs one
warning and keeps RCCL.

## Optional: bursty prefill on an idle server

When several requests reach an idle server at once and their prefill takes
more than one batch, overlap scheduling launches the next prefill batch before
it processes the results of the previous one. The first batch's requests then
get their first token only after the next batch has also run. To process each
prefill batch's results before the next prefill batch launches, add one
variable to the MTP command above and keep all of its flags:

```bash
export SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=1
```

* It only changes back-to-back prefill (extend) batches. Decode and
  target-verify batches keep overlap scheduling. Unlike
  `--disable-overlap-schedule`, decode stays asynchronous.
* It helps only when prefill batches run back to back, for example a burst
  that does not fit in one batch. A short request that arrives alone sees no
  change.
* SGLang notes that this setting may slightly hurt throughput. Every pair of
  back-to-back prefill batches loses overlap, including the chunks of one
  long prompt. Measure steady-load throughput on the target workload before
  enabling it.
* The recv-skipper settings (`--scheduler-recv-interval` with
  `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_*`) can also gather a burst into one
  batch. They are not recommended here: the window counts scheduler
  iterations rather than time, so its length depends on the host and the load.

Measured with the MTP command above on 2x MI308X (gfx942), ROCm 7.2.4, SGLang
v0.5.22. The default was run twice, before and after the switch, to show
run-to-run drift. Burst TTFT in ms, median / mean / P99, for N requests sent
together to an idle server:

| Prompt | N | Default | Default, repeat | Switch on |
|---|---|---|---|---|
| 2048 | 1 | 233 / 238 / 269 | 229 / 240 / 293 | 228 / 230 / 242 |
| 2048 | 2 | 453 / 455 / 464 | 477 / 474 / 505 | 344 / 343 / 460 |
| 2048 | 4 | 585 / 585 / 590 | 613 / 614 / 646 | 587 / 498 / 591 |
| 2048 | 8 | 1003 / 1001 / 1018 | 1046 / 1045 / 1102 | 1000 / 907 / 1017 |
| 4096 | 1 | 275 / 282 / 314 | 282 / 292 / 330 | 274 / 274 / 277 |
| 4096 | 2 | 538 / 539 / 543 | 549 / 552 / 571 | 411 / 410 / 552 |
| 4096 | 4 | 947 / 950 / 978 | 950 / 971 / 1027 | 948 / 780 / 951 |
| 4096 | 8 | 1813 / 1731 / 1836 | 1868 / 1785 / 1913 | 1159 / 1299 / 1834 |

In every wave the first request of the burst was admitted alone (splits
`1+1`, `1+3`, `1+7`, and `1+4+3` for 4096 x 8). The switch returns
that request after its own batch instead of after the next one. When the
median request falls in the second batch (N=4, and 2048 x 8), only the mean
improves. P99 is set by the last batch. With the switch it is within or below
the range of the two default runs.

Steady-load output throughput in tok/s, mean of 4 runs (`bench_serving`
random, 128 prompts, `--max-concurrency 8`). Request rates inf and 2 were run
twice each; the concurrency cap holds both at the same achieved 1.0 / 1.3
req/s, so they are pooled:

| Input / output | Default | Default, repeat | Switch on |
|---|---|---|---|
| 2048 / 512 | 512 | 509 | 503 |
| 6144 / 256 | 331 | 342 | 336 |

In the 2048 / 512 runs at request rate 2, the switch was 2-5% lower (mean 487
vs 498-515 for the two default runs; single runs 472.5 / 502.3 vs 513.5 /
516.9 / 507.8 / 488.0). The other runs were within noise. Median ITL is
9.7-9.8 ms in all runs. GSM8K (5-shot, chat, 200 questions): 0.98 / 0.985
default, 0.99 with the switch.

## Benchmark

```bash
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30080 \
  --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" --dataset-name random \
  --random-input-len 4096 --random-output-len 2048 --random-range-ratio 1.0 \
  --num-prompts 64 --max-concurrency 32 --request-rate inf
```

## Boundaries

- Use BF16 KV and page size 64. TP greater than 1 without EP is validated only
  for the PTPC checkpoint without MTP (the long-context configuration above,
  TP2/EP1); otherwise use EP, as the MTP configuration does, because the
  expert intermediate width is 640.
- Leave PLE embeddings in Native ATOM; do not enable PLE embedding offload.
  The memory fraction above leaves room for plugin-owned QSA indexer caches.
- Validation covers text requests, concurrent mixed lengths, long prompts,
  slot reuse, and eager fallback above the largest captured batch followed by
  graph replay. It does not establish
  multimodal MTP, radix caching, performance gains, or correctness at other
  parallel sizes and speculative horizons.
- On a SGLang upgrade, review recognition and MTP EntryClass mapping separately.
  Upstream Flash recognition does not replace Native ATOM compute or the
  plugin's graph padding and post-draft metadata refresh contracts. Also check
  that `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`, an internal SGLang
  scheduler variable, still exists and keeps the same meaning.