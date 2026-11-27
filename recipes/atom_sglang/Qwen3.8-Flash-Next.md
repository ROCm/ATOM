# Qwen3.8-Flash-Next on SGLang-ATOM

Flash uses `qwen4_exp` / `Qwen4ExpForConditionalGeneration`. Target and MTP
compute run through Native ATOM `Qwen4Exp` and `Qwen4ExpMTP`; the plugin
adapts SGLang scheduling, QSA KV pools, PLE state and HC hidden storage.
Run this recipe inside the ATOM SGLang release Docker image with its bundled
ATOM, SGLang, and AITER. The
[release image definition](../../docker/sglang_release.dockerfile) currently
pins SGLang v0.5.22. This integration requires SGLang 0.5.20 or
newer; upstream recognizes `qwen4_exp`, and the recognition patch adds only
the MTP draft-arch and HC-width adapters.
The bundled AITER must include [ROCm/aiter#8083](https://github.com/ROCm/aiter/pull/8083)
(`db870a851` or later), which provides the QSA support queries used by ATOM.

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

## Single-node prefill/decode disaggregation

Native Flash PD requires Mooncake, PP=1, and `--disable-radix-cache` on both
workers. Use matching model, page size, SSM dtype, and speculative settings.
The following TP2/EP2 configuration uses four MI308X GPUs and MTP step=3.
The transport is TCP, so RDMA devices are not required. This path uses
SGLang's PD scheduler with the ATOM model backend; the router backend is
`sglang`.

In **both worker terminals**, set the checkpoint path and define this helper:

```bash
export MODEL_PATH=/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16
export SGLANG_PLUGINS=atom_sglang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=atom.plugin.sglang.models
export SGLANG_USE_AITER=1
export ATOM_QSA_FLYDSL_ATTN=1
export MOONCAKE_PROTOCOL=tcp
export MC_FORCE_TCP=1
export OMP_NUM_THREADS=4

run_pd_worker() {
  HIP_VISIBLE_DEVICES="$2" CUDA_VISIBLE_DEVICES="$2" \
  python -m sglang.launch_server \
    --model-path "$MODEL_PATH" --served-model-name qwen-flash \
    --trust-remote-code --host 0.0.0.0 --port "$3" \
    --tp-size 2 --ep-size 2 --attention-backend aiter \
    --kv-cache-dtype bf16 --page-size 64 \
    --context-length 132096 --max-running-requests 64 \
    --mem-fraction-static 0.95 --disable-radix-cache \
    --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 64 \
    --speculative-algorithm EAGLE --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 256 \
    --mamba-ssm-dtype float32 --watchdog-timeout 1200 \
    --disaggregation-mode "$1" --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port "$4"
}
```

In the prefill terminal, run `run_pd_worker prefill 0,1 7081 8998`.
In the decode terminal, run `run_pd_worker decode 2,3 7082 8999`.
Select free GPU pairs and bind CPUs to their local NUMA nodes for your host.
After both `/health` endpoints return 200, start the router in a third terminal:

```bash
atomesh launch --host 0.0.0.0 --port 7080 --pd-disaggregation \
  --prefill http://127.0.0.1:7081 8998 --decode http://127.0.0.1:7082 \
  --backend sglang --policy round_robin --request-timeout-secs 300
```

Send requests to `http://127.0.0.1:7080/v1` with model `qwen-flash`.
SGLang applies its AITER long-context 0.85 memory factor to the requested
0.95 fraction, giving an effective fraction of 0.8075.
At step=3, use 256 persistent Mamba slots with the 64-request FP32 SSM
configuration above. The 330-slot step=2 setting leaves no KV budget on the
decode worker once its additional speculative intermediate states are reserved.

### Hybrid state transfer and verification

ATOM registers the buffers used by Native compute before CUDA graph capture:

| State | Addressing / transport |
| --- | --- |
| Target and MTP K/V | Existing SGLang paged KV transfer |
| GDN convolution and SSM | Existing Mamba physical-slot transfer |
| Target and MTP QSA raw/compressed keys | Full-KV-page indices, using the flat `QSA_COMPRESSED` state component |
| Native PLE convolution and n-gram history | Mamba slot siblings, including reset/copy/host round-trip |

SGLang's own QSA buffers are not written by Native compute. Their descriptors
are replaced with Native's actual page buffers; Native raw keys must not use
SGLang's request-indexed `QSA_PENDING` ring indices. Index caches include the
same padding page as K/V, so the final physical page remains addressable.

For a **single-node correctness run**, set the same new directory in both
worker environments **before startup**:

```bash
export ATOM_FLASH_PD_VERIFY_DIR=/tmp/flash-pd-checks
```

P hashes every registered K/V and state buffer before its final send; D hashes
the corresponding destination pages/slots after transfer and before its first
decode. Every buffer is compared with SHA-256. A mismatch writes a diagnostic
JSON file and raises; success logs `ATOM Flash PD byte verification passed`.
For the 48-layer target plus one MTP layer, each rank checks 128 buffers,
including SGLang's two auxiliary PLE buffers. Source and destination physical
IDs may differ.

This diagnostic synchronizes the GPU and copies selected data to the CPU.
It is disabled by default; unset the variable on both sides and restart before
measuring performance. Do not combine it with
`--disaggregation-enable-kv-checksum`: SGLang's single state-index checksum does
not represent the combination of Native QSA page indices and Mamba slot indices.

## MI308X pure TP2 / TP4 with BF16 MTP

For the `Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16` checkpoint on gfx942, use
`--ep-size 1`. ATOM keeps the logical expert shard at 320 (TP2) or 160
(TP4), and stores it at 320 or 192 respectively. Both the FP8 target and
BF16 MTP fuse the shared expert into slot 512, so AITER sees 513 experts
and top-k 11. The target and draft retain their checkpoint dtypes.

The 64-aligned layout and BF16 MTP fusion are limited to gfx942 with 80 CU,
TP2/TP4, EP disabled and DP1, hidden size 2560, and routed/shared intermediate
size 640. They require the fused 513-expert/top-k-11 layout and runtime CT
PTPC FP8 or BF16 weights. Other configurations retain the previous policy;
this guard does not query AITER's tuned-row coverage.

Run the following inside the release container. Leave `AITER_CONFIG_FMOE`
unset so AITER discovers the tuning tables automatically:

```bash
export MODEL_PATH=/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16
export CUDA_VISIBLE_DEVICES=0,1
export HIP_VISIBLE_DEVICES=0,1
export GPU_ARCHS=gfx942
export SGLANG_PLUGINS=atom_sglang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=atom.plugin.sglang.models
export SGLANG_USE_AITER=1

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

With `AITER_CONFIG_FMOE` unset, AITER automatically merges its default MoE
table with `aiter/configs/model_configs/*tuned_fmoe*.csv`, including this
model's FP8 PTPC and BF16 tables. Explicit CSV overrides are for debugging;
leave them unset for normal serving. The tables contain FlyDSL, ASM and
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
* batch-1 MoE: the scoped gfx942/80 CU pure TP2/TP4 setup (EP1, DP1) uses
  AITER for both CT PTPC FP8 and BF16. Outside it, `moe_decode_single_token`
  remains available for compatible fused FP8 experts with per-channel scales
  and intermediate width 320; other calls use their normal FusedMoE path.
* QSA indexer scoring: matrix cores for prefill, CUDA cores for decode-sized
  batches.

The MTP drafter's single layer keeps separate HC `mix`/`combine` operations.
For the MI308X pure TP2/TP4 configuration above, its BF16 shared expert is
fused into the MoE; other MTP layouts retain the standalone shared expert.

## Optional: quantized prefill all-reduce (lossy)

Prefill all-reduces of 16k-token chunks (84 MB at TP2) take about 1.8 ms on
RCCL. AITER's quick all-reduce with INT6 codes halves that
(`AITER_QUICK_REDUCE_QUANTIZATION=INT6`); it only engages above its minimum
message size, so decode all-reduces stay exact. It perturbs prefill
activations (max ~3% of a tensor's range per reduction), so validate accuracy
on the target workload before enabling it.

## Optional: bursty prefill on an idle server

When several requests reach an idle server at once and their prefill takes
more than one batch, overlap scheduling launches the next prefill batch before
it processes the results of the previous one. The first batch's requests then
get their first token only after the next batch has also run. To process each
prefill batch's results before the next prefill batch launches, add one
variable to the MTP command in [Text serving with MTP](#text-serving-with-mtp)
and keep all of its flags:

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

Measurements retained from the upstream recipe used the TP2/EP2 command in
[Text serving with MTP](#text-serving-with-mtp) on 2x MI308X (gfx942), ROCm
7.2.4, SGLang v0.5.22. The default was run twice, before and after the switch,
to show run-to-run drift. Burst TTFT in ms, median / mean / P99, for N requests sent
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
- When updating the release image, review recognition and MTP EntryClass
  mapping separately. Upstream Flash recognition does not replace Native ATOM
  compute or the plugin's graph padding and post-draft metadata refresh contracts. Also check
  that `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`, an internal SGLang
  scheduler variable, still exists and keeps the same meaning.
