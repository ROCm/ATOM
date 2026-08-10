# Kimi-K3 with ATOM SGLang Plugin

[Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) combines KDA
linear-attention layers, MLA full-attention layers, and an MXFP4 latent MoE.
This recipe serves its text-only backbone through the ATOM SGLang plugin.

The validated configuration requires eight MI355 (gfx950) GPUs with TP8.
Prefix caching is not supported in this configuration. Speculative decoding is
supported through the separate **DSpark** draft checkpoint (see
[DSpark Speculative Decoding](#dspark-speculative-decoding)); the base recipe
below runs the target model on its own.

## Preparing Environment

Pull the latest SGLang development image and install the KDA dependency:

```bash
docker pull rocm/atom-dev:sglang-latest
pip install "fla-core==0.5.1" "flash-linear-attention==0.5.1"
pip install -e /path/to/ATOM --no-deps
```

## Launching Server

```bash
MODEL_PATH=${MODEL_PATH:-moonshotai/Kimi-K3}
PORT=${PORT:-8000}
TP=${TP:-8}

export SGLANG_PLUGINS=atom_sglang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=atom.plugin.sglang.models
export SGLANG_USE_AITER=1

export ATOM_LOADER_NUM_THREADS=16
export ATOM_SYNC_AFTER_LOAD=1
export ATOM_DIST_TIMEOUT_SECONDS=3600

export ATOM_USE_TRITON_GEMM=1
export AITER_USE_GROUPED_GEMM=0
export ATOM_USE_TRITON_MOE=0
export AITER_FLYDSL_FORCE=1
export AITER_FORCE_GFX1250=0

# SGLang allocates 128-token hybrid-cache pages. ATOM MLA keeps its native
# per-token page layout when translating those pages for the attention kernel.
export ATOM_MLA_PAGE_SIZE=1
export ATOM_USE_UNIFIED_ATTN=1
export ATOM_FORCE_ATTN_TRITON=1
export AITER_LOG_LEVEL=WARNING

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --page-size 128 \
    --context-length 16384 \
    --mem-fraction-static 0.93 \
    --disable-radix-cache \
    2>&1 | tee kimi-k3-tp8-sglang-server.log
```

The plugin keeps KDA recurrent state in fp32 and translates SGLang's hybrid
KDA/MLA cache metadata to ATOM attention metadata. Radix cache must stay
disabled because KDA recurrent state cannot be reconstructed from MLA KV pages.

## Smoke Test

```bash
curl "http://127.0.0.1:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL_PATH}\",
      \"prompt\": \"Question: What is 17 + 25? Answer:\",
      \"max_tokens\": 32,
      \"temperature\": 0
    }"
```

The deterministic response starts with `42`.

## Accuracy Validation

Run the same GSM8K 5-shot evaluation used by the ATOM vLLM Kimi-K3 recipe:

```bash
OUTPUT_PATH=./kimi-k3-sglang-gsm8k

lm_eval \
    --model local-completions \
    --model_args "model=${MODEL_PATH},base_url=http://127.0.0.1:${PORT}/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --output_path "${OUTPUT_PATH}"
```

The ATOM vLLM TP8 reference is `0.9553` flexible-extract exact match. SGLang
nightly CI initially uses `0.94` as its pass threshold and should record a
SGLang-specific baseline after the first complete run.

Use a freshly started server for each reported accuracy run. Reusing a warm
Kimi-K3 server for back-to-back accuracy measurements is not a validated
baseline protocol.

## Serving Benchmark

The SGLang CI catalog covers 1024/1024 and 8192/1024 input/output lengths:

```bash
ISL=${ISL:-8192}
OSL=${OSL:-1024}
CONC=${CONC:-16}
NUM_PROMPTS=$((CONC * 10))

python -m atom.benchmarks.benchmark_serving \
    --model="${MODEL_PATH}" \
    --backend=sglang \
    --base-url="http://127.0.0.1:${PORT}" \
    --dataset-name=random \
    --random-input-len="${ISL}" \
    --random-output-len="${OSL}" \
    --random-range-ratio=0.8 \
    --num-prompts="${NUM_PROMPTS}" \
    --max-concurrency="${CONC}" \
    --request-rate=inf \
    --ignore-eos \
    --save-result \
    --percentile-metrics="ttft,tpot,itl,e2el"
```

## DSpark Speculative Decoding

Kimi-K3 supports **DSpark** block-parallel speculative decoding through a
separate draft checkpoint,
[Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark). The
draft runs one `TARGET_VERIFY` backbone pass per step to propose a whole block
of `gamma = 7` tokens (`--speculative-num-draft-tokens 8` = 7 drafts + 1 bonus),
which the target then verifies. The draft is a pure-MLA model; its KV cache is
allocated independently of the target and kept in bf16 regardless of the
target's `--kv-cache-dtype`.

### Launching Server (DSpark)

DSpark reuses the **same plugin/ATOM environment as the target-only recipe** and
only adds the speculative arguments. The block below is self-contained; the
`export` lines are identical to [Launching Server](#launching-server). The draft
model plus its KV pool need extra memory, so DSpark uses a smaller
`--context-length` / `--mem-fraction-static` than the target-only recipe.

> DSpark requires an SGLang build that ships the native `dspark_components`
> speculative worker (`sglang.srt.speculative.dspark_components`). Use the latest
> `rocm/atom-dev:sglang-latest` image; older baked SGLang releases without it
> cannot serve `--speculative-algorithm DSPARK`.

```bash
MODEL_PATH=${MODEL_PATH:-moonshotai/Kimi-K3}
DRAFT_PATH=${DRAFT_PATH:-Inferact/Kimi-K3-DSpark}
PORT=${PORT:-8000}
TP=${TP:-8}

# Plugin wiring (identical to the target-only recipe).
export SGLANG_PLUGINS=atom_sglang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=atom.plugin.sglang.models
export SGLANG_USE_AITER=1

export ATOM_LOADER_NUM_THREADS=16
export ATOM_SYNC_AFTER_LOAD=1
export ATOM_DIST_TIMEOUT_SECONDS=3600

export ATOM_USE_TRITON_GEMM=1
export AITER_USE_GROUPED_GEMM=0
export ATOM_USE_TRITON_MOE=0
export AITER_FLYDSL_FORCE=1
export AITER_FORCE_GFX1250=0

# K3 full-attention MLA keeps its native per-token page layout over SGLang's
# 128-token hybrid KV pool.
export ATOM_MLA_PAGE_SIZE=1
export ATOM_USE_UNIFIED_ATTN=1
export ATOM_FORCE_ATTN_TRITON=1
export AITER_LOG_LEVEL=WARNING

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --page-size 128 \
    --context-length 8192 \
    --mem-fraction-static 0.85 \
    --disable-radix-cache \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path "${DRAFT_PATH}" \
    --speculative-num-draft-tokens 8 \
    --cuda-graph-max-bs 4 \
    2>&1 | tee kimi-k3-dspark-tp8-sglang-server.log
```

The KDA (linear-attention) layers advance their recurrent state through ATOM's
native `KimiKDAAttention` speculative branch; the few MLA (full-attention)
layers verify the drafted block against the committed paged prefix. Both the
target verify graph and the draft graph are CUDA-graph captured. Pass
`--disable-cuda-graph` to fall back to eager (useful for debugging).

### Smoke Test (DSpark)

The smoke test is identical to the base recipe and still returns `42`; the only
difference is speculative acceptance in the server log (`accept len`).

### Accuracy and Acceptance

Run the same GSM8K 5-shot evaluation as the base recipe. Speculative decoding
must not change the target's greedy output, so accuracy matches the target-only
recipe within MoE non-determinism (validated: `0.9767` flexible-extract under
CUDA Graph, versus the target-only `~0.9553` vLLM reference). The server logs the
average accepted length per step; the validated DSpark configuration accepts
`~4.3` of the 8 verified tokens per step (`accept len` in the decode-batch logs).

### Serving Benchmark (DSpark)

Reuse the base benchmark command against the DSpark server. Because the draft KV
pool shrinks the number of concurrent slots, benchmark speculative decoding at
lower concurrency (for example `CONC=1`, `2`, `4`) where block-parallel
speculation gives the largest per-request latency reduction.

## Current Scope

- Text generation only; the vision tower and multimodal projector are skipped.
- TP8 on MI355/gfx950 is the validated deployment.
- FP8 KV cache and CUDA Graph capture/replay are enabled.
- Prefix caching is disabled.
- Speculative decoding is available via the DSpark draft checkpoint (below).
