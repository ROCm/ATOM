# Kimi-K3 with ATOM vLLM Plugin Backend

This recipe serves the text-only Kimi-K3 backbone
(`KimiK3ForConditionalGeneration`) through the ATOM vLLM out-of-tree plugin.
Kimi-K3 combines KDA recurrent-attention layers, MLA full-attention layers, and
an MXFP4 latent MoE.

The validated configuration requires eight MI355 (gfx950) GPUs with TP8.

## Prerequisites

Use the ATOM vLLM OOT image and install the KDA dependency used by the native
Kimi-K3 implementation:

```bash
docker pull rocm/atom-dev:vllm-latest
pip install "fla-core==0.5.1" "flash-linear-attention==0.5.1"
```

Install the target ATOM checkout into the same environment:

```bash
pip install -e /path/to/ATOM --no-deps
```

## Launch

```bash
ATOM_ROOT=/path/to/ATOM
MODEL=/data/amd_int/models/Kimi-K3
CHAT_TEMPLATE=.github/scripts/kimi_k3_chat_template.jinja

cd "${ATOM_ROOT}"

export AITER_LOG_LEVEL=WARNING
export ATOM_LOADER_USE_THREADPOOL=1
export ATOM_LOADER_THREADPOOL_WORKERS=16
export ATOM_SYNC_AFTER_LOAD=1
export ATOM_DIST_TIMEOUT_SECONDS=3600

export ATOM_USE_TRITON_GEMM=1
export AITER_USE_GROUPED_GEMM=0
export ATOM_USE_TRITON_MOE=0
export AITER_FLYDSL_FORCE=1
export AITER_FORCE_GFX1250=0

vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --language-model-only \
    --kv-cache-dtype fp8 \
    --max-model-len 16384 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.93 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --chat-template "${CHAT_TEMPLATE}" \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

The plugin keeps KDA temporal state in fp32, registers every KDA layer through
vLLM's hybrid/Mamba cache contract, and uses ATOM's MLA backend for full
attention. vLLM may increase the physical attention block size so its MLA and
KDA pages have equal byte size; this is expected.

Prefix caching must stay disabled because KDA recurrent state cannot be
reconstructed from the paged MLA cache alone.

## Smoke test

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [
        {
          \"role\": \"user\",
          \"content\": \"Question: What is 17 + 25? Answer:\"
        }
      ],
      \"max_tokens\": 32,
      \"temperature\": 0
    }"
```

The deterministic response gives the answer `42`.

## Accuracy validation

```bash
lm_eval \
    --model local-chat-completions \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --model_args "model=${MODEL},base_url=http://localhost:8000/v1/chat/completions,num_concurrent=65,max_retries=1,max_gen_toks=2048,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --output_path /tmp/kimi_k3_gsm8k_chat
```

Validated on the full 1319-example GSM8K test set with TP8 and
`FULL_AND_PIECEWISE` CUDA Graph:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9591|±  |0.0055|
|     |       |strict-match    |     5|exact_match|↑  |0.9598|±  |0.0054|
```

Raw result JSON is written below `/tmp/kimi_k3_gsm8k_chat/`. The CI
accuracy threshold is `0.95`.

Use a freshly started server for each reported accuracy run, matching the
native Kimi-K3 validation protocol. Back-to-back evaluations on a warm server
are not used as baselines for this model.

## Current scope

- Text generation only; the vision tower and multimodal projector are skipped.
- TP8 on MI355/gfx950 is the validated deployment.
- Prefix caching is disabled; asynchronous scheduling is enabled.
- Speculative decoding is not enabled for this model.
