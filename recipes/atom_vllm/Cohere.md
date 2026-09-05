# Cohere Command R with ATOM vLLM Plugin Backend

This recipe shows how to run Cohere Command R models (e.g., `CohereLabs/c4ai-command-r7b-12-2024`, `CohereForAI/c4ai-command-r-plus`) with the ATOM vLLM plugin backend on AMD Instinct GPUs. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

### Command R 7B (TP=1, MI300X)

```bash
vllm serve CohereLabs/c4ai-command-r7b-12-2024 \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 128 \
    --max-model-len 131072 \
    --swap-space 32 \
    --no-enable-prefix-caching
```

### Command R+ 104B (TP=4, MI300X)

```bash
vllm serve CohereForAI/c4ai-command-r-plus \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 64 \
    --max-model-len 131072 \
    --no-enable-prefix-caching
```

## Step 3: Performance Benchmark

```bash
vllm bench serve \
    --host localhost \
    --port 8000 \
    --model CohereLabs/c4ai-command-r7b-12-2024 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 512 \
    --num-prompts 100 \
    --request-rate inf \
    --max-concurrency 8
```

## Notes

- **Tied embeddings**: `CohereConfig.tie_word_embeddings=True` by default. The `lm_head` shares weights with `embed_tokens` — no separate `lm_head` checkpoint key is expected.
- **LayerNorm**: Cohere uses standard LayerNorm (with bias) rather than RMSNorm. ATOM uses the AITER-accelerated `layernorm2d_fwd` / `layernorm2d_fwd_with_add` kernels.
- **Q/K norm**: Some Cohere variants set `use_qk_norm=True` in the config. ATOM handles this automatically per-layer.
- **Long-context**: Command R models support up to 128K context. Use `--max-model-len 131072` with `--enable-chunked-prefill --max-num-batched-tokens 65536` for stable prefill on MI300X.
