# Qwen3.8-Flash-Next Usage Guide

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is a
multimodal MoE model combining Gated DeltaNet (GDN), sparse attention (QSA),
and n-gram memory. ATOM supports text and image inference with BF16 or FP8
weights and BF16 activations/KV cache.

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

BF16 is the only supported KV cache format. The block size must be divisible
by `indexer_compress_ratio` (4); 64 is used in this example.

Choose a GPU with enough memory for the weights, recurrent state, and KV
cache. Adjust `--max-num-seqs`, `--max-model-len`, and
`--max-num-batched-tokens` for the available memory and workload.
In particular, `--max-num-seqs` defaults to 512; lowering it reduces the
preallocated recurrent-state memory. These are resource controls, not
model-specific requirements.

## Usage notes and limitations

- For image requests, use `--no-enable_prefix_caching` and sufficient cache
  capacity to avoid preemption until the image caching/recompute limitations
  are resolved. Image prompts must fit within `--max-num-batched-tokens`.
- Preserve the checkpoint's quantization exclusions. Quantized GDN input
  projections are not supported.
- Chunked text prefill and full decode CUDA graphs are supported. Piecewise
  compilation/CUDA graphs are not supported.
- MTP/speculative decoding, pipeline parallelism, context parallelism,
  DP attention, TBO, and external KV transfer/offload are not supported.
- This example covers FP8 with TP1. Pure TP2 is not currently validated;
  it can fail MoE warmup on gfx950 due to an MoE dispatch limitation.
- Video inference and long-context accuracy have not been validated.

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
capacity.
