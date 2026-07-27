# GPT-OSS Usage Guide

[GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b) is a Mixture-of-Experts model from OpenAI featuring GQA with alternating sliding window attention, attention sinks, and SwiGLU activation. ATOM provides built-in support via `GptOssForCausalLM`.

## Preparing environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All the operations below will be executed inside the container.

## Launching server

### Single GPU (FP8 KV Cache)

GPT-OSS-120B fits on a single MI300X/MI355X GPU:

```bash
python -m atom.entrypoints.openai_server \
  --model openai/gpt-oss-120b \
  --kv_cache_dtype fp8 \
  --gpu-memory-utilization 0.5
```

### Multi-GPU with DP Attention + Expert Parallel

Scale across 2 GPUs with data-parallel attention and expert parallelism:

```bash
python -m atom.entrypoints.openai_server \
  --model openai/gpt-oss-120b \
  --kv_cache_dtype fp8 -tp 2 \
  --enable-dp-attention --enable-expert-parallel \
  --gpu-memory-utilization 0.5
```

Tips on server configuration:
- Single GPU is sufficient for this model — no TP required.
- `--gpu-memory-utilization 0.5` is recommended to leave room for KV cache growth.
- DP attention + EP mode does not require `--trust-remote-code` since ATOM has built-in `GptOssForCausalLM`.
- Sliding window attention is applied on even-indexed layers automatically.

### Eagle3 Speculative Decoding

GPT-OSS-120B can use NVIDIA's Eagle3 draft models for lossless throughput gains:

```bash
python -m atom.entrypoints.openai_server \
  --model openai/gpt-oss-120b \
  --kv_cache_dtype fp8 \
  --gpu-memory-utilization 0.5 \
  --method eagle3 \
  --num-speculative-tokens 1 \
  --draft-model nvidia/gpt-oss-120b-Eagle3-throughput
```

Example draft models:
- `nvidia/gpt-oss-120b-Eagle3-throughput` — throughput-optimized Eagle3 draft
- `nvidia/gpt-oss-120b-Eagle3-short-context` — short-context Eagle3 draft

When the draft checkpoint omits these weights, ATOM shares the target `lm_head`, and can
share `embed_tokens` when running without pipeline parallelism and with compatible sharding
(e.g. TP=1 or `ATOM_EAGLE_REPLICATE_EMBED=0`).

## Performance baseline

The following script can be used to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=openai/gpt-oss-120b --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=${ISL} --random-output-len=${OSL} \
  --random-range-ratio=0.8 \
  --num-prompts=$(( $CONC * 10 )) \
  --max-concurrency=$CONC \
  --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics="ttft,tpot,itl,e2el"
```

> Live performance tracking: [rocm.github.io/ATOM/benchmark-dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/)

### Accuracy test

We verified the lm_eval accuracy on gsm8k dataset with command:
```bash
lm_eval --model local-chat-completions --apply_chat_template \
        --model_args model=openai/gpt-oss-120b,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=65,max_retries=3,max_gen_toks=2048,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```

CI accuracy threshold: `flexible-extract ≥ 0.88`.
