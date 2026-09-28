# Qwen3.5 Usage Guide

[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) is a large language model developed by the Qwen team from Alibaba Cloud. It features several key architectural innovations:
* A hybrid attention design mixing Full Attention and Gated-Delta-Net (GDN) linear attention with a 1:3 ratio (`full_attention_interval=4`)
* A highly sparse Mixture-of-Experts (MoE) structure with 512 routed experts
* Native Multi-Token Prediction (MTP) with 1 reusable draft layer for speculative decoding

## Preparing environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All the operations below will be executed inside the container.

## Launching Qwen3.5 with ATOM

### Serving on 4xMI355 GPUs (TP4, FP8 KV Cache)

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --kv_cache_dtype fp8 -tp 4
```

### Serving with MTP Speculative Decoding

MTP-1 reduces per-token latency with ~94% acceptance rate:

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --kv_cache_dtype fp8 -tp 4 \
  --method mtp --num-speculative-tokens 1
```

MTP-3 for higher throughput (~83% acceptance rate, 3.49 avg tokens/fwd):

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --kv_cache_dtype fp8 -tp 4 \
  --method mtp --num-speculative-tokens 3
```

## Performance Metrics

### Benchmarking

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=Qwen/Qwen3.5-397B-A17B-FP8 --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=1024 --random-output-len=1024 \
  --random-range-ratio=0.8 \
  --num-prompts=1280 --max-concurrency=128 \
  --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics="ttft,tpot,itl,e2el"
```

### Accuracy

Install `lm-eval` first:
```bash
pip install lm-eval[api]
```

Run the evaluation:
```bash
lm_eval --model local-completions \
  --model_args model=Qwen/Qwen3.5-397B-A17B-FP8,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 3
```

Reference values (TP4, FP8 KV Cache, MTP-1):
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     3|exact_match|↑  |0.8613|±  |0.0095|
|     |       |strict-match    |     3|exact_match|↑  |0.8484|±  |0.0099|
```
