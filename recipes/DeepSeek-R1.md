# DeepSeek-R1 Usage Guide

[DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) is a reasoning-focused Mixture-of-Experts (MoE) large language model developed by DeepSeek. It features Multi-head Latent Attention (MLA) with LoRA-compressed QKV projections and Multi-Token Prediction (MTP) for speculative decoding. The model weights are natively stored in FP8. ATOM provides built-in support for both the FP8 original and MXFP4 quantized variants.

## Preparing environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All the operations below will be executed inside the container.

## Launching server

### FP8 on 8xMI300X/MI355X GPUs (TP8 + FP8 KV Cache)

```bash
python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-R1-0528 \
  --kv_cache_dtype fp8 -tp 8
```

### FP8 with MTP Speculative Decoding (Recommended)

MTP provides ~60% throughput improvement with 3 speculative tokens:

```bash
python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-R1-0528 \
  --kv_cache_dtype fp8 -tp 8 \
  --method mtp --num-speculative-tokens 3
```

### MXFP4 Quantized

```bash
python -m atom.entrypoints.openai_server \
  --model amd/DeepSeek-R1-0528-MXFP4 \
  --kv_cache_dtype fp8 -tp 8
```

### MXFP4 with MTP

```bash
python -m atom.entrypoints.openai_server \
  --model amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4 \
  --kv_cache_dtype fp8 -tp 8 \
  --method mtp --num-speculative-tokens 3
```

Tips on server configuration:
- Always use `--kv_cache_dtype fp8` for better memory efficiency.
- MTP with `--num-speculative-tokens 3` provides the best throughput/latency tradeoff.
- `--num-speculative-tokens 1` is more conservative with lower overhead per step.
- Set `AITER_LOG_LEVEL=WARNING` before starting to suppress aiter kernel log noise.
- Clear compile cache before restarting: `rm -rf /root/.cache/atom/*`

## Performance baseline

The following script can be used to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=deepseek-ai/DeepSeek-R1-0528 --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=${ISL} --random-output-len=${OSL} \
  --random-range-ratio=0.8 \
  --num-prompts=$(( $CONC * 10 )) \
  --max-concurrency=$CONC \
  --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics="ttft,tpot,itl,e2el"
```

Performance on 8xMI300X GPUs with the following environment:
- Docker image: rocm/atom:latest.
- ATOM: main branch.

### FP8 (TP8, FP8 KV Cache)

| ISL  | OSL  | Concurrency | Output Throughput (tok/s) | Total Throughput (tok/s) | Mean TPOT (ms) |
| ---- | ---- | ----------- | ------------------------- | ------------------------ | -------------- |
| 1024 | 1024 | 128         | 4,274                     | 8,558                    | 28.8           |
| 1024 | 1024 | 256         | 6,039                     | 12,071                   | 40.8           |

### FP8 + MTP3 (TP8, FP8 KV Cache, 3 speculative tokens)

| ISL  | OSL  | Concurrency | Output Throughput (tok/s) | Total Throughput (tok/s) | Mean TPOT (ms) |
| ---- | ---- | ----------- | ------------------------- | ------------------------ | -------------- |
| 1024 | 1024 | 128         | 6,913                     | 13,856                   | 17.5           |
| 1024 | 1024 | 256         | 7,284                     | 14,583                   | 33.0           |

> Live performance tracking: [rocm.github.io/ATOM/benchmark-dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/)

### Accuracy test

We verified the lm_eval accuracy on gsm8k dataset with command:
```bash
lm_eval \
  --model local-completions \
  --model_args model=deepseek-ai/DeepSeek-R1-0528,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k \
  --num_fewshot 5
```

Reference accuracy on 8 GPUs (FP8, FP8 KV Cache):
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9553|±  |0.0057|
|     |       |strict-match    |     5|exact_match|↑  |0.9538|±  |0.0058|
```

CI accuracy threshold: `flexible-extract ≥ 0.94` (FP8), `≥ 0.93` (MXFP4).
