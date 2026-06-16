# MiniMax-M3 BF16 Usage Guide

[MiniMax-M3](https://huggingface.co/amd/MiniMax-M3) is a MiniMax-M3 BF16 checkpoint supported by the native ATOM OpenAI-compatible server path.

This recipe tracks the MiniMax-M3 BF16 support added in [ROCm/ATOM PR #1238](https://github.com/ROCm/ATOM/pull/1238). The validated setup also uses the AITER branch `M3_mi355`.

## Preparing Environment

Pull the latest development image:

```bash
docker pull rocm/atom-dev:vllm-latest
```

All commands below should be run inside the container. The example assumes the model is available at:

```bash
model_path=/shared/data/amd_int/models/MiniMax-M3/
```

If `/shared/data/amd_int/models` is mounted as `/models` inside the container, use:

```bash
model_path=/models/MiniMax-M3/
```

## Launching Server

### BF16 on 8xMI355X GPUs (TP8)

```bash
model_path=/shared/data/amd_int/models/MiniMax-M3/
export ATOM_USE_TRITON_MOE="${ATOM_USE_TRITON_MOE:-1}"

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  -tp 8 --server-port 8013 --trust-remote-code --gpu-memory-utilization 0.7 \
  --block-size 128 \
  --no-enable_prefix_caching
```

Use the container path if the model directory is mounted at `/models`:

```bash
model_path=/models/MiniMax-M3/
```

## Smoke Test

```bash
curl -X POST "http://localhost:8013/v1/completions" \
     -H "Content-Type: application/json" \
     -d '{
         "prompt": "The capital of China is",
         "temperature": 0,
         "top_p": 1,
         "top_k": 1,
         "repetition_penalty": 1.0,
         "presence_penalty": 0,
         "frequency_penalty": 0,
         "stream": false,
         "ignore_eos": false,
         "n": 1,
         "seed": 123,
         "max_tokens": 10
     }'
```

Reference response from the validated run:

```json
{
  "model": "/models/MiniMax-M3/",
  "choices": [
    {
      "text": " Beijing. The the capital of China is Beijing.",
      "finish_reason": "max_tokens"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 10,
    "total_tokens": 15
  }
}
```

## Serving Benchmark

The following script can be used to benchmark online serving throughput and latency:

```bash
model_path=/shared/data/amd_int/models/MiniMax-M3/
ISL=1024
OSL=1024
CONC=128

python -m atom.benchmarks.benchmark_serving \
  --model="$model_path" \
  --backend=vllm \
  --base-url=http://localhost:8013 \
  --dataset-name=random \
  --random-input-len="${ISL}" \
  --random-output-len="${OSL}" \
  --random-range-ratio=0.8 \
  --num-prompts=$(( CONC * 10 )) \
  --max-concurrency="${CONC}" \
  --request-rate=inf \
  --ignore-eos \
  --save-result \
  --percentile-metrics="ttft,tpot,itl,e2el"
```

## Accuracy Test

Run GSM8K 5-shot with `lm_eval`:

```bash
model_path=/shared/data/amd_int/models/MiniMax-M3/

lm_eval --model local-completions \
        --model_args model=$model_path,base_url=http://localhost:8013/v1/completions,num_concurrent=65,max_retries=1,tokenized_requests=False,trust_remote_code=True \
        --tasks gsm8k \
        --num_fewshot 5 2>&1 | tee ./m3-accuracy.log
```

Validated GSM8K result:

```text
local-completions ({'model': '/models/MiniMax-M3/', 'base_url': 'http://localhost:8013/v1/completions', 'num_concurrent': 65, 'max_retries': 1, 'tokenized_requests': False}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9181|±  |0.0076|
|     |       |strict-match    |     5|exact_match|↑  |0.9181|±  |0.0076|
```

