# MiniMax-M3 Usage Guide

[MiniMax-M3](https://huggingface.co/amd/MiniMax-M3) and [MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4) are supported by the native ATOM OpenAI-compatible server path.

This recipe tracks the MiniMax-M3 BF16 support added in [ROCm/ATOM PR #1238](https://github.com/ROCm/ATOM/pull/1238). The validated setup also uses the AITER branch `M3_mi355`.

## Preparing Environment

Pull the latest development image:

```bash
docker pull rocm/atom-dev:latest
```

## BF16 on 8xMI355 GPUs

### Launching Server

Native ATOM enables CUDAGraph by default. The command below lists the decode
CUDAGraph capture sizes explicitly.

```bash
model_path=amd/MiniMax-M3
export ATOM_USE_TRITON_MOE="${ATOM_USE_TRITON_MOE:-1}"

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  -tp 8 --server-port 8013 --trust-remote-code --gpu-memory-utilization 0.7 \
  --block-size 128 \
  --no-enable_prefix_caching \
  --cudagraph-capture-sizes '[1,2,4,8,16,32,48,64,128,256,512]'
```

### Smoke Test

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
  "model": "amd/MiniMax-M3",
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

### Accuracy Test

Run GSM8K 5-shot with `lm_eval`:

```bash
model_path=amd/MiniMax-M3

lm_eval --model local-completions \
        --model_args model=$model_path,base_url=http://localhost:8013/v1/completions,num_concurrent=65,max_retries=1,tokenized_requests=False,trust_remote_code=True \
        --tasks gsm8k \
        --num_fewshot 5 2>&1 | tee ./m3-accuracy.log
```

Validated GSM8K result:

```text
local-completions ({'model': 'amd/MiniMax-M3', 'base_url': 'http://localhost:8013/v1/completions', 'num_concurrent': 65, 'max_retries': 1, 'tokenized_requests': False}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9181|±  |0.0076|
|     |       |strict-match    |     5|exact_match|↑  |0.9181|±  |0.0076|
```

### Serving Benchmark

The following script can be used to benchmark online serving throughput and latency:

```bash
model_path=amd/MiniMax-M3
ISL=8192
OSL=1024
CONC=16

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

Reference result from the validated run on 8xMI355X GPUs:

```text
Successful requests:                     160
Benchmark duration (s):                  233.51
Total input tokens:                      1175032
Total generated tokens:                  146426
Request throughput (req/s):              0.69
Output token throughput (tok/s):         627.05
Total Token throughput (tok/s):          5658.99
Mean TTFT (ms):                          556.88
Median TTFT (ms):                        335.52
P99 TTFT (ms):                           3610.92
Mean TPOT (ms):                          24.02
Median TPOT (ms):                        24.12
P99 TPOT (ms):                           27.21
Mean ITL (ms):                           24.01
Median ITL (ms):                         20.16
P99 ITL (ms):                            235.31
Mean E2EL (ms):                          22530.64
Median E2EL (ms):                        22406.40
P99 E2EL (ms):                           26977.54
```

## FP4 on 4xMI355 GPUs

### Launching Server

The following command starts the MiniMax-M3-MXFP4 checkpoint:

```bash
model_path=amd/MiniMax-M3-MXFP4

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --tensor-parallel-size 4 \
  --server-port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --block-size 128 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768 2>&1 | tee m3-mxfp4-server.log
```

### Accuracy Test

Run GSM8K 5-shot with `lm_eval`:

```bash
model_path=amd/MiniMax-M3-MXFP4
BS=65

lm_eval \
  --model local-chat-completions \
  --model_args "model=$model_path,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=32,max_gen_toks=16384" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size "${BS}" \
  --apply_chat_template \
  --fewshot_as_multiturn 2>&1 | tee m3-mxfp4-bs65-accuracy.log
```

Validated GSM8K result:

```text
local-chat-completions ({'model': 'amd/MiniMax-M3-MXFP4', 'base_url': 'http://127.0.0.1:8000/v1/chat/completions', 'num_concurrent': 32, 'max_gen_toks': 16384}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 65
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9386|±  |0.0066|
|     |       |strict-match    |     5|exact_match|↑  |0.9393|±  |0.0066|
```

### Serving Benchmark

The following script can be used to benchmark online serving throughput and latency:

```bash
model_path=amd/MiniMax-M3-MXFP4
ISL=8192
OSL=1024
CONC=16

python -m atom.benchmarks.benchmark_serving \
  --model="$model_path" \
  --backend=vllm \
  --base-url=http://localhost:8000 \
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

Reference result from the validated run on 4xMI355 GPUs:

```text
Successful requests:                     160
Benchmark duration (s):                  194.46
Total input tokens:                      1175032
Total generated tokens:                  146426
Request throughput (req/s):              0.82
Output token throughput (tok/s):         753.00
Total Token throughput (tok/s):          6795.63
Mean TTFT (ms):                          487.13
Median TTFT (ms):                        280.03
P99 TTFT (ms):                           3126.69
Mean TPOT (ms):                          20.03
Median TPOT (ms):                        20.15
P99 TPOT (ms):                           22.52
Mean ITL (ms):                           20.03
Median ITL (ms):                         16.51
P99 ITL (ms):                            196.23
Mean E2EL (ms):                          18815.54
Median E2EL (ms):                        18857.89
P99 E2EL (ms):                           22813.32
```
