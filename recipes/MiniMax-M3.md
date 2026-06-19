# MiniMax-M3 Usage Guide

[MiniMax-M3](https://huggingface.co/amd/MiniMax-M3), [MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4), and [MiniMax-M3-MXFP8](https://huggingface.co/amd/MiniMax-M3-MXFP8) are supported by the native ATOM OpenAI-compatible server path.

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



## MXFP4/MXFP8 on 4xMI355 GPUs

### Launching Server

MXFP4 and MXFP8 use the same launch path.  Set
`model_path=amd/MiniMax-M3-MXFP8 run_name=m3-mxfp8` to run the MXFP8
checkpoint.

```bash
model_path=${model_path:-amd/MiniMax-M3-MXFP4}
run_name=${run_name:-m3-mxfp4}
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --tensor-parallel-size 4 \
  --server-port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --block-size 128 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768 2>&1 | tee "${run_name}-server.log"
```

### Accuracy Test

Run GSM8K 5-shot with `lm_eval`:

```bash
model_path=${model_path:-amd/MiniMax-M3-MXFP4}
run_name=${run_name:-m3-mxfp4}
BS=65

lm_eval \
  --model local-chat-completions \
  --model_args "model=$model_path,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=32,max_gen_toks=16384" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size "${BS}" \
  --apply_chat_template \
  --fewshot_as_multiturn 2>&1 | tee "${run_name}-bs65-accuracy.log"
```

Validated MXFP4 GSM8K result:

```text
local-chat-completions ({'model': 'amd/MiniMax-M3-MXFP4', 'base_url': 'http://127.0.0.1:8000/v1/chat/completions', 'num_concurrent': 32, 'max_gen_toks': 16384}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 65
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9363|±  |0.0067|
|     |       |strict-match    |     5|exact_match|↑  |0.9371|±  |0.0067|
```

Validated MXFP8 GSM8K result:

```text
local-chat-completions ({'model': 'amd/MiniMax-M3-MXFP8', 'base_url': 'http://127.0.0.1:8000/v1/chat/completions', 'num_concurrent': 32, 'max_gen_toks': 16384}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 65
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9469|±  |0.0062|
|     |       |strict-match    |     5|exact_match|↑  |0.9477|±  |0.0061|
```

### Serving Benchmark

The following script can be used to benchmark online serving throughput and latency:

```bash
model_path=${model_path:-amd/MiniMax-M3-MXFP4}
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

Reference MXFP4 results from the validated run on 4xMI355 GPUs:

| CONC | Requests | Duration (s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) | Output tok/s | Total tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 40 | 73.27 | 260.77 | 791.33 | 7.50 | 8.33 | 502.35 | 4515.86 |
| 8 | 80 | 85.64 | 295.52 | 1144.91 | 8.78 | 9.29 | 864.87 | 7693.44 |
| 16 | 160 | 114.35 | 383.04 | 2200.03 | 11.73 | 12.84 | 1280.47 | 11555.95 |
| 32 | 320 | 163.86 | 512.32 | 4477.16 | 16.74 | 19.12 | 1807.32 | 16161.65 |
| 64 | 640 | 242.49 | 831.98 | 8566.28 | 25.00 | 29.83 | 2432.75 | 21928.25 |

Reference MXFP8 results from the validated run on 4xMI355 GPUs:

| CONC | Requests | Duration (s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) | Output tok/s | Total tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 40 | 82.00 | 268.02 | 564.13 | 8.43 | 8.66 | 448.82 | 4034.60 |
| 8 | 80 | 103.52 | 323.33 | 1284.59 | 10.67 | 11.31 | 715.51 | 6364.77 |
| 16 | 160 | 143.25 | 414.95 | 2411.41 | 14.80 | 16.44 | 1022.17 | 9224.81 |
| 32 | 320 | 208.34 | 565.02 | 4936.02 | 21.42 | 24.16 | 1421.47 | 12711.25 |
| 64 | 640 | 305.81 | 893.93 | 9610.43 | 31.69 | 37.31 | 1929.04 | 17387.94 |

## EAGLE3 Speculative Decoding

EAGLE3 runs a small single-layer draft model alongside the MiniMax-M3 target to
propose multiple tokens per step, which the target then verifies. It is lossless
with respect to the target's greedy output. The draft checkpoint is
[`Inferact/MiniMax-M3-EAGLE3`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3).
Enable it by adding three flags to any of the server commands above:

- `--method eagle3`
- `--draft-model Inferact/MiniMax-M3-EAGLE3`
- `--num-speculative-tokens 3`

### Launching Server

The following starts the MXFP4 target with the EAGLE3 draft on 4xMI355 (the FP4
server command above plus the three speculative-decoding flags):

```bash
model_path=amd/MiniMax-M3-MXFP4
draft_path=Inferact/MiniMax-M3-EAGLE3

export AITER_QUICK_REDUCE_QUANTIZATION=INT4

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --tensor-parallel-size 4 \
  --server-port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --block-size 128 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768 \
  --method eagle3 \
  --draft-model "$draft_path" \
  --num-speculative-tokens 3 2>&1 | tee m3-mxfp4-eagle3-server.log
```

### Accuracy Test

Run GSM8K 5-shot with `lm_eval` (identical to the non-speculative test):

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
  --fewshot_as_multiturn 2>&1 | tee m3-mxfp4-eagle3-bs65-accuracy.log
```

Validated GSM8K result (strict-match matches the non-speculative MXFP4 baseline):

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9439|±  |0.0064|
|     |       |strict-match    |     5|exact_match|↑  |0.9447|±  |0.0063|
```

### Acceptance Rate

With `--num-speculative-tokens 3` the draft proposes 3 tokens per step. Measured
over the full GSM8K run:

```text
Average accepted tokens / forward : 3.20    (1 verified + up to 3 speculated)
Overall draft acceptance rate     : 73.45%  (accepted / total proposed draft tokens)
Accepted-count distribution       : {0: 14.20%, 1: 12.11%, 2: 12.83%, 3: 60.86%}
```

Draft tokens are accepted sequentially — position *i* is kept only if every earlier
position was kept — so the per-position acceptance rate is the marginal probability
that each draft slot is accepted (`P(accepted ≥ i)`):

| Draft position | Per-position acceptance rate |
|----------------|------------------------------|
| 1st            | 85.8%                        |
| 2nd            | 73.7%                        |
| 3rd            | 60.9%                        |

The three per-position rates average to the 73.45% overall rate, and 60.86% of
steps accept all 3 speculated tokens.
