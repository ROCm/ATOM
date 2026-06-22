# MiniMax-M3 MXFP4 Usage Guide

[MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4) is supported by
the native ATOM OpenAI-compatible server path.

## Preparing Environment

Pull the latest development image:

```bash
docker pull rocm/atom-dev:latest
```

## MXFP4 on 4xMI355 GPUs

### Launching Server

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
  --max-num-batched-tokens 32768 \
  --no-enable_prefix_caching 2>&1 | tee "${run_name}-server.log"
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

### Serving Benchmark

The following script can be used to benchmark online serving throughput and
latency:

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
