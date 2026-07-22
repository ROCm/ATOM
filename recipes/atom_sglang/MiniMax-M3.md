# MiniMax-M3 with ATOM SGLang Backend

This recipe shows how to run `MiniMax-M3` with the
SGLang-ATOM backend. MiniMax-M3 uses ATOM's native MiniMax-M3 model
implementation through SGLang's external model package interface. SGLang keeps
the server API, scheduler, request lifecycle, and sampling flow, while ATOM owns
the model, weight loading, quantized kernels, and MiniMax-M3 sparse attention
implementation.

## Step 1: Pull the SGLang-ATOM Docker

```bash
docker pull rocm/atom-dev:sglang-latest
```

Launch a container from this image and run the remaining commands inside the
container.

## Step 2: Launch SGLang-ATOM Server

The SGLang-ATOM backend keeps the standard SGLang CLI, server APIs, and general
usage flow compatible with upstream SGLang. For general server options and API
usage, users can refer to the [official SGLang documentation](https://docs.sglang.ai/).

Before launching the server, export the SGLang-ATOM settings:

```bash
# Expose 4 MI355 GPUs for TP=4.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Introduce ATOM as external model and processor packages of SGLang.
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=atom.plugin.sglang.models

export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_USE_AITER=1
export SGLANG_AITER_FP8_PREFILL_ATTN=0
export ATOM_FORCE_ATTN_TRITON=1
export TORCHINDUCTOR_COMPILE_THREADS=128
```

### MiniMax-M3 MXFP4 (TP=4)

```bash
model_path=${model_path:-amd/MiniMax-M3-MXFP4}
PORT=${PORT:-8000}
TP=${TP:-4}

python3 -m sglang.launch_server \
    --model-path "${model_path}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --attention-backend aiter \
    --mem-fraction-static 0.8 \
    --page-size 128 \
    --context-length 32768 \
    --max-running-requests 128 \
    --disable-radix-cache \
    --disable-cuda-graph 2>&1 | tee minimax-m3-mxfp4-sglang-server.log
```

Notes:

- `SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models` makes SGLang load
  ATOM's MiniMax-M3 model wrapper instead of an upstream SGLang model.
- `SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=atom.plugin.sglang.models` registers the
  text-only MiniMax-M3 processor placeholder required by SGLang's multimodal
  processor setup path.
- MiniMax-M3 sparse attention requires `--page-size 128`.
- `--disable-radix-cache` keeps the evaluation path aligned with the validated
  MiniMax-M3 configuration.

For a different GPU count, adjust `TP` and expose the target GPUs with
`CUDA_VISIBLE_DEVICES`.

## Step 3: Performance Benchmark

The SGLang benchmark workflow uses the `bench_serving` client for performance
benchmarking.

```bash
git clone --depth 1 https://github.com/kimbochen/bench_serving.git /tmp/bench_serving

ISL=8192
OSL=1024
CONC=16
RANDOM_RANGE_RATIO=0.8
RESULT_DIR=./benchmark-results
RESULT_FILENAME=minimax-m3-mxfp4-sglang-tp${TP}-${ISL}-${OSL}-${CONC}-${RANDOM_RANGE_RATIO}.json

python3 /tmp/bench_serving/benchmark_serving.py \
    --model="${model_path}" \
    --backend=sglang \
    --base-url="http://127.0.0.1:${PORT}" \
    --dataset-name=random \
    --random-input-len="${ISL}" \
    --random-output-len="${OSL}" \
    --random-range-ratio "${RANDOM_RANGE_RATIO}" \
    --num-prompts="$(( CONC * 10 ))" \
    --max-concurrency="${CONC}" \
    --trust-remote-code \
    --num-warmups="$(( 2 * CONC ))" \
    --request-rate=inf \
    --ignore-eos \
    --save-result \
    --percentile-metrics="ttft,tpot,itl,e2el" \
    --result-dir="${RESULT_DIR}" \
    --result-filename="${RESULT_FILENAME}"
```

### Optional: Enable Profiling

If you want to collect profiling trace, set the SGLang profiling environment
variables before launching the server, and add `--profile` to the benchmark
client command.

```bash
export SGLANG_PROFILE_RECORD_SHAPES=1
export SGLANG_PROFILE_WITH_STACK=1
export SGLANG_TORCH_PROFILER_DIR=./profile_sglang/
```

Then append `--profile` to the `benchmark_serving.py` command in Step 3.

## Step 4: Accuracy Validation

Run GSM8K 5-shot with `lm_eval`:

```bash
BS=65

lm_eval \
    --model local-chat-completions \
    --model_args "model=${model_path},base_url=http://127.0.0.1:${PORT}/v1/chat/completions,num_concurrent=32,max_gen_toks=16384" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size "${BS}" \
    --apply_chat_template \
    --fewshot_as_multiturn 2>&1 | tee minimax-m3-mxfp4-sglang-gsm8k.log
```

Reference GSM8K accuracy on 4xMI355 GPUs with the environment above:

```text
local-chat-completions ({'model': '/shared/data/amd_int/models/MiniMax-M3-MXFP4', 'base_url': 'http://127.0.0.1:8000/v1/chat/completions', 'num_concurrent': 32, 'max_gen_toks': 16384}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 65
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9378|±  |0.0067|
|     |       |strict-match    |     5|exact_match|↑  |0.9386|±  |0.0066|
```
