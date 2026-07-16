# DeepSeek-V4 with ATOM SGLang Backend

This recipe shows how to run `deepseek-ai/DeepSeek-V4-Pro` with the SGLang-ATOM backend. For background on the SGLang-ATOM integration, see [Introduce ATOM as external model package of SGLang](https://github.com/ROCm/ATOM/issues/359).

`DeepSeek-V4-Pro` uses ATOM's native DeepSeek V4 model implementation through SGLang's external model package interface. SGLang keeps the server API, scheduler, request lifecycle, and sampling flow, while ATOM owns the model, weight loading, DeepSeek V4 cache views, and attention kernels.

## Step 1: Pull the SGLang-ATOM Docker

```bash
docker pull rocm/atom-dev:sglang-latest
```

Launch a container from this image and run the remaining commands inside the container.

## Step 2: Launch SGLang-ATOM Server

The SGLang-ATOM backend keeps the standard SGLang CLI, server APIs, and general usage flow compatible with upstream SGLang. For general server options and API usage, users can refer to the [official SGLang documentation](https://docs.sglang.ai/).

Before launching the server, export the SGLang-ATOM settings:

```bash
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_USE_AITER=1
export SGLANG_DSV4_FP4_EXPERTS=true
# Introduce ATOM as external model package of SGLang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
```

### DeepSeek-V4-Pro with FP8 KV Cache (TP=8)

```bash
TP=8

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V4-Pro \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.9 \
    --swa-full-tokens-ratio 0.1 \
    --max-running-requests 256 \
    --page-size 256 \
    --disable-radix-cache \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4
```

Notes:

- `SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models` makes SGLang load ATOM's model wrapper instead of the upstream SGLang DeepSeek V4 model.
- `--disable-radix-cache` is required for the current SGLang-ATOM DeepSeek V4 bridge.
- The recipe is validated on 8-GPU MI355 runners with TP=8.

## Step 3: Performance Benchmark

This recipe uses the `bench_serving` client for performance benchmarking.

```bash
git clone --depth 1 https://github.com/kimbochen/bench_serving.git /tmp/bench_serving

ISL=1024
OSL=1024
CONC=8
RANDOM_RANGE_RATIO=0.8
RESULT_DIR=./benchmark-results
RESULT_FILENAME=deepseek-v4-pro-sglang-tp${TP}-${ISL}-${OSL}-${CONC}-${RANDOM_RANGE_RATIO}.json

python3 /tmp/bench_serving/benchmark_serving.py \
    --model=deepseek-ai/DeepSeek-V4-Pro \
    --backend=sglang \
    --base-url=http://127.0.0.1:8000 \
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

If you want to collect profiling trace, set the SGLang profiling environment variables before launching the server, and add `--profile` to the benchmark client command.

```bash
export SGLANG_PROFILE_RECORD_SHAPES=1
export SGLANG_PROFILE_WITH_STACK=1
export SGLANG_TORCH_PROFILER_DIR=./profile_sglang/
```

Then append `--profile` to the `benchmark_serving.py` command in Step 3.

## Step 4: Accuracy Validation

```bash
lm_eval --model local-completions \
        --model_args model=deepseek-ai/DeepSeek-V4-Pro,base_url=http://localhost:8000/v1/completions,num_concurrent=8,max_retries=1,tokenized_requests=False,trust_remote_code=True \
        --tasks gsm8k \
        --num_fewshot 5
```

Reference accuracy on 8xMI355X GPUs with the environment above:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9530|±  |0.0058|
|     |       |strict-match    |     5|exact_match|↑  |0.9530|±  |0.0058|
```
