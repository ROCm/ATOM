# MiniMax-M2.5 with ATOM SGLang Backend

This recipe shows how to run `MiniMax-M2.5` with the SGLang-ATOM backend.
MiniMax-M2.5 support in the SGLang plugin is introduced in
[[Feat] Support MiniMax-M2.5 in SGLang Plugin](https://github.com/ROCm/ATOM/pull/1170).

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
# Introduce ATOM as external model package of SGLang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
```

### MiniMax-M2.5 with FP8 KV Cache (TP=4)

```bash
TP=4
PORT=8000

python3 -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2.5 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "${TP}" \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.9 \
    --context-length 16384 \
    --max-total-tokens 65536 \
    --max-running-requests 128 \
    --page-size 16 \
    --disable-radix-cache
```

For a different GPU count, adjust `TP` and expose the target GPUs with
`CUDA_VISIBLE_DEVICES`.

## Step 3: Performance Benchmark

The SGLang benchmark workflow uses the `bench_serving` client for performance
benchmarking.

```bash
git clone --depth 1 https://github.com/kimbochen/bench_serving.git /tmp/bench_serving

TP=4
PORT=8000

ISL=1024
OSL=1024
CONC=64
RANDOM_RANGE_RATIO=1.0
RESULT_DIR=./benchmark-results
RESULT_FILENAME=minimax-m2.5-sglang-tp${TP}-${ISL}-${OSL}-${CONC}-${RANDOM_RANGE_RATIO}.json

python3 /tmp/bench_serving/benchmark_serving.py \
    --model=MiniMaxAI/MiniMax-M2.5 \
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

```bash
lm_eval \
    --model local-completions \
    --model_args model="${MODEL_PATH}",base_url="http://localhost:${PORT}/v1/completions",num_concurrent=128,max_retries=1,tokenized_requests=False,trust_remote_code=True \
    --tasks gsm8k \
    --num_fewshot 3 \
    --log_samples \
    --output_path minimax_sglang_validation/lm_eval_minimax_m25_bf16kv_conc64_full
```

Expected GSM8K accuracy from the validation run in
[PR #1170](https://github.com/ROCm/ATOM/pull/1170):

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     3|exact_match|↑  |0.9348|±  |0.0068|
|     |       |strict-match    |     3|exact_match|↑  |0.9325|±  |0.0069|
```
