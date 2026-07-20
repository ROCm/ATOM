# DeepSeek-V3.2 with ATOM SGLang Backend

This recipe shows how to run `deepseek-ai/DeepSeek-V3.2` with the SGLang-ATOM backend. For background on the SGLang-ATOM integration, see [Introduce ATOM as external model package of SGLang](https://github.com/ROCm/ATOM/issues/359).

## Step 1: Pull the SGLang-ATOM Docker

```bash
docker pull rocm/atom-dev:sglang-latest
```

Launch a container from this image and run the remaining commands inside the container.

## Step 2: Launch SGLang-ATOM Server

The SGLang-ATOM backend keeps the standard SGLang CLI, server APIs, and general usage flow compatible with upstream SGLang. For general server options and API usage, users can refer to the [official SGLang documentation](https://docs.sglang.ai/).

Before launching the server, export the SGLang-ATOM settings used by the benchmark workflow:

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_AITER_FP8_PREFILL_ATTN=0
export SGLANG_USE_AITER=1
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
# Introduce ATOM as external model package of SGLang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_ENABLE_TORCH_COMPILE=1
```

### DeepSeek-V3.2 with BF16 KV Cache (TP=4)

```bash
TP=4

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3.2 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --attention-backend aiter \
    --mem-fraction-static 0.8 \
    --disable-radix-cache
```

### DeepSeek-V3.2 with FP8 KV Cache (TP=4)

```bash
TP=4

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3.2 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --disable-radix-cache
```

For an 8-GPU run, set `TP=8` and expose the target GPUs with `CUDA_VISIBLE_DEVICES`.

## Step 3: Performance Benchmark

The SGLang benchmark workflow uses the `bench_serving` client for performance benchmarking.

```bash
git clone --depth 1 https://github.com/kimbochen/bench_serving.git /tmp/bench_serving

ISL=8192
OSL=1024
CONC=64
RANDOM_RANGE_RATIO=0.8
RESULT_DIR=./benchmark-results
RESULT_FILENAME=deepseek-v3.2-sglang-tp${TP}-${ISL}-${OSL}-${CONC}-${RANDOM_RANGE_RATIO}.json

python3 /tmp/bench_serving/benchmark_serving.py \
    --model=deepseek-ai/DeepSeek-V3.2 \
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
        --model_args model=deepseek-ai/DeepSeek-V3.2,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=1,max_gen_toks=512,tokenized_requests=False,trust_remote_code=True \
        --tasks gsm8k \
        --num_fewshot 5
```
