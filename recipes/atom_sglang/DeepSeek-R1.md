# DeepSeek-R1 with ATOM SGLang Backend

This recipe shows how to run `deepseek-ai/DeepSeek-R1-0528` or `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` with the SGLang-ATOM backend. For background on the SGLang-ATOM integration, see [Introduce ATOM as external model package of SGLang](https://github.com/ROCm/ATOM/issues/359).

## Step 1: Pull the SGLang-ATOM Docker

```bash
docker pull rocm/atom-dev:sglang-latest
```

Launch a container from this image and run the remaining commands inside the container.

## Step 2: Launch SGLang-ATOM Server

The SGLang-ATOM backend keeps the standard SGLang CLI, server APIs, and general usage flow compatible with upstream SGLang. For general server options and API usage, users can refer to the [official SGLang documentation](https://docs.sglang.ai/).

Before launching the server, export the same SGLang-ATOM settings used by the benchmark workflow:

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_AITER_FP8_PREFILL_ATTN=0
export SGLANG_USE_AITER=1
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
# Introduce ATOM as external model package of SGLang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
```

### DeepSeek with FP8 (TP=8)

Users can use this command to launch the FP8 server with the same settings as the SGLang benchmark workflow.

```bash
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-R1-0528 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache
```

### DeepSeek with FP8 (TP=4)

```bash
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-R1-0528 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache
```

### DeepSeek with MXFP4 (TP=8)

AMD Instinct MI355X GPU supports MXFP4 computation instructions, and users can use the following command to launch the MXFP4 server on MI355X. For MXFP4 model weight, we suggest using the checkpoint quantized from AMD Quark.

```bash
python3 -m sglang.launch_server \
    --model-path amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache
```

### DeepSeek with MXFP4 (TP=4)

```bash
python3 -m sglang.launch_server \
    --model-path amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache
```

## Step 3: Performance Benchmark

The SGLang benchmark workflow uses the `bench_serving` client for performance benchmarking. The following example matches the workflow command pattern for an MXFP4 TP4 case.

```bash
git clone --depth 1 https://github.com/kimbochen/bench_serving.git /tmp/bench_serving

ISL=8192
OSL=1024
CONC=64
RANDOM_RANGE_RATIO=0.8
RESULT_DIR=./benchmark-results
RESULT_FILENAME=deepseek-r1-fp4-tp4-${ISL}-${OSL}-${CONC}-${RANDOM_RANGE_RATIO}.json

python3 /tmp/bench_serving/benchmark_serving.py \
    --model=amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4 \
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

For FP8 or TP8 cases, keep the same benchmark command and replace `--model` with the checkpoint used in Step 2.

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
        --model_args model=amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4,base_url=http://localhost:8000/v1/completions,num_concurrent=65,max_retries=1,tokenized_requests=False,trust_remote_code=True \
        --tasks gsm8k \
        --num_fewshot 3
```
