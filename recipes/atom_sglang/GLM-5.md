# GLM-5 / GLM-5.2 with ATOM SGLang Plugin

This recipe shows how to run GLM-5 and GLM-5.2 FP8 models with the SGLang-ATOM plugin. GLM-5 uses sparse MLA and is architecturally similar to DeepSeek-V3.2; in the SGLang-ATOM plugin it is exposed through `GlmMoeDsaForCausalLM`. GLM-5.2 uses the same model family and adds IndexShare, where shared sparse MLA layers reuse the index cache produced by the preceding full sparse MLA layer.

## Step 1: Pull the SGLang-ATOM Docker

```bash
docker pull rocm/atom-dev:sglang-latest
```

Launch a container from this image and run the remaining commands inside the container.

## Step 2: Launch SGLang-ATOM Server

The SGLang-ATOM backend keeps the standard SGLang CLI, server APIs, and general usage flow compatible with upstream SGLang. For general server options and API usage, users can refer to the [official SGLang documentation](https://docs.sglang.ai/).

### GLM-5 FP8 (TP=4)

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_AITER_FP8_PREFILL_ATTN=0
export SGLANG_USE_AITER=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_ENABLE_TORCH_COMPILE=1

TP=4
PORT=9000

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path zai-org/GLM-5-FP8 \
    --host localhost \
    --port "${PORT}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache
```

### GLM-5.2 FP8

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export SGLANG_USE_AITER=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models

MODEL_PATH=zai-org/GLM-5.2-FP8
# Or use a local checkpoint path, for example:
# MODEL_PATH=/shared/data/amd_int/models/GLM-5.2-FP8
TP=8
PORT=9000

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host localhost \
    --port "${PORT}" \
    --trust-remote-code \
    --tp-size "${TP}" \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --kv-cache-dtype fp8_e4m3
```

### GLM-5.2 FP8 with online quant

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export SGLANG_USE_AITER=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models

MODEL_PATH=zai-org/GLM-5.2-FP8
# Or use a local checkpoint path, for example:
# MODEL_PATH=/shared/data/amd_int/models/GLM-5.2-FP8
TP=8
PORT=9000
MODEL_LOADER_EXTRA_CONFIG='{"online_quant_config":{"global_quant_config":"ptpc_fp8","layer_quant_config":{"model.layers.*.mlp.experts":"mxfp8"},"exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate"]}}'

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host localhost \
    --port "${PORT}" \
    --trust-remote-code \
    --tp-size "${TP}" \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --kv-cache-dtype fp8_e4m3 \
    --model-loader-extra-config "${MODEL_LOADER_EXTRA_CONFIG}"
```

### GLM-5.2 FP8 with online quant on MI308

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export SGLANG_AITER_FP8_PREFILL_ATTN=0
export SGLANG_ENABLE_TORCH_COMPILE=1
export SGLANG_USE_AITER=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models

MODEL_PATH=zai-org/GLM-5.2-FP8
# Or use a local checkpoint path, for example:
# MODEL_PATH=/shared/data/amd_int/models/GLM-5.2-FP8
TP=8
PORT=9000
MODEL_LOADER_EXTRA_CONFIG='{"online_quant_config":{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate"]}}'

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host localhost \
    --port "${PORT}" \
    --trust-remote-code \
    --tp-size "${TP}" \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend aiter \
    --model-loader-extra-config "${MODEL_LOADER_EXTRA_CONFIG}"
```

`online_quant_config` must be nested under `model_loader_extra_config`. The ATOM plugin consumes this nested key before SGLang's default model loader validates the remaining loader config. Putting `global_quant_config`, `layer_quant_config`, or `exclude_layer` at the top level will make SGLang reject the config.

For an 8-GPU run, set `TP=8` and expose the target GPUs with `CUDA_VISIBLE_DEVICES`.


### GLM-5.2 MXFP4 with online quant

Use this case with the Quark MXFP4 checkpoint and online quantize the
non-expert weights to `ptpc_fp8`.

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export SGLANG_USE_AITER=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models

MODEL_PATH=amd/GLM-5.2-MXFP4
# Or use a local checkpoint path, for example:
# MODEL_PATH=/shared/data/amd_int/models/GLM-5.2-MXFP4
TP=4
PORT=9000
MODEL_LOADER_EXTRA_CONFIG='{"online_quant_config":{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}}'

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host localhost \
    --port "${PORT}" \
    --trust-remote-code \
    --tp-size "${TP}" \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --kv-cache-dtype fp8_e4m3 \
    --model-loader-extra-config "${MODEL_LOADER_EXTRA_CONFIG}"
```

## Step 3: Performance Benchmark

The SGLang benchmark workflow uses the `bench_serving` client for performance benchmarking.

```bash
git clone --depth 1 https://github.com/kimbochen/bench_serving.git /tmp/bench_serving

ISL=1024
OSL=1024
CONC=64
RANDOM_RANGE_RATIO=1.0
RESULT_DIR=./benchmark-results
RESULT_FILENAME=glm-5.2-sglang-tp${TP}-${ISL}-${OSL}-${CONC}-${RANDOM_RANGE_RATIO}.json

python3 /tmp/bench_serving/benchmark_serving.py \
    --model="${MODEL_PATH}" \
    --backend=sglang \
    --base-url=http://127.0.0.1:${PORT} \
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

The sparse MLA mechanism contains an indexer that selects the top-k tokens it deems most relevant for each query from the KV cache. To evaluate this path, use requests with context longer than the selected top-k window when possible.

```bash
lm_eval --model local-completions \
        --model_args "model=${MODEL_PATH},base_url=http://localhost:${PORT}/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
        --tasks gsm8k \
        --num_fewshot 5
```
