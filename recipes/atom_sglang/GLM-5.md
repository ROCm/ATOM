# GLM-5 / GLM-5.2 with ATOM SGLang Plugin

[GLM-5](https://huggingface.co/zai-org/GLM-5-FP8) is an advanced Mixture-of-Experts (MoE) large language model developed by Zhipu AI (THUDM). Its architecture is structurally similar to DeepSeek-V3.2, featuring sparse Multi-head Latent Attention (MLA). This guide covers deploying GLM-5 and GLM-5.2 through the ATOM SGLang plugin.

> The newer [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2-FP8) is also supported. It shares the same `glm_moe_dsa` architecture and adds **IndexShare**, where shared sparse MLA layers reuse the index cache produced by the preceding full sparse MLA layer.

Here is the support matrix for GLM-5.2 across different hardware platforms:

| Hardware | Data Type | Model | Parallelism | MTP Support | Recipe Section |
| --- | --- | --- | --- | --- | --- |
| MI355 | FP4 | [amd/GLM-5.2-MXFP4](https://huggingface.co/amd/GLM-5.2-MXFP4) | TP4 | ✅ | [MI355 FP4](#mi355-fp4) |
| MI355 | FP8 | [zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) | TP4 | ✅ | [MI355 FP8](#mi355-fp8) |
| MI300X | FP8 | [zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) | TP8 | ✅ | [MI300X / MI308X FP8](#mi300x-mi308x-fp8) |
| MI308X | FP8 | [zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) | TP8 | ✅ | [MI300X / MI308X FP8](#mi300x-mi308x-fp8) |

## Preparing Environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom-dev/
```bash
docker pull rocm/atom-dev:sglang-latest
```

Launch a container from this image and run the remaining commands inside the container. The examples below use the standard SGLang server entrypoint and expose ATOM model implementations through `SGLANG_EXTERNAL_MODEL_PACKAGE`.

## GLM-5.2 Recipes

MI355 supports both FP4 and FP8 deployments, whereas MI300X and MI308X support FP8 deployments only. Recipe configurations may differ across platforms to account for hardware-specific capabilities.

### MI355

<a id="mi355-fp4"></a>

#### GLM-5.2 MXFP4 Server

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export SGLANG_USE_AITER=1
export SGLANG_ENABLE_TORCH_COMPILE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
MODEL_PATH=amd/GLM-5.2-MXFP4
# Or use a local checkpoint path, for example:
# MODEL_PATH=/shared/data/amd_int/models/GLM-5.2-MXFP4
TP=4
MODEL_LOADER_EXTRA_CONFIG='{"online_quant_config":{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}}'

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host localhost \
    --port 8015 \
    --trust-remote-code \
    --tp-size "${TP}" \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --kv-cache-dtype fp8_e4m3 \
    --model-loader-extra-config "${MODEL_LOADER_EXTRA_CONFIG}" \
    2>&1 | tee glm-server-mxfp4-tp4-sglang.log
```

<a id="mi355-fp8"></a>

#### GLM-5.2 FP8 Server

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export SGLANG_USE_AITER=1
export SGLANG_ENABLE_TORCH_COMPILE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
MODEL_PATH=zai-org/GLM-5.2-FP8
# Or use a local checkpoint path, for example:
# MODEL_PATH=/shared/data/amd_int/models/GLM-5.2-FP8
TP=4
MODEL_LOADER_EXTRA_CONFIG='{"online_quant_config":{"global_quant_config":"ptpc_fp8","layer_quant_config":{"model.layers.*.mlp.experts":"per_block_fp8"}, "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate"]}}'

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host localhost \
    --port 8015 \
    --trust-remote-code \
    --tp-size "${TP}" \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --kv-cache-dtype fp8_e4m3 \
    --model-loader-extra-config "${MODEL_LOADER_EXTRA_CONFIG}" \
    2>&1 | tee glm-server-fp8-tp4-sglang.log
```

### MI300X / MI308X
On MI300X/MI308X, TP=8 is needed due to the memory limitations.
Note `online_quant_config` for the difference compared to MI355. On MI300X/MI308X, both attention linear layers and MoE experts are online-quantized to PTPC-FP8, leveraging the high-performance kernels on these platforms.

<a id="mi300x-mi308x-fp8"></a>

#### GLM-5.2 FP8 Server

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
MODEL_LOADER_EXTRA_CONFIG='{"online_quant_config":{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate"]}}'

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host localhost \
    --port 8015 \
    --trust-remote-code \
    --tp-size "${TP}" \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --page-size 1 \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --model-loader-extra-config "${MODEL_LOADER_EXTRA_CONFIG}" \
    2>&1 | tee glm-server-fp8-tp8-sglang.log
```

## Performance Benchmark

The SGLang benchmark workflow uses the `bench_serving` client for performance benchmarking.

```bash
git clone --depth 1 https://github.com/kimbochen/bench_serving.git /tmp/bench_serving

ISL=1024
OSL=1024
CONC=64
RANGE_RATIO=0.8
NUM=$(( CONC * 3 ))
RESULT_DIR=./tmp/glm-mxfp4-benchmark
RESULT_FILENAME=glm-mxfp4_${ISL}_${OSL}_${CONC}.json

python3 /tmp/bench_serving/benchmark_serving.py \
    --model="${MODEL_PATH}" \
    --backend=vllm \
    --base-url=http://localhost:8015 \
    --dataset-name=random \
    --random-input-len="${ISL}" \
    --random-output-len="${OSL}" \
    --random-range-ratio "${RANGE_RATIO}" \
    --num-prompts="${NUM}" \
    --max-concurrency="${CONC}" \
    --trust-remote-code \
    --request-rate=inf \
    --num-warmups="$(( 2 * CONC ))" \
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

## Accuracy Validation

The sparse MLA mechanism contains an indexer that selects the top-k tokens it deems most relevant for each query from the KV cache. For GLM-5, the top-2048 tokens are selected from the context by the indexer. To evaluate its accuracy, it is recommended to use requests with context longer than 2048 so that the indexer can be tested. In `lm_eval`, this can be set by increasing the `num_fewshot=20` to increase the context length.

```bash
MODEL_PATH=amd/GLM-5.2-MXFP4
PORT=8015

lm_eval --model local-completions \
        --model_args model="${MODEL_PATH}",base_url=http://localhost:${PORT}/v1/completions,num_concurrent=65,max_retries=3,tokenized_requests=False,trust_remote_code=True \
        --tasks gsm8k \
        --num_fewshot 20
```
