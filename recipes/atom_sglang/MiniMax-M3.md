# MiniMax-M3 MXFP4/MXFP8 Usage Guide

[MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4) and [MiniMax-M3-MXFP8](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8) are supported by the native ATOM OpenAI-compatible server path.

## Preparing Environment

Pull the latest development image:

```bash
docker pull rocm/atom-dev:sglang-latest
```

## MXFP4 on 4xMI355 GPUs

### Launching Server

```bash
model_path=${model_path:-amd/MiniMax-M3-MXFP4}
run_name=${run_name:-m3-mxfp4}
# Introduce ATOM as external model and processor packages of SGLang.
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=atom.plugin.sglang.models

export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_USE_AITER=1
export ATOM_FORCE_ATTN_TRITON=1
export SGLANG_ENABLE_TORCH_COMPILE=1
export TORCHINDUCTOR_COMPILE_THREADS=128
MODEL_LOADER_EXTRA_CONFIG='{"online_quant_config":{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}}'
JSON_MODEL_OVERRIDE_ARGS='{"use_index_cache": true, "index_topk_freq": 4}'

PORT=${PORT:-8000}
TP=${TP:-4}

python3 -m sglang.launch_server \
    --model-path "${model_path}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --mem-fraction-static 0.8 \
    --page-size 128 \
    --context-length 32768 \
    --max-running-requests 128 \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --model-loader-extra-config "${MODEL_LOADER_EXTRA_CONFIG}" \
    --json-model-override-args "${JSON_MODEL_OVERRIDE_ARGS}" \
    --kv-cache-dtype fp8_e4m3 \
    --disable-radix-cache  2>&1 | tee minimax-m3-mxfp4-sglang-server.log 2>&1 | tee "${run_name}-server.log"
```

## MXFP8 on 4xMI355 GPUs

### Launching Server

For the MXFP8 model, online quant is used to convert the linear weights in attention module and first 3 dense MLP layers to PTPC FP8 format, which are originally equipped with 1*32 block scale.
The MoE weights keep unchanged. Check **--online_quant_config** in the script below for more details.

```bash
model_path=${model_path:-MiniMaxAI/MiniMax-M3-MXFP8}
run_name=${run_name:-m3-mxfp8}
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=atom.plugin.sglang.models

export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_USE_AITER=1
export ATOM_FORCE_ATTN_TRITON=1
export SGLANG_ENABLE_TORCH_COMPILE=1
export TORCHINDUCTOR_COMPILE_THREADS=128
MODEL_LOADER_EXTRA_CONFIG='{"online_quant_config":{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}}'
JSON_MODEL_OVERRIDE_ARGS='{"use_index_cache": true, "index_topk_freq": 4}'

PORT=${PORT:-8000}
TP=${TP:-4}

python3 -m sglang.launch_server \
    --model-path "${model_path}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --mem-fraction-static 0.8 \
    --page-size 128 \
    --context-length 32768 \
    --max-running-requests 128 \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --model-loader-extra-config "${MODEL_LOADER_EXTRA_CONFIG}" \
    --json-model-override-args "${JSON_MODEL_OVERRIDE_ARGS}" \
    --kv-cache-dtype fp8_e4m3 \
    --disable-radix-cache 2>&1 | tee "${run_name}-server.log"
```


### Accuracy Test

Run GSM8K 5-shot with `lm_eval`:

```bash
model_path=${model_path:-amd/MiniMax-M3-MXFP4}
run_name=${run_name:-m3-mxfp4}
BS=65

lm_eval \
    --model local-chat-completions \
    --model_args "model=${model_path},base_url=http://127.0.0.1:${PORT}/v1/chat/completions,num_concurrent=64" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size "${BS}" \
    --apply_chat_template \
    --fewshot_as_multiturn 2>&1 | tee "${run_name}-bs64-accuracy.log"
```

Validated MXFP4 GSM8K result:

```text
local-chat-completions ({'model': '/shared/data/amd_int/models/MiniMax-M3-MXFP4/', 'base_url': 'http://127.0.0.1:8000/v1/chat/completions', 'num_concurrent': 64}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 65
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9363|±  |0.0067|
|     |       |strict-match    |     5|exact_match|↑  |0.9371|±  |0.0067|
```

Validated MXFP8 GSM8K result:

```text
local-chat-completions ({'model': '/shared/data/amd_int/models/MiniMax-M3-MXFP8/', 'base_url': 'http://127.0.0.1:8000/v1/chat/completions', 'num_concurrent': 64}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 65
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9454|±  |0.0063|
|     |       |strict-match    |     5|exact_match|↑  |0.9454|±  |0.0063|
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
  --backend=sglang \
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