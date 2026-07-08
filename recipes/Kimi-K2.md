# Kimi-K2 Native ATOM Usage Guide

[Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) is a native multimodal agentic model developed by Moonshot AI, built through continual pretraining on approximately 15 trillion mixed visual and text tokens atop Kimi-K2-Base.
[Kimi-K2.7-Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code) is a native multimodal code model in the same Kimi-K2 family.

ATOM currently supports the **text-only** backbone of Kimi-K2.5 and Kimi-K2.7-Code (i.e. the DeepseekV3-style MoE language model with MLA attention) in the **MXFP4** quantized variant:

| Variant | Quantization | Description |
|---------|-------------|-------------|
| **MXFP4** | Quark MXFP4 (w4a4, e8m0 scales, group_size=32) | Routed MoE expert weights in microscale FP4 format. Activations quantised dynamically at runtime. |

## Preparing environment
Pull the nightly docker from https://hub.docker.com/r/rocm/atom/.
All the operations below will be executed inside the container.

## Launching server
ATOM supports running the model with different parallelism, e.g., tensor parallel, expert parallel, data parallel.

### Kimi-K2.5 MXFP4 variant on 8xMI355 GPUs (TP8)

```bash
#!/bin/bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export AITER_AR_1STAGE_MAX_KB=512

HSA_NO_SCRATCH_RECLAIM=1 python -m atom.entrypoints.openai_server \
    --model amd/Kimi-K2.5-MXFP4 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --kv_cache_dtype fp8 \
    --no-enable_prefix_caching \
    --max-num-batched-tokens 32768 \
    --max-model-len 32768 \
    --online_quant_config '{"global_quant_config": "", "layer_quant_config": {"model.layers.*.self_attn.fused_qkv_a_proj": "ptpc_fp8", "model.layers.*.self_attn.q_b_proj": "ptpc_fp8", "model.layers.*.self_attn.kv_b_proj": "ptpc_fp8", "model.layers.*.self_attn.o_proj": "ptpc_fp8"}}'
```

### Kimi-K2.7-Code MXFP4 variant on 4xMI355 GPUs (TP4)

This command follows the native ATOM benchmark configuration for `amd/Kimi-K2.7-Code-MXFP4`.

```bash
#!/bin/bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export AITER_AR_1STAGE_MAX_KB=512

HSA_NO_SCRATCH_RECLAIM=1 python -m atom.entrypoints.openai_server \
    --model amd/Kimi-K2.7-Code-MXFP4 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --kv_cache_dtype fp8 \
    --no-enable_prefix_caching \
    --max-num-batched-tokens 32768 \
    --max-model-len 32768 \
    --online_quant_config '{"global_quant_config": "", "layer_quant_config": {"model.layers.*.self_attn.fused_qkv_a_proj": "ptpc_fp8", "model.layers.*.self_attn.q_b_proj": "ptpc_fp8", "model.layers.*.self_attn.kv_b_proj": "ptpc_fp8", "model.layers.*.self_attn.o_proj": "ptpc_fp8"}}'
```

### BF16 (unquantized) on 4×MI355 GPUs (TP4)

```bash
#!/bin/bash
export HIP_VISIBLE_DEVICES=0,1,2,3

python -m atom.entrypoints.openai_server \
    --model moonshotai/Kimi-K2.5 \
    --trust-remote-code \
    -tp 4 \
    --kv_cache_dtype fp8
```

**Notes**:
- The `--trust-remote-code` flag is required for loading the model's custom tokenizer.
- Kimi-K2.5 and Kimi-K2.7-Code use a DeepseekV3-style architecture with MLA attention, so they leverage the same optimized kernels (MLA, FP8 KV cache, etc.) as DeepSeek models.
- For the MXFP4 variant, `HSA_NO_SCRATCH_RECLAIM=1` is recommended for stability.
- Non-MoE layers (attention, shared experts, dense MLPs) remain in BF16 for all quantized variants.

## Performance baseline

The following script can be used to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
    --model=<model-path> --backend=vllm --base-url=http://localhost:$PORT \
    --trust-remote-code --dataset-name=random \
    --random-input-len=${ISL} --random-output-len=${OSL} \
    --random-range-ratio 0.8 \
    --num-prompts=$(( $CONC * 10 )) \
    --max-concurrency=$CONC \
    --request-rate=inf --ignore-eos \
    --save-result --result-dir=${result_dir} --result-filename=$RESULT_FILENAME.json \
    --percentile-metrics="ttft,tpot,itl,e2el"
```

### Accuracy test
You can verify accuracy using the lm_eval framework:
```bash
lm_eval \
--model local-completions \
--model_args model=<model-path>,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True \
--tasks gsm8k \
--num_fewshot 3
```

## Architecture Details

Kimi-K2.5 and Kimi-K2.7-Code are multimodal models (`KimiK25ForConditionalGeneration`) that wrap:
- **Language model**: A DeepseekV3-style MoE transformer with MLA attention (61 layers, 7168 hidden size, 64 attention heads, 384 routed experts, 8 experts per token)
- **Vision encoder**: MoonViT3d (not loaded in text-only mode)
- **MM Projector**: PatchMerger (not loaded in text-only mode)

ATOM loads only the language model backbone, skipping vision and projector weights for efficient text-only inference.
