# Kimi-K2.5 Usage Guide

[Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) is a native multimodal agentic model developed by Moonshot AI, built through continual pretraining on approximately 15 trillion mixed visual and text tokens atop Kimi-K2-Base.

ATOM currently supports the **text-only** backbone of Kimi-K2.5 (i.e. the DeepseekV3-style MoE language model with MLA attention). The model uses native INT4 quantization (`compressed-tensors`, group_size=32) for the routed MoE expert weights.

## Preparing environment
Pull the nightly docker from https://hub.docker.com/r/rocm/atom/.
All the operations below will be executed inside the container.

## Launching server
ATOM supports running the model with different parallelism, e.g., tensor parallel, expert parallel, data parallel.
Here we consider the parallelism of TP4 as an example.

### Serving on 4xMI355 GPUs

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
- Kimi-K2.5 uses a DeepseekV3-style architecture with MLA attention, so it leverages the same optimized kernels (MLA, FP8 KV cache, etc.) as DeepSeek models.
- The model uses native INT4 quantization for routed MoE expert weights via `compressed-tensors`.

## Performance baseline

The following script can be used to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
    --model=moonshotai/Kimi-K2.5 --backend=vllm --base-url=http://localhost:$PORT \
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
--model_args model=moonshotai/Kimi-K2.5,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True \
--tasks gsm8k \
--num_fewshot 3
```

## Architecture Details

Kimi-K2.5 is a multimodal model (`KimiK25ForConditionalGeneration`) that wraps:
- **Language model**: A DeepseekV3-style MoE transformer with MLA attention (61 layers, 7168 hidden size, 64 attention heads, 384 routed experts, 8 experts per token)
- **Vision encoder**: MoonViT3d (not loaded in text-only mode)
- **MM Projector**: PatchMerger (not loaded in text-only mode)

ATOM loads only the language model backbone, skipping vision and projector weights for efficient text-only inference.
