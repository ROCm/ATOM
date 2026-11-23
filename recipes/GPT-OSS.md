# GPT-OSS Usage Guide

[GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b) is a Mixture-of-Experts model from OpenAI featuring GQA with alternating sliding window attention, attention sinks, and SwiGLU activation. ATOM provides built-in support via `GptOssForCausalLM`.

## Preparing environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All the operations below will be executed inside the container.

## Launching server

### Single GPU (FP8 KV Cache)

GPT-OSS-120B fits on a single MI300X/MI355X GPU:

```bash
python -m atom.entrypoints.openai_server \
  --model openai/gpt-oss-120b \
  --kv_cache_dtype fp8 \
  --gpu-memory-utilization 0.5
```

### Multi-GPU with DP Attention + Expert Parallel

Scale across 2 GPUs with data-parallel attention and expert parallelism:

```bash
python -m atom.entrypoints.openai_server \
  --model openai/gpt-oss-120b \
  --kv_cache_dtype fp8 -tp 2 \
  --enable-dp-attention --enable-expert-parallel \
  --gpu-memory-utilization 0.5
```

Tips on server configuration:
- Single GPU is sufficient for this model — no TP required.
- `--gpu-memory-utilization 0.5` is recommended to leave room for KV cache growth.
- DP attention + EP mode does not require `--trust-remote-code` since ATOM has built-in `GptOssForCausalLM`.
- Sliding window attention is applied on even-indexed layers automatically.

## Performance baseline

The following script can be used to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=openai/gpt-oss-120b --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=${ISL} --random-output-len=${OSL} \
  --random-range-ratio=0.8 \
  --num-prompts=$(( $CONC * 10 )) \
  --max-concurrency=$CONC \
  --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics="ttft,tpot,itl,e2el"
```

> Live performance tracking: [rocm.github.io/ATOM/benchmark-dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/)

### Accuracy test

We verified the lm_eval accuracy on gsm8k dataset with command:
```bash
lm_eval --model local-chat-completions --apply_chat_template \
        --model_args model=openai/gpt-oss-120b,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=65,max_retries=3,max_gen_toks=2048,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```

CI accuracy threshold: `flexible-extract ≥ 0.88`.

## SGLang backend scope

The commands above use the native ATOM server. When launching GPT-OSS through
`sglang.launch_server`, the explicit `--moe-runner-backend triton_kernel` path
requires `--ep-size 1`: [SGLang validates this before startup](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/arg_groups/model_hook.py#L562).
Adding `--ep-size 8` is rejected even with `--dp-size 8 --enable-dp-attention`.
This restriction belongs to that SGLang runner; the native ATOM DP + EP example
above does not establish EP support for it or for other SGLang MoE backends.

Use the SGLang and Triton-kernels versions selected together by the
[release workflow](../.github/workflows/docker-release.yaml) and
[ATOM base image](../docker/atom_release.dockerfile). A `routing` name error can
follow an earlier optional `triton_kernels` import failure, so check the actual
package versions and import error before adding a routing shim. See
[#2065](https://github.com/ROCm/ATOM/issues/2065) for the old-image report and the
current import-check evidence; an import check alone does not validate model
startup, graph capture, or accuracy.
