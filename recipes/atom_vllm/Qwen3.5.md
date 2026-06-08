# Qwen3.5 with ATOM vLLM Plugin Backend

This recipe shows how to run `Qwen3.5-397B-A17B-FP8`, `Qwen3.5-397B-A17B`, and `Qwen3.5-397B-A17B-MXFP4` with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

### Qwen3.5-397B-A17B-FP8 (TP=4/TP=8)

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export ATOM_USE_CUSTOM_ALL_GATHER=0
export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=0
export GATED_DELTA_RULE_TRITON_AUTOTUNE=1
TP=8

vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
    --host localhost \
    --port 8000 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size "${TP}" \
    --attention-backend ROCM_AITER_FA \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.8 \
    --no-enable-prefix-caching
```

### Qwen3.5-397B-A17B (TP=8)

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export ATOM_USE_CUSTOM_ALL_GATHER=0
export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=0
export GATED_DELTA_RULE_TRITON_AUTOTUNE=1

vllm serve Qwen/Qwen3.5-397B-A17B \
    --host localhost \
    --port 8000 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 \
    --attention-backend ROCM_AITER_FA \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.8 \
    --no-enable-prefix-caching
```

### Qwen3.5-397B-A17B-MXFP4 (TP=4)

```bash
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export ATOM_USE_CUSTOM_ALL_GATHER=0
export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=0
export GATED_DELTA_RULE_TRITON_AUTOTUNE=1

vllm serve amd/Qwen3.5-397B-A17B-MXFP4 \
    --host localhost \
    --port 8000 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching
```

**Important**: ATOM-vLLM no longer supports disabling only ATOM attention while keeping ATOM models active. Use `ATOM_DISABLE_VLLM_PLUGIN=1` for a pure vLLM run.

The following environment variables are relevant for Qwen3.5:

- `ATOM_USE_CUSTOM_ALL_GATHER=0`: Disables custom all-gather for compatibility with Qwen3.5 model architecture
- `ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=0`: Disables FP8 blockscale weight preshuffle
- `AITER_QUICK_REDUCE_QUANTIZATION=INT4`: **Performance optimization** - enables INT4 quantization for quick reduce operations, which can significantly improve TTFT (Time To First Token) performance. **Note**: This optimization may introduce a risk of accuracy degradation. For accuracy-critical workloads, consider validating with your specific use case.

## Step 3: Performance Benchmark

Users can use the default vllm bench commands for performance benchmarking.

```bash
ISL=1000
OSL=100
CONC=4

vllm bench serve \
    --backend vllm \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --model Qwen/Qwen3.5-397B-A17B-FP8 \
    --dataset-name random \
    --random-input-len "${ISL}" \
    --random-output-len "${OSL}" \
    --random-range-ratio 0.0 \
    --max-concurrency "${CONC}" \
    --num-prompts "$(( CONC * 8 ))" \
    --trust_remote_code \
    --num-warmups "${CONC}" \
    --request-rate inf \
    --ignore-eos \
    --disable-tqdm \
    --save-result \
    --percentile-metrics ttft,tpot,itl,e2el
```

### Optional: Enable Profiling

If you want to collect profiling trace, you can use the same API as default vLLM to add `--profiler-config "$profiler_config"` to the `vllm serve` command above.

```bash
profiler_config=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":true,"torch_profiler_record_shapes":true}' \
    "${your-profiler-dir}")
```

## Step 4: Accuracy Validation

```bash
lm_eval --model local-completions \
        --model_args model=Qwen/Qwen3.5-397B-A17B-FP8,base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```

### Qwen3.5-397B-A17B accuracy example

```bash
lm_eval --model local-completions \
        --model_args model=Qwen/Qwen3.5-397B-A17B,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```

Reference result (TP=8):

```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     3|exact_match|↑  |0.8506|±  |0.0098|
|     |       |strict-match    |     3|exact_match|↑  |0.8378|±  |0.0102|
```


## Key Environment Variables

- `ATOM_USE_CUSTOM_ALL_GATHER=0`: **Required** - disables custom all-gather for compatibility with Qwen3.5 model architecture
- `ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=0`: **Required** - disables FP8 blockscale weight preshuffle
- `AITER_QUICK_REDUCE_QUANTIZATION=INT4`: **Performance optimization** - enables INT4 quantization for quick reduce operations
  - **Benefit**: Significantly improves TTFT (Time To First Token) performance by reducing communication overhead during tensor parallelism all-reduce operations
  - **Risk**: May cause slight accuracy degradation due to lower quantization precision
  - **Recommendation**: Use for latency-sensitive workloads where TTFT is critical. For accuracy-critical applications, validate with your specific dataset or consider removing this flag


## Performance baseline

The following script can be used to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
    --model=Qwen/Qwen3.5-397B-A17B-FP8 --backend=vllm --base-url=http://localhost:8000 \
    --dataset-name=random \
    --random-input-len=${ISL} --random-output-len=${OSL} \
    --random-range-ratio 0.0 \
    --num-prompts=$(( CONC * 8 )) \
    --max-concurrency=$CONC \
    --request-rate=inf --ignore-eos \
    --save-result --result-dir=${result_dir} --result-filename=$RESULT_FILENAME.json \
    --percentile-metrics="ttft,tpot,itl,e2el"
```
The performance number on 8 ranks is provided as a reference, with the following environment:
- docker image: rocm/atom-dev:vllm-latest.
- ATOM: main branch.

| ISL  | OSL  | Concurrency | Num Prompts | Output Throughput (tok/s) | Total Throughput (tok/s) |
| ---- | ---- | ----------- | ----------- | ------------------------- | ------------------------ |
| 1024 | 1024 | 4           | 40          | 363.93                    | 699.51                   |
| 1024 | 1024 | 8           | 80          | 707.23                    | 1407.70                  |
| 1024 | 1024 | 16          | 160         | 1276.43                   | 2564.45                  |
| 1024 | 1024 | 32          | 320         | 2186.24                   | 4350.59                  |
| 1024 | 1024 | 64          | 640         | 3442.65                   | 6991.11                  |

### Accuracy baseline 
We verified the lm_eval accuracy on gsm8k dataset with command:
```bash
lm_eval \
--model local-completions \
--model_args model=Qwen/Qwen3.5-397B-A17B-FP8,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
--tasks gsm8k \
--num_fewshot 3
```

Here is the reference value when deploying on 8 ranks:
```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     3|exact_match|↑  |0.8613|±  |0.0095|
|     |       |strict-match    |     3|exact_match|↑  |0.8491|±  |0.0099|
```

# Multimodal input example
Since Qwen3.5 is a multimodal model, you can also pass it with image input, you can refer to below script for image request

Let's take this dog image as example, this image located at `ATOM/recipes/atom_vllm/dog.png`.
<img src="./dog.png" width="400">

```bash 
PORT=${1:-8000}

# Convert image to base64
IMAGE_BASE64=$(base64 -w 0 ATOM/recipes/atom_vllm/dog.png)
curl -X POST "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-397B-A17B-FP8",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image? Describe it in detail."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,'"$IMAGE_BASE64"'"
            }
          }
        ]
      }
    ],
    "max_tokens": 256,
    "temperature": 0,
    "top_p": 1,
    "seed": 123,
    "stream": false
  }' | python3 -m json.tool
  ```

And you are expected to receive the response like following:
```bash
{
    "id": "chatcmpl-8191fab14490fc74",
    "object": "chat.completion",
    "created": 1773631986,
    "model": "Qwen/Qwen3.5-397B-A17B-FP8",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The user wants a detailed description of the image provided.\n\n1.  **Subject Identification:** The main subject is a dog. It looks like a Golden Retriever or a similar mix (maybe a Nova Scotia Duck Tolling Retriever or a mix with a spaniel). It has golden-brown fur.\n2.  **Composition:** It's a close-up portrait, focusing on the head and upper chest. The background is blurred (bokeh), suggesting a shallow depth of field.\n3.  **Physical Features - Head:**\n    *   **Ears:** Floppy, medium-sized, covered in slightly longer, feathery fur. They are set high on the head.\n    *   **Eyes:** Large, dark brown, expressive. They are looking slightly upward and to the left (viewer's left). There are catchlights (reflections) in the eyes, indicating a light source.\n    *   **Forehead:** Smooth, with a slight stop (indentation) between the eyes. The fur is short and sleek here.\n    *   **Nose:** Black, wet-looking, prominent. The nostrils are clearly visible.\n    *   **Muzzle:** Tapered but sturdy. There",
                "refusal": null,
                "annotations": null,
                "audio": null,
                "function_call": null,
                "tool_calls": [],
                "reasoning": null
            },
            "logprobs": null,
            "finish_reason": "length",
            "stop_reason": null,
            "token_ids": null
        }
    ],
    "service_tier": null,
    "system_fingerprint": null,
    "usage": {
        "prompt_tokens": 1048,
        "total_tokens": 1304,
        "completion_tokens": 256,
        "prompt_tokens_details": null
    },
    "prompt_logprobs": null,
    "prompt_token_ids": null,
    "kv_transfer_params": null
}

```