# Qwen3.5 with ATOM vLLM Plugin Backend

This recipe shows how to run `Qwen3.5-35B-A3B-Instruct-FP8` and `Qwen3.5-397B-A5B-Instruct-FP8` with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

### Qwen3.5-35B-A3B (TP=2)

```bash
export ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1

vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --max-model-len 16384 \
    --no-enable-prefix-caching
```

### Qwen3.5-397B-A5B (TP=8)

```bash
export ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1

vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --max-model-len 16384 \
    --no-enable-prefix-caching
```

**Important**: `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1` is required for Qwen3.5 because it uses a hybrid architecture with both linear attention (GatedDeltaNet) and full attention layers. This env var ensures full attention layers use vLLM's default implementation.

## Step 3: Performance Benchmark

Users can use the default vllm bench commands for performance benchmarking.

```bash
vllm bench serve \
    --host localhost \
    --port 8000 \
    --model Qwen/Qwen3.5-35B-A3B-FP8 \
    --dataset-name random \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --random-range-ratio 0.8 \
    --max-concurrency 64 \
    --num-prompts 640 \
    --trust_remote_code \
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
        --model_args model=Qwen/Qwen3.5-35B-A3B-FP8,base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```


## Key Environment Variables

- `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1`: **Required** - disables ATOM attention plugin to use vLLM's implementation for full attention layers


## Performance baseline

The following script can be used to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
    --model=Qwen/Qwen3.5-397B-A17B-FP8 --backend=vllm --base-url=http://localhost:8000 \
    --dataset-name=random \
    --random-input-len=${ISL} --random-output-len=${OSL} \
    --random-range-ratio 0.8 \
    --num-prompts=$(( $CONC * 10 )) \
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
    "model": "/mnt/data/pretrained_model/Qwen/Qwen3.5-35B-A3B-FP8",
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
    "model": "/mnt/data/pretrained_model/Qwen/Qwen3.5-35B-A3B-FP8",
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