# Qwen3.5/Qwen3-Next with ATOM-plugined SGLang

This recipe shows how to run `Qwen/Qwen3.5-397B-A17B-FP8` or `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` with the ATOM-plugined SGLang backend. The related PR is [here](https://github.com/ROCm/ATOM/issues/359).

## Step 1: Pull the SGLang-ATOM Docker Image

```bash
docker pull rocm/atom-dev:sglang-latest
```

Launch a container from this image and run the remaining commands inside the container.

## Step 2: Launch ATOM-plugined SGLang Server

The ATOM-plugined SGLang backend keeps the standard SGLang CLI, server APIs, and general usage flow compatible with upstream SGLang. For general server options and API usage, users can refer to the [official SGLang documentation](https://docs.sglang.ai/).

Before launching the server, export the same ATOM-plugined settings used by the benchmark workflow:

```bash
# Introduce ATOM as external model package of SGLang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
```

### Qwen3.5/Qwen3-Next

Users can use this command to launch the FP8 server with the same settings as the SGLang benchmark workflow.

```bash

export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=0

model_path=[Qwen/Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) # [Qwen/Qwen3-Next-80B-A3B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8)
tp=4

python3 -m sglang.launch_server \
  --model-path ${model_path} \
  --port 8000 \
  --tensor-parallel-size ${tp} \
  --mem-fraction-static 0.9 \
  --reasoning-parser qwen3 \
  --disable-radix-cache
```

## Step 3: Performance Benchmark

The SGLang benchmark workflow uses the `bench_serving` CLI for performance benchmarking, more details can be found in [https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_serving.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_serving.py).

```bash

ISL=4096
OSL=1024
CONC=64
RANDOM_RANGE_RATIO=0.8
RESULT_FILENAME=${model}-tp${tp}-${ISL}-${OSL}-${CONC}-${RANDOM_RANGE_RATIO}.json

python3 -m sglang.bench_serving --backend sglang-oai-chat \
    --model ${model_path} \
    --base-url=http://127.0.0.1:30000 \
    --max-concurrency 16 \ 
    --num-prompts "$(( CONC * 5 ))" \ 
    --request-rate inf \
    --dataset-name random \
    --random-input-len ${ISL} \
    --random-output-len ${OSL} \
    --random-range-ratio ${RANDOM_RANGE_RATIO} \
    --warmup-requests $(( CONC * 2 )) \
    --disable-ignore-eos \ 
    --output-file ${RESULT_FILENAME} \
    --trust-remote-code
```


### Optional: Enable Profiling

If you want to collect profiling trace, set the SGLang profiling environment variables before launching the server, and add `--profile` to the benchmark CLI.

```bash
export SGLANG_PROFILE_RECORD_SHAPES=1
export SGLANG_PROFILE_WITH_STACK=1
export SGLANG_TORCH_PROFILER_DIR=./profile_sglang/
```

Then append `--profile` to the `sglang.bench_serving` command in Step 3.

## Step 4: Accuracy Validation

```bash

lm_eval --model local-completions \
        --model_args model=${model_path},base_url=http://localhost:30000/v1/completions,num_concurrent=65,max_retries=1,tokenized_requests=False,trust_remote_code=True \
        --tasks gsm8k \
        --num_fewshot 3 \
        --trust_remote_code
```


## CI (MI308)

Besides the FP8 cases, the MI308 SGLang-ATOM nightlies cover the PTPC
checkpoints [`sammysun0711/Qwen3.5-35B-A3B-PTPC-FP8`](https://huggingface.co/sammysun0711/Qwen3.5-35B-A3B-PTPC-FP8)
and [`sammysun0711/Qwen3.5-397B-A17B-PTPC-FP8`](https://huggingface.co/sammysun0711/Qwen3.5-397B-A17B-PTPC-FP8),
with and without DFlash speculative decoding:

| Case | Draft model | GSM8K gate |
|---|---|---|
| `MI308 Qwen3.5-35B-A3B-PTPC-FP8 TP4` | — | 0.9409 |
| `MI308 Qwen3.5-35B-A3B-PTPC-FP8 TP1 DFlash` | `z-lab/Qwen3.5-35B-A3B-DFlash` | 0.9409 |
| `MI308 Qwen3.5-397B-A17B-PTPC-FP8 TP8` | — | 0.9603 |
| `MI308 Qwen3.5-397B-A17B-PTPC-FP8 TP8 DFlash` | `z-lab/Qwen3.5-397B-A17B-DFlash` | 0.9554 |

* Server: `aiter` attention and linear-attention prefill, FP8 KV,
  `--chunked-prefill-size 32768`, `--reasoning-parser qwen3`. The
  non-speculative cases disable the radix cache. The DFlash cases keep it,
  with `--mamba-radix-cache-strategy extra_buffer`, which requires
  `--page-size 64`.
* DFlash: `--speculative-algorithm DFLASH --speculative-num-draft-tokens 16
  --speculative-draft-attention-backend triton`, with the draft passed through
  `--speculative-draft-model-path`. With `aiter` attention, DFlash requires
  `SGLANG_USE_AITER_UNIFIED_ATTN=1`; the server refuses to start without it.
* The 35B cases set `AITER_BYPASS_TUNE_CONFIG=1`, as the existing MI308 35B
  PTPC case does. The tuned CK MoE row for this checkpoint cannot be
  dispatched on gfx942.
* GSM8K: lm_eval on the first 200 examples (`lm_eval_limit`), 5-shot, chat
  completions, greedy, up to 16384 generated tokens. The gates are 97% of the
  0.97 (35B), 0.99 (397B) and 0.985 (397B DFlash) references. The DFlash cases
  have an acceptance gate of 0: it passes at any acceptance rate, but fails if
  `/server_info` reports no speculative metrics.
* Benchmark: 8000 input / 500 output tokens at concurrency 1, 128 prompts,
  10 warmups. The shape comes from each case's `nightly_param_lists`.
