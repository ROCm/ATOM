# GLM-5 Usage Guide

[GLM-5](https://huggingface.co/zai-org/GLM-5-FP8) is an advanced Mixture-of-Experts (MoE) large language model developed by Zhipu AI (THUDM). Its architecture is structurally similar to DeepSeek v3.2, featuring Multi-head Latent Attention (MLA). This guide covers deploying the FP8 version of GLM-5 on AMD GPUs with ATOM.

> The newer [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2-FP8) is also supported — it shares the same `glm_moe_dsa` architecture and adds **IndexShare**. See [GLM-5.2 (IndexShare)](#glm-52-indexshare) below.

Here is the support matrix for GLM-5.2 across different hardware platforms:

| Hardware | Data Type | Model | Parallelism | MTP Support | Recipe Section |
| --- | --- | --- | --- | --- | --- |
| MI355 | FP4 | [amd/GLM-5.2-MXFP4](https://huggingface.co/amd/GLM-5.2-MXFP4) | TP4 | ✅ | [MI355 FP4](#mi355-fp4) |
| MI355 | FP8 | [zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) | TP4 | ✅ | [MI355 FP8](#mi355-fp8) |
| MI300X | FP8 | [zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) | TP8 | ✅ | [MI300X / MI308X FP8](#mi300x-mi308x-fp8) |
| MI308X | FP8 | [zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) | TP8 | ✅ | [MI300X / MI308X FP8](#mi300x-mi308x-fp8) |

## Preparing environment
Pull the latest docker from https://hub.docker.com/r/rocm/atom-dev/ :
```bash
docker pull rocm/atom-dev:latest
```
All the operations in the next will be executed inside the container.

## Launching server
ATOM supports running the model with different parallelism, e.g., tensor parallel, expert parallel, data parallel. The examples below are organized by hardware and use the current ATOM server entrypoint.

### MI355

<a id="mi355-fp4"></a>

#### GLM-5.2 MXFP4 Server

```bash
#!/bin/bash

model_path=amd/GLM-5.2-MXFP4
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
TP=4

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --server-port 8000 \
  --kv_cache_dtype fp8 \
  --no-enable_prefix_caching \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate", "*expert*"]}' \
  -tp $TP 2>&1 | tee server.log &
```

#### GLM-5.2 MXFP4 MTP Server

```bash
#!/bin/bash

model_path=amd/GLM-5.2-MXFP4
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
TP=4

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --server-port 8004 \
  --kv_cache_dtype fp8 \
  --no-enable_prefix_caching \
  --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate", "model.layers.[0-9].mlp.*expert*","model.layers.[1-6][0-9].mlp.*expert*","model.layers.7[0-7].mlp.*expert*"]}' \
  --num-speculative-tokens 3 \
  --method mtp \
  -tp $TP 2>&1 | tee server_mtp.log &
```

<a id="mi355-fp8"></a>

#### GLM-5.2 FP8 Server

```bash
#!/bin/bash

model_path=zai-org/GLM-5.2-FP8
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
TP=4

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --server-port 8000 \
  --kv_cache_dtype fp8 \
  --no-enable_prefix_caching \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "layer_quant_config":{"model.layers.*.mlp.experts":"per_block_fp8"}, "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate"]}' \
  -tp $TP 2>&1 | tee server.log &
```

#### GLM-5.2 FP8 MTP Server

```bash
#!/bin/bash

model_path=zai-org/GLM-5.2-FP8
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
TP=4

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --server-port 8004 \
  --kv_cache_dtype fp8 \
  --no-enable_prefix_caching \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "layer_quant_config":{"model.layers.*.mlp.experts":"per_block_fp8"}, "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate"]}' \
  --num-speculative-tokens 3 \
  --method mtp \
  -tp $TP 2>&1 | tee server_mtp.log &
```

### MI300X / MI308X

<a id="mi300x-mi308x-fp8"></a>

#### GLM-5.2 FP8 Server

```bash
#!/bin/bash

model_path=zai-org/GLM-5.2-FP8
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
TP=4

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --server-port 8000 \
  --kv_cache_dtype fp8 \
  --no-enable_prefix_caching \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate"]}' \
  -tp $TP 2>&1 | tee server.log &
```

#### GLM-5.2 FP8 MTP Server

```bash
#!/bin/bash

model_path=zai-org/GLM-5.2-FP8
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
TP=4

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --server-port 8004 \
  --kv_cache_dtype fp8 \
  --no-enable_prefix_caching \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate"]}' \
  --num-speculative-tokens 3 \
  --method mtp \
  -tp $TP 2>&1 | tee server_mtp.log &
```

### Offline Inference with DP Attention + Expert Parallel

```bash
#!/bin/bash

python -m atom.examples.simple_inference --model zai-org/GLM-5-FP8 -tp 8 --enable-dp-attention --enable-expert-parallel
```

Tips on server configuration:
- We suggest using fp8 kv cache for better memory efficiency in the serving mode.
- DP attention + EP MoE mode does not support fp8 kv cache when gqa=8, so `--kv_cache_dtype fp8` should not be used with `--enable-dp-attention --enable-expert-parallel`.
- GLM-5 reuses the DeepSeek v3 implementation in ATOM (MLA attention, MoE routing), so all DeepSeek v3 optimizations apply automatically.
- No `--trust-remote-code` is needed since ATOM has built-in support for `GlmMoeDsaForCausalLM`.



## Performance baseline

The following script can be used to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
    --model=zai-org/GLM-5-FP8 --backend=vllm --base-url=http://localhost:7777 \
    --dataset-name=random \
    --random-input-len=${ISL} --random-output-len=${OSL} \
    --random-range-ratio 1.0 \
    --num-prompts=$(( $CONC * 10 )) \
    --max-concurrency=$CONC \
    --request-rate=inf --ignore-eos \
    --save-result --result-dir=${result_dir} --result-filename=$RESULT_FILENAME.json \
    --percentile-metrics="ttft,tpot,itl,e2el"
```
The performance number on 8 ranks is provided as a reference, with the following environment:
- docker image: rocm/atom:latest.
- ATOM: zlr/glm5 branch.

| ISL  | OSL  | Concurrency | Num Prompts | Output Throughput (tok/s) | Total Throughput (tok/s) |
| ---- | ---- | ----------- | ----------- | ------------------------- | ------------------------ |
| 1024 | 1024 | 4           | 40          | 151.13                    | 303.73                   |
| 1024 | 1024 | 8           | 80          | 285.37                    | 568.63                   |
| 1024 | 1024 | 16          | 160         | 528.32                    | 1062.26                  |
| 1024 | 1024 | 32          | 320         | 925.64                    | 1848.35                  |
| 1024 | 1024 | 64          | 640         | 1605.75                   | 3212.22                  |
| 1024 | 1024 | 128         | 1280        | 2738.57                   | 5483.16                  |

Here are the steps to reinstall ATOM/AITER in the docker, if you are trying to verify with other specific commits:
```bash
# uninstall existing ATOM/AITER
pip uninstall -y atom amd-aiter

cd PATH_TO_ATOM
# normally ATOM is already installed in develop mode
# you may just do checkout without reinstall
git checkout specific_branch_or_commit
pip install -e .

cd PATH_TO_AITER
rm -rf aiter/jit/build aiter/jit/*.so
git checkout specific_branch_or_commit
git submodule sync && git submodule update --init --recursive
python setup.py develop
```

### Accuracy test
We verified the lm_eval accuracy on gsm8k dataset with command:
```bash
lm_eval \
--model local-completions \
--model_args model=zai-org/GLM-5-FP8,base_url=http://localhost:7777/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
--tasks gsm8k \
--num_fewshot 5
```

Here is the reference value when deploying on 8 ranks:
```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.93|±  |0.0256|
|     |       |strict-match    |     5|exact_match|↑  | 0.93|±  |0.0256|
```

## GLM-5.2 (IndexShare)

[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2-FP8) builds on the same `glm_moe_dsa` architecture as GLM-5 and adds **IndexShare**: the DSA indexer is computed only on `"full"` attention layers and reused by the following `"shared"` layers (the per-layer schedule is declared in `indexer_types`). Shared layers carry no indexer weights of their own. ATOM detects this schedule and enables the indexer cache automatically — no extra flags required.

Tips on server configuration:
- Use the FP8, MXFP4, or MXFP4 MTP server recipes above for GLM-5.2.
- Use `--kv_cache_dtype fp8` with the optimized GLM-5.2 server recipes unless you are intentionally comparing against the older bf16 KV-cache baseline.
- No `--trust-remote-code` is needed — ATOM has built-in support for `GlmMoeDsaForCausalLM`.

### Performance baseline

Reference numbers on 8×MI355X (TP8, FP8 weights, bf16 KV cache), using the benchmark command above with `--random-range-ratio 0.8`:

| ISL  | OSL  | Concurrency | Output Throughput (tok/s) | Total Throughput (tok/s) | Median TTFT (ms) | Median TPOT (ms) |
| ---- | ---- | ----------- | ------------------------- | ------------------------ | ---------------- | ---------------- |
| 1024 | 1024 | 1   | 79   | 158   | 102 | 12.5 |
| 1024 | 1024 | 16  | 841  | 1690  | 95  | 18.5 |
| 1024 | 1024 | 64  | 2074 | 4148  | 107 | 30.0 |
| 8192 | 1024 | 1   | 73   | 669   | 409 | 13.2 |
| 8192 | 1024 | 16  | 645  | 5818  | 418 | 23.3 |
| 8192 | 1024 | 64  | 1210 | 10853 | 483 | 51.3 |

## GLM-5.2 Prefill Context Parallel (PCP)

Prefill Context Parallel accelerates **long-context prefill** (large ISL) by
round-robin splitting the prompt tokens across an extra parallel dimension
(`world = tp × pcp`). Only the query side is sharded — every rank keeps the
**full KV cache**, so decode, the KV-cache layout, and accuracy are unchanged.
The dominant `O(S²)` DSA indexer scoring and the sparse MLA attention run on
`1/pcp` of the queries per rank, cutting TTFT on long inputs.

Enable it with `--prefill-context-parallel-size` (`-pcp`). `pcp` is orthogonal
to `-tp`, and the two multiply into the number of GPUs used
(`GPUs = tp × pcp`). MTP speculative decoding is supported — the draft's prefill
pass is split and gathered the same way.

### Serving on 4 GPUs (TP2 × PCP2)

```bash
model_path=amd/GLM-5.2-MXFP4
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --server-port 8000 \
  --kv_cache_dtype fp8 \
  --max-num-batched-tokens 32768 \
  -tp 2 -pcp 2 2>&1 | tee server_pcp.log &
```
