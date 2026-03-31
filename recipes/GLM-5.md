# GLM-5 Usage Guide

[GLM-5](https://huggingface.co/zai-org/GLM-5-FP8) is an advanced Mixture-of-Experts (MoE) large language model developed by Zhipu AI (THUDM). Its architecture is structurally similar to DeepSeek v3.2, featuring Multi-head Latent Attention (MLA). This guide covers deploying the FP8 version of GLM-5 on AMD GPUs with ATOM.

## Preparing environment
Pull the latest docker from https://hub.docker.com/r/rocm/atom-dev/ :
```bash
docker pull rocm/atom-dev:latest
```
All the operations in the next will be executed inside the container.

## Launching server
ATOM supports running the model with different parallelism, e.g., tensor parallel, expert parallel, data parallel.

### Serving on 8xMI355 GPUs (TP8 + FP8 KV Cache)

```bash
#!/bin/bash

python -m atom.entrypoints.openai_server --model zai-org/GLM-5-FP8 -tp 8 --kv_cache_dtype fp8 --port 5678 --server-port 7777
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
