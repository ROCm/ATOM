# Qwen3-235B-A22B Usage Guide

[Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) is an advanced large language model created by the Qwen team from Alibaba Cloud. This is a guide on running the model on AMD GPUs with ATOM.
In particular, we focus on deploying the fp8 model of Qwen3-235B-A22B on MI355 in this guide.

## Preparing environment
Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:rocm_7.2_preview_gfx950_latest
```
All the operations in the next will be executed inside the container.

## Launching server
ATOM supports running the model with different parallelism, e.g., tensor parallel, expert parallel, data parallel.
Here we consider the parallelism of TP8 + EP8 as an example. 

### Serving on 8xMI355 GPUs

```bash
#!/bin/bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

python -m atom.entrypoints.openai_server --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 -tp 8 --kv_cache_dtype fp8 --enable-expert-parallel --max-model-len 16384 --max-num-batched-tokens 20000
```
Tips on server configuration:
- We suggest always using fp8 kv cache for better memory efficiency.
- Quick allreduce is enabled for prefill to reduce TTFT.
- QK norm + rope + cache quant are fused into one kernel by enabling ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1.
- The max-model-len is set to secure the performance of gluon pa decode kernel, which will be used when bs=64.
- The max-num-batched-tokens is set based on our benchmark settings, i.e., ISL is selected from [1000,4000,10000]. This argument will affect TTFT and users can adjust it according to the scenarios.



## Performance baseline

We used the following script to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
    --model=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 --backend=vllm --base-url=http://localhost:$PORT \
    --dataset-name=random \
    --random-input-len=${ISL} --random-output-len=${OSL} \
    --random-range-ratio 1.0 \
    --num-prompts=$(( $CONC * 4 )) \
    --max-concurrency=$CONC \
    --request-rate=inf --ignore-eos \
    --save-result --result-dir=${result_dir} --result-filename=$RESULT_FILENAME.json \
    --percentile-metrics="ttft,tpot,itl,e2el"
```
The performance number for both 4 and 8 ranks are provided as a reference, with the following environment:
- docker image: rocm/atom:rocm_7.2_preview_gfx950_latest, pulled on 1/7/2026.
- ATOM: the one comes with the image, main branch, commit 2f713064。
- AITER: yadai/moe_tile_config_v1, commit 88857df.

|              |               |                 |             | TP8 + EP8    |              |              | TP4 + EP4        |              |              |
| ------------ | ------------- | --------------- | ----------- | ------------ | ------------ | ---------------- | ------------ | ------------ | ---------------- |
| ISL | OSL | Concurrency | Num Prompts | Mean TTFT (ms) | Mean TPOT (ms) | Total Throughput | Mean TTFT (ms) | Mean TPOT (ms) | Total Throughput |
| 1000  | 1000 | 256 | 1024 | 2452.83 | 20.55 | 22239.78 | 3214.23 | 27.79 | 16506.71 |
| 1000  | 1000 | 128 | 512  | 1326.08 | 16.47 | 14380.66 | 1743.22 | 21.15 | 11182.11 |
| 4000  | 1000 | 128 | 512  | 4656.93 | 21.32 | 24632.52 | 6215.98 | 26.79 | 19388.74 |
| 4000  | 1000 | 64  | 256  | 2444.89 | 17.52 | 16024.13 | 3252.3  | 19.88 | 13837.46 |
| 10000 | 1000 | 64  | 256  | 6144.75 | 22.8  | 24318.56 | 8340.75 | 26.84 | 20015.67 |
| 10000 | 1000 | 32  | 128  | 3208.08 | 19.08 | 15799.19 | 4330.11 | 21.44 | 13663.56 |

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
--model_args model=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8,base_url=http://localhost:8000/v1/completions,num_concurrent=100,max_retries=3,tokenized_requests=False \
--tasks gsm8k \
--num_fewshot 5
```

Here is the reference value when deploying on 8 ranks:
```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8810|±  |0.0089|
|     |       |strict-match    |     5|exact_match|↑  |0.8719|±  |0.0092|
```

