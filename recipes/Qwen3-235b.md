# Qwen3-235B-A22B Usage Guide

[Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) is an advanced large language model created by the Qwen team from Alibaba Cloud. This is a guide on running the model on AMD GPUs with ATOM.
In particular, we focus on deploying the fp8 model of Qwen3-235B-A22B on MI355 in this guide.

## Preparing environment
Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:gfx950_latest
```
All the oparations in the next will be executed inside the container.

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
The performance number for both 4 and 8 ranks are provided as a baseline:
|              |               |                 |             | TP8 + EP8    |              |              | TP4 + EP4        |              |              |
| ------------ | ------------- | --------------- | ----------- | ------------ | ------------ | ---------------- | ------------ | ------------ | ---------------- |
| ISL | OSL | Concurrency | Num Prompts | Mean TTFT (ms) | Mean TPOT (ms) | Total Throughput | Mean TTFT (ms) | Mean TPOT (ms) | Total Throughput |
| 1000         | 1000          | 256             | 1024        | 2523.16      | 23.06        | 19994.9          | 3315.62      | 28.91        | 15879.98         |
| 1000         | 1000          | 128             | 512         | 1396.19      | 18.8         | 12670.42         | 1820.53      | 23.62        | 10061.07         |
| 4000         | 1000          | 128             | 512         | 4897.86      | 23.96        | 22173.88         | 6479.95      | 29.62        | 17725.37         |
| 4000         | 1000          | 64              | 256         | 2585.34      | 20.09        | 14111.39         | 3427.88      | 24.1         | 11629.43         |
| 10000        | 1000          | 64              | 256         | 6502.95      | 25.88        | 21741.39         | 8765.1       | 31.4         | 17529.47         |
| 10000        | 1000          | 32              | 128         | 3378.36      | 23.34        | 13179.94         | 4525.85      | 23.55        | 12544.81         |


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

