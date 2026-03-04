# Qwen3-NEXT Usage Guide

[Qwen3-NEXT](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) is an advanced large language model created by the Qwen team from Alibaba Cloud.  It features several key improvements:
* A Gated-Delta-Net (GDN) structure
* A highly sparse Mixture-of-Experts (MoE) structure
* A Zero-Centered RMSNorm structure
* A multi-token prediction mechanism for faster inference

## Preparing environment
Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All the operations in the next will be executed inside the container.

## Launching Qwen3-Next with ATOM

You can use 8x MI308/MI355/MI350 to launch this model. Here we consider the parallelism of TP8 as an example.


### Serving on 8xMI355 GPUs

```bash
#!/bin/bash

python -m atom.entrypoints.openai_server --model Qwen/Qwen3-Next-80B-A3B-Instruct -tp 8
```

### Advanced Configuration with MTP

`Qwen3-Next` also supports Multi-Token Prediction (MTP in short), you can launch the model server with the following arguments to enable MTP.

```bash
python -m atom.entrypoints.openai_server --model Qwen/Qwen3-Next-80B-A3B-Instruct -tp 8 --method mtp
```


## Performance Metrics

### Benchmarking

We use the following script to demonstrate how to benchmark `Qwen/Qwen3-Next-80B-A3B-Instruct`.

```bash
python -m atom.benchmarks.benchmark_serving \
    --model=Qwen/Qwen3-Next-80B-A3B-Instruct --backend=vllm --base-url=http://localhost:$8000 \
    --dataset-name=random \
    --random-input-len=2048 --random-output-len=1024 \
    --random-range-ratio 1.0 \
    --num-prompts=100 \
    --max-concurrency=4 \
    --request-rate=inf --ignore-eos \
    --save-result --percentile-metrics="ttft,tpot,itl,e2el" \
    --result-dir=./ --result-filename="result.json"
```

### Accuracy

We use gsm8k dataset for accuracy test, please Install `lm-eval` firstly.

```
pip install lm-eval[api]
```

Run the evaluation:

```bash
lm_eval --model local-completions \
        --model_args model=/mnt/raid0/models/Qwen3-Next-80B-A3B-Instruct,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
        --tasks gsm8k
```

Here is the reference value when deploying on 8 ranks:
```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8604|±  |0.0093|
|     |       |strict-match    |     5|exact_match|↑  |0.8399|±  |0.0098|
```

