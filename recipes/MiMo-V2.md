# MiMo-V2-Flash MiMo-V2.5-Pro Usage Guide

[MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash) is a high-performance Mixture-of-Experts (MoE) large language model developed by Xiaomi. It features several key architectural innovations:
* A hybrid attention design mixing Full Attention and Sliding Window Attention (SWA) with a 1:5 ratio
* A highly sparse MoE structure with 256 routed experts and sigmoid top-8 routing with 309B total parameters and 15B active parameters
* Natively trained Multi-Token Prediction (MTP) with 3 independent draft layers for speculative decoding

[MiMo-V2.5-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro) is a 1.02T-parameter Mixture-of-Experts model with 42B active parameters, built on a hybrid-attention architecture with a 1M-token context window.

## Preparing environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All the operations below will be executed inside the container.

## Launching server

### Serving MiMo-V2-Flash on 4xMI355X GPUs (TP4, FP8 KV Cache)

```bash
python -m atom.entrypoints.openai_server \
  --model XiaomiMiMo/MiMo-V2-Flash \
  --kv_cache_dtype fp8 -tp 4 --trust-remote-code
```

### Serving MiMo-V2.5-Pro on 8xMI355X GPUs (TP8, FP8 KV Cache)

```bash
python -m atom.entrypoints.openai_server \
  --model XiaomiMiMo/MiMo-V2-Pro \
  --kv_cache_dtype fp8 -tp 8 --trust-remote-code
```

### Serving MiMo-V2-Flash on 4xMI355X GPUs (TP4, BF16 KV Cache)

```bash
python -m atom.entrypoints.openai_server \
  --model XiaomiMiMo/MiMo-V2-Flash \
  -tp 4 --trust-remote-code
```

### Serving MiMo-V2-Flash with MTP Speculative Decoding

```bash
# only support num-speculative-tokens=1 now
python -m atom.entrypoints.openai_server \
  --model XiaomiMiMo/MiMo-V2-Flash \
  --kv_cache_dtype fp8 -tp 4 --trust-remote-code \
  --method mtp
```

### Serving MiMo-V2.5-Pro with MTP Speculative Decoding

```bash
# only support num-speculative-tokens=1 now
python -m atom.entrypoints.openai_server \
  --model XiaomiMiMo/MiMo-V2.5-Pro \
  --kv_cache_dtype fp8 -tp 8 --trust-remote-code \
  --method mtp
```

## Performance Metrics

The following script can be used to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=XiaomiMiMo/MiMo-V2-Flash --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=${ISL} --random-output-len=${OSL} \
  --random-range-ratio=0.8 \
  --num-prompts=$(( $CONC * 10 )) \
  --max-concurrency=$CONC \
  --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics="ttft,tpot,itl,e2el"
```

### Accuracy test

We use gsm8k dataset for accuracy test. Install `lm-eval` first:

```bash
pip install lm-eval[api]
```

Run the evaluation for MiMo-V2-Flash:

```bash
lm_eval \
  --model local-completions \
  --model_args model=XiaomiMiMo/MiMo-V2-Flash,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k \
  --num_fewshot 5
```
Here is the reference value when deploying with tp4 fp8 kvcache:
```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8279|±  |0.0104|
|     |       |strict-match    |     5|exact_match|↑  |0.8211|±  |0.0106|
```

Run the evaluation for MiMo-V2.5-Pro:
```bash
lm_eval \
  --model local-completions \
  --model_args model=XiaomiMiMo/MiMo-V2.5-Pro,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k \
  --num_fewshot 5
```
Here is the reference value when deploying with tp8 fp8 kvcache mtp1:
```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9401|±  |0.0065|
|     |       |strict-match    |     5|exact_match|↑  |0.9386|±  |0.0066|
```