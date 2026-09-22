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
  --model XiaomiMiMo/MiMo-V2.5-Pro \
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

### Serving MiMo-V2.5-Pro with the SGLang plugin

The configuration below uses 8xMI355X GPUs, FP8 KV cache, chunked prefill,
and one-step NEXTN/MTP decoding. It is validated with the ATOM SGLang development
image `rocm/atom-dev:sglang-v0.5.17-nightly_20260921`, using this ATOM source tree
and the image's unmodified SGLang 0.5.17. Install the ATOM checkout with
`pip install -e .` inside the image before launching. The ATOM plugin supplies
the MiMo MTP prefill compatibility path needed by this SGLang release.

Set the socket interface names to an interface present on your host; the
validated host uses `eno0`.

```bash
export GLOO_SOCKET_IFNAME=eno0
export NCCL_SOCKET_IFNAME=eno0
export MORI_SOCKET_IFNAME=eno0
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_USE_AITER=1
export SGLANG_ENABLE_TORCH_COMPILE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_USE_AITER_UNIFIED_ATTN=1
export SGLANG_AITER_UNIFIED_VERIFY=1
export TORCHINDUCTOR_COMPILE_THREADS=128
export ATOM_FORCE_ATTN_TRITON=1
export ATOM_LOADER_PREFETCH=false
export ATOM_LOADER_NUM_THREADS=1
unset SGLANG_AITER_KV_CACHE_LAYOUT

python3 -m sglang.launch_server \
  --model XiaomiMiMo/MiMo-V2.5-Pro \
  --host localhost --port 10086 \
  --watchdog-timeout 1800 --trust-remote-code \
  --tp-size 8 --mem-fraction-static 0.8 --max-running-requests 48 \
  --disable-radix-cache --disable-hybrid-swa-memory \
  --attention-backend aiter --speculative-draft-attention-backend aiter \
  --page-size 64 --chunked-prefill-size 4096 --kv-cache-dtype fp8_e4m3 \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":4}' \
  --weight-loader-disable-mmap --skip-server-warmup \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path XiaomiMiMo/MiMo-V2.5-Pro \
  --speculative-num-steps 1 --speculative-num-draft-tokens 1 \
  --speculative-eagle-topk 1
```

- Use the same model ID for the draft: this checkpoint contains the MiMo MTP
  weights. SGLang constructs a separate draft runner, and its weight-loading
  pass can read the checkpoint again. This configuration does not merge the
  target and draft loading passes.
- The MiMo wrapper honors SGLang's weight iterator and loader options.
  `--weight-loader-disable-mmap` requests eager reads instead of memory mapping;
  the four shard-reader threads and one ATOM staging thread are the tested
  settings. Reader concurrency and host RAM usage should be considered together
  when tuning another machine.
- `--disable-hybrid-swa-memory` works around excessive SWA-pool retractions in
  SGLang 0.5.17. It allocates an ordinary KV pool while retaining MiMo's
  sliding-window attention calculation. It uses more KV memory than a hybrid
  pool. SGLang's later SWA allocator changes should be evaluated separately
  before removing this flag on a newer version.
- Keep `--speculative-num-steps 1` and `--speculative-eagle-topk 1` for this
  configuration. Tune the request limit and memory fraction together; 64 client
  requests can queue behind the server's 48-request limit.
- This configuration captures target-verification CUDA graphs without enabling
  SGLang's full `torch.compile` mode. In 0.5.17,
  `SGLANG_ENABLE_TORCH_COMPILE` is overwritten from the `--enable-torch-compile`
  CLI setting, so the environment variable alone does not enable that mode.
  The CLI flag is not included in this validated configuration.

The workload used to validate serving is:

```bash
python3 -m sglang.bench_serving \
  --backend sglang --model XiaomiMiMo/MiMo-V2.5-Pro \
  --host localhost --port 10086 \
  --dataset-name random --random-input-len 4096 \
  --random-output-len 1024 --random-range-ratio 1.0 \
  --flush-cache --seed 12345 --num-prompts 640 \
  --warmup-requests 128 --max-concurrency 64
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
