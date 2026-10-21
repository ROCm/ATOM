# Qwen3.5 Usage Guide

[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) is a large language model developed by the Qwen team from Alibaba Cloud. It features several key architectural innovations:
* A hybrid attention design mixing Full Attention and Gated-Delta-Net (GDN) linear attention with a 1:3 ratio (`full_attention_interval=4`)
* A highly sparse Mixture-of-Experts (MoE) structure with 512 routed experts
* Native Multi-Token Prediction (MTP) with 1 reusable draft layer for speculative decoding

## Preparing environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All the operations below will be executed inside the container.

## Launching Qwen3.5 with ATOM

### Serving on 4xMI355 GPUs (TP4, FP8 KV Cache)

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --kv_cache_dtype fp8 -tp 4
```

### Serving with MTP Speculative Decoding

MTP-1 reduces per-token latency with ~94% acceptance rate:

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --kv_cache_dtype fp8 -tp 4 \
  --method mtp --num-speculative-tokens 1
```

MTP-3 for higher throughput (~83% acceptance rate, 3.49 avg tokens/fwd):

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --kv_cache_dtype fp8 -tp 4 \
  --method mtp --num-speculative-tokens 3
```

## Performance Metrics

### Benchmarking

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=Qwen/Qwen3.5-397B-A17B-FP8 --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=1024 --random-output-len=1024 \
  --random-range-ratio=0.8 \
  --num-prompts=1280 --max-concurrency=128 \
  --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics="ttft,tpot,itl,e2el"
```

### Accuracy

Install `lm-eval` first:
```bash
pip install lm-eval[api]
```

Run the evaluation:
```bash
lm_eval --model local-completions \
  --model_args model=Qwen/Qwen3.5-397B-A17B-FP8,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 3
```

Reference values (TP4, FP8 KV Cache, MTP-1):
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     3|exact_match|↑  |0.8613|±  |0.0095|
|     |       |strict-match    |     3|exact_match|↑  |0.8484|±  |0.0099|
```

## Fused decode (mono)

On a TP4 deployment with fp8 KV cache and KV block size 16 (the commands above), every decoder layer of a
decode step of up to 16 rows runs as one persistent kernel launch (`atom/models/qwen3_5/mono/`): the Gated
DeltaNet or gated full attention, the MoE with its shared expert, and both TP all-reduces. A plain decode
serves up to 16 requests; an MTP verify (MTP-1 to MTP-3) serves whole requests (MTP-3: up to 4), on ReplaySSM
or the snapshot GDN state. With the FP8 checkpoint the MTP draft's decoder layer runs on the same kernel. FP8
and MXFP4 checkpoints are both supported. It is on by default; `ATOM_MONO_ENABLE=0` turns it off. Wider steps,
prefill and any other deployment keep the original path; the conditions are listed under `ATOM_MONO_ENABLE`
in the [environment variable guide](../docs/environment_variables.md).

Median TPOT on MI355X (TP4, fp8 KV, `--no-enable_prefix_caching`, ISL 8192 / OSL 1024, the client of
`.github/scripts/atom_test.sh`; the original path is the same server with `ATOM_MONO_ENABLE=0`, one node):

| Checkpoint | Concurrency | Original | Mono | Speedup |
|---|---|---|---|---|
| FP8 | 1 | 7.74 ms | 2.30 ms | 3.37x |
| FP8 | 4 | 8.66 ms | 4.19 ms | 2.07x |
| FP8 | 8 | 10.04 ms | 6.63 ms | 1.51x |
| FP8 | 16 | 13.65 ms | 11.61 ms | 1.18x |
| FP8, MTP-3 | 1 | 2.44 ms | 1.49 ms | 1.64x |
| FP8, MTP-3 | 4 | 3.57 ms | 3.08 ms | 1.16x |
| MXFP4 | 1 | 6.79 ms | 2.58 ms | 2.63x |
| MXFP4 | 4 | 7.99 ms | 4.04 ms | 1.98x |
| MXFP4 | 8 | 9.49 ms | 6.15 ms | 1.54x |
| MXFP4 | 16 | 12.42 ms | 10.30 ms | 1.21x |

MTP-3 at 8 and 16 requests verifies 32 and 64 rows a step, which the original path runs. GSM8K 3-shot
(flexible-extract, the accuracy CI's server) is within stderr (about 0.009) of the original path, mono /
original: FP8 0.8673 / 0.8734 at 65 concurrent requests and 0.8772 / 0.8681 at 8, MTP-3 0.8719 / 0.8726 at 65
and 0.8688 / 0.8719 at 4 (acceptance 84.5% / 84.4%), MXFP4 0.8552 / 0.8605 at 65 and 0.8514 / 0.8575 at 8.
