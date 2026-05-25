# DeepSeek-V4 Usage Guide

[DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) is a million-token-context Mixture-of-Experts (MoE) large language model from DeepSeek. It builds on the V3.2 architecture with hash-based expert routing (3 hash layers + sigmoid + bias), a Compressed Sparse Attention (CSA) indexer that selects top-1024 prior tokens per query, and Multi-Latent Attention (MLA) with LoRA-compressed QKV projections. Weights are stored natively in FP8 (E4M3) with UE8M0 block-scaled scales. ATOM ships built-in support via the `DeepseekV4ForCausalLM` architecture — no `--trust-remote-code` is needed.

## Preparing environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All the operations below will be executed inside the container.

## Launching server

### FP8 on 8xMI355X GPUs (TP8 + FP8 KV Cache)

```bash
ATOM_USE_TRITON_MOE=1 \
python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --kv_cache_dtype fp8 -tp 8
```

Tips on server configuration:
- **`ATOM_USE_TRITON_MOE=1` is required.** V4-Pro routes 6 experts out of 384 with hash-based selection; the triton MoE backend is the only path that handles the FP8 E4M3 + UE8M0 block-scaled weights correctly. Launching without this env silently falls back to a numerically incorrect path and GSM8K accuracy drops from ~0.95 to ~0.6.
- Use `--kv_cache_dtype fp8` for memory efficiency. The CSA indexer's compressed K cache is stored separately in FP8 regardless.
- Set `AITER_LOG_LEVEL=WARNING` before starting to suppress aiter kernel log noise.
- Clear compile cache before restarting after code changes: `rm -rf /root/.cache/atom/*`
- V4-Pro reuses the DeepSeek-V3 config schema; V4-specific fields (compress ratios, hash layers, index head dims) are read from the HF config automatically.

### PD Disaggregation with Mooncake (Prefill/Decode Separation)

Run prefill and decode on separate nodes with Mooncake RDMA KV cache transfer.

#### 1. Start Proxy (on producer node)

```bash
python -m atom.kv_transfer.disaggregation.proxy --port 10001
```

#### 2. Start Producer (prefill node)

```bash
export LOCAL_IP=<this-node-ip>

AITER_BF16_FP8_MOE_BOUND=0 \
ATOM_MOE_GU_ITLV=1 \
ATOM_DISABLE_MMAP=true \
NCCL_SOCKET_IFNAME=lo \
AITER_LOG_LEVEL=WARNING \
python -m atom.entrypoints.openai_server \
  --model /data/models/DeepSeek-V4-Pro/ \
  --kv_cache_dtype fp8 \
  -tp 8 \
  --server-port 8003 \
  --kv-transfer-config '{
    "kv_role": "kv_producer",
    "kv_connector": "mooncake",
    "proxy_ip": "'"${LOCAL_IP}"'",
    "proxy_ping_port": 36367,
    "http_port": 8003
  }' \
  2>&1 | tee producer.log
```

#### 3. Start Consumer (decode node)

```bash
export PRODUCER_IP=<producer-node-ip>

AITER_BF16_FP8_MOE_BOUND=0 \
ATOM_MOE_GU_ITLV=1 \
ATOM_DISABLE_MMAP=true \
NCCL_SOCKET_IFNAME=eno0 \
AITER_LOG_LEVEL=WARNING \
python -m atom.entrypoints.openai_server \
  --model /data/models/DeepSeek-V4-Pro/ \
  --kv_cache_dtype fp8 \
  -tp 8 \
  --server-port 8004 \
  --kv-transfer-config '{
    "kv_role": "kv_consumer",
    "kv_connector": "mooncake",
    "proxy_ip": "'"${PRODUCER_IP}"'",
    "proxy_ping_port": 36367,
    "http_port": 8004
  }' \
  2>&1 | tee consumer.log
```

#### 4. Send Requests

```bash
curl -s http://${PRODUCER_IP}:10001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"1 2 3 4 5","max_tokens":10,"temperature":0}'
```

> **Note:** `AITER_BF16_FP8_MOE_BOUND=0` and `ATOM_MOE_GU_ITLV=1` are required for V4-Pro's hash-routed MoE to work correctly in PD mode. See the [PD disaggregation guide](pd_disaggregation_guide.md) for architecture details and MORI-IO backend setup.

## Performance baseline

The following script can be used to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=deepseek-ai/DeepSeek-V4-Pro --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=${ISL} --random-output-len=${OSL} \
  --random-range-ratio=1.0 \
  --num-prompts=$(( $CONC * 10 )) \
  --max-concurrency=$CONC \
  --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics="ttft,tpot,itl,e2el"
```

Performance on 8xMI355X GPUs with the following environment:
- Date measured: 2026-05-23.
- Docker image: rocm/atom:latest.
- ATOM: `feat/v4-swa-write-tok-n-guard-opus-default` branch (commit bf9b133e).
- `ATOM_USE_TRITON_MOE=1`, `--kv_cache_dtype fp8`.

The numbers below are a snapshot. For the latest data tracked across commits, see [rocm.github.io/ATOM/benchmark-dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/).

### FP8 (TP8, FP8 KV Cache) — no MTP

| ISL  | OSL  | Concurrency | Num Prompts | Output Throughput (tok/s) | Total Throughput (tok/s) | Mean TPOT (ms) |
| ---- | ---- | ----------- | ----------- | ------------------------- | ------------------------ | -------------- |
| 1024 | 1024 | 4           | 40          | 195.31                    | 392.53                   | 19.66          |
| 1024 | 1024 | 8           | 80          | 367.43                    | 732.14                   | 21.09          |
| 1024 | 1024 | 16          | 160         | 668.02                    | 1343.15                  | 23.19          |
| 1024 | 1024 | 32          | 320         | 1145.71                   | 2287.81                  | 26.90          |
| 1024 | 1024 | 64          | 640         | 1808.69                   | 3618.19                  | 33.96          |
| 1024 | 1024 | 128         | 1280        | 2847.24                   | 5700.73                  | 43.26          |
| 1024 | 1024 | 256         | 2560        | 4289.93                   | 8575.71                  | 57.55          |

### FP8 (TP8, FP8 KV Cache) — MTP-3

Add `--method mtp --num-speculative-tokens 3` to the server launch. MTP-3
trades a small amount of memory for ~1.5–2× lower TPOT and ~1.3–1.5×
higher total throughput at the same concurrency.

| ISL  | OSL  | Concurrency | Num Prompts | Output Throughput (tok/s) | Total Throughput (tok/s) | Mean TPOT (ms) |
| ---- | ---- | ----------- | ----------- | ------------------------- | ------------------------ | -------------- |
| 1024 | 1024 | 4           | 40          | 528.46                    | 1061.26                  | 7.25           |
| 1024 | 1024 | 8           | 80          | 907.09                    | 1806.44                  | 8.17           |
| 1024 | 1024 | 16          | 160         | 1391.13                   | 2795.02                  | 10.95          |
| 1024 | 1024 | 32          | 320         | 2159.04                   | 4308.13                  | 14.01          |
| 1024 | 1024 | 64          | 640         | 3222.33                   | 6441.40                  | 18.75          |
| 1024 | 1024 | 128         | 1280        | 4376.29                   | 8755.90                  | 27.77          |
| 1024 | 1024 | 256         | 2560        | 5701.20                   | 11388.96                 | 43.06          |

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
  --model_args model=deepseek-ai/DeepSeek-V4-Pro,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k \
  --num_fewshot 5
```

Reference accuracy on 8xMI355X GPUs (FP8, FP8 KV Cache, `ATOM_USE_TRITON_MOE=1`,
measured 2026-05-23 at commit bf9b133e):

**no MTP**:
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9553|±  |0.0057|
|     |       |strict-match    |     5|exact_match|↑  |0.9560|±  |0.0056|
```

**MTP-3** (`--method mtp --num-speculative-tokens 3`, average acceptance ≈ 64.5%):
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9530|±  |0.0058|
|     |       |strict-match    |     5|exact_match|↑  |0.9538|±  |0.0058|
```
