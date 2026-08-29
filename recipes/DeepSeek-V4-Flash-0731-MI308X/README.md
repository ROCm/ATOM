# DeepSeek-V4-Flash-0731 / MI308X Optimization and Reproduction Guide

## Performance Summary and Quick Start

This recipe targets four MI308X GPUs and keeps the topology at TP4/DP1/PCP1. It does not obtain performance gains by using additional GPUs. Across 54 matched scenarios against the original DeepSeek-V4-Flash-0731 deployment:

- Throughput improved in 54/54 scenarios, with a **133.58%** geometric-mean gain.
- TTFT improved in 53/54 scenarios, with a **60.50%** geometric-mean improvement.
- TPOT improved in 54/54 scenarios, with a **58.23%** geometric-mean improvement.

The complete recipe includes ATOM source changes, AITER kernel policies, measured tuning tables, a digest-pinned Dockerfile, and a production entrypoint.

Build the image from the ATOM repository root:

```bash
docker build \
  -f recipes/DeepSeek-V4-Flash-0731-MI308X/Dockerfile \
  -t dsv4-0731-atom:pr-repro .
```

Start the server:

```bash
export MODEL_DIR=/volumes/oss1/models

docker run -d \
  --ipc=host --network=host --privileged \
  --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --group-add video --shm-size=128G \
  --name atom_dsv4_0731_optimized \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -v "$MODEL_DIR":/data/models \
  dsv4-0731-atom:pr-repro
```

benchmark test:

```bash
docker exec atom_dsv4_0731_optimized \
  python -m atom.benchmarks.benchmark_serving \
    --model=DeepSeek-V4-Flash-0731 \
    --backend=vllm \
    --base-url=http://localhost:8000 \
    --dataset-name=random \
    --tokenizer=/data/models/DeepSeek-V4-Flash-0731 \
    --random-input-len=1000 \
    --random-output-len=1024 \
    --random-range-ratio=1.0 \
    --num-prompts=10 \
    --max-concurrency=1 \
    --request-rate=inf \
    --ignore-eos \
    --save-result \
    --percentile-metrics=ttft,tpot,itl,e2el
```

This command runs only one short scenario. It verifies server availability and reports basic TTFT, TPOT, and throughput without relying on a private benchmark script. A performance matrix can be built by varying the concurrency, input length, and number of prompts.

## Representative Performance Results

| Scenario | Original TPS | Optimized TPS | TPS Gain | Original TTFT | Optimized TTFT | Original TPOT | Optimized TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1/i1K | 222.97 | 310.01 | +39.04% | 924.24 ms | 199.40 ms | 7.97 ms | 6.19 ms |
| c8/i64K | 4,142.35 | 11,955.90 | +188.62% | 24,409.76 ms | 8,963.06 ms | 98.70 ms | 33.51 ms |
| c64/i1K | 2,704.59 | 5,324.71 | +96.88% | 2,088.11 ms | 1,300.71 ms | 42.99 ms | 20.86 ms |
| c128/i64K | 4,530.69 | 13,976.16 | +208.47% | 543,958.47 ms | 164,461.74 ms | 1,260.54 ms | 416.92 ms |
| c256/i64K | 4,456.30 | 13,931.48 | +212.62% | 1,814,569.44 ms | 581,164.43 ms | 1,770.24 ms | 431.00 ms |

## Included Optimizations

This directory contains the following reproducible components:

- Cyclic, balanced DeepSeek-V4 FP8 prefill-indexer row sharding.
- Online conversion support for MXFP4 source weights.
- DSV4 QKV padding, MoE, prefill, and sparse-attention path optimizations.
- AITER HCA, FP8 MQA, MHC, and sparse-attention policy optimizations.
- Measured GEMM, BF16, blockscale, and FMoE tuning tables.
- A Dockerfile with a digest-pinned base image.
- An `entrypoint.sh` containing the measured server configuration.

The indexer assigns query rows cyclically across tensor-parallel ranks to balance causal-window work. Each rank gathers only compact `int32` top-k output, and the original token order is restored after the collective. The decode path is unchanged. Enable the feature with:

```bash
ATOM_INDEXER_PREFILL_ROW_SHARD=1
```

### Row-Sharding Contribution

The isolated row-sharding comparison used TP4/DP1/PCP1, MTP3, MBT 131072, and the `c8/i64K/o1024` scenario:

| Variant | Total Throughput | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|
| Before row sharding | 10,819.48 TPS | 26,338.50 ms | 20.57 ms |
| Balanced cyclic row sharding | 11,749.00 TPS | 24,122.95 ms | 19.19 ms |
| Improvement | **+8.59%** | **8.41% faster** | **6.71% faster** |

The initial contiguous row-sharding implementation reached 11,526.21 TPS, 24,402.79 ms TTFT, and 19.50 ms TPOT. Cyclic sharding added another 1.93% throughput and improved TTFT and TPOT by 1.15% and 1.59%, respectively.

## Optimized Deployment Details

The Dockerfile pins the following base image:

```text
rocm/atom-dev:nightly_202608201458@sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976
```

The build copies the current repository, applies the AITER changes and tuning tables stored in this directory, and installs the current ATOM source in editable mode. It does not download another ATOM checkout.

The image entrypoint contains the following measured configuration:

- TP4/DP1/PCP1.
- MTP6.
- BF16 KV cache.
- FP8 index cache.
- MBT 131072.
- Prefill chunk size 16384.
- Maximum active sequences 128.
- Online per-block FP8 quantization for expert weights.
- Prefix caching disabled.
- Balanced indexer row sharding enabled.

Monitor server startup with:

```bash
docker logs -f atom_dsv4_0731_optimized
```

`--max-num-seqs 128` limits the number of sequences executed concurrently by the engine. It does not prevent the benchmark client from submitting 256 concurrent requests; excess requests wait in the scheduler queue. A value of 128 is part of the measured best configuration.

## Host PTL Configuration

The measured MI308X host used PTL mode `VECTOR,F8`. Check the current setting without modifying it:

```bash
amd-smi static --limit --json
```

PTL is a host-level setting and is not modified by the Dockerfile. If the current value is not `VECTOR,F8`, configure it according to the host's operational policy.

## Original Deployment and Performance

The original deployment used `rocm/atom-dev:nightly_202608161502` with TP4, FP8 KV cache, MTP2, GPU memory utilization 0.83, and prefix caching disabled.

```bash
export MODEL_DIR=/volumes/oss1/models

docker run -it \
  --ipc=host --network=host --privileged \
  --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --group-add video --shm-size=128G \
  --name atom_dsv4_0731 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e NCCL_IB_GID_INDEX=3 \
  -w /workspace \
  -v "$MODEL_DIR":/data/models \
  rocm/atom-dev:nightly_202608161502
```

Run inside the container:

```bash
export ATOM_USE_TRITON_MOE=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export PYTHONUNBUFFERED=1

python -u -m atom.entrypoints.openai_server \
  --model /data/models/DeepSeek-V4-Flash-0731 \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --server-port 8000 \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --method mtp \
  --num-speculative-tokens 2 \
  --gpu-memory-utilization 0.83 \
  --no-enable-prefix-caching
```

The original matrix covered concurrency `1/8/16/32/64/128/256`, input lengths `1K/3K/8K/12K/16K/32K/64K/128K`, and output length 1024.

| Scenario | Original Throughput | Original TTFT | Original TPOT |
|---|---:|---:|---:|
| c1/i1K | 222.97 TPS | 924.24 ms | 7.97 ms |
| c8/i64K | 4,142.35 TPS | 24,409.76 ms | 98.70 ms |
| c64/i1K | 2,704.59 TPS | 2,088.11 ms | 42.99 ms |
| c128/i64K | 4,530.69 TPS | 543,958.47 ms | 1,260.54 ms |
| c256/i64K | 4,456.30 TPS | 1,814,569.44 ms | 1,770.24 ms |
