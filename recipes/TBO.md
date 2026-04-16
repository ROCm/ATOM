# TBO (Two-Batch Overlap) Guide

In multi-GPU MoE serving, expert-parallel communication (all-to-all, all_gather/reduce_scatter) can become a significant bottleneck, especially in multi-node deployments with limited inter-node bandwidth. Following [DeepSeek's system design](https://arxiv.org/abs/2501.12948), ATOM implements **Two-Batch Overlap (TBO)** to hide this communication latency behind useful compute.

TBO splits each batch into two micro-batches (ubatches) and pipelines them so that one ubatch's MoE communication overlaps with the other's attention compute. This effectively masks the communication cost while also reducing peak memory usage by halving the active batch size per step.

```
  Thread 0 (ubatch 0):  |-Attn₀-||--D₀--||--MoE₀--||--C₀--||Attn₀..
  Thread 1 (ubatch 1):  |--D₁--||--MoE₁-||--C₁---||--Attn₁-|

  Compute stream:  |-Attn₀-||-MoE₁-||--MoE₀--||-Attn₁-|
  Comm stream:     |--D₁---||-D₀---||--C₁----||-C₀----|
                   └overlap┘└overlap┘└overlap─┘└overlap┘

  D = MoE dispatch (all-to-all send)    C = MoE combine (all-to-all recv)
```

The two ubatches' compute and communication are fully interleaved: while ubatch 0 runs attention on the compute stream, ubatch 1's MoE dispatch runs concurrently on the comm stream. Then ubatch 1's MoE compute runs while ubatch 0's dispatch overlaps on comm, and so on. By default, TBO is applied to prefill only (`--enable-tbo prefill`). Use `--enable-tbo all` to also enable it for decode, though this may regress throughput.

## When to use TBO

- **Best fit**: DP attention + Expert Parallel ([MORI](https://github.com/ROCm/mori) all-to-all) on large MoE models, especially multi-node where communication cost is high
- **Also works**: DP attention + all_gather/reduce_scatter (without EP)
- **Requires**: `--enable-dp-attention` and at least 2 GPUs (`-tp 2` or higher)

## CLI usage

```bash
--enable-tbo                   # prefill-only TBO (default, recommended)
--enable-tbo prefill           # explicit prefill-only TBO
--enable-tbo all               # TBO for both prefill and decode
```

| Mode | Scope | Description |
|---|---|---|
| `prefill` (default) | Prefill only | Recommended. Overlaps prefill MoE communication with no decode overhead |
| `all` | Prefill + Decode | Also overlaps decode phase. May regress throughput due to ubatch splitting overhead in decode |

For MORI all-to-all backend selection, use a separate flag:

```bash
--all2all-backend                  # high-throughput (default)
--all2all-backend high-throughput  # explicit high-throughput mode
--all2all-backend low-latency     # AsyncLL MORI kernel for low-latency overlap
```

| Backend | MORI Kernel | Description |
|---|---|---|
| `high-throughput` (default) | IntraNode dispatch/combine | Best for throughput-oriented workloads with large batches |
| `low-latency` | AsyncLL (`dispatch_send/recv`, `combine_send/recv`) | CU-free communication. Better when minimizing per-token latency matters |

## Launching server

### GPT-OSS-120B: DP + EP + TBO (2 GPUs)

```bash
MORI_SHMEM_MODE=ISOLATION \
python -m atom.entrypoints.openai_server \
  --model openai/gpt-oss-120b \
  -tp 2 --port 5678 --server-port 7777 \
  --gpu-memory-utilization 0.4 \
  --enable-dp-attention \
  --enable-expert-parallel \
  --enable-tbo
```

### GPT-OSS-120B: DP + TBO without EP (2 GPUs)

```bash
TORCH_NCCL_BLOCKING_WAIT=1 \
python -m atom.entrypoints.openai_server \
  --model openai/gpt-oss-120b \
  -tp 2 --port 5678 --server-port 7777 \
  --gpu-memory-utilization 0.4 \
  --enable-dp-attention \
  --enable-tbo
```

### DeepSeek-R1: DP + EP + TBO (8 GPUs)

```bash
MORI_SHMEM_MODE=ISOLATION \
python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-R1-0528 \
  -tp 8 --port 5678 --server-port 7777 \
  --gpu-memory-utilization 0.4 \
  --enable-dp-attention \
  --enable-expert-parallel \
  --enable-tbo
```

Tips:
- `MORI_SHMEM_MODE=ISOLATION` is required when using `--enable-expert-parallel`.
- `TORCH_NCCL_BLOCKING_WAIT=1` is recommended for the non-EP path (all_gather/reduce_scatter).
- `--torch-profiler-dir ./log` can be added to collect traces for performance analysis.

## Performance baseline

Benchmark against a running server:

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=openai/gpt-oss-120b --backend=vllm --base-url=http://localhost:7777 \
  --dataset-name=random \
  --random-input-len=1024 --random-output-len=1024 \
  --num-prompts=512 --max-concurrency=512 \
  --request-rate=inf --ignore-eos
```

### GPT-OSS-120B on 2xMI355X (ISL=1024, OSL=1024)

#### DP + EP MORI + TBO vs DP + EP MORI

| Concurrency | Configuration | Output Throughput (tok/s) | Total Throughput (tok/s) | TBO Impact |
|---|---|---|---|---|
| 256 | dp + ep mori | 13,660 | 27,320 | — |
| 256 | dp + ep mori + tbo (all) | 10,688 | 21,376 | -21.8% |
| 256 | dp + ep mori + tbo (prefill only) | 14,206 | 28,411 | **+4.0%** |
| 512 | dp + ep mori | 15,449 | 30,898 | — |
| 512 | dp + ep mori + tbo (all) | 14,109 | 28,219 | -8.7% |
| 512 | dp + ep mori + tbo (prefill only) | 18,067 | 36,134 | **+16.9%** |

#### DP + TBO vs DP (all_gather/reduce_scatter)

| Concurrency | Configuration | Output Throughput (tok/s) | Total Throughput (tok/s) | TBO Impact |
|---|---|---|---|---|
| 256 | dp | 15,066 | 30,132 | — |
| 256 | dp + tbo (all) | 11,714 | 23,427 | -22.2% |
| 256 | dp + tbo (prefill only) | 15,380 | 30,759 | **+2.1%** |
| 512 | dp | 17,560 | 35,120 | — |
| 512 | dp + tbo (all) | 14,898 | 29,796 | -15.2% |
| 512 | dp + tbo (prefill only) | 19,944 | 39,889 | **+13.6%** |

#### DP + EP MORI + TBO + MTP 3 (Speculative Decoding)

| Concurrency | Configuration | Output Throughput (tok/s) | Total Throughput (tok/s) | TBO Impact |
|---|---|---|---|---|
| 256 | dp + ep mori + mtp3 | 6,375 | 12,764 | — |
| 256 | dp + ep mori + tbo + mtp3 | 6,590 | 13,196 | **+3.4%** |

> **Note**: TBO on both prefill+decode shows a throughput regression due to ubatch splitting and synchronization overhead in the decode phase. **Prefill-only TBO** eliminates this regression and delivers positive gains: up to **+16.9%** with EP MORI and **+13.6%** without EP at 512 concurrency. TBO also stacks with MTP speculative decoding (**+3.4%**).

## How it works

1. The batch is split into 2 micro-batches (ubatches)
2. Two threads execute the model forward pass concurrently on separate streams
3. MoE communication (MORI all-to-all or NCCL all_gather) runs on a dedicated comm stream
4. While ubatch 0 does MoE communication, ubatch 1 runs attention compute, and vice versa
5. A CPU event ring synchronizes the ping-pong scheduling between threads, ensuring correct ordering without busy-waiting
6. When decode TBO is enabled (`--enable-tbo all`), CUDAGraph capture records the entire dual-thread execution (including stream synchronization events) into a single graph for efficient replay during decode
