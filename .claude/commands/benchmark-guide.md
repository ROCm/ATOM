# Performance Benchmarking Guide

## Server Launch Examples

```bash
# Standard BF16
python -m atom.entrypoints.openai_server --model <model> --kv_cache_dtype fp8 -tp 8

# With MTP speculative decoding (improves throughput)
python -m atom.entrypoints.openai_server --model <model> --kv_cache_dtype fp8 -tp 8 \
  --method mtp --num-speculative-tokens 1

# With block-size tuning
python -m atom.entrypoints.openai_server --model <model> --kv_cache_dtype fp8 -tp 8 \
  --block-size 16
```

> **Before starting:** Kill old servers, clear compile cache (`rm -rf /root/.cache/atom/*`), set `AITER_LOG_LEVEL=WARNING`.

## Local Benchmark

```bash
# 1. Verify server is ready (VRAM must be > 0%)
curl -sf http://localhost:8000/v1/models
rocm-smi --showmemuse | grep "GPU Memory Allocated"

# 2. Run benchmark
python -m atom.benchmarks.benchmark_serving \
  --model=<model> --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random \                          # use synthetic random data
  --random-input-len=1024 --random-output-len=1024 \  # ISL/OSL
  --random-range-ratio=0.8 \                       # ±20% length variation
  --max-concurrency=128 \                          # concurrent request limit
  --num-prompts=1280 \                             # CONC*10, total requests
  --trust-remote-code \                            # required for custom models
  --num-warmups=256 \                              # CONC*2, warmup before measurement
  --request-rate=inf --ignore-eos \                # saturate server, fixed output length
  --save-result \                                  # write JSON result file
  --percentile-metrics="ttft,tpot,itl,e2el"        # metrics to report percentiles for
```

## Nightly Standard Parameters

CI Nightly uses the following ISL/OSL/concurrency combinations (aligned with InferenceX dsr1-fp8):

| Scenario | ISL | OSL | Concurrency |
|----------|-----|-----|-------------|
| Short-in Short-out | 1024 | 1024 | 1, 2, 4, 8, 16, 32, 64, 128, 256 |
| Short-in Long-out | 1024 | 8192 | 1, 2, 4, 8, 16, 32 |
| Long-in Short-out | 8192 | 1024 | 1, 2, 4, 8, 16, 32, 64, 128 |

## Profiling

```bash
# Start profiler (while server is running)
curl -X POST http://127.0.0.1:8000/start_profile

# Run tests...

# Stop profiler
curl -X POST http://127.0.0.1:8000/stop_profile
```

- Trace output directory: set via `ATOM_TORCH_PROFILER_DIR` env var
- Detailed profiling: set `ATOM_PROFILER_MORE=1` (enables `record_shapes`, `with_stack`, `profile_memory`)
- Offline profiling: `python -m atom.examples.profile_offline --model <model> --kv_cache_dtype fp8`

## Key Metrics

| Metric | Description |
|--------|-------------|
| TTFT | Time To First Token |
| TPOT | Time Per Output Token |
| ITL | Inter-Token Latency |
| E2EL | End-to-End Latency |
| Throughput | Output tokens/s |

## Critical Rules

- **Never run accuracy and performance tests simultaneously** — they interfere with each other's results. Always finish one before starting the other.
- **Report Total Token throughput (tok/s)**, not Output token throughput — Total includes input+output.
- **MTP models require `--use-chat-template`** — tokenizer mismatch without it causes inaccurate results.

## CI Benchmark Workflow

- File: `.github/workflows/atom-benchmark.yaml`
- Trigger: Nightly (01:00 Beijing time) or manual dispatch
- Models: DeepSeek-R1-0528, gpt-oss-120b
- Auto-compares against the last successful nightly run baseline
- Summary script: `.github/scripts/summarize.py`
