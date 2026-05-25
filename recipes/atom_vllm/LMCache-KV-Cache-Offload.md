# LMCache KV Cache Offload with ATOM vLLM Plugin Backend

This recipe shows how to use [LMCache](https://github.com/LMCache/LMCache) for CPU DRAM and NVMe KV cache offloading with the ATOM vLLM plugin backend. LMCache adds an L2 (CPU DRAM) and optional L3 (NVMe) caching tier on top of vLLM's HBM prefix cache, enabling higher concurrency for prefix-heavy workloads such as multi-turn agentic serving.

For background on the ATOM vLLM plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

## When to Use LMCache

LMCache is most beneficial for **agentic and multi-turn workloads** where:

- Conversations reuse 90–97% of their prefix across turns (e.g., Claude Code, Cursor, Devin)
- Many concurrent users compete for HBM KV cache capacity
- Input lengths are long (20K–150K tokens) with short outputs (100–500 tokens)

Under low concurrency where the working set fits in HBM, ATOM's native prefix cache is sufficient and LMCache adds unnecessary overhead.

## Benchmark Summary

Tested on MiniMax-M2.5 (456B MoE, FP8) with 2× AMD MI300X (TP=2), using 739 anonymized Claude Code agentic traces from [kv-cache-tester](https://github.com/callanjfox/kv-cache-tester).

**Stress test (32 users, 100K context, 20 min, `gpu-memory-utilization=0.78`):**

| Metric | ATOM HBM-only | ATOM + LMCache CPU | ATOM + LMCache CPU+NVMe |
|--------|:---:|:---:|:---:|
| Requests completed | 208 | 331 | 374 |
| TTFT avg | 84.0s | 47.1s | 38.7s |
| TTFT median | 80.1s | 32.9s | 34.2s |
| TTFT p95 | 207.3s | 150.2s | 88.2s |
| Input throughput (tok/s) | 6,975 | 11,664 | 13,462 |
| Output throughput (tok/s) | 82 | 120 | 144 |
| Cache hit rate | 75.6% | 81.3% | 83.4% |

LMCache CPU offload delivers **1.8× lower avg TTFT** and **59% more requests** vs HBM-only. Adding the NVMe tier further reduces **p95 TTFT by 41%**.

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Build LMCache from Source

The PyPI `lmcache` wheel ships CUDA-linked binaries that do not work on ROCm. Build from source with `BUILD_WITH_HIP=1` to get native HIP KV transfer kernels:

```bash
# Inside the container
git clone --depth 1 https://github.com/LMCache/LMCache.git /tmp/LMCache
cd /tmp/LMCache && BUILD_WITH_HIP=1 pip install -e . --no-build-isolation --no-deps

# Install required runtime dependencies (skip CUDA-only packages)
pip install aiofile aiofiles blake3 msgspec sortedcontainers redis pyzmq

# Uninstall any CUDA transitive deps that may have been pulled in
pip uninstall -y nixl nixl-cu12 nixl-cu13 cupy-cuda12x cupy-cuda13x \
    cufile-python cuda-pathfinder cuda-bindings cuda-toolkit 2>/dev/null
```

Verify the HIP backend loads:

```bash
python3 -c "import lmcache" 2>&1 | grep "Using backend"
# Expected: Using backend: lmcache.c_ops
# NOT:      Fallback to python backend lmcache.non_cuda_equivalents
```

> **Important:** If you see `non_cuda_equivalents`, the source build did not compile the HIP
> kernels. Re-run the build step and check for compilation errors. The Python fallback
> backend adds significant serialization overhead that negates the caching benefit.

## Step 3: Launch vLLM Server with LMCache

### CPU DRAM Offload (L2 tier)

```bash
export PYTHONHASHSEED=0
export VLLM_FLOAT32_MATMUL_PRECISION=high
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

# LMCache configuration
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=64.0  # GB of CPU DRAM for KV cache
export LMCACHE_CHUNK_SIZE=256

vllm serve <model_name_or_path> \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --gpu-memory-utilization 0.78 \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --max-num-batched-tokens 16384 \
    --async-scheduling \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### CPU DRAM + NVMe Offload (L2 + L3 tiers)

Add NVMe disk as a third caching tier below CPU DRAM:

```bash
# All the same env vars as above, plus:
export LMCACHE_LOCAL_DISK=file:///path/to/nvme/lmcache_disk
export LMCACHE_MAX_LOCAL_DISK_SIZE=200.0  # GB

# Same vllm serve command as above
```

Create the disk directory before launching: `mkdir -p /path/to/nvme/lmcache_disk`

### Key Configuration Notes

- **`PYTHONHASHSEED=0`** is mandatory. Without it, Python's randomized hash function causes TP workers to generate different cache keys for the same prompt, resulting in 0% cache hits.
- **`--enable-prefix-caching`** is required (not `--no-enable-prefix-caching`). LMCache uses vLLM's prefix-cache hash function for cache-key derivation.
- **`LMCACHE_MAX_LOCAL_CPU_SIZE`** should be sized generously — DRAM is cheap and evictions force recomputation. 64 GB is a good default for 2× MI300X.
- Do **not** set `LMCACHE_SAVE_DECODE_CACHE=true` — it synchronously offloads every decode step to CPU, which serializes the GPU pipeline and causes stalls.

## Step 4: Verify LMCache is Active

Check the server log for LMCache initialization on each TP worker:

```
(Worker_TP0) LMCache INFO: Using backend: lmcache.c_ops
(Worker_TP1) LMCache INFO: Using backend: lmcache.c_ops
```

Send the same prompt twice and verify cache hits:

```
LMCache: Reqid=...80e (1030 tok, 1st pass): hit tokens: 0      ← cold (correct)
LMCache: Reqid=...8cf (1030 tok, 2nd pass): hit tokens: 1024   ← warm hit ✅
```

## Step 5: Benchmark with Agentic Traces

For realistic evaluation, use the [kv-cache-tester](https://github.com/callanjfox/kv-cache-tester) trace replay tool with anonymized Claude Code conversation traces:

```bash
python3 trace_replay_tester.py \
    --api-endpoint http://127.0.0.1:8000 \
    --trace-directory traces \
    --output-dir results/ \
    --start-users 1 \
    --max-users 32 \
    --max-ttft 60.0 \
    --max-context 100000 \
    --test-duration 1200 \
    --recycle \
    --warm-prefix-pct 0.5 \
    --seed 42 \
    --tokenizer <model_name_or_path> \
    --skip-graphs --no-color
```

Use `--seed` for reproducible trace assignment when comparing configurations.

## Model Compatibility

LMCache works through vLLM's KV connector interface and is model-agnostic. It has been tested with:

- **MiniMax-M2.5** (MoE, TP=2/4)
- Other models supported by the ATOM vLLM plugin should work without modification

## References

- [LMCache GitHub](https://github.com/LMCache/LMCache)
- [LMCache Documentation](https://docs.lmcache.ai)
- [kv-cache-tester](https://github.com/callanjfox/kv-cache-tester) — agentic workload trace replay
- [Benchmarking LMCache for Multi-Turn Agentic Workloads on AMD MI300X](https://andyluo7.github.io/llm/amd/mi300x/vllm/lmcache/performance/2026/04/20/benchmarking-lmcache-multi-turn-agentic-mi300x/) — vanilla vLLM + LMCache benchmark
