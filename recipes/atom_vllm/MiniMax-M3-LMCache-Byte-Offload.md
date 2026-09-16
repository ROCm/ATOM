# MiniMax-M3 — LMCache KV offload on the vLLM plugin (byte codec)

MiniMax-M3 cannot use LMCache's own GPU connector. This recipe uses
`AtomLMCacheOffloadConnector`, which drives ATOM's `DenseKVByteCodec` from vLLM's
KV-connector API and leaves LMCache as a pure byte store.

For the generic plugin + `LMCacheConnectorV1` path (works on M2.5 and other dense
models), see [LMCache KV Cache Offload](LMCache-KV-Cache-Offload.md). That path
**does not work on M3** — see *Why a separate connector* below.

## Why a separate connector

M3 registers 117 KV tensors in three different physical layouts at once:

| layers | shape | note |
|---|---|---|
| 3 dense | `(nb, 1, 128, 256)` | K and V interleaved per token |
| 57 sparse | `(nb, 2, 128, 1, 128)` | K and V in two **separate** regions |
| 57 index caches | `(nb, 128, 128)` fp8 | DSA indexer keys |

LMCache's `normalize_kv_and_discover_format()` probes for one **global** format,
so it aborts with `currently unsupported kv_caches format with list depth 1 and
tensor dimension 4`. The per-layer-format connector (V3) is off by default and
hangs on M3; the multi-process path needs cupy, which LMCache's `platform/rocm`
does not provide.

`AtomLMCacheOffloadConnector` sidesteps the whole question: ATOM gathers whole
paged blocks into a chunk-major uint8 blob and LMCache only ever stores opaque
bytes, so no format probe ever runs.

A sparse layer is also not contiguous as a whole (`stride(1)` jumps the entire K
region), so it is split into `t[:, 0]` / `t[:, 1]` before handing it to the codec.
M3's fp8 KV scales live on the layer, not in vLLM's `kv_caches` dict, and are
fetched through `get_kv_transfer_scales()` — moving mantissas without them
dequantises a restored block against the previous occupant's scale, which is
silent corruption.

## Launch

```bash
export PYTHONHASHSEED=0              # mandatory, see Gotchas
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=20 # GiB **per TP rank**
export LMCACHE_CHUNK_SIZE=128        # must equal --block-size
export OFFLOAD_MIN_LOAD_TOKENS=256   # default 8192 disables the tier for chat-sized prompts
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

vllm serve /path/to/MiniMax-M3-MXFP4 \
  --served-model-name amd/MiniMax-M3-MXFP4 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8 \
  --block-size 128 \
  --max-model-len 131072 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 32768 \
  --kv-cache-dtype fp8 \
  --no-async-scheduling \
  --language-model-only \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector","kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector","kv_role":"kv_both"}'
```

Select the connector through vLLM's **out-of-tree entry point** (the
`kv_connector_module_path` above). vLLM validates `kv_transfer_config` while
building VllmConfig, which happens *before* platform plugins load, so naming the
connector without the module path fails config validation.

## Verify it is actually on

Four independent checks — all of them, because each can pass for the wrong reason:

```bash
# 1. vLLM's own factory, on every worker AND the EngineCore
grep "Creating v1 connector with name: AtomLMCacheOffloadConnector" server.log

# 2. the codec saw the whole model
grep "ATOM LMCache offload: registered 60 layers" server.log

# 3+4. the tier is queried AND returns data (must be > 0)
curl -s localhost:8000/metrics | grep -E 'external_prefix_cache_(hits|queries)'
```

Two identities must hold on any interval; they catch a miscounting tier that
still looks plausible:

```
prefix_cache_queries - prefix_cache_hits == external_prefix_cache_queries
prefix_cache_hits    + external_prefix_cache_hits == prompt_tokens_cached
```

The first says the two tiers are strictly serial (HBM first, LMCache only gets
what HBM missed); the second says nothing is double-counted.

## Measured (MiniMax-M3-MXFP4, TP4, MI355X)

Synthetic prefix-reuse load: 80,896-token shared prefix + 9,270 unique tail,
32 requests, concurrency 2, pool of 4 prefixes, KV capped to 262,144 tokens.
Paired runs — same seeds, same server flags, LMCache the only difference.

| | LMCache on | LMCache off |
|---|---:|---:|
| Total cache read | **76.65%** | 55.17% |
| external (LMCache) supplied | 629,504 tok | 0 |
| **TTFT** | **1,247 ms** | 1,766 ms |
| Output throughput | 133.19 tok/s | 125.18 tok/s |

**TTFT 29% lower, throughput 6% higher.** Run-to-run noise on this
configuration is 0.86% (throughput) and 0.11% (TTFT), so the effect is far
outside it.

Cross-checked against an independent source: aiperf's
`usage.prompt_tokens_details.cached_tokens` summed to 2,211,584 against the
server's `prompt_tokens_cached` delta of 2,211,584 — exact.

### Sizing matters more than anything else here

The same code on a larger KV pool measures **56.81%** instead of 76.65%, because
HBM alone then holds ~2 of the 8 prefixes and LMCache only patches the edges.
Work out the budget before benchmarking:

```
free for caching = KV pool - concurrency x ISL
```

At KV 524,288 / conc 4 / ISL 90,166 that leaves 163,624 tokens = 2.02 prefixes;
at KV 262,144 / conc 2 it leaves 81,812 = 1.01 prefixes, which pushes ~75% of all
reuse onto LMCache. **If HBM can hold the whole working set, this tier cannot
help and a benchmark will show nothing.**

Also compute the ceiling before calling a number low:

```
ceiling = prefix/ISL - (pool x prefix)/(ISL x requests)
```

which is 78.50% for the config above — the measured 76.65% is **97.6% of what is
physically reachable**. The per-request unique tail can never hit, and each
prefix must be computed once. Published agentic numbers (~96%) come from
multi-turn replay where the unique tail is a small fraction of each turn; they
are not comparable to a synthetic prefix load.

### Agentic replay, and what an oversized pool looks like

Same server, `--public-dataset semianalysis_cc_traces_weka_062126`, 8 trajectory
sessions, 900 s, but with the **default KV pool (7,610,112 tokens)**:

| | |
|---|---:|
| Total cache read (aiperf) | **97.12%** |
| HBM tier | 91.36% |
| external tier queried | 2,578,390 tok |
| **external tier hit** | **0** |

Two things worth reading carefully.

**97.12% is the number to compare against published agentic results (~96%)** —
same kind of workload, same kind of measurement. The 76.65% from the synthetic
load above is not a worse result, it is a different ceiling.

**The external tier hit nothing, and that is correct here.** The pool holds
7.6M tokens, so HBM alone absorbs 91.36%; what it misses is content appearing
for the first time, which no cache can hold. The connector was fully live — it
was queried 2,578,390 times, exactly the HBM miss volume — it simply had no
reusable bytes to offer. **An oversized pool makes this tier look useless.**
Cap the pool below the working set before concluding anything about it.

**Do not compare the two sides' absolute token counts on an agentic run.** Here
the server reported 27,274,624 cached tokens and aiperf 10,992,256 — a 2.48x
gap, where every other run in this recipe matched exactly. The denominators
differ: aiperf counts only the profiling phase, the server counters also include
warmup and trajectory reconstruction. Percentages remain comparable; absolute
counts do not.

## Gotchas

- **`PYTHONHASHSEED=0` is mandatory.** Without it each TP rank derives a
  different cache key for the same prompt and the hit ratio collapses to 0.
- **`LMCACHE_CHUNK_SIZE` must equal `--block-size` (128).** ATOM refuses a load
  whose HBM frontier is not chunk-aligned; with prefix caching on that frontier
  is block-aligned, so at chunk 256 roughly every other hit is dropped.
- **`OFFLOAD_MIN_LOAD_TOKENS` defaults to 8192**, which is above every prompt in
  a chat-sized workload — the tier would never serve anything.
- **`LMCACHE_MAX_LOCAL_CPU_SIZE` is per rank.** TP4 x 20 GiB locks 80 GiB of
  pinned memory; if free memory is low the allocation reclaims page cache and
  each worker can take minutes. Startup looks hung — `EngineCore` prints
  `No available shared memory broadcast block found in 60 seconds` once a minute.
  That line alone is **not** a failure: a healthy startup prints it 5 times too.
  Check whether the workers are still emitting log lines instead.
- **Saves are fire-and-forget.** A request returning does not mean its KV has
  landed. Benchmarks that measure immediately after warm-up systematically
  under-report external hits; allow a settle period.
- **`--enable-prompt-tokens-details` is required for client-side verification.**
  Without it aiperf silently reports an empty prompt-cache column rather than an
  error.
- **Stop the server with `podman restart`, not `pkill`.** Killing TP workers
  leaves zombies holding GPU memory (82 GiB/card observed); only restarting the
  container releases it.

## Related

- [LMCache KV Cache Offload](LMCache-KV-Cache-Offload.md) — generic plugin path
  (`LMCacheConnectorV1`), does not support M3's layouts
- [MiniMax-M3](MiniMax-M3.md) — base serving recipe
- `recipes/MiniMax-M3-Agentic-Offload.md` — ATOM **native** backend offload with
  agentic replay, DP2 sizing ladder
