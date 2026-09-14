# GLM-5.2 — LMCache KV offload on the vLLM plugin (byte codec)

GLM-5.2 (`GlmMoeDsaForCausalLM`) cannot use LMCache's own GPU connector. This
recipe uses `AtomLMCacheOffloadConnector`, which drives ATOM's `DenseKVByteCodec`
from vLLM's KV-connector API and leaves LMCache as a pure byte store.

**Read *Measured* before turning this on.** The connector is correct — a
two-pass check restores 99.84% of a flooded-out prefix and recovers every marker
— but on GLM-5.2 at the bandwidths measured here it costs throughput at both
working points tested (-12.31% and -48.41% over 600 s). It is a tool for
workloads with a small, hot prefix set, not a default.

For the generic plugin + `LMCacheConnectorV1` path (works on dense models), see
[LMCache KV Cache Offload](LMCache-KV-Cache-Offload.md). That path does **not**
work on GLM-5.2 — see *Why a separate connector* below. For the same connector on
MiniMax-M3, see
[MiniMax-M3 LMCache Byte Offload](MiniMax-M3-LMCache-Byte-Offload.md).

## Why a separate connector

At TP=4 GLM-5.2 registers **99 KV entries in two physical layouts**:

| entries | shape | dtype | what |
|---|---|---|---|
| 78 | `(nb, 64, 576)` | uint8 | MLA latent (`kv_lora_rank` 512 + rope 64), K and V fused |
| 21 | `(nb, 64, 132)` | uint8 | DSA indexer keys — 128 fp8 bytes + 4 bytes of scale **packed into the row** |

LMCache's `normalize_kv_and_discover_format()` probes for one **global** format
and aborts when the registration is not uniform. The byte codec sidesteps the
question entirely: ATOM gathers whole paged blocks into a chunk-major uint8 blob
and LMCache only ever stores opaque bytes, so no format probe runs. **Stock
LMCache 0.4.5 (shipped in `rocm/atom-dev:vllm-latest`) is sufficient** — unlike
the official-connector route, no source build of 0.5.x is needed.

Three GLM-5.2 specifics the mapping had to handle rather than assume:

**Indexer entries are folded onto their attention layer.** vLLM registers the
DSA indexer key cache as its own KV entry, but its bytes belong to the owning
layer. GLM spells it `<p>.indexer.k_cache` → `<p>.attn` (M3 spells it
`<p>.index_cache` → `<p>`). The GLM pairing is the one
`AiterMlaSparseIndexerMetadataBuilder` itself uses
(`attention_prefix = layer_name.removesuffix(".attn")`), so it is the model's
own convention, not a guess. Folding is keyed on the name, never on the shape —
a real layer that merely looked indexer-shaped would be restored under a
neighbour's key.

**Only 21 of 78 layers own an indexer.** GLM-5.2's IndexShare lets "shared"
layers reuse the preceding "full" layer's indexer, so most layers have no
indexer entry at all. The mapping must not invent an empty slot for them — that
would change the codec's per-block byte stride.

**No scale hook is needed.** M3 keeps fp32 KV scales on the layer object, outside
vLLM's `kv_caches` dict, and must report them through `get_kv_transfer_scales()`.
GLM-5.2's indexer packs its scale into the moved bytes, and its MLA layers carry
only vLLM's scalar per-tensor `_k_scale`, which is constant. Both tensors are
block-major and contiguous, so each travels whole and nothing is left behind.

**One KV cache group.** The MLA layers and the indexer layers share a block size
and a common `MLAAttentionSpec` base, so `UniformTypeKVCacheSpecs` merges them
into a single group with one block table and one `num_blocks` (48,699 measured
at the default pool). The codec addresses every segment with that one block
table, and `build_kv_cache_tensors` hard-fails if the registered tensors ever
disagree on block count — two groups would still divide evenly often enough to
pass the codec's own check and then slice the smaller tensor at the wrong
granularity, with nothing logged.

## Launch

```bash
export PYTHONHASHSEED=0               # mandatory, see Gotchas
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=180 # GiB **per TP rank**, size for the whole run
export LMCACHE_CHUNK_SIZE=64          # must equal --block-size
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1

vllm serve /path/to/GLM-5.2-MXFP4 \
  --served-model-name amd/GLM-5.2-MXFP4 \
  --trust-remote-code \
  --load-format fastsafetensors \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --block-size 64 \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --additional-config '{"online_quant_config": {"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate", "*expert*"]}}' \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector","kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector","kv_role":"kv_both","kv_load_failure_policy":"recompute"}'
```

Select the connector through vLLM's **out-of-tree entry point** (the
`kv_connector_module_path` above). vLLM validates `kv_transfer_config` while
building `VllmConfig`, which happens *before* platform plugins load, so naming
the connector without the module path fails config validation.

`--enable-prefix-caching` is required. The base GLM-5.2 recipe passes
`--no-enable-prefix-caching`; LMCache keys prefixes, so without prefix caching
every lookup is dead on arrival.

## Verify it is actually on

Four independent checks — all of them, because each can pass for the wrong reason:

```bash
# 1. vLLM's own factory, on every worker AND the EngineCore
grep "Creating v1 connector with name: AtomLMCacheOffloadConnector" server.log

# 2. the codec saw the whole model, with the indexers folded in
#    (78, not 99 -- if this says 99 the fold rule did not fire)
grep "ATOM LMCache offload: registered 78 layers" server.log

# 3+4. the tier is queried AND returns data (must be > 0)
curl -s localhost:8330/metrics | grep -E 'external_prefix_cache_(hits|queries)'
```

Two identities must hold on any interval; they catch a miscounting tier that
still looks plausible:

```
prefix_cache_queries - prefix_cache_hits == external_prefix_cache_queries
prefix_cache_hits    + external_prefix_cache_hits == prompt_tokens_cached
```

The first says the two tiers are strictly serial (HBM first, LMCache only gets
what HBM missed); the second says nothing is double-counted.

**Check them on a drained server, not under load.** The
`external_prefix_cache_*` pair is recorded when a request is *admitted*;
`prompt_tokens_by_source` is recorded when it *finishes*. With a 16 s TTFT there
are always requests in flight, so the two sides cover different request sets and
the second identity is off by whatever is mid-prefill -- in one 600 s run,
admission had seen 305 requests and the by-source counters only 168, and the
identity missed by 4.5M tokens. Drive the load to zero first, then read
`/metrics`; on a quiesced server it closes exactly.

## Sizing: do this before benchmarking

Per rank, per token, GLM-5.2 at TP=4 moves

```
78 x 576 B (MLA)  +  21 x 132 B (indexer)  =  47,700 B  (~46.6 KiB)
```

The default pool is **48,699 blocks = 3,116,736 tokens**, which is far larger
than any synthetic working set — HBM alone absorbs everything and the external
tier has nothing left to do. An oversized pool makes this tier look useless.
Cap it below the working set with `--num-gpu-blocks-override` before concluding
anything:

```
free for caching = KV pool - concurrency x ISL
```

At `--num-gpu-blocks-override 8192` (524,288 tokens ≈ 25 GB/rank) with
concurrency 8 and a 32,768-token prompt, 262,144 tokens are resident in flight
and the remaining 262,144 hold **~9 prefixes**. A 16-prefix pool therefore sits
almost entirely in HBM and leaves the external tier nothing to do — measured, an
off arm at 16 prefixes already served 75.97% of its prompt tokens from HBM. Use
at least 64 prefixes so ~55 of them are out of HBM reach; see Measured.

The CPU tier must hold **the whole run**, not the prefix pool. At 64 prefixes
the prefixes alone are 64 x 1.27 GiB = 81.5 GiB/rank, and every request adds a
unique cache-bust tail (~0.18 GiB), which over 600 s comes to ~84 GiB/rank. The
40 GiB that fits a 16-prefix pool is exhausted ~150 s into a 600 s run, and once
full the tier logs `Failed to allocate memory block ... no memory is available`
(27,695 times in one run) and starts resolving lookups to chunks that are gone
by retrieve time. Size it for the run: `LMCACHE_MAX_LOCAL_CPU_SIZE=180`.

## Measured

gfx950 x8 (this connector on GPUs 0-3), TP=4, `amd/GLM-5.2-MXFP4`,
`--num-gpu-blocks-override 8192`, block size 64, `LMCACHE_CHUNK_SIZE=64`,
`PYTHONHASHSEED=0`. aiperf `random` dataset, 28,672-token shared prefix +
4,096-token per-request unique tail, 512 output tokens, concurrency 8,
`--cache-bust first-turn-suffix`, 600 s measurement window with a 60 s grace
period, `--use-server-token-count`. Each pair uses one seed for both arms.

Two working points, differing only in how many distinct prefixes circulate:

| metric | 16 prefixes, off | 16 prefixes, on | 64 prefixes, off | 64 prefixes, on |
|---|---|---|---|---|
| seed | 904117 | 904117 | 517293 | 517293 |
| requests ok / error | 472 / 0 | 419 / 47 | 303 / 0 | 168 / 0 |
| total throughput tok/s | 25,864 | 22,679 (**-12.31%**) | 16,420 | 8,471 (**-48.41%**) |
| TTFT avg ms | 1,832 | 2,020 | 4,494 | 16,569 |
| ITL avg ms | 16.54 | 18.43 | 22.83 | 16.92 |
| prompt tokens read from cache | 75.97% | 83.84% | 14.94% | 58.64% |

The 47 errors in the 16-prefix on arm are the `kv_load_failure_policy` default
biting (see Gotchas); they are gone at 64 prefixes once the policy is `recompute`
and the CPU tier is sized for the whole run. **Both arms of the 64-prefix pair
completed with zero errors**, so that pair is the one to read. The 16-prefix on
arm also *under-reports its own cost*: aiperf excludes failed requests from
throughput, but the server still prefilled all 47 of them.

Where the prompt tokens came from, 64-prefix on arm
(`vllm:prompt_tokens_by_source_total`, delta over the window):

```
external_kv_transfer  2,598,720   47.2%      <- this connector
local_cache_hit         629,632   11.4%      <- HBM prefix cache
local_compute         2,276,824   41.4%      <- re-prefilled
```

So the tier is genuinely carrying the load — and throughput still halves.

### Why it loses here

The offload path, not the GPU, is the saturated resource. From the on arm's
`Retrieved`/`Stored` lines (summed over the 4 ranks, then divided by 4):

| | per rank | call time | bandwidth | per token |
|---|---|---|---|---|
| retrieve | 7,079,168 tok | 424.4 s | 0.796 GB/s | 60.0 us |
| store | 2,666,752 tok | 400.8 s | 0.317 GB/s | 150.3 us |

Those 825 s of transfer sit inside a 600 s wall clock. The link is not
uniformly slow — half the retrieve calls run at 3.891 GB/s or better — but the
slow tail eats the time: p10 is 0.228 GB/s, p90 is 6.36 s per call and the worst
is 12.9 s, which is exactly where the 16.5 s TTFT comes from. Retrieve and store
are contending for the same host path (401 s/rank of store overlaps the 424 s of
retrieve), so aggregate bandwidth lands at a fifth of a typical call.

Compare against simply re-prefilling. The off arm computed 8,445,197 prompt
tokens in 600 s = **14,075 tok/s, i.e. 71.0 us per token**. With a 2.65x reuse
factor (7.08M retrieved against 2.67M stored), the amortised cost of a token
served from the tier is

```
60.0 us (retrieve)  +  150.3 us / 2.65 (store)  =  116.6 us  =  1.64x recompute
```

which is the whole result: on GLM-5.2 it is cheaper to recompute the prefix than
to move it. The observed drop is larger still (1.94x) — the rest is the
connector's own bookkeeping: per-step lookups and the scheduling deferral while
an async load is outstanding.

Note what this is *not*. GLM-5.2's KV is already compressed — 576 B per token
per MLA layer, against ~2 KiB for an 8-head GQA layer at fp8 — so the tier is
moving less per layer than it would on a dense model. The volume is only large
because there are 99 of them (78 MLA + 21 indexer = 46.6 KiB/token/rank,
**1.27 GiB per rank** for a 28,672-token prefix). What makes it lose is the
other side of the ratio: sparse MLA prefill is *fast*. Recomputing a token costs
71.0 us and moving one costs 60.0 us — the two are within 20% of each other, so
there is no headroom for the store traffic to hide in. A model whose prefill is
several times more expensive per KV byte leaves that headroom; this one does
not.

### What would have to change

Break-even needs the amortised transfer cost below 71 us/token, so either

- **more reuse** — solving `60.0 + 150.3/r < 71.0` gives **r > 13.7**: at the
  measured bandwidths a prefix must be read back about fourteen times before the
  tier pays for storing it. A small set of very hot prefixes (system prompts,
  few-shot blocks) is the shape that works; the 64-way rotation measured here,
  at r = 2.65, is nowhere near it;
- **more bandwidth** — the median call already does 3.891 GB/s, so the deficit
  is contention rather than the link. Halving the store traffic, or giving
  retrieve its own stream, moves the aggregate more than faster memory would;
- **fewer bytes** — there is little left here. The KV is already the MLA latent
  at fp8; dropping the 21 indexer layers and recomputing them saves 5.8% of the
  volume, which does not move r = 13.7. Bytes are not the lever on this model.

The honest summary: the connector is correct and the tier does carry the load
(47.2% of prompt tokens at 64 prefixes), but on GLM-5.2 at these bandwidths it
costs more than it saves. Ship it behind a flag and turn it on for workloads
with a small, hot prefix set — not as a default.

Do not read the 16-prefix row as "a smaller loss". It is a different failure: at
16 prefixes the off arm already served 75.97% of its prompt tokens from HBM, so
the external tier had almost nothing left to do and its 12.31% is close to pure
overhead. Neither working point is a configuration to ship; they bracket the
behaviour.

### Correctness: two-pass restore check

Throughput aside, the tier has to give back what it stored. A single pass only
ever SAVEs, so it proves nothing. `tools/kv_offload_twopass_check.py` runs
pass 1 (N marker prompts) → flood (F unrelated long prompts, to evict HBM) → pass 2 (the same N
prompts). Each prompt hides a random marker at the very start behind ~20K tokens
of filler and asks for it back under greedy decoding, so the answer depends on
the exact bytes of the offloaded prefix.

| | pass-2 prompt tokens | pass-2 served from cache | marker recall | text identical to pass 1 |
|---|---|---|---|---|
| off (noise floor) | 160,200 | **0** | 8/8 | 0/8 |
| on | 160,200 | **159,936 (99.84%)** | 8/8 | 0/8 |

The off arm's zero is what makes this a test: the flood really did evict
everything, so the on arm's 99.84% can only have come back through LMCache, and
the marker sits in the first blocks of that restored prefix. Zero alloc
failures, zero KV load failures, zero empty retrieves.

Reproduce with:

```bash
python3 tools/kv_offload_twopass_check.py --url http://127.0.0.1:8330 \
  --model amd/GLM-5.2-MXFP4 --label on --n 8 --flood 24 --out-dir results/
```

**Do not use byte-identical output as the criterion.** Both arms score 0/8,
including the one with no connector attached — a prefix-cache hit is not
bit-reproducible against a cold run. The noise floor is 0, so 0 carries no
signal; marker recall is the criterion that does.

## Gotchas

- **`PYTHONHASHSEED=0` is mandatory.** Without it each TP rank derives a
  different cache key for the same prompt and the hit ratio collapses to 0.
- **`LMCACHE_CHUNK_SIZE` must equal `--block-size` (64).** ATOM refuses a load
  whose HBM frontier is not chunk-aligned; with prefix caching that frontier
  advances in whole blocks, so at chunk 256 three of every four hits are dropped
  into `HBM prefix is not chunk-aligned ... re-prefill`.
- **`kv_load_failure_policy` defaults to `"fail"`, which is wrong for a cache.**
  When a chunk is evicted between the scheduler-side lookup and the worker-side
  retrieve, the connector correctly reports the unfilled blocks — and vLLM then
  marks the whole request `FINISHED_ERROR`, so the user gets an empty 500. An
  offload tier is a cache; the only correct answer to a miss is to re-prefill
  those blocks. Set `"kv_load_failure_policy":"recompute"` explicitly. Left at
  the default, one 600 s run lost **10.09% of its requests** (47 of 466) to
  `InvalidInferenceResultError`.
- **`LMCACHE_MAX_LOCAL_CPU_SIZE` is per rank.** TP4 x 180 GiB locks 720 GiB of
  pinned memory. Pinning is itself a cost: when comparing against another
  offload stack, match this number, or the comparison is measuring page-cache
  reclaim rather than the cache. Size it for the whole run, not the prefix pool
  — see Sizing.
- **Saves are fire-and-forget.** A request returning does not mean its KV has
  landed. Benchmarks that measure immediately after warm-up systematically
  under-report external hits; allow a settle period.
- **`--enable-prompt-tokens-details` is required for client-side verification**,
  and aiperf needs `--use-server-token-count` to read it. Without the pair,
  aiperf silently reports an empty prompt-cache column rather than an error.
- **Stop the server with `podman restart`, not `pkill -9`.** Killing TP workers
  leaves zombies holding GPU memory; only restarting the container releases it.
- **`Failed to import Triton kernels ... triton_kernels.matmul_ogs` at startup is
  benign** — it is the gpt-oss MXFP4 path, printed once per TP worker.

## Related

- [MiniMax-M3 LMCache Byte Offload](MiniMax-M3-LMCache-Byte-Offload.md) — the
  same connector on M3's three-layout registration
- [LMCache KV Cache Offload](LMCache-KV-Cache-Offload.md) — generic plugin path
  (`LMCacheConnectorV1`), does not support GLM-5.2's registration
- [GLM-5](GLM-5.md) — base serving recipe
