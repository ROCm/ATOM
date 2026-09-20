# GLM-5.2 — LMCache KV offload on the vLLM plugin (byte codec)

GLM-5.2 (`GlmMoeDsaForCausalLM`) cannot use LMCache's own GPU connector. This
recipe uses `AtomLMCacheOffloadConnector`, which drives ATOM's `DenseKVByteCodec`
from vLLM's KV-connector API and leaves LMCache as a pure byte store.

On a 64-prefix rotation this is worth **+50.4%** throughput against the same
workload with the tier off, and it gives back what it stored — a two-pass check
restores 99.84% of a flooded-out prefix and recovers every marker.

That number is recent. The first working version of this connector *lost* 48.41%
at this same working point, and the recipe said so. It was not bandwidth-bound,
as that analysis concluded; it was livelocked. *Measured* keeps both numbers,
because the wrong one was arrived at carefully and is the more useful of the two
to read. The 16-prefix working point has **not** been re-measured since the fix,
so take no number for it from this recipe.

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

The 64-prefix point is the one to read: both arms complete with zero errors and
the tier has real work to do. (At 16 prefixes the off arm already serves 75.97%
of its prompt tokens from HBM, so the external tier has little left to carry;
that point brackets the behaviour rather than describing a configuration to
ship.)

| metric | 64 prefixes, off | 64 prefixes, on |
|---|---|---|
| seed | 441907 | 441907 |
| requests ok / error | 312 / 0 | 464 / 0 |
| output throughput tok/s | 261.64 | **393.50 (+50.4%)** |
| total throughput tok/s | 17,007 | **25,578 (+50.4%)** |
| TTFT avg ms | 4,086 | 1,406 |
| TTFT p50 ms | 4,232 | 672 |
| ITL avg ms | 22.62 | 17.58 |

Where the prompt tokens came from (`vllm:prompt_tokens_by_source_total`, delta
over the window):

| source | off | on |
|---|---|---|
| `external_kv_transfer` | 0 | 9,708,800 (63.9%) |
| `local_cache_hit` | 2,192,960 (21.4%) | 3,112,704 (20.5%) |
| `local_compute` | 8,030,918 (78.6%) | 2,383,238 (15.7%) |

The last row is the mechanism in one line: recomputed prompt tokens fall from
8.03M to 2.38M, a 3.4x cut in prefill work, and prefill is what this workload is
short of.

### The -48.41% run, and why its analysis was wrong

The first measurement of this working point (seed 517293) read 16,420 -> 8,471
tok/s, **-48.41%**. The recipe then attributed it to the link: 825 s of transfer
summed across ranks inside a 600 s wall clock, a slow tail with 12.9 s retrieve
calls, an amortised 116.6 us/token against 71.0 us to recompute = 1.64x, and the
conclusion that a prefix had to be read back ~14 times to break even.

Every one of those numbers was measured correctly. The inference from them was
wrong, because the transfer volume they were computed from was itself the
symptom.

A lookup hit covering the whole prompt is decremented by one so the request has
something left to compute. LMCache resolves at chunk granularity, so when the
prompt length is an exact multiple of the chunk size that decrement walks off a
chunk boundary and names tokens the tier does not hold — the load can never
satisfy its own `ret_mask` check. And a failed load left no record, so the next
scheduler pass looked up, hit, parked the request in `WAITING_FOR_REMOTE_KVS`,
and failed again, forever, holding its KV blocks and its concurrency slot.

One request in that run was retried **137 times at ~1.45 GiB per attempt**; a
second run reached 528. That is where the 825 s of transfer and the 12.9 s tail
came from — not from a saturated link. Six of eight workers dead-ended one at a
time, the last at t=563 s, costing 26.88% of the run's slot-seconds against a
26.93% throughput drop. Both defects are fixed; `total_suppressed_load_retries`
counts the suppressed retries so the failure mode is visible if it returns.

**The lesson worth keeping:** a cost model built on measured volume is only as
good as the assumption that the volume was necessary. Before fitting a
break-even against transfer bytes, check that the bytes were work.

### Where the remaining cost is

With the livelock gone the path is host-bound, not link-bound. A 2.91 MiB chunk
takes ~2.1 ms to store or retrieve while the raw link moves it in ~0.06 ms, so
~97% of the time is host-side — and it is the same in both directions and on
LMCache 0.4.5 and 0.5.5rc4, which is why upgrading LMCache does not move it
(measured: 0.5.5rc4 is 5.72% *slower* end to end).

py-spy on a live unprofiled server put 46% of the save thread and 56% of the
load thread inside one call: the staging ring's wait on the slot it is about to
reuse. That wait blocks because a blocking runtime call drops the GIL to sleep
and then waits out a switch interval to get it back — ~22 ms (save) / ~35 ms
(load) per blocking wrap, against 0.158 ms of GPU work per group. Deepening the
ring is the fix, and it is nearly free (a slot is `nblocks*8` pinned bytes):

| host staging slots | output tok/s | TTFT p50 ms | ITL ms |
|---|---|---|---|
| 4 | 353.77 | 1,425.71 | 18.87 |
| 32 | 383.15 | 742.31 | 18.00 |
| 128 | 393.50 | 672.32 | 17.58 |

Monotone on all three against a 1.23% cross-sweep spread, so 128 is where the
marginal gain stopped being worth another arm rather than a measured optimum.
Set `OFFLOAD_STAGING_PROBE=1` to time each runtime call in the staging path
separately; it perturbs what it measures, so read it for shape, not for level.

Two things are *not* the lever on this model. Bytes: GLM-5.2's KV is already the
MLA latent at fp8 — 576 B per token per MLA layer against ~2 KiB for an 8-head
GQA layer — and dropping the 21 indexer layers saves 5.8% of the volume.
LMCache's version: see above.

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
