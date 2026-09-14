# GLM-5.2 — LMCache KV offload on the vLLM plugin (byte codec)

GLM-5.2 (`GlmMoeDsaForCausalLM`) cannot use LMCache's own GPU connector. This
recipe uses `AtomLMCacheOffloadConnector`, which drives ATOM's `DenseKVByteCodec`
from vLLM's KV-connector API and leaves LMCache as a pure byte store.

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
export LMCACHE_MAX_LOCAL_CPU_SIZE=40  # GiB **per TP rank**
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
  --kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector","kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector","kv_role":"kv_both"}'
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
and the remaining 262,144 hold ~9 of a 16-prefix pool — so roughly 44% of all
reuse has to come from the external tier. That is the regime this connector is
for.

On the CPU side, 40 GiB/rank holds ~900K tokens; a 16 x 28,672-token prefix pool
is 458,752 tokens ≈ 21.9 GB/rank, which fits.

## Gotchas

- **`PYTHONHASHSEED=0` is mandatory.** Without it each TP rank derives a
  different cache key for the same prompt and the hit ratio collapses to 0.
- **`LMCACHE_CHUNK_SIZE` must equal `--block-size` (64).** ATOM refuses a load
  whose HBM frontier is not chunk-aligned; with prefix caching that frontier
  advances in whole blocks, so at chunk 256 three of every four hits are dropped
  into `HBM prefix is not chunk-aligned ... re-prefill`.
- **`LMCACHE_MAX_LOCAL_CPU_SIZE` is per rank.** TP4 x 40 GiB locks 160 GiB of
  pinned memory. Pinning is itself a cost: when comparing against another
  offload stack, match this number, or the comparison is measuring page-cache
  reclaim rather than the cache.
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
