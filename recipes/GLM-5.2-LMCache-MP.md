# GLM-5.2 with the LMCache Multiprocess Server

This recipe exercises ATOM's native GLM-5.2 cache layout through LMCache's
standalone multiprocess (MP) server. The connector transfers both cache planes:

- the 576-element MLA latent cache;
- the padded DSA index cache (144 bytes per token for the standard GLM-5.2
  configuration).

They are registered as two physical LMCache groups that share ATOM's single
paged-block ID space. This is different from the DeepSeek-V4 PAGE+SLOT layout
and from the legacy in-process `lmcache_offload` connector.

GLM-5.2 is deliberately not registered with LMCache's pure-MLA rank-collapse
optimization. IndexShare makes all TP workers consumers of the combined cache;
each TP rank therefore gets an independent LMCache KV rank and object key. This
prevents multiple workers from consuming a single rank-0 lookup lock and makes
rank-local retrieve failures visible.

## Current Scope

- GLM-5.2 only (`model_type == "glm_moe_dsa"`)
- one LMCache MP server
- tensor parallelism (TP)
- no PP, DCP, PCP, data parallelism, or DP Attention yet
- LMCache's `auto` (ROCm IPC handle) or `lmcache_driven` transfer mode;
  `engine_driven` cannot represent the two physical cache groups
- LMCache native ATOM MP API from
  [LMCache#4662](https://github.com/LMCache/LMCache/pull/4662)

ATOM's Docker images install the v0.5.3 ROCm wheel for its prebuilt storage
extensions, then overlay the complete Python package from the pinned
LMCache#4662 revision and rebuild both `lmcache_native` and the HIP `c_ops`
extension against the image's PyTorch. This full overlay is required:
overriding only LMCache's vLLM adapter does not install
`lmcache.integration.atom`, `EngineType.ATOM`, or ATOM cache-format detection.
Until LMCache#4662 merges, external images must fetch that PR ref (or the same
immutable commit) rather than installing the v0.5.3 wheel alone.

The connector uses a layout-specific model namespace, so its objects cannot be
mixed with objects written by vLLM's connector or ATOM's legacy connector.

## 1. Start LMCache First

Run the standalone server before starting ATOM. The example allocates 200 GiB
of host DRAM and uses 256-token chunks. `cupy-rocm-7-0` is required by
LMCache's `lmcache_driven` GPU cache context; ATOM's Docker images install it.

```bash
export LMCACHE_PORT=${LMCACHE_PORT:-5555}
export TP=${TP:-8}

python -m lmcache.v1.multiprocess.http_server \
  --host 0.0.0.0 \
  --port "${LMCACHE_PORT}" \
  --http-host 0.0.0.0 \
  --http-port 8080 \
  --max-workers "${TP}" \
  --l1-size-gb 200 \
  --eviction-policy LRU \
  --chunk-size 256 \
  --supported-transfer-mode lmcache_driven \
  2>&1 | tee lmcache-mp.log
```

The server's `--chunk-size` must be an integer multiple of ATOM's KV block
size. The connector checks this during cache registration and fails fast if the
two geometries are incompatible.

## 2. Start ATOM

The first validation should use the non-MTP configuration below. Once cold/warm
reuse is confirmed, the normal GLM-5.2 MTP flags can be added.

```bash
export MODEL_PATH=${MODEL_PATH:-zai-org/GLM-5.2-FP8}
export TP=${TP:-8}
export LMCACHE_PORT=${LMCACHE_PORT:-5555}

export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export OFFLOAD_MIN_LOAD_TOKENS=256

python -m atom.entrypoints.openai_server \
  --model "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --server-port 8000 \
  --kv_cache_dtype fp8 \
  --block-size 16 \
  --no-enable_prefix_caching \
  --online_quant_config \
    '{"global_quant_config":"ptpc_fp8","layer_quant_config":{"model.layers.*.mlp.experts":"per_block_fp8"},"exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate"]}' \
  --kv-transfer-config \
    "{\"kv_connector\":\"lmcache_mp\",\"kv_role\":\"offload\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":${LMCACHE_PORT},\"lmcache.mp.lookup_timeout\":10.0,\"lmcache.mp.mq_timeout\":30.0,\"lmcache.mp.heartbeat_interval\":1.0,\"lmcache.mp.mp_transfer_mode\":\"lmcache_driven\"}}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  2>&1 | tee atom-glm52-lmcache-mp.log
```

`LMCacheMPConnector` is accepted as an alias for `lmcache_mp`.

`OFFLOAD_MIN_LOAD_TOKENS` is ATOM's minimum useful remote-load span. The default
is 8192. The example uses `256` for a functional smoke test; restore the
performance-oriented value for normal serving. Local HBM prefix caching is
disabled above so a repeated request must exercise LMCache rather than being
satisfied by ATOM's in-process cache.

The MP server owns storage configuration. Legacy variables such as
`LMCACHE_LOCAL_CPU`, `LMCACHE_MAX_LOCAL_CPU_SIZE`, and `LMCACHE_CHUNK_SIZE` do
not configure this path; use the `lmcache server` flags instead.

## 3. Check Cold and Warm Requests

Send the same long prompt twice. The first request stores complete 256-token
chunks; the second should report an LMCache lookup hit and retrieve the suffix
not already present in ATOM's HBM prefix cache.

For TP8, a cold request that stores `N` chunks should add `8 * N` L1 objects.
Every GPU should then report the same retrieved token count in LMCache's
`L0<->L1 stats`. Checking object counts and per-rank transfer errors is more
reliable than requiring generated text to be byte-identical: the GLM-5.2 FP8
MoE path can select different greedy tokens even on repeated requests that do
not use LMCache.

Useful startup and runtime checks include:

```text
GLM-5.2 LMCache MP registered rank=... latent_layers=... index_layers=...
[OFFLOAD-LOOKUP] ... hit=...
[OFFLOAD-LOAD-EMIT] ...
```

LMCache's HTTP frontend uses port 8080 by default:

```bash
curl -sS http://127.0.0.1:8080/status
```

## Connector Options

Options live under `kv_connector_extra_config`:

| Option | Default | Purpose |
| --- | ---: | --- |
| `lmcache.mp.host` | `tcp://localhost` | MP server host |
| `lmcache.mp.port` | `5555` | MP server ZMQ port |
| `lmcache.mp.lookup_timeout` | `30.0` | ATOM lookup polling deadline after control-plane calls return |
| `lmcache.mp.lookup_poll_interval` | `0.01` | Lookup polling interval |
| `lmcache.mp.mq_timeout` | `300.0` | LMCache control-plane request timeout |
| `lmcache.mp.heartbeat_interval` | `10.0` | LMCache server heartbeat interval |
| `lmcache.mp.mp_transfer_mode` | `auto` | Must be `auto` or `lmcache_driven` |

`lmcache.mp.server_urls` is also accepted for compatibility with LMCache, but
ATOM currently requires exactly one URL.

TP8 registration creates eight large GPU IPC contexts. A 30-second MQ timeout
was used for validation. The same timeout bounds the first request after an MP
server failure; after it expires the request fails closed to recomputation.
Use a smaller value only after confirming that startup and heartbeat
re-registration fit within it on the target node.

That recomputation path applies when lookup fails or the worker drops a transfer
before submission. If the server becomes unhealthy after device work is already
in flight, ATOM keeps the request pending until LMCache's device future proves
the copy is terminal. A server that dies permanently in that window requires an
ATOM engine restart; recomputing immediately could race a late GPU write into
reused KV blocks.

Lookup currently runs synchronously on the scheduler thread. A single submit or
status query can block for up to `lmcache.mp.mq_timeout`, so
`lmcache.mp.lookup_timeout` alone is not a hard end-to-end deadline. Configure
both timeouts to fit the deployment's failure budget.

Remote capacity must include every TP rank. For the standard FP8 layout each
rank stores 56,160 bytes per token (`78 * (576 + 144)`), so TP8 consumes
449,280 bytes per cached token before allocator overhead.

## Known Limitations

- This initial adapter is for native ATOM GLM-5.2; it is not a generic model
  connector.
- PP, DCP, PCP, DP, and DP Attention fail during connector construction.
- `LMCACHE_MP_TRANSFER_MODE=engine_driven` (or its extra-config equivalent)
  fails during connector construction because LMCache's engine-driven
  path cannot express the two physical cache groups.
- A real ROCm multi-GPU cold/store/warm/load run is required before treating a
  new image or model checkpoint as validated.
- The connector has no engine-level shutdown callback yet; LMCache registrations
  are released when the ATOM worker process exits.
- Lookup admission is synchronous; an unresponsive control plane can stall the
  scheduler for up to the configured MQ timeout before recomputation begins.
- A permanent server failure during an in-flight GPU transfer is fail-stop for
  the affected request; restart the ATOM engine to recover it safely.
