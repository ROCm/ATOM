# PAGE-backed native checkpoints with LMCache MP

ATOM's `lmcache_mp` connector has two capability-selected paths:

- ordinary attention backends publish PAGE views and use PAGE-only transfer;
- stateful attention backends additionally publish a
  `PagedStateCheckpointSpec` and `execute_paged_state_copies`, enabling one
  combined PAGE/native-STATE transfer.

The connector does not inspect model names or layout-ID prefixes. A new model
can reuse the native path by implementing those shared contracts. The native
path pins existing READY checkpoint PAGE units; it does not take another
snapshot of the request's Active SLOT. Every request keeps its normal fixed
SLOT while running.

The PAGE-only path covers both sparse MLA and MHA/GQA layouts. GLM-5.2's MLA
KV and index cache are byte-identical across TP, so automatic rank collapse
stores one copy while every rank retrieves it. MiniMax-M3 publishes its GQA
KV, scale, and NSA index-cache planes as zero-copy PAGE views and deliberately
keeps one stored shard per TP rank. Both layouts use chunk completion events to
release save-source PAGE blocks before the remote store becomes terminal.

## Run

Install the matching ATOM and LMCache changes. The LMCache build must include
per-group `null_block_id`, automatic object grouping for non-default null
policies, and sparse null handling. Run the MP server on the same host, with GPU
IPC access to the ATOM worker allocations:

```bash
lmcache server --host 127.0.0.1 --port 5555 \
  --chunk-size 256 \
  --supported-transfer-mode lmcache_driven --l1-size-gb 64
```

For example, add the following options to a DSv4 launch that already publishes
the native-state contract:

```bash
export LMCACHE_CHUNK_SIZE=256
export OFFLOAD_MAX_PENDING_SAVES=2
export OFFLOAD_MIN_LOAD_TOKENS=8192
export OFFLOAD_MIN_SAVE_TOKENS=8192

python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-V4-Pro --kv_cache_dtype fp8 -tp 8 \
  --enable_prefix_caching --state-checkpoint-interval-tokens 8192 \
  --kv-transfer-config '{
    "kv_connector": "lmcache_mp",
    "kv_role": "offload",
    "kv_connector_extra_config": {
      "lmcache.mp.host": "tcp://127.0.0.1",
      "lmcache.mp.port": 5555,
      "lmcache.mp.tp_rank_collapse": true
    }
  }'
```

The ATOM configured chunk size must equal the MP server's chunk size. Both must
align to ATOM's PAGE/hash block size. Native checkpoints are produced by the
attention backend's existing checkpoint policy, so their cadence must provide
the desired reusable boundaries. A prefix is loadable only where PAGE KV and a
complete STATE checkpoint both exist on all TP ranks.

`lmcache.mp.max_pinned_state_bytes` optionally limits native checkpoint sources
and temporary restore images together. Its default is
`OFFLOAD_MAX_PENDING_SAVES * units_per_checkpoint * page_unit_bytes`, per TP
worker's geometry. PAGE KV sources continue to use normal request ownership.
Candidates consume no PAGE or image pin until admission. If a request finishes
while waiting, admission resolves the original token/hash chain back through
the live prefix index and stores only its still-resident contiguous prefix.
`OFFLOAD_MIN_SAVE_TOKENS` (default 8192) suppresses a late save whose remaining
prefix is too small. The shared save limit defaults to
`max(2, 2 * OFFLOAD_COPY_WORKERS)` when not configured.

The namespace includes model/PAGE geometry, TP size, speculation configuration,
native layout and image sizes, Hugging Face commit identity when available,
and `lmcache.mp.model_revision` if supplied. Set that revision string when
replacing weights in an existing local model directory. Local directory names
alone cannot identify changed weight contents.

## Lifetime and representation

Engine group 0 contains PAGE views and declares `null_block_id=None`, so PAGE 0
is ordinary data. Native image ordinal `j` uses engine group `1+j`, aliases the
same PAGE allocation, and declares a one-chunk recurrent window with null ID
`-1`. Every ordinal is present at the checkpoint endpoint; earlier chunks use
all-null STATE groups. The non-default null policies make LMCache automatically
separate PAGE from STATE and group all STATE ordinals into one object.
The final image region is trimmed at `image_bytes`, preserving the original
physical PAGE stride.

The scheduler dispatches one combined PAGE/STATE save generation at a time per
request, with round-robin admission and count/byte bounds. It acquires the exact
READY image only after admission. An IPC producer event orders MP reads after
native checkpoint creation. Source-safe events release PAGE leases chunk by
chunk and release the READY STATE image once its endpoint is safe; terminal
completion then settles the logical operation. Decode-only PAGEs and the live
SLOT are not save sources and can be returned as soon as the request ends.
Failed saves roll back the watermark for at most three attempts at that
boundary. Native saves never reclaim an uncertain DMA lease by elapsed time.

Restore loads PAGE KV only for `[hbm, lmcache)` and restores the endpoint's full
native image into the request's already allocated fixed SLOT. If the local HBM
hit is not chunk-aligned, ATOM first prefills to the next chunk boundary and
then parks the request for the aligned remainder, provided that remainder meets
`OFFLOAD_MIN_LOAD_TOKENS`. The native codec runs on a dedicated stream with a
separate descriptor slot; completion is polled by event and never synchronizes
the compute stream. After successful H2D and SLOT restore, the temporary STATE
PAGE units are atomically adopted as an unpinned `READY` checkpoint, so a later
request can hit it locally until normal LRU eviction. An aborted request keeps
its allocations until the same exact completion arrives. Transport exceptions
without proof of device completion retain the lease; elapsed time and server
heartbeat failure do not free DMA sources or destinations.

## Current scope and constraints

- Native ATOM with one MP server supports TP, single-host DP, and single-host
  DP-attention with EP. Each DP replica keeps a private request-session scope
  while sharing the same content-addressed model namespace, so equal prefixes
  remain reusable across replicas. Under DP-attention, ATOM folds TP into DP;
  each EngineCore therefore uses a one-worker LMCache group and early-free
  quorum of one.
- Multi-node DP, PP, PCP, DCP, and engine-driven transfers are rejected.
  Multi-node DP needs one GPU-local LMCache server per host plus server routing;
  one server cannot import GPU IPC allocations from another host.
- PAGE tensors are registered as zero-copy `uint8` views. They are opaque cache
  storage, so byte views preserve FP8/BF16 bit patterns and keep ROCm's
  raw-pointer fallback from applying numerical dtype conversions.
- DSv4 declares both PAGE and native STATE byte-identical across TP ranks.
  `lmcache.mp.tp_rank_collapse=auto` therefore collapses TP automatically;
  explicit `true` is also accepted after the worker validates both declarations.
  One rank stores each object and every rank retrieves it (`num_kv_readers=TP`).
- GLM-5.2 (`glm_moe_dsa`) uses the same fully replicated sparse-MLA PAGE rule:
  one TP rank stores, all TP ranks retrieve, and non-writers report immediate
  source-safety so early release waits only for the real writer DMA.
- MiniMax-M3 uses GQA PAGE shards plus its NSA index cache. It publishes the
  complete layout to LMCache MP but leaves TP replication at `1`, so every TP
  rank stores and retrieves its own shard while retaining chunk-wise early free.
- External restore supports both zero-HBM and incremental local-prefix cases.
  Lookups truncate the token list to
  `floor((prompt_tokens - 1) / chunk_size) * chunk_size`, and PAGE/STATE must
  reach the same real endpoint.
- Pending saves do not pin source PAGEs. Once admitted, only the verified,
  continuous hash-matching prefix is acquired, and only source-unsafe chunks
  remain leased after request teardown. Running requests never give their SLOT
  to the MP connector.
- The transport aliases native buffers directly, but LMCache may use its own
  GPU transfer buffers internally. This removes an additional ATOM SLOT image,
  rather than promising a completely copy-free transport.

## Validation

CPU contracts cover exact READY leases, generation replay, eviction and reset,
byte budgets, fair admission, cancellation, failures, full-prompt boundaries,
native image byte order and strided tail registration. LMCache tests cover
null markers, serialization, sparse STATE lookup and capability negotiation.

Run the real independent-process CUDA/ROCm transport test from the matching
LMCache checkout:

```bash
python -m pytest -xvs tests/v1/multiprocess/test_native_state_alias_gpu.py
```

It uses synthetic cache contents and validates the shared transport contract.
The repository-level DSv4-Pro TP8 test additionally covers a forced full remote
restore, READY reuse, incremental restore, second local reuse, exact next-token
comparison with fresh prefill, per-rank retrieve counts, one-writer storage,
and exact promoted ranges.
