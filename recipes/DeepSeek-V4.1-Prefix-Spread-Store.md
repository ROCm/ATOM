# V4.1 prefix-spread with native Mooncake Store

Use a shared Mooncake Store to recover a published prefix after `prefix_spread`
routes a request to another DP rank. This path does not run LMCache MP. HBM
prefix caching remains enabled: local hits stay local, and Store restores only
the missing PAGEs. V4.1 reconstructs its per-request STATE by replay, which
needs encoder SWA bounded replay (the default); startup refuses the Store
without it.

## Store and model setup

Provision a Mooncake master and a CPU DRAM owner accessible to every model rank.
The master alone provides no cache capacity: mount owner segments and verify
put/get before starting the model. For RDMA, register the owner memory using
the deployment's supported hugepage/MR setup. This is a CPU-backed shared pool,
not a remote-HBM D2D pool. Reuse the model, GPU mapping and standard ROCm runtime
setup from your existing V4.1 deployment.

Example for a master at `STORE_IP:19871`, metadata at port 19872, and an owner
reachable through `rdma0`:

```bash
export ATOM_KV_OFFLOAD=mooncake_store
export ATOM_KV_OFFLOAD_EXTRA_CONFIG='{
  "mooncake_store.master": "STORE_IP:19871",
  "mooncake_store.metadata": "http://STORE_IP:19872/metadata",
  "mooncake_store.protocol": "rdma",
  "mooncake_store.rdma_devices": "rdma0",
  "mooncake_store.local_hostname": "MODEL_IP",
  "mooncake_store.chunk_tokens": 512,
  "mooncake_store.load_pool_mib": 1024,
  "mooncake_store.save_pool_mib": 256
}'
export ATOM_FRONTEND_PREFIX_HASH=1
export OFFLOAD_MIN_LOAD_TOKENS=4096
export OFFLOAD_MAX_PENDING_SAVES=8
# Bound coalescing for both waiting requests and running partial prefills.
# These are scheduling bounds, not a 250 ms end-to-end TTFT guarantee.
export ATOM_PREFILL_DELAYER_MAX_QUEUE_MS=250
export ATOM_PREFILL_DELAYER_PARTIAL_MAX_TICKS=2
# Required in every Mooncake process (see the offload README).
export MC_NUM_QP_PER_EP=1
unset ATOM_DP_SESSION_AFFINITY
```

Replace the address placeholders and choose a supported requester NIC. Add
these options to the V4.1 model command:

```bash
--enable-dp-attention --enable_prefix_caching \
--dp-load-balance prefix_spread
```

The Store extra config names the shared pool. Do not also pass a second
standalone offload connector in `--kv-transfer-config`. This is a
single-node (non-P/D) setup: V4.1 does not admit a `multi` connector, so the
Store cannot be combined with Mooncake P/D in the same server. Store capacity is a deployment choice, not reserved by these flags.

If using `mooncake_store.pools` to name requester NICs, every NIC alias must
resolve to the same `(master, metadata)` pair. Active `prefix_spread` rejects
distinct per-NIC pools: the destination worker cannot load another pool's
prefix. Other routing policies can still use distinct pools.

## Validation and performance

Before interpreting throughput, verify positive `loaded_tokens`, zero
`load_failures`, and total prefix reuse alongside the actual prefill work.
Finding a Store key is not proof that its KV was loaded. Keep request seeds,
GPU assignment, prefill budget, HBM memory fraction and warmup identical across
comparisons.

Native Store lookup is synchronous. KV restoration starts at admission.

The minimum applies to the missing, chunk-aligned prefix beyond the selected
rank's local HBM hit. For example, an owner with 16384 cached tokens and an
idle target with 13312 cached tokens differ by 3072: the request still goes
to the target, skips Store loading and recomputes the remainder. Short prompts
continue to participate in prefix routing.

The final recipe uses 4096 as a configurable, unmeasured policy cutoff, not a
measured break-even point. Measurements used
V4.1-Flash on MI355X, TP2/DPA2,
512-token Store chunks, a 16384-token published owner prefix and a 16385-token
request. Two warm pairs per difference gave these idle-target TTFT medians:

| Missing prefix tokens | Load (ms) | Recompute (ms) |
| ---: | ---: | ---: |
| 3072 | 399 | 277 |
| 6144 | 396 | 367 |
| 8192 | 386 | 407 |

For this comparison only, the loading arm allowed 512-token loads; the recipe
uses 4096 as an initial policy cutoff: loading 3072 tokens was clearly slower,
while the 6144- and 8192-token comparisons were closer to break-even. This is
not a measured optimum: loading at 6144 was still about 29 ms slower in this
idle case, and no 4096-token crossover measurement was taken. The final 4096
setting therefore permits that measured slower 6144-token load; deployments
prioritizing the measured idle-case latency can raise the minimum to 8192.
Tune the minimum
for the deployment's load and latency goals. With two real decoding requests
filling the owner, a separate measurement using an 8192-token minimum found that
an 8192-token load took a median 756 ms versus 1526 ms for recomputation, with
the owner still decoding when the request completed. These are request TTFTs,
including replay and scheduling, rather than isolated copy times. They do not
establish an optimum for other models, parallelism or links.

The queue-age bound releases a waiting suffix after a load. It does not cover
a partial prefill already in the running list; the separate two-tick partial
bound prevents that request from waiting through the default 100-tick partial
coalescing window. Both settings were tested together. Their effect on
throughput was not benchmarked; this recipe favors prompt latency.
