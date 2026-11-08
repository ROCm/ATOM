# V4.1 prefix-spread with native Mooncake Store

Use a shared Mooncake Store to recover a published prefix after `prefix_spread`
routes a request to another DP rank. This path does not run LMCache MP. HBM
prefix caching remains enabled: local hits stay local, and Store restores only
the missing PAGEs. V4.1 reconstructs its per-request STATE by replay; a backend
without that explicit replay contract is rejected at KV registration.

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
  "mooncake_store.load_pool_mib": 1024,
  "mooncake_store.save_pool_mib": 256
}'
export ATOM_FRONTEND_PREFIX_HASH=1
export OFFLOAD_MIN_LOAD_TOKENS=512
export OFFLOAD_MIN_SAVE_TOKENS=512
export OFFLOAD_MAX_PENDING_SAVES=8
export OFFLOAD_PROFILE=1
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
standalone offload connector in `--kv-transfer-config`. The single-node path
above does not configure a P/D handoff; P/D composition must be configured
separately. Store capacity is a deployment choice, not reserved by these flags.

## Validation and performance

Before interpreting throughput, verify positive `loaded_tokens`, zero
`load_failures`, and total prefix reuse alongside the actual prefill work.
Finding a Store key is not proof that its KV was loaded. Keep request seeds,
GPU assignment, prefill budget, HBM memory fraction and warmup identical across
comparisons.

A single-node MI355X TP2/DPA2 AgentX C128 run used 128 GiB CPU cache, a 16K
prefill budget, memory utilization 0.9, DSpark5, seed 42, and a 900-second
measurement target. MP and native Store used prefix-spread; main used HBM with
DP session affinity. The three runs were sequential on the same GPU pair:

| Metric | Main HBM/session | Prefix-spread MP | Prefix-spread Store |
| --- | ---: | ---: | ---: |
| TTFT mean (s) | 2.50 | 3.52 | 2.40 |
| TTFT P90 (s) | 5.68 | 8.05 | 5.54 |
| TTFT P99 (s) | 10.47 | 13.34 | 8.96 |
| ITL P50 (ms) | 31.35 | 33.40 | 31.24 |
| Output tok/s/GPU | 1239.6 | 1124.8 | 1193.0 |
| Total prefix reuse | 94.18% | 94.28% | 94.46% |

Store improved mean TTFT by 31.8% and output throughput by 6.1% relative to MP
in this run. Relative to main, output throughput was 3.8% lower; this is not
evidence of a general performance advantage over HBM/session. One run per
configuration does not establish variance or results at larger concurrency.

Native Store lookup is synchronous. KV restoration starts at admission.
