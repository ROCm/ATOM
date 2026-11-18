# DeepSeek V4.1: DP4 share-cache with a UMBP CPU tier

This is a single MI355X-node configuration using four GPUs, TP4 with DP
attention, and one shared UMBP DRAM pool. The connector transfers V4.1 PAGE
regions, including index data. Request SLOT/STATE regions remain local and
are reconstructed using bounded replay. Keep encoder SWA bounded replay
enabled; an unbounded state layout and DCP greater than one are refused.

Use the UMBP-capable build of the `hyi/umbp-store` branch with this change.
The earlier successful V4.1 restore smoke used a separate source snapshot;
it does not certify every subsequent PR head or establish a throughput gain.

## Deployment

Follow [UMBP-KV-Offload.md](UMBP-KV-Offload.md) for the Mori dependency,
standalone server, IPC sharing and memory locking. Model and Store processes
must see the same GPU assignment and shared IPC namespace. Reserve a real
1 TiB DRAM pool, shared by all four ranks, rather than four independent pools.
Check available host memory before allocation and inspect the native server's
allocated capacity after startup. A configured capacity or a running container
does not prove that the backing memory exists.

Inside the deployment container, start the server as described in that guide
with these settings:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export UMBP_ADDR=unix:///tmp/v41-umbp/server.sock
export UMBP_DRAM_CAPACITY=1099511627776
export UMBP_DRAM_NUMA_NODE=0,1
export UMBP_DRAM_USE_HUGEPAGES=0
export UMBP_DRAM_PREFAULT_THREADS=16
export UMBP_SSD_ENABLED=0
export UMBP_STANDALONE_METRICS_PORT=9877
```

Use the same DeepSeek-V4.1-Flash-1017 checkpoint and matching frontend encoder
in both arms. The encoder belongs to the model deployment and is not supplied
by UMBP. The comparison described here uses **BF16 KV and FP8 index**, rather
than the FP8 KV/FP4 index configuration of the general V4.1 agentic recipe.

## Model and comparison arms

Run A and B sequentially on the same four GPUs with fresh model instances,
the same CPU placement, checkpoint, source and image. Keep these common flags:

```bash
export MODEL_PATH=/models/dsv41
export OMP_NUM_THREADS=4
export ATOM_NUMA_BIND=0
export ATOM_DISABLE_MMAP=true
export AITER_LOG_LEVEL=WARNING

COMMON=(
  --model "$MODEL_PATH" --served-model-name DeepSeek-V4.1-Flash-1017
  --trust-remote-code --host 0.0.0.0 --server-port 8000
  --tensor-parallel-size 4 --enable-dp-attention
  --kv_cache_dtype bf16 --index-cache-dtype fp8
  --gpu-memory-utilization 0.9 --max-num-seqs 64
  --max-model-len 1048576
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384
  --enable_prefix_caching --block-size 16
  --state-checkpoint-interval-tokens 8192
  --level 3 --cudagraph-mode FULL
  --method dspark --num-speculative-tokens 5
  --spec-decode-acceptance-length 3.51
  --tool-call-parser dsml_v41 --timeout-keep-alive 900
)
```

Retain the deployment's identical graph-capture and other execution settings
across arms. Fixed acceptance length 3.51 produces synthetic performance
outputs. Keep any deployment-specific confidence schedule identical; the
recorded comparison disables it. Remove forced acceptance for accuracy tests.

A is original ATOM without offload or share-cache. Remove inherited offload
settings and inherited prefill-delayer overrides; this leaves the actual
source defaults in effect. Preserve session affinity and the default
`least_tokens` policy:

```bash
unset ATOM_KV_OFFLOAD ATOM_KV_OFFLOAD_EXTRA_CONFIG
unset ATOM_ENABLE_PREFILL_DELAYER ATOM_PREFILL_DELAYER_TARGET_FILL
export ATOM_DP_SESSION_AFFINITY=1
python3 -u -m atom.entrypoints.openai_server "${COMMON[@]}" \
  --dp-load-balance least_tokens
```

B enables the shared UMBP tier and share-cache routing. A Store key is looked
up asynchronously; a transfer is only complete after the normal worker and
cross-rank completion checks. HBM prefix caching remains enabled.

```bash
export ATOM_DP_SESSION_AFFINITY=0
export ATOM_ENABLE_PREFILL_DELAYER=1
export ATOM_PREFILL_DELAYER_TARGET_FILL=0.8
python3 -u -m atom.entrypoints.openai_server "${COMMON[@]}" \
  --dp-load-balance share_cache \
  --kv-transfer-config '{"kv_connector":"umbp_store","kv_role":"offload","kv_connector_extra_config":{"umbp_store.address":"unix:///tmp/v41-umbp/server.sock","umbp_store.chunk_tokens":256,"umbp_store.load_pool_mib":1024,"umbp_store.save_pool_mib":256}}'
```

The 1024/256 MiB values are per-worker staging pools, not Store capacity.
Do not pass another offload connector at the same time. This recipe is not a
P/D or `multi` connector recipe.

## Native gate and AgentX

Before starting B's formal window, publish a real nonempty prefix on one
rank, restore it on another rank and verify positive loaded tokens and
successful native GETs. Then close a streaming request after a nonempty token
and verify request, transfer and resource ownership drain on every rank.
Check native failure and quarantined-slot counters and preserve the logs.
Health checks and cache-key hits alone are insufficient.

Use `agentx-harness==1.0.13` and the same dataset, tokenizer, seed and client
settings across arms. Each valid formal window is 900 seconds, excluding
model compilation and client warmup:

```bash
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX='atom:'
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1
agentx profile \
  --url http://localhost:8000 --model DeepSeek-V4.1-Flash-1017 \
  --tokenizer "$MODEL_PATH" --tokenizer-trust-remote-code \
  --concurrency 256 --benchmark-duration 900 \
  --server-metrics http://localhost:8000/metrics \
  --output-artifact-dir ./agentx-v41-ARM-c256 \
  --public-dataset semianalysis_cc_traces_weka_062126
```

Replace `ARM` with A or B. Preserve exact model/client argv, source tree and
image identities, native capacity, per-rank metrics and client exports.
Report TTFT mean/P50/P90/P99, ITL, E2E, output throughput, total prefix reuse,
actual prefill work, completed requests and tail cancellations. The configured
300-second cancellation/drain limit can truncate the tail; record this limit
instead of treating canceled requests as successful completions. A warmup
failure has no valid formal result and must be reported separately.
