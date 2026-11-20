# DeepSeek V4.1 6P16D / C1228 with shared Store

2026-11-20 update: new performance runs explicitly use FP8 KV and FP8 index.
Historical BF16 measurements retain their original dtype labels. The latest
measured 8P16D deployment and the completed NoEP128 capacity experiment are documented in
[8P16D FP8 recipe](DeepSeek-V4.1-8P16D-C1228-FP8-Share-Cache.md).
This document retains its six-P topology and original public-source validation boundary.

This recipe records the 22-GPU AgentX topology used for V4.1-Flash-1017:
three P instances with two GPUs each, and four D instances with four GPUs
each. P uses `share_cache`; engine session affinity is disabled. The shared
CPU-backed Mooncake Store has 1574 GiB (1690069630976 bytes) of mounted owner
capacity. The master process alone does not allocate this capacity.

For the public master/owner entry points, see the
[Mooncake Store configuration and deployment notes](../atom/kv_transfer/offload/README.md#configuration).
They show `mooncake_master` and `mooncake_client` owner commands and the
SDK/device requirements. Allocate dedicated owners with total mounted capacity
1690069630976 bytes; verify the size-unit interpretation of your client build.
Choose owner NUMA nodes and NICs explicitly, keep requester NICs separate,
and verify actual memory registration and successful put/get. Historical
owners used deployment-specific service scripts; those scripts are not public
dependencies supplied by this branch. Do not run the all-in-one launcher on
shared hosts without inspecting its allocation and lifecycle operations.

## Source and validation boundary

The documentation base is ATOM main
`10d17e2f53915b183b26573aac78e1c4b6fccdb1`. It has not been GPU validated for
this deployment. Its V4.1 admission allows `multi` with Mooncake P/D and
Mooncake Store, explicit `kv_role`, and encoder SWA bounded replay. This is
verified in `atom/models/deepseek_v41/config.py::validate_runtime_config`.
Admission support does not establish a 22-GPU E2E result.

Historical common-fix experiments used adapted versions of public changes
[#3097](https://github.com/ROCm/ATOM/pull/3097),
[#3323](https://github.com/ROCm/ATOM/pull/3323), and
[#3324](https://github.com/ROCm/ATOM/pull/3324), identified in the frozen
evidence by `e5d05fe`, `71201db`, and `5d536db`. These abbreviations describe
lineage, not a public reproducible composite checkout. The historical
composite and container image are not advertised as public release artifacts.
Pin the public main commit above for a new deployment and record dependency
versions. This normalized topology recipe is a public starting point, not an
exact reproduction of the historical composite or its measured performance.

## Deployment configuration

Configure host IPs, GPU lists, requester/owner RDMA devices and ports explicitly.
No historical host is reserved by this recipe. Each P process has two dynamic
attention DP ranks; each D process has four. The router must discover the
actual ranks. Do not map session IDs to a fixed rank or freeze P-to-D pairs.

| Role | Instances | GPUs per instance | KV / index dtype | HBM fraction | Sequences | Batched tokens | Prefill chunk |
|---|---:|---:|---|---:|---:|---:|---:|
| P | 3 | 2 | fp8 / fp8 | 0.85 | 64 | 32768 | 65536 |
| D | 4 | 4 | fp8 / fp8 | 0.75 | 24 | 1024 | 16384 |

All instances use prefix caching, block size 128, state checkpoint interval
8192, level 3, FULL cudagraphs, dspark with five speculative tokens,
`{"confidence_schedule":false}`, and the `dsml_v41` tool parser. P capture
sizes are `[1,2,4,8,16,24,32,40,48,64]`; D sizes are `[1,2,4,8,16,24]`.
Model and tokenizer must refer to the same V4.1-Flash-1017 checkpoint.

P transfer configuration is a `multi` connector with a Mooncake
`kv_producer` and a `mooncake_store` `offload` connector. D uses a Mooncake
`kv_consumer`. Each instance has its own HTTP, handshake and DP rendezvous
ports. Store requester load/save pools are both 1024 MiB. Set
`OFFLOAD_MIN_LOAD_TOKENS=512`, `OFFLOAD_LOAD_WORKERS=4`,
`OFFLOAD_COPY_WORKERS=4`, `ATOM_FRONTEND_PREFIX_HASH=1`,
`ATOM_DP_SESSION_AFFINITY=0`, `MC_NUM_QP_PER_EP=1`, and
`ATOM_MOONCAKE_MATCHED_RAILS=auto` consistently. Network interface, GID and
HIP-DMABUF settings depend on the deployed NIC/runtime.

D uses `--dp-load-balance least_requests`. The historical model argv did not
set `--scheduler-delay-factor` or `--prefill-admit-policy`; the pinned main
defaults are 0.0 and `interval`. Encoder SWA bounded replay defaults to true;
keep it enabled for Store and verify the resolved configuration.
Do not introduce the experimental V4 short-restore flags or V6 scheduling
candidate. Neither demonstrated an overall performance benefit. The
#3350 + two-stream evidence covers bounded stability; it is not a completed
C1228 performance result.

## P and D command template

Run this template three times on the configured P allocations and four times
on the configured D allocations. Export `ROLE=P` or `ROLE=D`, `ROLE_IP`,
`GPUS`, `HTTP_PORT`, `HANDSHAKE_PORT`, `DP_MASTER_PORT`, `DP_BASE_PORT`,
`NET_IFACE` and `GID_INDEX` separately for each instance. P also requires
`STORE_MASTER`, `STORE_METADATA`, `REQUESTER_RDMA` and `OWNER_RDMA`. The
master/metadata variables are `IP:port` and `http://IP:port/metadata`.
Run inside a prepared ROCm/Mooncake environment with the checkpoint mounted
at `MODEL_PATH`, GPU/RDMA access and unlimited memlock. This template does
not allocate hosts or stop services.

```bash
export HIP_VISIBLE_DEVICES="$GPUS" ATOM_HOST_IP="$ROLE_IP"
export ATOM_MONO_ENABLE=1 ATOM_FRONTEND_PREFIX_HASH=1
export ATOM_DP_SESSION_AFFINITY=0 MC_NUM_QP_PER_EP=1
export ATOM_MOONCAKE_MATCHED_RAILS=auto ATOM_DISABLE_MORI_EP=1
export ATOM_USE_CUSTOM_ALL_GATHER=0 ATOM_DSPARK_DISABLE_COMPILE=1
export ATOM_DISABLE_MMAP=true PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN MC_LOG_LEVEL=INFO
# Historical network settings; verify against your actual NIC/driver.
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME="$NET_IFACE"
export GLOO_SOCKET_IFNAME="$NET_IFACE" MC_GID_INDEX="$GID_INDEX"
export MOONCAKE_DISABLE_HIP_DMABUF=1 HSA_ENABLE_IPC_MODE_LEGACY=1
case "$ROLE" in
  P)
    TP=2; HBM=0.85; SEQS=64; BATCH=32768; CHUNK=65536
    CAPTURE='[1,2,4,8,16,24,32,40,48,64]'; POLICY=share_cache
    export OFFLOAD_MIN_LOAD_TOKENS=512 OFFLOAD_LOAD_WORKERS=4 OFFLOAD_COPY_WORKERS=4
    ;;
  D)
    TP=4; HBM=0.75; SEQS=24; BATCH=1024; CHUNK=16384
    CAPTURE='[1,2,4,8,16,24]'; POLICY=least_requests
    ;;
  *) echo 'ROLE must be P or D' >&2; exit 1 ;;
esac
KV_CONFIG=$(python3 - <<'PY'
import json, os
e = os.environ
pd = {"kv_connector": "mooncake",
      "kv_role": "kv_producer" if e["ROLE"] == "P" else "kv_consumer",
      "proxy_ip": e["ROLE_IP"], "http_port": int(e["HTTP_PORT"]),
      "handshake_port": int(e["HANDSHAKE_PORT"]), "protocol": "rdma"}
if e["ROLE"] == "P":
    extra = {"mooncake_store.master": e["STORE_MASTER"],
             "mooncake_store.metadata": e["STORE_METADATA"],
             "mooncake_store.protocol": "rdma",
             "mooncake_store.local_hostname": e["ROLE_IP"],
             "mooncake_store.rdma_devices": e["REQUESTER_RDMA"],
             "mooncake_store.owner_rdma_devices": e["OWNER_RDMA"],
             "mooncake_store.load_pool_mib": 1024,
             "mooncake_store.save_pool_mib": 1024}
    pd = {"kv_connector": "multi", "connectors": [pd,
          {"kv_connector": "mooncake_store", "kv_role": "offload",
           "kv_connector_extra_config": extra}]}
print(json.dumps(pd))
PY
)
python3 -m atom.entrypoints.openai_server --model "$MODEL_PATH" \
  --trust-remote-code -tp "$TP" --enable-dp-attention \
  --data-parallel-master-ip "$ROLE_IP" \
  --data-parallel-master-port "$DP_MASTER_PORT" \
  --data-parallel-base-port "$DP_BASE_PORT" --server-port "$HTTP_PORT" \
  --moe-backend standard --all2all-backend none --dp-load-balance "$POLICY" \
  --kv_cache_dtype fp8 --index-cache-dtype fp8 \
  --gpu-memory-utilization "$HBM" --max-num-seqs "$SEQS" \
  --max-num-batched-tokens "$BATCH" --attn-prefill-chunk-size "$CHUNK" \
  --enable_prefix_caching --block-size 128 --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL --cudagraph-capture-sizes "$CAPTURE" \
  --method dspark --num-speculative-tokens 5 --tool-call-parser dsml_v41 \
  --dspark-config '{"confidence_schedule":false}' \
  --kv-transfer-config "$KV_CONFIG" --disable-uvicorn-access-log
```

The legacy `-tp` + DP-attention convention resolves attention DP width from
the instance GPU width. Verify two/four discovered ranks per instance; these
are independent process groups, not global six/sixteen-rank groups. GPU
lists on one host must be disjoint; assign nonoverlapping port ranges for
all engine sockets, not only the HTTP ports.

## Router and encoding ingress

The public-main router command uses deployment addresses and explicit rank
discovery:

```bash
atomesh launch --host "$ROUTER_IP" --port "$ROUTER_PORT" \
  --prometheus-port "$ROUTER_METRICS_PORT" --pd-disaggregation --dp-aware \
  --prefill-policy cache_aware --decode-policy cache_aware \
  --cache-threshold 1.0 --balance-abs-threshold 0 \
  --balance-rel-threshold 1.0 --atom-pd-rank-mapping-policy none \
  --backend atom --model-path "$MODEL_PATH" --disable-circuit-breaker \
  --request-timeout-secs 1800 \
  --prepare-queue-timeout-ms 5000 --prepare-timeout-ms 15000 \
  --prefill "$P0_URL" --prefill "$P1_URL" --prefill "$P2_URL" \
  --decode "$D0_URL" --decode "$D1_URL" --decode "$D2_URL" --decode "$D3_URL"
```

The historical frozen router command omitted `--dp-aware` and added the
experimental `--prefill-context-pools-baseline`; that experimental flag is
omitted here. The current command explicitly enables discovery. These
differences require runtime validation and prevent claiming byte-for-byte
reproduction of the historical router.

AgentX sends chat requests through an encoding ingress before atomesh. The
ingress must use the exact V4.1 tokenizer/chat encoding, preserve streaming,
tool payloads, request IDs and cancellation, and forward token IDs to the
router. Historical capacity was four encoder processes, sixteen workers,
1280 queued requests, 2048 connections, 512 keepalive connections and 600 s
keepalive. Segment cache was disabled (0 MiB). Its frozen candidate script is
not a public dependency supplied by this branch; a substitute requires
encoding equivalence and cancellation validation. Sending chat directly to
an incompatible token-only endpoint is not an equivalent reproduction.

The frozen ingress argv template was:

```bash
python3 "$ENCODER_SCRIPT" --model "$MODEL_PATH" \
  --backend "$ROUTER_URL" --host "$ENCODER_IP" --port "$ENCODER_PORT" \
  --processes 4 --workers 16 --queued 1280 --max-connections 2048 \
  --max-keepalive-connections 512 --timeout-keep-alive 600 --segment-cache-mib 0
```

`ENCODER_SCRIPT` must name the frozen implementation or an independently
validated equivalent. It is not a script supplied by public ATOM main.

## AgentX command

Use an AIPerf build supporting the AgentX scenario and the flags below. Record
its public source/version and dataset revision. The frozen trace revision was
`23f152f6f0f9399a85901b89a6458def0ef16729`; trace SHA256 was
`29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`.

```bash
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true
export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
aiperf profile --scenario inferencex-agentx-mvp --url "$ENCODING_URL" \
  --endpoint /v1/chat/completions --endpoint-type chat --streaming \
  --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" --tokenizer-trust-remote-code \
  --concurrency 1228 --benchmark-duration 1800 --stats-interval 30 \
  --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 --use-server-token-count \
  --no-gpu-telemetry --num-dataset-entries 393 --slice-duration 1.0 \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --output-artifact-dir "$OUTPUT_DIR" --no-server-metrics
```

## Acceptance and reporting

Normal decoding omits `--spec-decode-acceptance-length`. Report the measured
acceptance length and output quality separately. A performance-only comparison
may add `--spec-decode-acceptance-length 3.51` to every P and D command. This
artificial constraint is not real acceptance or accuracy validation. Historical
real D acceptance was approximately 2.871–2.890; forced runs reported about
3.510. Do not compare those modes as if they were the same workload.

Before formal issuance, verify the configured owner capacity, actual memory
registration, empty initial Store for a fresh experiment, model identities,
every discovered DP rank, positive cross-rank Store restores, and identical
warmup. These checks must use only the experiment's allocated assets. This
recipe does not stop services, clear a shared Store or reuse another run's
hosts automatically.

1800 seconds is the issuance window, not necessarily the throughput
denominator. Preserve completed, failed, cancelled and unknown request counts,
natural trajectory drain, tail credits, elapsed time and the exact throughput
denominator. A previous forced run had 69246 successes from 69338 sent, 92
missing tail credits and forced completion after the cancellation wait. It did
not establish natural whole-DAG drain. Its mean TTFT was 11.207 s, P99 47.151 s,
ITL 8.485 ms and output throughput 31726.10 tokens/s (1442.10 per GPU). A
historical different-source/node baseline had 11.395 s, 37.847 s, 7.500 ms and
32577.68 tokens/s respectively. This is not evidence of overall improvement
or a fair same-source performance guarantee.
These historical measurements used BF16 KV with FP8 index; the updated FP8 KV
command template does not change their recorded precision or constitute a
rerun of those measurements.
