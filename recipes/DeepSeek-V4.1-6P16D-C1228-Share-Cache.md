# DeepSeek V4.1 8P16D / C1228 with shared Store

Updated 2026-11-20 in the existing recipe file: FP8 KV/index, P EP2, D NoEP128,
C1228/900s. The filename retains its original six-P name.

This recipe records the 24-GPU AgentX topology used for V4.1-Flash-1017:
four P instances with two GPUs each, and four D instances with four GPUs
each. P uses ordinary EP2 with `share_cache`; D retains TP4/DP attention and
TP-sharded MoE with expert EP disabled (NoEP). Engine session affinity is disabled. The shared
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

The performance table below belongs to the frozen measured runtime, image
`sha256:b0029ebfa2f7fa67f7b65791c999e610b624676ee723e528ef7c07e40ca5b374`.
P source is `/tmp/lirzhang-wide-ep8p16d-c1228-20261119/source-putretry-81ba44e7`;
D source is the same root's `source-no3550`. Parent identity is
`19d4aa9615631cc817ca673edc8696dc88b26b05`, with runtime ABI overlays.
The measured frontend is the fastokens Python facade and Rust P/D router;
this file does not bundle those private overlays or claim main/latest E2E
reproduction. Per-role actual PLANs remain the source/runtime authority.


The documentation base is ATOM main
`10d17e2f53915b183b26573aac78e1c4b6fccdb1`. It has not been GPU validated for
this deployment. Its V4.1 admission allows `multi` with Mooncake P/D and
Mooncake Store, explicit `kv_role`, and encoder SWA bounded replay. This is
verified in `atom/models/deepseek_v41/config.py::validate_runtime_config`.
Admission support alone does not reproduce the measured 24-GPU result.

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
| P | 4 | 2 | fp8 / fp8 | 0.85 | 64 | 32768 | 65536 |
| D | 4 | 4 | fp8 / fp8 | 0.75 | 128 | 1024 | 16384 |

All instances use prefix caching, CLI block size 16 (V4.1 resolves it to 128), state checkpoint interval
8192, level 3, FULL cudagraphs, dspark with five speculative tokens,
`{"confidence_schedule":false}`, and the `dsml_v41` tool parser. P capture
sizes are `[1,2,4,8,16,24,32,40,48,64]`; D sizes are
`[1,2,4,8,16,24,32,48,64,96,128]`. `max-num-seqs=128` applies per D attention
lane, not to global C1228. With five speculative tokens, verification uses up
to six rows/request (768 rows/lane), within the 1024-token budget. Actual
128 graph capture passed; the workload did not prove replay at BS128.
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

Run this template four times on the configured P allocations and four times
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
export ATOM_MOONCAKE_MATCHED_RAILS=auto
unset ATOM_DISABLE_MORI_EP
export ATOM_DSV41_MOE_FP8_ACTIVATIONS=0
export ATOM_USE_CUSTOM_ALL_GATHER=0 ATOM_DSPARK_DISABLE_COMPILE=1
export ATOM_DISABLE_MMAP=true PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN MC_LOG_LEVEL=INFO
# Historical network settings; verify against your actual NIC/driver.
export NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME="$NET_IFACE"
export GLOO_SOCKET_IFNAME="$NET_IFACE" MC_GID_INDEX="$GID_INDEX"
export MOONCAKE_DISABLE_HIP_DMABUF=1 HSA_ENABLE_IPC_MODE_LEGACY=1
case "$ROLE" in
  P)
    TP=2; HBM=0.85; SEQS=64; BATCH=32768; CHUNK=65536
    CAPTURE='[1,2,4,8,16,24,32,40,48,64]'; POLICY=share_cache
    EP_FLAGS=(--enable-expert-parallel)
    export OFFLOAD_MIN_LOAD_TOKENS=512 OFFLOAD_LOAD_WORKERS=4 OFFLOAD_COPY_WORKERS=4
    ;;
  D)
    TP=4; HBM=0.75; SEQS=128; BATCH=1024; CHUNK=16384
    CAPTURE='[1,2,4,8,16,24,32,48,64,96,128]'; POLICY=least_requests
    EP_FLAGS=()
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
  --trust-remote-code -tp "$TP" --enable-dp-attention "${EP_FLAGS[@]}" \
  --data-parallel-master-ip "$ROLE_IP" \
  --data-parallel-master-port "$DP_MASTER_PORT" \
  --data-parallel-base-port "$DP_BASE_PORT" --server-port "$HTTP_PORT" \
  --moe-backend standard --all2all-backend high-throughput --dp-load-balance "$POLICY" \
  --kv_cache_dtype fp8 --index-cache-dtype fp8 \
  --gpu-memory-utilization "$HBM" --max-num-seqs "$SEQS" \
  --max-num-batched-tokens "$BATCH" --attn-prefill-chunk-size "$CHUNK" \
  --enable_prefix_caching --block-size 16 --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL --cudagraph-capture-sizes "$CAPTURE" \
  --method dspark --num-speculative-tokens 5 --tool-call-parser dsml_v41 \
  --dspark-config '{"confidence_schedule":false}' \
  --kv-transfer-config "$KV_CONFIG" --disable-uvicorn-access-log \
  --spec-decode-acceptance-length 3.51
```

The legacy `-tp` + DP-attention convention resolves attention DP width from
the instance GPU width. Verify two/four discovered ranks per instance; these
are independent process groups, not global eight/sixteen-rank groups. Ordinary DEP4 means four independent
D EP4 groups; it is not wideEP16. To reproduce the measured ordinary DEP4/32
reference, D used EP_FLAGS=(--enable-expert-parallel), SEQS=32 and capture
`[1,2,4,8,16,24,32]`. The current best TTFT template above is NoEP128. GPU
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
  --prefill "$P0_URL" --prefill "$P1_URL" --prefill "$P2_URL" --prefill "$P3_URL" \
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

Untested alternative: start every P and D instance with
`USE_ATOMESH_ENTRYPOINTS=1` (ATOM standalone with the Rust frontend), which
renders and tokenizes each leg in Rust as ATOM renders it, and send AgentX's
chats to the router directly. The router command above is unchanged. This
has no GPU result for this deployment; see `docs/atomesh_rust_frontend.md`.

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
  --concurrency 1228 --benchmark-duration 900 --stats-interval 30 \
  --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 --use-server-token-count \
  --no-gpu-telemetry --num-dataset-entries 393 --slice-duration 1.0 \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --output-artifact-dir "$OUTPUT_DIR" --no-server-metrics
```

## Acceptance and reporting

The template fixes acceptance to 3.51 to match the measured performance
experiment. For normal decoding, omit `--spec-decode-acceptance-length` from
every P and D command and report measured acceptance and output quality. This
artificial constraint is not real acceptance or accuracy validation. Historical
real D acceptance was approximately 2.871–2.890; forced runs reported about
3.510. Do not compare those modes as if they were the same workload.

Before formal issuance, verify the configured owner capacity, actual memory
registration, a fresh Store for a cold experiment or verified compatible retained state, model identities,
every discovered DP rank, positive cross-rank Store restores, and validated
warmup/reuse. Reuse hot P/Store and compatible compilation assets when
retained; the measured rerun used a verified native warmup-skip handoff with
real profiling requests. This is not full cold GPU-snapshot restoration.
New D capacity requires initialization, graph capture and real P-to-D smoke. These checks must use only the experiment's allocated assets. This
recipe does not stop services, clear a shared Store or reuse another run's
hosts automatically.

900 seconds is the issuance window, not necessarily the throughput
denominator. Preserve completed, failed, cancelled and unknown request counts,
natural trajectory drain, tail credits, elapsed time and the exact throughput
denominator. The latest NoEP128 and NoEP48 formal results are:

| Metric | NoEP48 | NoEP128 (current template) |
|---|---:|---:|
| TTFT mean | 2.422806 s | 2.149373 s |
| TTFT P50 | 1.554338 s | 1.576900 s |
| TTFT P90 | 3.558231 s | 3.402836 s |
| TTFT P99 | 23.621053 s | 14.453256 s |
| ITL mean | 8.442352 ms | 8.598623 ms |
| ITL P90 | 10.492757 ms | 10.648370 ms |
| Native output throughput | 32682.6821 tok/s | 32848.6325 tok/s |
| Usage cache hit rate | 97.403940% | 97.255420% |

The 128 run has 42971 unique successful requests, 43027 native terminal requests
and 56 unACKed cancellation credits. Native throughput uses 1230.000912809 s
(900 s issue + 30 s grace + 300 s cancellation wait). The 929.264854808 s
successful-request span is a separate denominator. All model queues drained
and final native load/save/transfer/OOM errors were zero. Store had 958
allocation-start refusals, 31 batch-start failures, 1105 failed items, two
evictions and 16 retry windows recovering 395 keys with no exhausted retry.
Usage hit is not HBM hit or zero Store cost.

P/Store stayed hot; all four D groups restarted for 128, whereas 48 retained D2.
Initial D HBM and closed-loop trajectories are not fully paired. The 0.51%
throughput increase is not independently established as statistically significant;
128 is tested, not proven optimal. Relative to historical BF16/NoEP24 8P16D,
mean TTFT changed from 8.371578 to 2.149373 s, output throughput from 28008.1527
to 32848.6325 tok/s, and ITL from 8.0394 to 8.598623 ms. This is a whole-configuration
historical comparison. Earlier six-P BF16 results remain 11.207 s TTFT,
8.485 ms ITL and 31726.10 tok/s output throughput, with the different-source 11.395 s
reference; updating the FP8 command does not rewrite their recorded precision.

Evidence: `outputs/pd-overnight-20261119/noep128-all4D-formal-B-20261120/RESULT.md`,
actual per-role PLANs, FINAL-RECONCILIATION and INDEPENDENT-FINAL-REVIEW.
Raw C1228 JSONL SHA256:
`7cda1bf273c393c1c050904b3baf94489692df645c0aacc9661f917d76da6b12`.
