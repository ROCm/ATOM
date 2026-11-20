# DeepSeek V4.1 8P16D / C1228 / FP8 KV / shared Store

Updated 2026-11-20 (Asia/Shanghai). This records the measured deployment,
including its private runtime dependencies; it is not a claim that an arbitrary
main/latest checkout reproduces the measured result. The recipe branch contains documentation and command templates; it does not
bundle the private runtime overlays or model checkpoint.

## Reviewed NoEP48 and NoEP128 results

Both D NoEP48 and D NoEP128 completed formal runs and independent review.
For the current TTFT priority, use the tested 128 setting; retain 48 as the
comparison with slightly lower ITL. This does not establish the optimal
capacity or a statistically significant throughput improvement.

| Metric | NoEP48 | NoEP128 | Change |
|---|---:|---:|---:|
| TTFT mean | 2.422806 s | 2.149373 s | -11.29% |
| TTFT P50 | 1.554338 s | 1.576900 s | +1.45% |
| TTFT P90 | 3.558231 s | 3.402836 s | -4.37% |
| TTFT P99 | 23.621053 s | 14.453256 s | -38.81% |
| ITL mean | 8.442352 ms | 8.598623 ms | +1.85% |
| Native output throughput | 32682.6821 tok/s | 32848.6325 tok/s | +0.51% |
| Usage cache hit rate | 97.403940% | 97.255420% | -0.149 percentage points |

The 128 run has 42971 unique successful requests, 43027 sent/native terminal
requests and 56 unACKed cancellation credits under the original protocol.
All eight model queues drained; final native load/save/transfer/OOM errors
were zero. Store experienced capacity pressure (958 allocation-start
refusals, 31 batch-start failures, 1105 failed items and two evictions),
with 16 retry windows recovering 395 keys and zero exhausted retries.
These are measured pressure events, not evidence of zero Store overhead.

| Role | Instances | GPUs / instance | Attention lanes / instance | Expert decomposition | KV / index | HBM fraction |
|---|---:|---:|---:|---|---|---:|
| P | 4 | 2 | 2 | EP2 | FP8 / FP8 | 0.85 |
| D | 4 | 4 | 4 | TP-sharded MoE, no expert EP | FP8 / FP8 | 0.75 |

P uses `share_cache`; D uses `least_requests`. The shared Mooncake RDMA
Store has 1574 GiB mounted owner capacity (798 GiB + 776 GiB). Preserve its
namespace, model identity and key layout when reusing the resident state.
The observed deployment uses g41 for P, g29/g35 for D; g15 is borrowed and
must not be used by this run. Addresses, GPU lists, RDMA rails and ports must
come from the actual per-role manifests.

## D capacity configuration

`--max-num-seqs` applies to each attention lane's scheduler in this DPA4
deployment. It is not a global C1228 limit. KV admission, pending remote KV,
token budget and available memory can make the running batch smaller.

| Variant | Max requests / lane | Total configured slots across 16 D lanes | FULL graph capture sizes |
|---|---:|---:|---|
| Previous measured reference | 48 | 768 | `[1,2,4,8,16,24,32,48]` |
| Tested setting for current TTFT priority | 128 | 2048 | `[1,2,4,8,16,24,32,48,64,96,128]` |

These are parameter fragments for the actual D launch, not a complete server
command. Retain the existing per-role connector, GPU, port and source settings:

```bash
# Common D settings, including explicit FP8 KV (index FP8 alone is insufficient).
--kv_cache_dtype fp8 --index-cache-dtype fp8 --block-size 16 \
--gpu-memory-utilization 0.75 --max-num-batched-tokens 1024 \
--cudagraph-mode FULL --num-speculative-tokens 5

# Measured reference:
--max-num-seqs 48 --cudagraph-capture-sizes '[1,2,4,8,16,24,32,48]'

# Tested setting for current TTFT priority:
--max-num-seqs 128 --cudagraph-capture-sizes '[1,2,4,8,16,24,32,48,64,96,128]'
```

With five speculative tokens, the target uses up to six verification query
rows per request: 128 requests require 768 rows per lane, below the existing
1024 token budget. The default 512 TokenGraph cap applies to a different
TR_HYBRID path; this experiment retains the measured FULL ReqGraph mode.
Actual graph128 capture, STATE/PAGE capacity and workspace passed native
startup validation for the measured run; repeat this gate on another runtime.
The 900-second issue observation has 534192 lane-steps: 1121 exceeded BS48
(0.20985%), none exceeded BS64. Capturing graph128 does not prove it was
replayed in this workload, and the optimum capacity remains unmeasured.
NoEP still retains DPA4 and TP4;
omit the expert-parallel enable flag instead of removing attention DP.

## Exact source and execution boundary

Runtime image ID: `sha256:b0029ebfa2f7fa67f7b65791c999e610b624676ee723e528ef7c07e40ca5b374`.

- P source: `/tmp/lirzhang-wide-ep8p16d-c1228-20261119/source-putretry-81ba44e7`.
- D source: `/tmp/lirzhang-wide-ep8p16d-c1228-20261119/source-no3550`.
- Parent source identity: `19d4aa9615631cc817ca673edc8696dc88b26b05`.
- Five runtime ABI overlays are identified by SHA in each group's PLAN.
- Frontend: existing Python facade with fastokens final encoding and ordinary
  Rust HTTP P/D router. This does not deploy the complete #3372/#3489
  EngineCore/engine_v1 runtime.

Use the actual PLAN/parent manifests as the launch authority; these source
paths include runtime-specific patches and are not a portable public recipe.
Keep fixed acceptance 3.51 for this throughput experiment; it is not an
accuracy validation.

## Complete per-instance launch templates

Set `ROLE`, `ROLE_IP`, `GPUS`, `MODEL_PATH`, `HTTP_PORT`, `HANDSHAKE_PORT`,
`DP_MASTER_PORT`, `DP_BASE_PORT`, `NET_IFACE` and `GID_INDEX` per instance.
P also requires `STORE_MASTER`, `STORE_METADATA`, `REQUESTER_RDMA`, and
`OWNER_RDMA` as described in the [Store deployment recipe](DeepSeek-V4.1-6P16D-C1228-Share-Cache.md).
Run P four times on disjoint two-GPU groups, D four times on disjoint four-GPU
groups. The default below is P EP2 plus D NoEP128. `D_EXPERT_MODE=ep4`
enables ordinary EP4 on each D group; the measured ordinary EP4 reference
used `D_MAX_NUM_SEQS=32`. An EP4/128 combination has not been formally measured.

These templates normalize addresses and omit diagnostic hooks; actual per-role
PLANs remain the authority for the recorded result. Run in the exact prepared
runtime or revalidate CLI, V4.1 admission, codecs and real P-to-D transfer on
another checkout. Do not infer a performance rerun from successful parsing.

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
    TP=4; HBM=0.75; SEQS=${D_MAX_NUM_SEQS:-128}; BATCH=1024; CHUNK=16384
    POLICY=least_requests
    case "${D_EXPERT_MODE:-noep}" in
      noep) EP_FLAGS=() ;;
      ep4) EP_FLAGS=(--enable-expert-parallel) ;;
      *) echo 'D_EXPERT_MODE must be noep or ep4' >&2; exit 1 ;;
    esac
    case "$SEQS" in
      32) CAPTURE='[1,2,4,8,16,24,32]' ;;
      48) CAPTURE='[1,2,4,8,16,24,32,48]' ;;
      128) CAPTURE='[1,2,4,8,16,24,32,48,64,96,128]' ;;
      *) echo 'D_MAX_NUM_SEQS must be 32, 48 or 128' >&2; exit 1 ;;
    esac
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
  --spec-decode-acceptance-length "${ACCEPTANCE_LENGTH:-3.51}"
```

For Docker, preserve GPU/RDMA devices, `--ipc host`, `--network host`,
`--cap-add IPC_LOCK`, `--ulimit memlock=-1:-1`, and sufficient nofile limits.
Assign disjoint CPU budgets to models, encoding/router and Store owners;
validate effective worker affinity instead of assuming the container mask is
inherited unchanged. Do not copy the recorded P2 single-CPU worker binding
into a new deployment: its origin is recorded, and expanding it did not
improve the controlled wave. Keep model/Engram/compilation cache mounts when
reusing the runtime. No script in this recipe stops or allocates services.

P retains a 65536 attention-prefill chunk setting, but its scheduler query
budget is 32768; these two fields do not both mean the amount of actual work
performed in a forward. CLI block size 16 resolves to 128 for this V4.1
layout; verify the resolved PAGE/STATE/index configuration on startup.
`high-throughput` is a selected backend setting; on D NoEP it does not imply
that an EP dispatch runs. Check runtime group widths and trace kernel paths.

## Ordinary DEP and wideEP boundaries

| Mode | P | D | Status |
|---|---|---|---|
| Current TTFT setting | Four TP2/DPA2/EP2 groups | Four TP4/DPA4/EP1 groups, 128 slots/lane | Formal C1228/900s measured |
| Ordinary DEP reference | Four TP2/DPA2/EP2 groups | Four TP4/DPA4/EP4 groups, 32 slots/lane | Formal C1228/900s measured |
| wideEP reference | EP across P8 | EP across D16 | Historical experiment, not current best |

DEP means expert parallelism across the attention DP ranks in each group.
Four independent EP4 D groups are not one EP16 group. Enabling
`--enable-expert-parallel` on four processes does not create cross-node EP16.
wideEP additionally needs global DP/EP rendezvous, ranks, cross-node transport,
compatible MegaMoE and the exact warmup/addressing changes; it cannot be
reproduced by toggling this group's flag. No new RCCL metadata or busy-round
sync-interval change is claimed for the current NoEP128 result.

## Four-P / four-D router and encoding ingress

Use the compatible Rust router build and Python encoding facade. The current
measured fastokens facade is a runtime dependency, not bundled in this branch.
The following portable router template explicitly discovers DP ranks; exact
historical router flags differ as documented in the six-P recipe. Set all four
P URLs and four D URLs and inspect actual rank discovery before issuance.

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

Preserve tokenizer/chat/tool encoding, request IDs, streaming and cancellation.
Do not silently replace a token-only backend with chat payloads. For a
benchmark-only queue policy, record changed rejection/timeouts separately
from production QoS; disabling them is not a model service-time improvement.

## C1228 / 900-second AgentX issuance

The following is the original native issuance command normalized to deployment
URLs and output paths. The measured same-runtime rerun additionally installed
a verified resident-state warmup-skip receipt. On a fresh runtime run complete
warmup once, save compatible assets at a drained point, then validate reuse.
Do not copy a fixed benchmark ID or fabricate a restoration receipt across
models, dtypes, Store namespaces or client revisions.

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

A local compilation cache is not a full KV snapshot. Retain the original
native 30-second grace and 300-second cancel-credit wait. Only WARMUP can use
the validated replay handoff; PROFILING must reach the real P/D backend.
A missing helper means use the complete native warmup and report that fact.
The measured 128 Store/P remained resident while all four D groups restarted;
this does not prove complete cold GPU-state restoration.

## Fast iteration and statistics

Preserve hot P and resident Store. Reuse compatible compilation caches and
the verified native client warmup-skip handoff. New D capacity requires
necessary model initialization and FULL graph capture, followed by a real
Store-to-P-to-D smoke and health gate. This is resident-state reuse, not a
verified full GPU snapshot cold restore.

Keep the 900-second issue window and the original grace/cancel protocol.
The measured 48 reference uses a native throughput denominator of
1230.001682841 seconds, and 128 uses 1230.000912809 seconds; do not substitute 900 seconds or the separate
success-span definition. Preserve canceled/unACKed credits and include the
first measured minute. The 48 reference retained D2 while restarting three
D groups; the 128 experiment restarts all four D groups, so D HBM initial
states are not fully paired. Report whole-window and minute-level results,
cache tiers, restore work, P queue/query work and D batch/waiting/step time.

Do not record GPU KV snapshots during formal measurement. Save at a drained
point and validate dtype/layout/namespace before importing. New runs default
to FP8 KV; historical BF16 measurements remain labeled BF16.

Local evidence root:
`C:/Users/lirzhang/Documents/Codex/outputs/pd-overnight-20261119/`.
Measured results: `FINAL-0830.md`.
128 manifests: `noep128-all4D-formal-B-20261120/D*/PLAN.json`.
Final 128 results: `noep128-all4D-formal-B-20261120/RESULT.md`.
Independent review: `noep128-all4D-formal-B-20261120/INDEPENDENT-FINAL-REVIEW.json`.
Aggregate SHA256: `3b44a82d39314a4356fd1791aa164b69173bbc0e01364c423463bd18c32e2b19`.
Large raw JSONL stays on g35, SHA256:
`7cda1bf273c393c1c050904b3baf94489692df645c0aacc9661f917d76da6b12`.
