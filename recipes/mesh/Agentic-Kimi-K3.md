# Kimi-K3 AgentX with PD Disaggregation and On-Demand Checkpoints

Run Kimi-K3 with one TP8/DCP8 prefill server and one TP8/DCP8 decode server
on two 8×MI355X nodes. This recipe uses DSpark4, ReplaySSM, native prefix
caching, and LMCache CPU offload on the prefill node for the SemiAnalysis/Weka
AgentX workload.

The prefill CPU pool remains **224 GiB per TP rank**, shared by paged KV and KDA
state. `OFFLOAD_STATE_CPU_SIZE=128` caps retained state at **128 GiB per rank**;
when state reaches that cap, approximately 96 GiB of the pool remains for KV.
This is a shared allocator, not two separately reserved pools. Plan for
**1.75 TiB of CPU cache capacity across the eight prefill ranks**, plus process
and model-loading memory.

`ATOM_STATE_CHECKPOINT_DEMAND=1` enables checkpoints at observed shared-prefix
targets. `--state-checkpoint-interval-tokens -1` keeps prompt-end anchors and
on-demand checkpoints while disabling the periodic checkpoint ladder. A
prefix can be reused only where both its KV and corresponding state remain
available; enabling on-demand does not eliminate capacity-driven eviction.

The matching cases are in
[models_atomesh.yaml](../../.github/benchmark/models_atomesh.yaml), under
`Kimi-K3-MXFP4` → `nightly`:

| CONC | Case suffix | `max-num-seqs` | Decode graph capture sizes | Comm-group reuse |
|---:|---|---:|---|---:|
| 48 | `dspark4-agentic-lmcache-1m-c48` | 96 | 1…480 | 0 |
| 56 | `dspark4-agentic-lmcache-1m-c56` | 112 | 1…560 | 1 |
| 72 | `dspark4-agentic-lmcache-1m-c72` | 144 | 1…720 | 1 |
| 88 | `dspark4-agentic-lmcache-1m-c88` | 176 | 1…880 | 1 |

Every case has the prefix `kimi-k3-mxfp4-1p1d-tp8-dcp8-` and uses the same
128 GiB state cap, 224 GiB shared CPU pool, and on-demand setting. The
commands below default to C72. Start fresh servers for each concurrency.
These are configuration recommendations; performance at the 128 GiB state
cap must be measured with the corresponding CI run.

## Prerequisites

Use the same ATOM revision and container on both nodes, with atomesh,
Mooncake, and LMCache installed. Both nodes need the Kimi-K3 target and DSpark
draft weights available locally, and RDMA connectivity between them. See the
[PD guide](../pd_disaggregation_guide.md) for networking setup and the
[single-node AgentX recipe](../Agentic-Kimi-K3.md#0-container-prerequisites)
for the Triton and local draft-model prerequisites.

## Common configuration

Run this block in the serving shell on **both nodes**. Set `ATOM_HOST_IP` to
the current node's reachable IP and use the same `CONC` on both nodes.

```bash
export MODEL_PATH="${MODEL_PATH:-/data/Kimi-K3}"
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/data/Kimi-K3-DSpark}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3}"
export CONC="${CONC:-72}"
export ATOM_HOST_IP="${ATOM_HOST_IP:?Set this to the current node IP}"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export AITER_SITUV2_A4W4=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_FLYDSL_STAGE2_FP8=1
export ATOM_GDN_SSM_DTYPE=fp16
export ATOM_USE_FLYDSL_GATHER_KV_B_PROJ=1
export ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1
export ATOM_ENABLE_REPLAYSSM=1
export ATOM_STATE_CHECKPOINT_DEMAND=1

case "${CONC}" in
  48) export AITER_REUSE_IDENTICAL_COMM_GROUPS=0 ;;
  56|72|88) export AITER_REUSE_IDENTICAL_COMM_GROUPS=1 ;;
  *) echo "Supported CONC values: 48,56,72,88" >&2; exit 2 ;;
esac

MAX_NUM_SEQS=$((2 * CONC))
ONLINE_QUANT_CONFIG='{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}'
COMMON_ARGS=(
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0
  --trust-remote-code
  --tensor-parallel-size 8
  --decode-context-parallel-size 8
  --kv_cache_dtype fp8
  --block-size 128
  --enable_prefix_caching
  --gpu-memory-utilization 0.90
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens 8192
  --state-checkpoint-interval-tokens -1
  --method dspark
  --draft-model "${DRAFT_MODEL_PATH}"
  --num-speculative-tokens 4
  --spec-decode-acceptance-length 3.36
  --online_quant_config "${ONLINE_QUANT_CONFIG}"
  --tool-call-parser kimi_k3
)
```

The acceptance-length override is for performance benchmarking. Omit it for
correctness evaluation; see [DSpark](../DSpark.md).

## Prefill node

The `multi` connector sends KV to decode through Mooncake and stores reusable
KV/state locally through LMCache. The CPU size settings apply only here.

```bash
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=224
export OFFLOAD_STATE_CPU_SIZE=128
export LMCACHE_CHUNK_SIZE=1024
export LMCACHE_NUMA_MODE=auto
export ATOM_NUMA_BIND=1
export ATOM_NUMA_NODE=0,0,0,0,1,1,1,1
export ATOM_AUTO_NUMA_BIND=0
export OFFLOAD_PROFILE=1
export OFFLOAD_GPU_STAGING_CHUNKS=32

PREFILL_KV_TRANSFER_CONFIG="{\"kv_connector\":\"multi\",\"connectors\":[{\"kv_connector\":\"mooncake\",\"kv_role\":\"kv_producer\",\"proxy_ip\":\"${ATOM_HOST_IP}\",\"handshake_port\":6301},{\"kv_connector\":\"lmcache_offload\",\"kv_role\":\"offload\"}]}"
python3 -m atom.entrypoints.openai_server \
  "${COMMON_ARGS[@]}" \
  --server-port 8010 \
  --kv-transfer-config "${PREFILL_KV_TRANSFER_CONFIG}" \
  2>&1 | tee "prefill-kimik3-agentx-c${CONC}.log"
```

Adapt the explicit NUMA mapping if the host topology differs. At startup,
check for `kimi_k3 offload: state tier up` in each rank's log; the state tier
reports its entry size and retention capacity.

## Decode node

After applying the common block in a separate shell on the decode node,
start the Mooncake consumer. Decode uses no LMCache CPU offload in this recipe.
Its dense graph range matches CI's `cudagraph: auto` calculation:
`2 * CONC * (1 + 4)` query tokens.

```bash
GRAPH_MAX=$((MAX_NUM_SEQS * 5))
CUDAGRAPH_CAPTURE_SIZES="[$(seq -s, 1 "${GRAPH_MAX}")]"
DECODE_KV_TRANSFER_CONFIG="{\"kv_connector\":\"mooncake\",\"kv_role\":\"kv_consumer\",\"proxy_ip\":\"${ATOM_HOST_IP}\",\"handshake_port\":6301}"
python3 -m atom.entrypoints.openai_server \
  "${COMMON_ARGS[@]}" \
  --server-port 8020 \
  --level 3 \
  --cudagraph-mode FULL \
  --cudagraph-capture-sizes "${CUDAGRAPH_CAPTURE_SIZES}" \
  --kv-transfer-config "${DECODE_KV_TRANSFER_CONFIG}" \
  2>&1 | tee "decode-kimik3-agentx-c${CONC}.log"
```

## Router and AgentX client

Run the router on the client host, using reachable addresses for both servers.

```bash
export PREFILL_IP="${PREFILL_IP:?Set the prefill node IP}"
export DECODE_IP="${DECODE_IP:?Set the decode node IP}"

atomesh launch \
  --host 0.0.0.0 --port 8000 \
  --pd-disaggregation \
  --prefill "http://${PREFILL_IP}:8010" \
  --decode "http://${DECODE_IP}:8020" \
  --policy random \
  --backend atom \
  --log-level info \
  2>&1 | tee router-kimik3-agentx.log
```

In another client shell, follow
[Run the AgentX Profile](../Agentic-Kimi-K3.md#2-run-the-agentx-profile) with
`CONC=72`, `PORT=8000`, and the same model/tokenizer paths and served model
name. The router must be running on that host for the example's localhost
URL. CI pins SemiAnalysis AIPerf to
`754356e9a39acc6cc6afb242d123bb57c3fb6f75`, uses
`inferencex-agentx-mvp` / `semianalysis_cc_traces_weka_062126`, and profiles
each concurrency for 3,600 seconds with 10 warmup requests per lane.

For CI, select `ATOMesh Benchmark`, suite `nightly`, model `Kimi-K3-MXFP4`,
and one exact case name, for example
`kimi-k3-mxfp4-1p1d-tp8-dcp8-dspark4-agentic-lmcache-1m-c72`.
