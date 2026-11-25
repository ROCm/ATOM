# shellcheck shell=bash
# p_env.sh: the PREFILL role (PP4 x TP1, kv_producer, Mooncake Store tier) of the MiniMax-M3.1 AgentX P/D deployment.
# Sources: our PP4 x TP1 PR stacks (defaults as on the PR branches, Jasen/m31-pp-*), the merged PP4 build's server
# (its e2e server.sh, arm B), the 1P2D C220 reference cell
# (m31best cells/cell-p1d2s-c220if8b-*.json, 101.6-101.7K) and the Store draft-window stack ROCm/ATOM#3603.
# ganyi's C17 adds nothing here: its prefill flags are his aggregated TP4 SP prefill (flags.json class prefill-sp).
#
# Usage (inside the prefill container):
#   source p_env.sh
#   python3 -m atom.entrypoints.openai_server "${P_SERVER_ARGS[@]}" --server-port <P> --port <EP> ...
#   p_env_dump        # NAME=VALUE of every exported variable (for ATOMESH_PREFILL_ENV_*); the Store settings are
#                     # read by the launcher (pd_server_atom.sh) from the prefill role env, on decode nodes as well.
# Inputs (all optional):
#   MODEL, AL         as in d_env.sh (AL must equal the decode's: 4.04 perf, "real" accuracy)
#   CONC              offered concurrency, sizes --max-num-seqs (2 x CONC, as the C220 cell's 440)
#   P_SEQS P_MNBT P_MEM            capacity overrides
#   P_STORE_LAYOUT    shared (default: one master, owners on rdma4-7, the measured C220 layout) | per_nic
#   MOONCAKE_STORE_OWNERS / MOONCAKE_STORE_DECODE_OWNERS   per-node owner plans (size them per node, see below)
#   ROLE_IP HANDSHAKE_PORT          as in d_env.sh
# Needs: the same build and image as the decode (no mixed versions); PP4 PR heads only after CONTROL.md says
# "PP4 HEADS READY"; the PP4 aiter PR tables (aiter-m31-mxfp8-rows, m31-tp1-moe-rows) in aiter-merge's
# model_configs (this role sets no AITER_CONFIG_* override); /root/.cache/atom cleared after a code change.

MODEL=${MODEL:-/share_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4}
AL=${AL:-4.04}
CONC=${CONC:-220}

# --- start clean: ganyi's TP4 decode switches and his aggregated prefill / scheduling switches stay off here.
# ATOM_AK_MXFP8_GEMM in particular would route the TP1 shared expert (6144x6144, 6144x3072 = his TP4 dense shapes)
# to amd-kernels and around the PP4 TP1 GEMM path; the AITER_CONFIG_* overrides would drop aiter's model_configs merge.
unset AITER_QUICK_REDUCE_QUANTIZATION AITER_CONFIG_GEMM_BF16 AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE \
  ATOM_AK_MXFP8_GEMM ATOM_M3_AK_INSERT ATOM_M3_AK_DECODE ATOM_M3_AK_DECODE_QOUT ATOM_M3_AK_MOE ATOM_M3_FUSED_DECODE \
  ATOM_MINIMAX_AK_AR_NORM ATOM_P1_FRONT_QKV_CHAIN ATOM_P1_AR_QKV_CODES ATOM_P1_L4_PULLQ ATOM_P1_L2_D2 ATOM_P1_AK_WIDE \
  ATOM_P1_INDEXER_SPLIT4 ATOM_P1_INDEXER_SPLIT4_WIDE ATOM_P1_INDEXER_SPLIT4_MIN8 ATOM_P1_K2_SCORER ATOM_DG_LMHEAD_AK \
  ATOM_P1_LMHEAD_RING ATOM_M3_DRAFT_AK_AR_NORM ATOM_P1_DRAFT_MARKOV_PERSIST ATOM_P1_DRAFT_FC_AK ATOM_P1_DSPARK_AUX_NEXT \
  ATOM_P1_MOE_TILES ATOM_P1_MOE_IDROWS ATOM_P1_MBF_ASM ATOM_P1_MBF_BACK ATOM_M3_SP_PREFILL ATOM_P1_SP_SHARED_LOCAL \
  ATOM_M3_PREFILL_FAST ATOM_M3_PREFILL_ATTN ATOM_M3_PREFILL_INSERT ATOM_M3_PREFILL_TOKEN_GRAPH ATOM_TOKEN_GRAPH_SIZES \
  ATOM_SCHED_PREFILL_DUTY ATOM_SCHED_DECODE_PREFILL_CAP ATOM_PREFILL_DECODE_INTERVAL ATOM_PREFILL_DELAYER_TARGET_FILL \
  ATOM_P1_SCHED_IDLE_PREFILL_CAP ATOM_P1_SCHED_SERIAL_MAXN HSA_ENABLE_IPC_MODE_LEGACY ATOM_KV_OFFLOAD \
  ATOM_KV_OFFLOAD_EXTRA_CONFIG

# --- 1. base and numerics (merged build server.sh arm B; colleague recipe env.sh) --------------------------------
# Shared expert stays its own MXFP8 MLP. NOTE: the C220 reference cell ran this role with
# ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=1 (lossy); this recipe does not, so its numbers are not directly comparable.
export ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=0
export AITER_LOG_LEVEL=WARNING PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 ARCH=gfx950
export ATOM_FORCE_ATTN_TRITON=1
export NCCL_IB_DISABLE=1 RCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=none

# --- 2. P/D transport (main; C220 reference cell; colleague recipe prefill.sh) ----------------------------------
export MC_NUM_QP_PER_EP=1 MC_DISABLE_HIP=1
export ATOM_PD_MINIMAX_LANDING=1    # PP/TP1 -> TP4 head-cut staging + decode landing (set on BOTH roles)
export ATOM_TOKENIZE_CACHE_MB=2048
# Chunked transfer; admission timeout 1500 s (LLM-Booster#617 livelock at the 300 s default).
if [[ -n "${ROLE_IP:-}" && -n "${HANDSHAKE_PORT:-}" ]]; then _ip=${ROLE_IP}; _hs=${HANDSHAKE_PORT}; else
  _ip='${ROLE_IP}'; _hs='${HANDSHAKE_PORT}'; fi
P_KV_TRANSFER_CONFIG="{\"kv_connector\":\"mooncake\",\"kv_role\":\"kv_producer\",\"proxy_ip\":\"${_ip}\",\"handshake_port\":${_hs},\"protocol\":\"rdma\",\"enable_chunked_transfer\":true,\"chunked_admission_timeout_s\":1500}"
unset _ip _hs
export PREFILL_KV_TRANSFER_CONFIG="${P_KV_TRANSFER_CONFIG}"

# --- 3. PP4 x TP1 (colleague recipe; PP4 PR defaults, written out so a run log shows them) -----------------------
export VLLM_PP_LAYER_PARTITION=16,16,16,12  # 60 target layers; the last stage also holds the DSpark draft's KV
export ATOM_PP_MAX_INFLIGHT=8   # = the IPC boundary's own default (2 x PP); also the C220 best's value under NCCL
export ATOM_PP_IPC=1            # Jasen/m31-pp-ipc: HIP IPC stage boundary on a MiniMax-M3 kv_producer, 1 rank/stage
export ATOM_M3_TP1_PREFILL=1    # atom-layer stack: TP1 fused norms / MXFP8 psc GEMMs / MoE route rows
export ATOM_M3_TP1_MOE_G2_P6=1  # atom-layer PR 4: ragged GEMM2 (second W2 copy, ~+18 GB/stage: the A/B knob for KV)
# No switch: the DSpark stacked context K/V (Jasen/m31-pp-dspark-ctx-kv) engages at TP1 on an fp8 DRAFT cache, which
# --spec-decode-kv-cache-dtype fp8 gives (main's default draft under an NVFP4 target is bf16 and would skip it).
# FP8-only work gates itself off under NVFP4 (checked on the PR heads, check.txt): tt16 (kv_cache_dtype == "fp8"),
# qknorm_v3 (kv and index fp8), the FP8 index score workspace (index_cache_dtype != "fp8" -> none), fused FP8 top-k.
export ATOM_MONO_ENABLE=0       # mono refuses PP anyway; main's default is on
export ATOM_ENABLE_METRICS_DEVICE_TIMER=1   # per-stage GPU busy for the reports (CI agentic suite sets it)
export ATOM_STEP_WATCHDOG_S=5 ATOM_STEP_KICK_S=0  # log long forwards; no GPU kick (no spin kernels on this role)

# --- 4. Mooncake Store tier (launcher: MOONCAKE_STORE=1 adds ATOM_KV_OFFLOAD=mooncake_store itself) -------------
export MOONCAKE_STORE=1
# Store draft-window stack ROCm/ATOM#3603: keep only each saved prompt's draft read window (prefill env; the decode
# needs the code). Eviction 0.89 / 0.02 with the draft split (0.88 / 0.02 without); main's launcher hard-codes
# 0.90 / 0.05, the harness launcher (Jasen/m31-agentx-launcher) reads these two.
export ATOM_MOONCAKE_STORE_DRAFT_WINDOW=1
export MOONCAKE_STORE_EVICTION_HIGH_WATERMARK=0.89 MOONCAKE_STORE_EVICTION_RATIO=0.02
export OFFLOAD_PROFILE=1 OFFLOAD_MIN_LOAD_TOKENS=0 OFFLOAD_LOAD_WORKERS=1
export MOONCAKE_STORE_PAGE_CACHE_DROP_DIRS=/share_nfs/models MOONCAKE_STORE_COMPACT_TIMEOUT=1800
# Owner plans "<numa>:<GiB>:<rdma>;...". Size per node: each NUMA keeps 128 GiB after its pins, one owner <= 832 GiB
# (ionic registration), and some NUMA nodes compact only ~814-871 GiB of huge pages (g27 / g08 on 11-18).
# shared (default, the measured C220 layout, one master): owners on rdma4-7, away from the stages' rdma0-3;
#   the P node's NUMA1 also hosts decode 0, hence its smaller owners; decode-only nodes put owners on both NUMAs.
# per_nic (one master per NIC; CONTEXT's "per-NIC pools"): every owner on exactly one of the stages' NICs
#   rdma0-3, each stage reads its own NIC's pool; untested in this deployment.
P_STORE_LAYOUT=${P_STORE_LAYOUT:-shared}
if [[ "${P_STORE_LAYOUT}" == per_nic ]]; then
  export MOONCAKE_STORE_POOLS=per_nic
  export MOONCAKE_STORE_OWNERS=${MOONCAKE_STORE_OWNERS:-0:512:rdma0;0:512:rdma1;1:384:rdma2;1:384:rdma3}
  export MOONCAKE_STORE_DECODE_OWNERS=${MOONCAKE_STORE_DECODE_OWNERS:-0:400:rdma0;0:400:rdma1;1:400:rdma2;1:400:rdma3}
else
  export MOONCAKE_STORE_POOLS=shared
  export MOONCAKE_STORE_OWNERS=${MOONCAKE_STORE_OWNERS:-0:576:rdma4;0:576:rdma5;1:384:rdma6;1:384:rdma7}
  export MOONCAKE_STORE_DECODE_OWNERS=${MOONCAKE_STORE_DECODE_OWNERS:-0:400:rdma4;0:400:rdma5;1:400:rdma6;1:400:rdma7}
fi
# Without the launcher (colleague recipe prefill.sh style) set instead, with this node's master:
#   ATOM_KV_OFFLOAD=mooncake_store
#   ATOM_KV_OFFLOAD_EXTRA_CONFIG={"mooncake_store.master":"<ip>:<port>","mooncake_store.metadata":"http://<ip>:<port>/metadata",
#     "mooncake_store.owner_rdma_devices":"rdma4,rdma5,rdma6,rdma7","mooncake_store.local_hostname":"<ip>","mooncake_store.protocol":"rdma"}
#   MC_MAX_MR_SIZE=1073741824 MC_TCP_BIND_ADDRESS=<ip>; owners with GLIBC_TUNABLES=glibc.malloc.hugetlb=1

# --- 5. server arguments (merged build arm B + P/D cell) ----------------------------------------------------------
# MNBT 8192: 16384 measured worse in 1P2D (C240 m16: ~4 batch/s) and +12% 128K C1 TTFT in the merged build.
P_SEQS=${P_SEQS:-$((2 * CONC))}
P_MNBT=${P_MNBT:-8192}
P_MEM=${P_MEM:-0.9}
# DSpark k=7 with the draft pool in fp8 (must match the decode: the Store PAGE namespace hashes the draft dtype and
# Mooncake P/D pairs only matching draft units); golden AL on both roles for perf runs.
P_SPEC=(--method dspark --draft-model "${MODEL}/dspark" --num-speculative-tokens 7 --spec-decode-kv-cache-dtype fp8)
[[ "${AL}" != real ]] && P_SPEC+=(--spec-decode-acceptance-length "${AL}")
P_SERVER_ARGS=(
  --model "${MODEL}" --trust-remote-code
  --tensor-parallel-size 1 --pipeline-parallel-size 4 --enforce-eager
  --kv_cache_dtype nvfp4 --index-cache-dtype nvfp4 --block-size 128 --enable-prefix-caching
  --max-model-len 1048576 --max-num-seqs "${P_SEQS}" --max-num-batched-tokens "${P_MNBT}"
  --gpu-memory-utilization "${P_MEM}"
  --default-chat-template-kwargs '{"thinking_mode": "enabled"}'
  --timeout-keep-alive 900
  "${P_SPEC[@]}"
  --kv-transfer-config "${P_KV_TRANSFER_CONFIG}"
)

P_ENV_NAMES=(ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4 AITER_LOG_LEVEL PYTHONHASHSEED PYTHONUNBUFFERED ARCH
  ATOM_FORCE_ATTN_TRITON NCCL_IB_DISABLE RCCL_IB_DISABLE NCCL_NET_PLUGIN MC_NUM_QP_PER_EP MC_DISABLE_HIP
  ATOM_PD_MINIMAX_LANDING ATOM_TOKENIZE_CACHE_MB PREFILL_KV_TRANSFER_CONFIG VLLM_PP_LAYER_PARTITION
  ATOM_PP_MAX_INFLIGHT ATOM_PP_IPC ATOM_M3_TP1_PREFILL ATOM_M3_TP1_MOE_G2_P6 ATOM_MONO_ENABLE
  ATOM_ENABLE_METRICS_DEVICE_TIMER ATOM_STEP_WATCHDOG_S ATOM_STEP_KICK_S MOONCAKE_STORE
  ATOM_MOONCAKE_STORE_DRAFT_WINDOW MOONCAKE_STORE_EVICTION_HIGH_WATERMARK MOONCAKE_STORE_EVICTION_RATIO
  OFFLOAD_PROFILE OFFLOAD_MIN_LOAD_TOKENS OFFLOAD_LOAD_WORKERS MOONCAKE_STORE_PAGE_CACHE_DROP_DIRS
  MOONCAKE_STORE_COMPACT_TIMEOUT MOONCAKE_STORE_POOLS MOONCAKE_STORE_OWNERS MOONCAKE_STORE_DECODE_OWNERS)

p_env_dump() {
  local n
  for n in "${P_ENV_NAMES[@]}"; do printf '%s=%s\n' "${n}" "${!n}"; done
  printf 'P_SERVER_ARGS=%s\n' "${P_SERVER_ARGS[*]}"
}
