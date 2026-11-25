# shellcheck shell=bash
# d_env.sh: the DECODE role (TP4, kv_consumer) of the MiniMax-M3.1 AgentX P/D deployment.
# Decode stack = ganyi's C17 recipe (origin/p1-c17-reproducible 759de87d9, recipe/env_armC17.sh chain), cut to what a
# P/D consumer runs at high concurrency and to the campaign's numerics rule. Every C17 item, its arm, class and
# numerics: flags.json; what is left out and why: excluded.txt.
#
# Usage (inside the decode container, before starting the server):
#   source d_env.sh                          # exports the role env, builds D_KV_TRANSFER_CONFIG and D_SERVER_ARGS
#   python3 -m atom.entrypoints.openai_server "${D_SERVER_ARGS[@]}" --server-port <P> --port <EP> ...
#   d_env_dump                               # NAME=VALUE of every variable this file exports (for ATOMESH_DECODE_ENV_*)
# Inputs (all optional):
#   MODEL     checkpoint (default the Quark MXFP4 M3.1-Flash; draft in $MODEL/dspark)
#   AL        4.04 = InferenceX golden acceptance length for DSpark k=7 (perf runs); "real" = real acceptance
#             (GSM8K / needle). Must equal the prefill's.
#   D_HALF    1 = GPUs 4-7 + NUMA1 + rdma4-7 (default), 0 = GPUs 0-3 + NUMA0 + rdma0-3 (a second decode on one node).
#             GPU selection itself (ROCR_VISIBLE_DEVICES=GPU-<uuid>,...) is the harness's.
#   D_NICS    override the ib_device list; "launcher" = the harness launcher's ${WORKER_RDMA_DEVICES} placeholder.
#   D_SEQS D_MNBT D_MEM D_CG   capacity overrides (defaults below)
#   ROLE_IP HANDSHAKE_PORT     when both are set the KV config is expanded; otherwise it keeps the CI launcher's
#                              "${ROLE_IP}" / ${HANDSHAKE_PORT} placeholders.
# Needs: the ONE build both roles run (atom-merge's tree) with ganyi's vendored packages at the tree root on
# PYTHONPATH (amd_kernels/ p1k2/ p1l2/ l4c_y/ df_draft_ar/), aiter-merge's aiter, a node-local
# FLYDSL_RUNTIME_CACHE_DIR (cold start compiles for ~7 min), and /root/.cache/atom cleared after a code change.

TUNED_CONFIGS=${TUNED_CONFIGS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../recipe/configs" && pwd)}
MODEL=${MODEL:-/share_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4}
AL=${AL:-4.04}
D_HALF=${D_HALF:-1}

# --- start from a clean slate: none of C17's prefill / aggregated / excluded switches leak in from a parent shell
unset AITER_QUICK_REDUCE_QUANTIZATION \
  ATOM_M3_SP_PREFILL ATOM_M3_SP_PREFILL_COMM ATOM_M3_SP_PREFILL_OPROJ ATOM_M3_SP_PREFILL_QGATHER \
  ATOM_M3_SP_PREFILL_MIN_TOKENS ATOM_P1_SP_HANDOFFS ATOM_P1_SP_QG_AK ATOM_P1_SP_A2A_UNPACK ATOM_P1_SP_OPROJ_AK \
  ATOM_P1_SP_MOE_SLOTS ATOM_P1_SP_MOE_COMPACT ATOM_P1_SP_MOE_COMPACT_MIN_TOKENS ATOM_P1_SP_SHARED_LOCAL \
  ATOM_P1_SP_SHARED_LOCAL_MIN_TOKENS ATOM_P1_SP_SHARED_LOCAL_SIDE_ROWS ATOM_M3_PREFILL_FAST ATOM_M3_PREFILL_ATTN \
  ATOM_M3_PREFILL_INSERT ATOM_M3_PREFILL_TOKEN_GRAPH ATOM_TOKEN_GRAPH_SIZES ATOM_SCHED_PREFILL_DUTY \
  ATOM_SCHED_DECODE_PREFILL_CAP ATOM_PREFILL_DECODE_INTERVAL ATOM_PREFILL_DELAYER_TARGET_FILL \
  ATOM_P1_SCHED_FINISH_HOLD_LEFT ATOM_P1_SCHED_FINISH_HOLD_MAX_MS ATOM_P1_SCHED_FINISH_HOLD_MAX_TOKENS \
  ATOM_P1_SCHED_ITL_RISK_MS ATOM_P1_SCHED_ITL_RISK_MAX_MS ATOM_P1_SCHED_IDLE_PREFILL_CAP ATOM_P1_SCHED_SERIAL_MAXN \
  ATOM_P1_SCHED_SERIAL_AGE_TOKENS ATOM_P1_SCHED_SERIAL_GAP_TOKENS ATOM_P1_MOE_TILES ATOM_P1_MOE_IDROWS \
  ATOM_P1_MBF_ASM ATOM_P1_MBF_BACK ATOM_DG_LMHEAD_AK ATOM_P1_LMHEAD_RING ATOM_P1_MONO_MAX_ROWS ATOM_MONO_STATS_EVERY \
  ATOM_P1_DRAFT_BS_LADDER ATOM_M3_FHMOE ATOM_M3_FHMOE_ACT

# --- 1. base and numerics (C17 serve.sh hard rules; colleague recipe env.sh; CLAUDE.md) --------------------------
# Shared expert stays its own MXFP8 MLP, bf16 all-reduce (no INT4/FP8 quick reduce), NVFP4 KV + index (args below).
export ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=0
export AITER_LOG_LEVEL=WARNING PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 ARCH=gfx950
# C17 serve.sh (also the merged PP4 build's server.sh): Triton attention fallbacks where available.
export ATOM_FORCE_ATTN_TRITON=1
# C17 serve.sh: legacy HSA IPC for amd-kernels' IPC buffers (split4 indexer, staged all-reduce, Markov pushes).
export HSA_ENABLE_IPC_MODE_LEGACY=1
# TP4 inside one node: RCCL needs no IB (colleague recipe env.sh; C17 serve.sh).
export NCCL_IB_DISABLE=1 RCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=none

# --- 2. P/D transport (main's launcher rules; 1P2D C220 reference cell; ROCm/ATOM#3306 / #2846 landing) ------------
# One QP per endpoint in every Mooncake process (must match the prefill and the Store); RDMA, not HIP IPC.
export MC_NUM_QP_PER_EP=1 MC_DISABLE_HIP=1
# PP/TP1 prefill -> TP4 decode landing (main's name of the experiment's ATOM_PD_MHA_LANDING; set on BOTH roles).
export ATOM_PD_MINIMAX_LANDING=1
# API-server tokenization cache for long multi-turn AgentX prompts (C220 reference cell; main default 512).
export ATOM_TOKENIZE_CACHE_MB=2048
# This decode's NICs: its half's own rail (#606: node-0 decode on rdma4-7, never the prefill stages' rdma0-3).
# D_NICS=launcher leaves the harness launcher's ${WORKER_RDMA_DEVICES} placeholder (Jasen/m31-agentx-launcher).
if [[ "${D_NICS:-}" == launcher ]]; then D_NICS='${WORKER_RDMA_DEVICES}'
elif [[ -z "${D_NICS:-}" ]]; then
  if [[ "${D_HALF}" == 0 ]]; then D_NICS=rdma0,rdma1,rdma2,rdma3; else D_NICS=rdma4,rdma5,rdma6,rdma7; fi
fi
# Decode admission on every decode (ROCm/ATOM#3147 #3165 #3173, on main): fraction / reserve are the decode-memory
# exploration's knobs.
D_PD_ADMISSION='{"enabled":true,"max_pending_kv_fraction":0.15,"num_reserved_decode_tokens":512}'
if [[ -n "${ROLE_IP:-}" && -n "${HANDSHAKE_PORT:-}" ]]; then _ip=${ROLE_IP}; _hs=${HANDSHAKE_PORT}; else
  _ip='${ROLE_IP}'; _hs='${HANDSHAKE_PORT}'; fi  # literal placeholders the CI launcher fills
D_KV_TRANSFER_CONFIG="{\"kv_connector\":\"mooncake\",\"kv_role\":\"kv_consumer\",\"proxy_ip\":\"${_ip}\",\"handshake_port\":${_hs},\"protocol\":\"rdma\",\"ib_device\":\"${D_NICS}\",\"pd_admission\":${D_PD_ADMISSION}}"
unset _ip _hs
export DECODE_KV_TRANSFER_CONFIG="${D_KV_TRANSFER_CONFIG}"

# --- 3. indexer CP4 = C17's split4 (arms B / C2 / C16), NOT main's ATOM_M3_INDEXER_CP ------------------------------
# C17's four-GPU decode indexer is amd-kernels' IndexDecodeTop16 on the replicated NVFP4 index cache (bit-identical
# ids). Main's indexer CP must stay off: attention_mha.py takes the AK and the fused NVFP4 decode only while
# indexer_cp_group is None, and ganyi's own config refuses main's CP under NVFP4 caches.
export ATOM_M3_INDEXER_CP=0
export ATOM_M3_AK_DECODE=1            # int1: AK index + attention <= 16 requests; the switch split4 initialises under
export ATOM_P1_INDEXER_SPLIT4=1       # B: four-GPU index top-16 (IPC buffers sized by max_num_seqs, <= 256 requests)
export ATOM_P1_INDEXER_SPLIT4_WIDE=1  # B: four-GPU SCORES into the fused decode from 17 requests (the C100+ indexer)
export ATOM_P1_INDEXER_SPLIT4_MIN8=6  # C2: 8-row route from 6 requests (bit-identical)
export ATOM_P1_K2_SCORER=rot2_wr2_h1024  # C16: overlay p1k2 fused scorer + top-16 at <= 16 requests (bit-identical)
# --- 4. decode attention (int1 / B / C / C14) ------------------------------------------------------------------
export ATOM_M3_AK_INSERT=1            # int1: norm + RoPE + NVFP4 K/V/index insert, same page bytes (bit-identical)
export ATOM_M3_AK_DECODE_QOUT=1       # B: AK route writes o_proj's MXFP8 input (bit-identical bytes)
export ATOM_P1_AK_WIDE=1              # C: AK block at 17-32 requests (fp16-P rounding, gated in arm C)
export ATOM_P1_L2_D2=1                # C14: overlay p1l2 split-combine variants (bit-identical)
export ATOM_M3_FUSED_DECODE=1         # int1: > 32 requests: aiter's top-16 selection + q8 union attention (fp32 order)

# --- 5. layer front, all-reduce and MXFP8 GEMMs (int1 / B / C14) --------------------------------------------------
export ATOM_AK_MXFP8_GEMM=1           # int1: TP4 MXFP8 linears on amd-kernels where they won (fp32 order); decode ONLY
export ATOM_P1_FRONT_QKV_CHAIN=1      # B: qkv GEMM + norms + RoPE + NVFP4 insert in one launch, 1-36 requests
export ATOM_P1_AR_QKV_CODES=1         # B: AR epilogue emits the qkv GEMM's MXFP8 codes (bit-identical)
export ATOM_P1_L4_PULLQ=1             # C14: staged pull exchange for 16-88 rows (overlay l4c_y, bit-identical)
export ATOM_MINIMAX_AK_AR_NORM=1      # B: AR + add + Gemma norm on amd-kernels' bf16-wire communicator, <= 64 requests
# C17's tuned aiter tables (byte copies of recipe/configs/, TP4 shapes). Each variable REPLACES aiter's
# default + model_configs/ merge: decode only, never on the TP1 prefill.
export AITER_CONFIG_GEMM_BF16=${TUNED_CONFIGS}/bf16_tuned_gemm_m31.csv
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE=${TUNED_CONFIGS}/mx8_tuned_gemm_m31.csv

# --- 6. MoE: aiter's fused_moe (a16w4 below 256 rows, a4w4 from 256), NOT C17's amd-kernels block ----------------
# ATOM_M3_AK_MOE runs MXFP4 activations at every row count (lossier below 256 rows) and its tiling flags
# (MOE_TILES / MOE_IDROWS / MBF_ASM / MBF_BACK) only act on it: excluded.txt, reported to the coordinator.
export ATOM_M3_AK_MOE=0
# Mono (C17: on, 8-row steps) runs the same a4w4 MoE, only at 1-request steps, and the engine blocks it while KV
# tier transfers are in flight (ROCm/ATOM#3369), i.e. always on a P/D decode. Main's default is on: switch it off.
export ATOM_MONO_ENABLE=0

# --- 7. DSpark draft and verify head (B / C3 / C14 / C16): draft-side or bit-identical --------------------------
export ATOM_M3_DRAFT_AK_AR_NORM=1 ATOM_M3_DRAFT_FUSED_CTX_ROPE=1 ATOM_M3_DRAFT_MARKOV_WIDE=1 ATOM_M3_DRAFT_POS_MAJOR=1
export ATOM_P1_DRAFT_MARKOV_PERSIST=1  # C16: one persistent Markov launch per rank, B <= 64 (falls back above)
export ATOM_P1_DRAFT_FC_AK=1           # C14: draft input fc <= 256 rows (+94 MB/rank packed copy)
export ATOM_P1_FAST_CONCAT=1           # C14: draft tap concat, same bytes
export ATOM_P1_DSPARK_AUX_NEXT=64      # C3: taps read the next AR tail on decode forwards >= 64 rows (bit-identical)
export ATOM_M3_VERIFY_HEAD_CHUNK_TUNED=1  # B: verify-head chunk per row count (same argmax ids)

# --- 8. decode prefix cache (C12) -------------------------------------------------------------------------------
# A finished request's generated blocks are evicted first: AgentX replays recorded assistant turns, so they never
# prefix-match the next turn; prompt blocks stay -> more decode-held prompt, fewer P->D bytes. A/B candidate.
export ATOM_P1_PREFIX_DEMOTE_DECODE=1

# --- 9. observability and stall safety (B / C; the CI agentic suite enables the device timer) -------------------
export ATOM_ENABLE_METRICS_DEVICE_TIMER=1
export ATOM_STEP_WATCHDOG_S=5 ATOM_STEP_KICK_S=5  # the kick can end a GPU spin livelock (split4 / AK AR spin on peers)

# --- 10. server arguments ----------------------------------------------------------------------------------------
# Capacity for C100+ per decode (colleague recipe tp4 mode: 256 / 16384 / graphs to 256; split4 caps at 256
# requests). C17's own values (MEM 0.85, MNBT 32768, max-num-seqs 2*CONC = 2-4, exact graphs 1..24) served
# 1-2 requests on an aggregated server. 0.90: no prefill-step VRAM spikes on a P/D decode (C17 lowered MEM for them).
D_SEQS=${D_SEQS:-256}
D_MNBT=${D_MNBT:-16384}
D_MEM=${D_MEM:-0.90}
D_CG=${D_CG:-[1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128,144,160,176,192,208,224,240,256]}
# DSpark k=7; the draft's own pool in fp8 on BOTH roles (C17: --dspark-config '{"draft_kv_dtype":"fp8"}'; on main
# that key is DeepSeek-V4.1-only and the M3 draft's dtype is --spec-decode-kv-cache-dtype; ROCm/ATOM#3616 moves an
# own-dtype draft over mooncake / mooncake_store). confidence_schedule stays at its default (off).
D_SPEC=(--method dspark --draft-model "${MODEL}/dspark" --num-speculative-tokens 7 --spec-decode-kv-cache-dtype fp8)
[[ "${AL}" != real ]] && D_SPEC+=(--spec-decode-acceptance-length "${AL}")
D_SERVER_ARGS=(
  --model "${MODEL}" --trust-remote-code
  --tensor-parallel-size 4
  --kv_cache_dtype nvfp4 --index-cache-dtype nvfp4 --block-size 128 --enable-prefix-caching
  --max-model-len 1048576 --max-num-seqs "${D_SEQS}" --max-num-batched-tokens "${D_MNBT}"
  --gpu-memory-utilization "${D_MEM}" --cudagraph-capture-sizes "${D_CG}"
  --default-chat-template-kwargs '{"thinking_mode": "enabled"}'
  --timeout-keep-alive 900
  "${D_SPEC[@]}"
  --kv-transfer-config "${D_KV_TRANSFER_CONFIG}"
)

D_ENV_NAMES=(ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4 AITER_LOG_LEVEL PYTHONHASHSEED PYTHONUNBUFFERED ARCH
  ATOM_FORCE_ATTN_TRITON HSA_ENABLE_IPC_MODE_LEGACY NCCL_IB_DISABLE RCCL_IB_DISABLE NCCL_NET_PLUGIN
  MC_NUM_QP_PER_EP MC_DISABLE_HIP ATOM_PD_MINIMAX_LANDING ATOM_TOKENIZE_CACHE_MB DECODE_KV_TRANSFER_CONFIG
  ATOM_M3_INDEXER_CP ATOM_M3_AK_DECODE ATOM_P1_INDEXER_SPLIT4 ATOM_P1_INDEXER_SPLIT4_WIDE ATOM_P1_INDEXER_SPLIT4_MIN8
  ATOM_P1_K2_SCORER ATOM_M3_AK_INSERT ATOM_M3_AK_DECODE_QOUT ATOM_P1_AK_WIDE ATOM_P1_L2_D2 ATOM_M3_FUSED_DECODE
  ATOM_AK_MXFP8_GEMM ATOM_P1_FRONT_QKV_CHAIN ATOM_P1_AR_QKV_CODES ATOM_P1_L4_PULLQ ATOM_MINIMAX_AK_AR_NORM
  AITER_CONFIG_GEMM_BF16 AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE ATOM_M3_AK_MOE ATOM_MONO_ENABLE
  ATOM_M3_DRAFT_AK_AR_NORM ATOM_M3_DRAFT_FUSED_CTX_ROPE ATOM_M3_DRAFT_MARKOV_WIDE ATOM_M3_DRAFT_POS_MAJOR
  ATOM_P1_DRAFT_MARKOV_PERSIST ATOM_P1_DRAFT_FC_AK ATOM_P1_FAST_CONCAT ATOM_P1_DSPARK_AUX_NEXT
  ATOM_M3_VERIFY_HEAD_CHUNK_TUNED ATOM_P1_PREFIX_DEMOTE_DECODE ATOM_ENABLE_METRICS_DEVICE_TIMER
  ATOM_STEP_WATCHDOG_S ATOM_STEP_KICK_S)

d_env_dump() {
  local n
  for n in "${D_ENV_NAMES[@]}"; do printf '%s=%s\n' "${n}" "${!n}"; done
  printf 'D_SERVER_ARGS=%s\n' "${D_SERVER_ARGS[*]}"
}
