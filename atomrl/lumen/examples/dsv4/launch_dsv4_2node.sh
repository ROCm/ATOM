#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Launch DSV4 Flash 2-node GRPO finetune from the head node with identical env on both ranks.
#
# Usage (on head node):
#   cd ~/ATOM/atomrl/lumen
#   MASTER_ADDR=<head-ip> WORKER_SSH=${USER}@<worker-host> \
#     bash examples/dsv4/launch_dsv4_2node.sh
#
# Optional overrides (exported to both nodes unless noted):
#   SKIP_PREPARE=1 GBS=256 NUM_ROLLOUT=10 DSV4_HC_MULT=4 IMAGE=lumen/dsv4-lumen:mi308x
#   WORKER_MODEL_DIR=/mnt/nvme0n1/${USER}/models  (worker-only host ckpt path; head uses MODEL_DIR)
#   Smoke (match pretrain batch): GBS=8 DSV4_KEEP_GBS=1 SEQ_LEN=512 DSV4_KEEP_SEQ_LEN=1 \
#     NUM_ROLLOUT=2 ROLLOUT_N_PROMPTS=1 ROLLOUT_N_PER_PROMPT=8 SMOKE_LEGACY_FAKE_ROLLOUT=1
#   Bisect: TP=4 PP=2 EP=1 DECODER_FIRST_PP_LAYERS=22 DECODER_LAST_PP_LAYERS=21

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"

DSV4_PROFILE="${DSV4_PROFILE:-flash}"
MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to head node IP}"
WORKER_ADDR="${WORKER_ADDR:-}"
WORKER_SSH="${WORKER_SSH:-}"
if [[ -z "${WORKER_SSH}" && -n "${WORKER_ADDR}" ]]; then
    WORKER_SSH="${USER}@${WORKER_ADDR}"
fi
WORKER_SSH="${WORKER_SSH:?Set WORKER_SSH (e.g. ${USER}@worker-host) or WORKER_ADDR}"
SSH_KEY="${SSH_KEY:-}"
PREFLIGHT_ID="$(date +%Y%m%d_%H%M%S)"
CONTAINER_PREFIX="lumen-dsv4-flash-finetune-node"
WORKER_MODEL_DIR="${WORKER_MODEL_DIR:-${MODEL_DIR}}"

COMMON_ENV=(
    "DSV4_PROFILE=${DSV4_PROFILE}"
    "PREFLIGHT_ID=${PREFLIGHT_ID}"
    "MASTER_ADDR=${MASTER_ADDR}"
    "AITER_DIR=${AITER_DIR}"
    "MASTER_PORT=${MASTER_PORT:-29500}"
    "MODEL_DIR=${MODEL_DIR}"
    "DATA_ROOT=${DATA_ROOT}"
    "DATA_DIR=${DATA_DIR:-${DATA_ROOT}/datasets}"
    "LOG_DIR=${LOG_DIR}"
    "SKIP_PREPARE=${SKIP_PREPARE:-1}"
    "GBS=${GBS:-256}"
    "DSV4_KEEP_GBS=${DSV4_KEEP_GBS:-0}"
    "SEQ_LEN=${SEQ_LEN:-4096}"
    "DSV4_KEEP_SEQ_LEN=${DSV4_KEEP_SEQ_LEN:-0}"
    "ROLLOUT_N_PROMPTS=${ROLLOUT_N_PROMPTS:-32}"
    "ROLLOUT_N_PER_PROMPT=${ROLLOUT_N_PER_PROMPT:-8}"
    "SMOKE_LEGACY_FAKE_ROLLOUT=${SMOKE_LEGACY_FAKE_ROLLOUT:-0}"
    "NUM_ROLLOUT=${NUM_ROLLOUT:-10}"
    "DSV4_HC_MULT=${DSV4_HC_MULT:-4}"
    "DEBUG_TRAIN_ONLY=${DEBUG_TRAIN_ONLY:-1}"
    "IMAGE=${IMAGE:-lumen/dsv4-lumen:mi308x}"
    "V4_SPARSE_MLA_BACKEND=${V4_SPARSE_MLA_BACKEND:-triton}"
    "V4_INDEXER_IMPL=${V4_INDEXER_IMPL:-aiter}"
    "V4_INDEXER_BLOCK_N=${V4_INDEXER_BLOCK_N:-64}"
    "V4_INDEXER_NUM_STAGES=${V4_INDEXER_NUM_STAGES:-1}"
    "OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
    "DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES:-180}"
    "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}"
    "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}"
    "NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
    "NCCL_IB_GDR_LEVEL=${NCCL_IB_GDR_LEVEL:-}"
    "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-}"
    "NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}"
    "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
    "MEGATRON_NO_BATCH_P2P_COMM=${MEGATRON_NO_BATCH_P2P_COMM:-1}"
    "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
    "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens14np0}"
    "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ens14np0}"
    "TP=${TP:-}"
    "PP=${PP:-}"
    "EP=${EP:-}"
    "ETP=${ETP:-}"
    "DECODER_FIRST_PP_LAYERS=${DECODER_FIRST_PP_LAYERS:-}"
    "DECODER_LAST_PP_LAYERS=${DECODER_LAST_PP_LAYERS:-}"
)

echo "════════════════════════════════════════════════"
echo "  2-node GRPO finetune launch  PREFLIGHT_ID=${PREFLIGHT_ID}"
echo "  Head   : ${MASTER_ADDR} (NODE_RANK=0)"
echo "  Worker : ${WORKER_SSH} (NODE_RANK=1)"
echo "  MODEL_DIR (head)  : ${MODEL_DIR}"
echo "  MODEL_DIR (worker): ${WORKER_MODEL_DIR}"
echo "  GBS=${GBS:-256}  SEQ_LEN=${SEQ_LEN:-4096}  NUM_ROLLOUT=${NUM_ROLLOUT:-10}  DSV4_HC_MULT=${DSV4_HC_MULT:-4}"
echo "  parallel : TP=${TP:-4} PP=${PP:-4} EP=${EP:-4} (override via env)"
echo "  NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}  IB_HCA=${NCCL_IB_HCA:-mlx5_0-7}  NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
echo "════════════════════════════════════════════════"

docker rm -f "${CONTAINER_PREFIX}0" "${CONTAINER_PREFIX}1" 2>/dev/null || true
_ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)
if [[ -n "${SSH_KEY}" ]]; then
    _ssh_opts+=(-i "${SSH_KEY}" -o IdentitiesOnly=yes)
fi
if ssh "${_ssh_opts[@]}" \
    "${WORKER_SSH}" "docker rm -f ${CONTAINER_PREFIX}0 ${CONTAINER_PREFIX}1 2>/dev/null || true"; then
    WORKER_REACHABLE=1
else
    WORKER_REACHABLE=0
    echo "[launch][WARN] worker unreachable (${WORKER_SSH}) - head will start; preflight waits for worker manifest"
fi

join_env() {
    local out=""
    local kv=""
    for kv in "${COMMON_ENV[@]}"; do
        out+="${kv} "
    done
    printf '%s' "${out}"
}

join_worker_env() {
    local out=""
    local kv=""
    for kv in "${COMMON_ENV[@]}"; do
        if [[ "${kv}" == MODEL_DIR=* ]]; then
            out+="MODEL_DIR=${WORKER_MODEL_DIR} "
        else
            out+="${kv} "
        fi
    done
    printf '%s' "${out}"
}

HEAD_LOG="${LOG_DIR}/lumen_dsv4_flash_finetune_launch_head_${PREFLIGHT_ID}.log"
WORKER_LOG="${LOG_DIR}/lumen_dsv4_flash_finetune_launch_worker_${PREFLIGHT_ID}.log"

cd "${LUMEN_DIR}"
nohup env NODE_RANK=0 $(join_env) \
    bash examples/dsv4/run_dsv4.sh \
    > "${HEAD_LOG}" 2>&1 &
HEAD_PID=$!
echo "[launch] head pid=${HEAD_PID} log=${HEAD_LOG}"

sleep 5

if [[ "${WORKER_REACHABLE}" == "1" ]]; then
    ssh "${_ssh_opts[@]}" "${WORKER_SSH}" \
        "cd ${LUMEN_DIR} && nohup env NODE_RANK=1 $(join_worker_env) \
            bash examples/dsv4/run_dsv4.sh \
            > ${WORKER_LOG} 2>&1 & echo worker_pid=\$! log=${WORKER_LOG}"
else
    echo "[launch][WARN] skip worker launch - bring up worker then run:"
    echo "  cd ${LUMEN_DIR} && NODE_RANK=1 $(join_worker_env) bash examples/dsv4/run_dsv4.sh"
fi

echo "[launch] done - tail training logs:"
echo "  head:   tail -f ${LOG_DIR}/lumen_dsv4_flash_finetune_node0_*.log"
echo "  worker: tail -f ${LOG_DIR}/lumen_dsv4_flash_finetune_node1_*.log"
