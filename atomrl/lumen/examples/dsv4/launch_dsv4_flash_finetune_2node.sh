#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Launch DSV4 Flash 2-node GRPO finetune from the head node with identical env on both ranks.
#
# Usage (on head node):
#   cd ~/ATOM/atomrl/lumen
#   MASTER_ADDR=<head-ip> WORKER_SSH=${USER}@<worker-host> \
#     bash examples/dsv4/launch_dsv4_flash_finetune_2node.sh
#
# Optional overrides (exported to both nodes):
#   SKIP_PREPARE=1 GBS=256 NUM_ROLLOUT=10 DSV4_HC_MULT=4 IMAGE=lumen/dsv4-lumen:mi308x

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"

MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to head node IP}"
WORKER_ADDR="${WORKER_ADDR:-}"
WORKER_SSH="${WORKER_SSH:-}"
if [[ -z "${WORKER_SSH}" && -n "${WORKER_ADDR}" ]]; then
    WORKER_SSH="${USER}@${WORKER_ADDR}"
fi
WORKER_SSH="${WORKER_SSH:?Set WORKER_SSH (e.g. ${USER}@worker-host) or WORKER_ADDR}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519_conductor}"
PREFLIGHT_ID="$(date +%Y%m%d_%H%M%S)"

COMMON_ENV=(
    "PREFLIGHT_ID=${PREFLIGHT_ID}"
    "MASTER_ADDR=${MASTER_ADDR}"
    "AITER_DIR=${AITER_DIR}"
    "MODEL_DIR=${MODEL_DIR}"
    "DATA_DIR=${DATA_DIR:-${DATA_ROOT}/datasets}"
    "LOG_DIR=${LOG_DIR}"
    "SKIP_PREPARE=${SKIP_PREPARE:-1}"
    "LOAD_CKPT=${LOAD_CKPT:-1}"
    "GBS=${GBS:-256}"
    "SEQ_LEN=${SEQ_LEN:-4096}"
    "NUM_ROLLOUT=${NUM_ROLLOUT:-10}"
    "DSV4_HC_MULT=${DSV4_HC_MULT:-4}"
    "DEBUG_TRAIN_ONLY=${DEBUG_TRAIN_ONLY:-1}"
    "IMAGE=${IMAGE:-lumen/dsv4-lumen:mi308x}"
    "V4_SPARSE_MLA_BACKEND=${V4_SPARSE_MLA_BACKEND:-triton}"
    "OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
    "DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES:-180}"
)

echo "════════════════════════════════════════════════"
echo "  2-node GRPO finetune launch  PREFLIGHT_ID=${PREFLIGHT_ID}"
echo "  Head   : ${MASTER_ADDR} (NODE_RANK=0)"
echo "  Worker : ${WORKER_SSH} (NODE_RANK=1)"
echo "  GBS=${GBS:-256}  SEQ_LEN=${SEQ_LEN:-4096}  NUM_ROLLOUT=${NUM_ROLLOUT:-10}  DSV4_HC_MULT=${DSV4_HC_MULT:-4}"
echo "  MLA=${V4_SPARSE_MLA_BACKEND:-triton}  OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
echo "  Logs: Miles format rollout/step/perf (grep finetune_loop in node0 log)"
echo "════════════════════════════════════════════════"

docker rm -f lumen-dsv4-flash-finetune-node0 lumen-dsv4-flash-finetune-node1 2>/dev/null || true
if ssh -o BatchMode=yes -i "${SSH_KEY}" -o IdentitiesOnly=yes -o ConnectTimeout=10 \
    "${WORKER_SSH}" 'docker rm -f lumen-dsv4-flash-finetune-node0 lumen-dsv4-flash-finetune-node1 2>/dev/null || true'; then
    WORKER_REACHABLE=1
else
    WORKER_REACHABLE=0
    echo "[launch][WARN] worker unreachable (${WORKER_SSH}) — head will start; preflight waits for worker manifest"
fi

join_env() {
    local out=""
    local kv=""
    for kv in "${COMMON_ENV[@]}"; do
        out+="${kv} "
    done
    printf '%s' "${out}"
}

HEAD_LOG="${LOG_DIR}/lumen_dsv4_flash_finetune_launch_head_${PREFLIGHT_ID}.log"
WORKER_LOG="${LOG_DIR}/lumen_dsv4_flash_finetune_launch_worker_${PREFLIGHT_ID}.log"

cd "${LUMEN_DIR}"
nohup env NODE_RANK=0 $(join_env) \
    bash examples/dsv4/run_dsv4_flash_finetune.sh \
    > "${HEAD_LOG}" 2>&1 &
HEAD_PID=$!
echo "[launch] head pid=${HEAD_PID} log=${HEAD_LOG}"

sleep 5

if [[ "${WORKER_REACHABLE}" == "1" ]]; then
    ssh -o BatchMode=yes -i "${SSH_KEY}" -o IdentitiesOnly=yes "${WORKER_SSH}" \
        "cd ${LUMEN_DIR} && nohup env NODE_RANK=1 $(join_env) \
            bash examples/dsv4/run_dsv4_flash_finetune.sh \
            > ${WORKER_LOG} 2>&1 & echo worker_pid=\$! log=${WORKER_LOG}"
else
    echo "[launch][WARN] skip worker launch — bring up worker then run:"
    echo "  cd ${LUMEN_DIR} && NODE_RANK=1 $(join_env) bash examples/dsv4/run_dsv4_flash_finetune.sh"
fi

echo "[launch] done — tail training logs:"
echo "  head:   tail -f ${LOG_DIR}/lumen_dsv4_flash_finetune_node0_*.log"
echo "  worker: tail -f ${LOG_DIR}/lumen_dsv4_flash_finetune_node1_*.log"
