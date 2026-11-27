#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# run_dsv4_4layer_finetune.sh — DSV4 4-layer GRPO full finetune on 8×MI308X.
#
# Full-parameter finetune (not Megatron pretrain / mock LM loss):
#   Native torchrun finetune_dsv4_megatron.py + fake_rollout.pt + GRPO policy loss.
#
# Default DEBUG_TRAIN_ONLY=1: actor-only GRPO with pre-generated GSM8K rollout
# (no SGLang / no Ray — stable on ROCm).
#
# Usage:
#   bash examples/dsv4/run_dsv4_4layer_finetune.sh
#   NUM_ROLLOUT=10 DSV4_HC_MULT=4 SKIP_PREPARE=1 bash examples/dsv4/run_dsv4_4layer_finetune.sh
#
# Live SGLang rollout (experimental): smoke_test_dsv4_lumen_grpo.py (Miles + Ray).
#
# Miles repo optional for checkpoint prep / realistic rollout generation (default: sibling of Lumen).
# For kernel-only mock pretrain smoke, use run_dsv4_4layer_pretrain.sh instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"
# shellcheck source=examples/dsv4/dsv4_docker_common.sh
source "${SCRIPT_DIR}/dsv4_docker_common.sh"

LOGFILE="${LOG_DIR}/lumen_dsv4_4layer_finetune_$(date +%Y%m%d_%H%M%S).log"

IMAGE="${IMAGE:-lumen/dsv4-lumen:mi308x}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-FP8-4layer}"
DSV4_HC_MULT="${DSV4_HC_MULT:-4}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/datasets}"

V4_SPARSE_MLA_BACKEND="${V4_SPARSE_MLA_BACKEND:-triton}"
V4_INDEXER_IMPL="${V4_INDEXER_IMPL:-aiter}"
V4_INDEXER_BLOCK_N="${V4_INDEXER_BLOCK_N:-64}"
V4_INDEXER_NUM_STAGES="${V4_INDEXER_NUM_STAGES:-1}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
GBS="${GBS:-256}"

# shellcheck source=examples/dsv4/dsv4_finetune_common.sh
source "${SCRIPT_DIR}/dsv4_finetune_common.sh"
dsv4_apply_finetune_batch_defaults

USE_BOOTSTRAP=0
BOOTSTRAP_MOUNT="${BOOTSTRAP_DIR}"
if [[ "${IMAGE}" == "lumen/tests:latest" || "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
    USE_BOOTSTRAP=1
    if [[ "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
        BOOTSTRAP_MOUNT=""
    elif [[ ! -f "${BOOTSTRAP_DIR}/.ready" ]]; then
        echo "[prepare] bootstrap missing — running prepare_bootstrap.sh"
        bash "${SCRIPT_DIR}/prepare_bootstrap.sh"
    fi
fi

if ! docker image inspect "${IMAGE}" &>/dev/null; then
    if [[ "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
        bash "${SCRIPT_DIR}/build_dsv4_lumen_image.sh"
    else
        echo "[ERROR] Image not found: ${IMAGE}"
        exit 1
    fi
fi

if ! ls /dev/kfd &>/dev/null; then
    echo "[ERROR] /dev/kfd not found — ROCm device not accessible."
    exit 1
fi

mkdir -p "${MODEL_DIR}" "${DATA_DIR}" "${LOG_DIR}" "${MODEL_DIR}/miopen-cache" "${TVM_CACHE_DIR}" "${PIP_CACHE_DIR}"

TORCH_DIST="${MODEL_DIR}/${MODEL_NAME}_torch_dist_hc${DSV4_HC_MULT}"

echo "════════════════════════════════════════════════"
echo "  Lumen DSV4 4-layer GRPO full finetune"
echo "  Image     : ${IMAGE}"
echo "  Mode      : native torchrun GRPO (debug-train-only)"
echo "  Logs      : Miles format rollout/step/perf per GRPO step"
echo "  Spec      : atomrl.lumen.models.dsv4.megatron.spec get_dsv4_spec"
echo "  Rollouts  : ${NUM_ROLLOUT}"
echo "  Batch     : GBS=${GBS} MBS=${MBS} seq_len=${SEQ_LEN}"
echo "  HC mult   : ${DSV4_HC_MULT} (AIter)"
echo "  SparseMLA : ${V4_SPARSE_MLA_BACKEND}"
dsv4_print_gemm_env
echo "  Ckpt      : ${TORCH_DIST}"
echo "  Miles     : ${MILES_DIR}"
echo "  Log       : ${LOGFILE}"
echo "════════════════════════════════════════════════"

DOCKER_MOUNTS=(
    -v "${LUMEN_DIR}:/workspace/Lumen"
    -v "${AITER_DIR}:/workspace/aiter"
    -v "${MILES_DIR}:/workspace/miles"
    -v "${MODEL_DIR}:/root/models"
    -v "${DATA_DIR}:/root/datasets"
    -v "${MODEL_DIR}/miopen-cache:/root/.config/miopen"
    -v "${TVM_CACHE_DIR}:/root/.cache/tvm-ffi"
    -v "${PIP_CACHE_DIR}:/root/.cache/pip"
)
if [[ "${USE_BOOTSTRAP}" -eq 1 && -n "${BOOTSTRAP_MOUNT}" ]]; then
    DOCKER_MOUNTS+=(-v "${BOOTSTRAP_MOUNT}:/bootstrap:ro")
fi

DOCKER_ENV=(
    -e LUMEN_DIR=/workspace/Lumen
    -e MILES_DIR=/workspace/miles
    -e LUMEN_DSV4_NATIVE_FINETUNE=1
    -e LUMEN_DSV4_PRETRAIN=1
    -e MODEL_DIR=/root/models
    -e DATA_DIR=/root/datasets
    -e MODEL_NAME="${MODEL_NAME}"
    -e DSV4_HC_MULT="${DSV4_HC_MULT}"
    -e SKIP_PREPARE="${SKIP_PREPARE}"
    -e NUM_ROLLOUT="${NUM_ROLLOUT}"
    -e GBS="${GBS}"
    -e MBS="${MBS}"
    -e SEQ_LEN="${SEQ_LEN}"
)
dsv4_docker_append_kernel_env
dsv4_docker_append_gemm_env
dsv4_docker_append_rocm_env
dsv4_docker_append_bootstrap_env

docker rm -f lumen-dsv4-finetune 2>/dev/null || true

docker run --rm \
    --name lumen-dsv4-finetune \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    --ipc=host \
    --network=host \
    --shm-size=128g \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1 \
    "${DOCKER_MOUNTS[@]}" \
    "${DOCKER_ENV[@]}" \
    "${IMAGE}" \
    bash /workspace/Lumen/examples/dsv4/run_dsv4_4layer_finetune_inner.sh \
    2>&1 | tee "${LOGFILE}"

echo ""
echo "Log saved to: ${LOGFILE}"
