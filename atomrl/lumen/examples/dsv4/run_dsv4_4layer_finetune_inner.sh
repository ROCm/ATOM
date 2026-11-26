#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Inner container entry for run_dsv4_4layer_finetune.sh — native torchrun GRPO (no Ray).
set -euo pipefail

cd /workspace/Lumen
# Megatron-only bootstrap: skip Ray/SGLang + miles/requirements.txt on every docker run.
export LUMEN_DSV4_NATIVE_FINETUNE=1
export LUMEN_DSV4_PRETRAIN=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-FP8-4layer}"
DSV4_HC_MULT="${DSV4_HC_MULT:-4}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
DEBUG_TRAIN_ONLY="${DEBUG_TRAIN_ONLY:-1}"
FAKE_ROLLOUT_DATA="${FAKE_ROLLOUT_DATA:-/root/models/fake_rollout.pt}"

if [[ "${DEBUG_TRAIN_ONLY}" != "1" ]]; then
    echo "[finetune] ERROR: native finetune path only supports DEBUG_TRAIN_ONLY=1"
    echo "  For live SGLang rollout use smoke_test_dsv4_lumen_grpo.py (Miles + Ray)."
    exit 1
fi

source examples/dsv4/dsv4_4layer_megatron_args.sh
# shellcheck source=examples/dsv4/dsv4_finetune_common.sh
source examples/dsv4/dsv4_finetune_common.sh
dsv4_apply_finetune_batch_defaults

# shellcheck source=examples/dsv4/setup_container_env.sh
source examples/dsv4/setup_container_env.sh
setup_dsv4_container_env /workspace/miles

dsv4_print_finetune_launch_info finetune

dsv4_resolve_finetune_ckpt "${MODEL_NAME}" "${DSV4_HC_MULT}"

if [[ "${SKIP_PREPARE}" != "1" && ! -f "${CKPT}/latest_checkpointed_iteration.txt" ]]; then
    if [[ ! -d /workspace/miles ]]; then
        echo "[ERROR] Checkpoint missing and MILES_DIR not mounted for prepare_dsv4_4layer_checkpoint.py"
        exit 1
    fi
    export PYTHONPATH="/workspace/ATOM/atomrl:/workspace/miles:${PYTHONPATH:-}"
    DSV4_HC_MULT="${DSV4_HC_MULT}" python examples/dsv4/prepare_dsv4_4layer_checkpoint.py
else
    echo "[prepare] torch_dist checkpoint already present — skipping"
fi

if [[ ! -f "${CKPT}/latest_checkpointed_iteration.txt" ]]; then
    echo "[finetune] ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

dsv4_prepare_rollout_on_shared_fs
dsv4_finetune_recompute_args

echo "[finetune] launching torchrun (native Megatron GRPO, no Ray) ..."
torchrun --nproc_per_node=8 --nnodes=1 \
    examples/dsv4/finetune_dsv4_megatron.py \
    "${DSV4_MODEL_ARGS[@]}" \
    "${DSV4_FINETUNE_TORCHRUN_ARGS[@]}" \
    "${RECOMPUTE_ARGS[@]}" \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    --context-parallel-size "${CP}" \
    --expert-model-parallel-size "${EP}" \
    --expert-tensor-parallel-size "${ETP}" \
    --micro-batch-size "${MBS}" \
    --global-batch-size "${GBS}" \
    --seq-length "${SEQ_LEN}" \
    --max-position-embeddings "${SEQ_LEN}" \
    --train-iters "${NUM_ROLLOUT}" \
    --num-rollout "${NUM_ROLLOUT}" \
    --rollout-data-path "${FAKE_ROLLOUT_DATA}" \
    --load "${CKPT}"

echo ""
echo "=== [done] Lumen DSV4 4-layer native GRPO finetune completed ==="
