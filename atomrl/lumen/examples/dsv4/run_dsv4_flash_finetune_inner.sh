#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Inner container entry for run_dsv4_flash_finetune.sh — 2-node native torchrun GRPO (no Ray).
set -euo pipefail

cd /workspace/Lumen
export LUMEN_DSV4_NATIVE_FINETUNE=1
export LUMEN_DSV4_PRETRAIN=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"

MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-FP8}"
DSV4_HC_MULT="${DSV4_HC_MULT:-4}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
DEBUG_TRAIN_ONLY="${DEBUG_TRAIN_ONLY:-1}"
FAKE_ROLLOUT_DATA="${FAKE_ROLLOUT_DATA:-/root/models/fake_rollout.pt}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR required}"
MASTER_PORT="${MASTER_PORT:-29500}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-180}"

if [[ "${DEBUG_TRAIN_ONLY}" != "1" ]]; then
    echo "[finetune-full] ERROR: native finetune path only supports DEBUG_TRAIN_ONLY=1"
    echo "  For live SGLang rollout use Miles run_deepseek_v4_mi300x_16gpu.sh."
    exit 1
fi

source examples/dsv4/dsv4_flash_megatron_args.sh
source examples/dsv4/dsv4_flash_mi300x_parallel.sh
# shellcheck source=examples/dsv4/dsv4_finetune_common.sh
source examples/dsv4/dsv4_finetune_common.sh
dsv4_apply_finetune_batch_defaults

# shellcheck source=examples/dsv4/setup_container_env.sh
source examples/dsv4/setup_container_env.sh
setup_dsv4_container_env /workspace/miles

dsv4_print_finetune_launch_info finetune-full

dsv4_resolve_finetune_ckpt "${MODEL_NAME}" "${DSV4_HC_MULT}"

if [[ "${SKIP_PREPARE}" != "1" && ! -f "${CKPT}/latest_checkpointed_iteration.txt" ]]; then
    if [[ ! -d /workspace/miles ]]; then
        echo "[ERROR] Checkpoint missing and MILES_DIR not mounted for prepare_dsv4_flash_checkpoint.py"
        exit 1
    fi
    export PYTHONPATH="/workspace/ATOM/atomrl:/workspace/miles:${PYTHONPATH:-}"
    python examples/dsv4/prepare_dsv4_flash_checkpoint.py
else
    echo "[prepare] torch_dist checkpoint already present — skipping (path=${CKPT})"
fi

if [[ ! -f "${CKPT}/latest_checkpointed_iteration.txt" ]]; then
    echo "[finetune-full] ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

dsv4_prepare_rollout_on_shared_fs
dsv4_finetune_recompute_args

echo "[finetune-full] launching torchrun ${NNODES}×${NPROC_PER_NODE} (native Megatron GRPO, no Ray) ..."

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    examples/dsv4/finetune_dsv4_megatron.py \
    "${DSV4_MODEL_ARGS[@]}" \
    "${DSV4_FINETUNE_TORCHRUN_ARGS[@]}" \
    "${RECOMPUTE_ARGS[@]}" \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    --decoder-first-pipeline-num-layers "${DECODER_FIRST_PP_LAYERS}" \
    --decoder-last-pipeline-num-layers "${DECODER_LAST_PP_LAYERS}" \
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
    --num-workers 0 \
    --no-pin-cpu-grads \
    --no-pin-cpu-params \
    --optimizer-cpu-offload \
    --use-precision-aware-optimizer \
    --overlap-cpu-optimizer-d2h-h2d \
    --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION}" \
    --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}" \
    --load "${CKPT}"

echo ""
echo "=== [done] Lumen DSV4 Flash full-model native GRPO finetune completed (node_rank=${NODE_RANK}) ==="
