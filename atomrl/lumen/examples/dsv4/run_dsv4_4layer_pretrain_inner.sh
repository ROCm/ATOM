#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Inner container entry for run_dsv4_4layer_pretrain.sh (sourced config + torchrun).
set -euo pipefail

cd /workspace/Lumen
export CUDA_DEVICE_MAX_CONNECTIONS=1

TRAIN_ITERS="${TRAIN_ITERS:-10}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-FP8-4layer}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
LOAD_CKPT="${LOAD_CKPT:-0}"

source examples/dsv4/dsv4_4layer_megatron_args.sh
# DSV4_HC_MULT / GBS / MBS / SEQ_LEN come from dsv4_4layer_megatron_args.sh

export LUMEN_DSV4_PRETRAIN=1
# shellcheck source=examples/dsv4/setup_container_env.sh
source examples/dsv4/setup_container_env.sh
setup_dsv4_container_env /workspace/miles

CKPT="/root/models/${MODEL_NAME}_torch_dist_hc${DSV4_HC_MULT}"
if [[ ! -f "${CKPT}/latest_checkpointed_iteration.txt" ]]; then
    FALLBACK="/root/models/${MODEL_NAME}_torch_dist"
    if [[ -f "${FALLBACK}/latest_checkpointed_iteration.txt" ]]; then
        echo "[prepare] using fallback checkpoint ${FALLBACK}"
        CKPT="${FALLBACK}"
    fi
fi
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

LOAD_ARGS=()
if [[ "${LOAD_CKPT}" == "1" && -f "${CKPT}/latest_checkpointed_iteration.txt" ]]; then
    LOAD_ARGS=(--load "${CKPT}" --no-load-optim --no-load-rng)
    echo "[pretrain] loading checkpoint ${CKPT} (dsv4-hc-mult=${DSV4_HC_MULT})"
else
    echo "[pretrain] training from random init (LOAD_CKPT=${LOAD_CKPT})"
fi

RECOMPUTE_ARGS=()
if [[ "${DSV4_ENABLE_RECOMPUTE:-1}" == "1" ]]; then
    RECOMPUTE_ARGS=(
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
    )
fi

echo "[pretrain] launching torchrun (native Megatron, no Ray) ..."
echo "[pretrain] batch: GBS=${GBS} MBS=${MBS} seq_len=${SEQ_LEN} (hc_mult=${DSV4_HC_MULT})"
torchrun --nproc_per_node=8 --nnodes=1 \
    examples/dsv4/pretrain_dsv4_megatron.py \
    "${DSV4_MODEL_ARGS[@]}" \
    --transformer-impl local \
    --disable-jit-fuser \
    --moe-router-freeze-gate \
    --freeze-e-score-correction-bias \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    --context-parallel-size "${CP}" \
    --expert-model-parallel-size "${EP}" \
    --expert-tensor-parallel-size "${ETP}" \
    --sequence-parallel \
    "${RECOMPUTE_ARGS[@]}" \
    --micro-batch-size "${MBS}" \
    --global-batch-size "${GBS}" \
    --seq-length "${SEQ_LEN}" \
    --max-position-embeddings "${SEQ_LEN}" \
    --train-iters "${TRAIN_ITERS}" \
    --mock-data \
    --split 100,0,0 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --accumulate-allreduce-grads-in-fp32 \
    --use-distributed-optimizer \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --clip-grad 1.0 \
    --log-interval 1 \
    --save-interval 1000000 \
    --eval-interval 1000000 \
    --eval-iters "${EVAL_ITERS:-1}" \
    "${LOAD_ARGS[@]}" \
    --distributed-backend nccl

echo ""
echo "=== [done] Lumen DSV4 Megatron pretrain smoke completed ==="
