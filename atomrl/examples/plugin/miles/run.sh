#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZhangDanyang-AMD
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Launch Qwen3-8B training on miles with AtomRL Megatron/Lumen hooks.
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the prepared AtomRL data root}"
: "${MILES_ROOT:?Set MILES_ROOT to the miles repository}"

STEPS="${STEPS:-3}"
MODEL="${DATA_ROOT}/models/Qwen3-8B-Base"
TRAIN_DATA="${DATA_ROOT}/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet"
LOG="${LOG:-${DATA_ROOT}/logs/miles-plugin-qwen3-8b-$(date +%Y%m%d-%H%M%S).log}"
TRAIN_FP8="${TRAIN_FP8:-0}"
mkdir -p "$(dirname "${LOG}")"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=7200
if [[ "${TRAIN_FP8}" == 1 ]]; then
    export LUMEN_FP8=1 FP8_PARAM_MANAGER=0 LUMEN_NORM=1
    export LUMEN_FP8_SCALING=blockwise2d LUMEN_FP8_FORMAT=fp8_e4m3
    export LUMEN_FP8_BLOCK_SIZE=128 LUMEN_FP8_ATTN=none
fi

python3 - <<'PY'
from atomrl.plugin.miles.hooks import (
    custom_megatron_before_log_prob_hook,
    custom_megatron_before_train_step_hook,
    custom_megatron_init,
    custom_megatron_post_save_hook,
    custom_model_provider,
    custom_update_weight_post_write,
    generate_rollout,
)
from atomrl.plugin.miles.lifecycle import setup_atom_manager, teardown_atom_manager

print("AtomRL miles hooks: importable")
PY

PLUGIN_ARGS=(
    --custom-megatron-init-path
        atomrl.plugin.miles.hooks.custom_megatron_init
    --custom-model-provider-path
        atomrl.plugin.miles.hooks.custom_model_provider
    --custom-megatron-before-train-step-hook-path
        atomrl.plugin.miles.hooks.custom_megatron_before_train_step_hook
    --custom-megatron-before-log-prob-hook-path
        atomrl.plugin.miles.hooks.custom_megatron_before_log_prob_hook
    --custom-megatron-post-save-hook-path
        atomrl.plugin.miles.hooks.custom_megatron_post_save_hook
    --custom-update-weight-post-write-path
        atomrl.plugin.miles.hooks.custom_update_weight_post_write
)

cd "${MILES_ROOT}"
python3 -u train.py \
    --hf-checkpoint "${MODEL}" \
    --prompt-data "${TRAIN_DATA}" \
    --input-key prompt \
    --num-rollout "${STEPS}" \
    --rollout-batch-size 128 \
    --train-micro-batch-size 1 \
    --num-generation 16 \
    --max-response-len 2048 \
    --lr 1.0e-6 \
    --lr-warmup-iters 10 \
    --loss-type grpo \
    --update-weight-transfer-mode broadcast \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    "${PLUGIN_ARGS[@]}" \
    2>&1 | tee "${LOG}"
