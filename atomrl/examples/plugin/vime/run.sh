#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Launch Qwen3-8B RL training on vime with AtomRL plugin hooks.
#
# AtomRL hooks are injected via vime's --custom-*-path CLI arguments:
#   - --custom-megatron-init-path:  sets up Lumen FP8 training
#   - --custom-model-provider-path: builds GPTModel with Lumen FP8
#   - --custom-generate-function-path: per-sample ATOM rollout (optional)
#
# Prerequisites:
#   - vime and atomrl both installed
#   - Megatron-LM (ROCm fork) installed or on PYTHONPATH
#   - Model and data downloaded to $DATA_ROOT
#
# Usage:
#   DATA_ROOT=/data bash examples/plugin/vime/run.sh           # 3-step smoke
#   DATA_ROOT=/data STEPS=200 bash examples/plugin/vime/run.sh # long run
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing models/ and data_cached/}"
: "${VIME_ROOT:?Set VIME_ROOT to the vime repo directory}"

STEPS="${STEPS:-3}"
MODEL="${DATA_ROOT}/models/Qwen3-8B-Base"
TRAIN_DATA="${DATA_ROOT}/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet"
LOG="${LOG:-${DATA_ROOT}/logs/vime-plugin-qwen3-8b-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"

# Enable FP8 training (set to 0 to disable)
TRAIN_FP8="${TRAIN_FP8:-0}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=7200

# FP8 environment variables (only when TRAIN_FP8=1)
if [ "$TRAIN_FP8" = "1" ]; then
    export LUMEN_FP8=1 FP8_PARAM_MANAGER=0 LUMEN_NORM=1
    export LUMEN_FP8_SCALING=blockwise2d LUMEN_FP8_FORMAT=fp8_e4m3
    export LUMEN_FP8_BLOCK_SIZE=128 LUMEN_FP8_ATTN=none
fi

# Verify hooks are importable
python3 -c "
from atomrl.plugin.vime.hooks import custom_megatron_init, custom_model_provider, generate_rollout
import inspect
sig = inspect.signature(custom_megatron_init)
assert list(sig.parameters) == ['args'], f'Wrong signature: {sig}'
print('AtomRL vime hooks: importable and signature-verified')
"

echo "=== vime + AtomRL plugin: Qwen3-8B Megatron ==="
echo "    STEPS=$STEPS  TRAIN_FP8=$TRAIN_FP8  LOG=$LOG"

# AtomRL plugin hooks (the core injection points)
PLUGIN_ARGS=(
    --custom-megatron-init-path  atomrl.plugin.vime.hooks.custom_megatron_init
    --custom-model-provider-path atomrl.plugin.vime.hooks.custom_model_provider
)

cd "$VIME_ROOT"
python3 -u train.py \
    --hf-checkpoint "$MODEL" \
    --prompt-data "$TRAIN_DATA" \
    --input-key prompt \
    --num-rollout "$STEPS" \
    --rollout-batch-size 128 \
    --train-micro-batch-size 1 \
    --num-generation 16 \
    --max-response-len 2048 \
    --lr 1.0e-6 \
    --lr-warmup-iters 10 \
    --loss-type grpo \
    --update-weight-transport nccl \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    "${PLUGIN_ARGS[@]}" \
    2>&1 | tee "$LOG"
