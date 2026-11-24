#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Launch Qwen3-8B GRPO on verl with AtomRL plugin (FSDP2 + ATOM).
#
# Prerequisites:
#   - verl and the unified ATOM/AtomRL distribution installed
#   - The verl.plugins entry-point is active (default: VERL_USE_EXTERNAL_PLUGINS=auto)
#   - Model and data downloaded to $DATA_ROOT
#
# Usage:
#   DATA_ROOT=/data bash examples/plugin/verl/run.sh          # 3-step smoke
#   DATA_ROOT=/data STEPS=200 bash examples/plugin/verl/run.sh  # long run
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing models/ and data_cached/}"
STEPS="${STEPS:-3}"
CONFIG="${CONFIG:-examples/plugin/verl/qwen3_8b_grpo.yaml}"
LOG="${LOG:-${DATA_ROOT}/logs/verl-plugin-qwen3-8b-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=7200
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export TORCHDYNAMO_DISABLE=1
export VLLM_USE_V1=1
export HIP_FORCE_DEV_KERNARG=1
export HSA_NO_SCRATCH_RECLAIM=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Clean GPU visibility — let Ray manage device assignment (runbook §7.2)
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES 2>/dev/null || true

# Verify AtomRL plugin is registered
python3 -c "
import verl  # triggers plugin auto-discovery
from verl.workers.rollout.base import _ROLLOUT_REGISTRY
from verl.workers.engine.base import EngineRegistry
assert ('atom', 'async') in _ROLLOUT_REGISTRY, 'ATOM rollout not registered'
print('AtomRL plugin: ATOM rollout registered')
try:
    cls = EngineRegistry.get_engine_cls('language_model', 'atomrl_fsdp2')
    print(f'AtomRL plugin: atomrl_fsdp2 engine registered -> {cls.__name__}')
except Exception as e:
    print(f'Warning: atomrl_fsdp2 engine not registered: {e}')
"

echo "=== verl + AtomRL plugin: Qwen3-8B GRPO ==="
echo "    STEPS=$STEPS  CONFIG=$CONFIG  LOG=$LOG"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_ABS="$(cd "$REPO_ROOT" && realpath "$CONFIG")"

# Copy our yaml into verl's config dir so Hydra defaults resolution works.
# Use pip show to avoid importing verl (which triggers apex hipify stdout noise).
VERL_PKG_DIR="$(pip show verl 2>/dev/null | grep ^Location: | awk '{print $2}')/verl"
VERL_CFG_DIR="$VERL_PKG_DIR/trainer/config"
cp "$CONFIG_ABS" "$VERL_CFG_DIR/_atomrl_plugin.yaml"

python3 -m verl.trainer.main_ppo \
  --config-name _atomrl_plugin \
  trainer.total_training_steps="$STEPS" \
  2>&1 | tee "$LOG"
