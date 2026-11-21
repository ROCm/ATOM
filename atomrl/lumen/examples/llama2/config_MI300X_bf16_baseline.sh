#!/bin/bash
# BF16 baseline config for diagnosing FP8/precision issues.
# Same as config_MI300X_1x8x1.sh but with FP8 disabled, no synthetic
# warmup, and standard-precision RoPE.
#
# Usage:
#   CONFIG=config_MI300X_bf16_baseline.sh bash run_finetune.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_MI300X_1x8x1.sh"

# ---- Override: disable FP8 completely ----------------------------------------
export FP8_TRAINING=0
export FP8_ACTIVATION=0
export FP8_WGRAD=0

# ---- Override: disable synthetic warmup (no FP8 = no warmup needed) ----------
export WARMUP_STEPS=0

# ---- Override: shorter run for quick diagnosis (192 steps = first eval) ------
export TRAIN_STEPS=384
export EVAL_EVERY=192

# ---- Override: evaluate more frequently for loss curve visibility ------------
export EVAL_INTERVAL=48
export EVAL_ITERS=5
