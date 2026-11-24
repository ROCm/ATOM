#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Single source of every path this example uses. Source it, never hardcode.
#
# Defaults assume the container layout the README sets up: one host directory
# bind-mounted at /work. Override any variable before sourcing to fit a
# different layout, e.g.
#
#   WORK=/scratch/flydsl . env.sh
#
: "${WORK:=/work}"
: "${ATOM_ROOT:=$WORK/ATOM}"
: "${ATOMRL_DIR:=$ATOM_ROOT/atomrl}"
: "${LUMEN_DIR:=$ATOMRL_DIR/lumen}"
: "${VEOMNI_DIR:=$WORK/VeOmni}"
: "${QWEN_IMAGE_DIR:=$WORK/models/Qwen-Image}"
: "${DATA_DIR:=$WORK/data/qwen_image_smoke}"
# Wan2.1-T2V-1.3B and the Tom-and-Jerry clips, for the video (T>1) comparison.
# Only needed by run_wan_10steps.sh; the image example does not touch them.
: "${WAN_DIR:=$WORK/models-extra/Wan2.1-T2V-1.3B}"
: "${WAN_DATA_DIR:=$WORK/data/tom-and-jerry-parquet}"
: "${OUT_DIR:=$WORK/outputs}"
: "${LOG_DIR:=$WORK/logs}"
: "${HF_HOME:=$WORK/hf-cache}"

: "${EXAMPLE_DIR:=$LUMEN_DIR/examples/perf-veomni}"

# VeOmni commit this example is pinned to. Later commits may move the condition
# model or the DiT trainer hook points that train_dit_lumen.py wraps.
: "${VEOMNI_COMMIT:=573848a00fcd7329c2411346c6f4a983e9f67e3f}"

export WORK ATOM_ROOT ATOMRL_DIR LUMEN_DIR VEOMNI_DIR QWEN_IMAGE_DIR DATA_DIR OUT_DIR LOG_DIR HF_HOME
export WAN_DIR WAN_DATA_DIR
export EXAMPLE_DIR VEOMNI_COMMIT

# Use the unified AtomRL/Lumen source without modifying installed dependencies.
export LUMEN_PYTHONPATH="$ATOMRL_DIR"
