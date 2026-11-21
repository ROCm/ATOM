#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Capture a rocprofv3 performance trace of Llama2-7B FSDP LoRA SFT, FP8
# blockwise2d, on 8 GPUs. Only GLOBAL rank 0 is profiled. A short run is
# enough — the trace covers model load, first-step JIT/compile, and a few
# steady-state training steps. View the resulting .pftrace at ui.perfetto.dev.
#
# Output (host): ${HOST_RESULTS}/rocprof/trace_rank0_results.pftrace
#
# Usage:
#   bash run_fsdp_lora_7b_rocprof.sh                 # 8 steps, runtime-trace
#   TRAIN_STEPS=12 bash run_fsdp_lora_7b_rocprof.sh  # more steady steps
#   LUMEN_ROCPROF_OPTS="--sys-trace" bash run_fsdp_lora_7b_rocprof.sh   # heavier
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Short by default: step 1 includes JIT/compile (~150 s for blockwise2d); a
# handful of steps after that is plenty for a steady-state trace. TRAIN_STEPS<10
# also avoids the step-10 eval so the trace stays pure training.
export MODE=fp8_blockwise2d
export TRAIN_STEPS="${TRAIN_STEPS:-8}"
export LUMEN_ROCPROF=1
export LUMEN_ROCPROF_OPTS="${LUMEN_ROCPROF_OPTS:--r}"

: "${HOST_MODEL:?Set HOST_MODEL to a HuggingFace Llama-2-7b directory}"
: "${HOST_DATA:?Set HOST_DATA to the directory that contains train.npy and validation.npy}"
export HOST_MODEL HOST_DATA
export HOST_RESULTS="${HOST_RESULTS:-${SCRIPT_DIR}/results/rocprof_7b_fp8_blockwise2d}"
export CONTAINER_NAME="${CONTAINER_NAME:-lumen_fsdp_7b_rocprof}"

# Trace default-heuristic kernels (no first-step tuning sweep). Set =1 to trace
# the tuned kernels instead (adds minutes of startup).
export PYTORCH_TUNABLEOP_ENABLED="${PYTORCH_TUNABLEOP_ENABLED:-0}"

bash "${SCRIPT_DIR}/run_fsdp_lora_7b.sh"
