#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

#
# LLaMA2-7B LoRA SFT — 8 GPUs, PyTorch FSDP backend.
#
# Thin convenience wrapper around run_fsdp_lora.sh that pins the 7B config
# and sensible 7B defaults. Precision/scaling is chosen by MODE:
#   HOST_MODEL=/path/to/llama2_7b_hf HOST_DATA=/path/to/npy \
#     bash run_fsdp_lora_7b.sh                              # bf16 (default)
#   MODE=fp8_delayed   bash run_fsdp_lora_7b.sh
#   MODE=fp8_blockwise bash run_fsdp_lora_7b.sh
#
# HOST_DATA is the directory mounted at /data. The 7B config reads
# /data/train.npy and /data/validation.npy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${MODE:-bf16}"
HOST_MODEL="${HOST_MODEL:?Set HOST_MODEL to a HuggingFace Llama-2-7b directory}"
HOST_DATA="${HOST_DATA:?Set HOST_DATA to the directory that contains train.npy and validation.npy}"

if [ ! -d "${HOST_MODEL}" ]; then
    echo "ERROR: 7B model dir not found: ${HOST_MODEL}"
    echo "  Set HOST_MODEL=/path/to/llama2_7b_hf (a HuggingFace LlamaForCausalLM dir)."
    exit 1
fi
if [ ! -d "${HOST_DATA}" ]; then
    echo "ERROR: data dir not found: ${HOST_DATA}"
    exit 1
fi

# Same public NousResearch fetch as the Megatron cases:
# examples/llama2/tokenizer, not the model directory.
source "${SCRIPT_DIR}/../scripts/ensure_tokenizer.sh"
example_tokenizer llama2 "${SCRIPT_DIR}/tokenizer" || exit 1

# Delegate to the generic launcher; results dir defaults to a per-mode path.
HOST_MODEL="${HOST_MODEL}" \
HOST_DATA="${HOST_DATA}" \
HOST_RESULTS="${HOST_RESULTS:-${SCRIPT_DIR}/results/fsdp_lora_7b_${MODE}}" \
CONFIG="config_MI308X_fsdp_lora_7b.sh" \
CONTAINER_NAME="${CONTAINER_NAME:-lumen_fsdp_7b_${MODE}}" \
MODE="${MODE}" \
IMAGE="${IMAGE:-lumen/llama2:latest}" \
bash "${SCRIPT_DIR}/run_fsdp_lora.sh"
