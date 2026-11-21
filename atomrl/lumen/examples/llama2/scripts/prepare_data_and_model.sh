#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# End-to-end data and model preparation for LLaMA2 SFT using Megatron-LM-AMD.
#
# This script:
#   1. (Optional) Downloads the LLaMA2 HuggingFace checkpoint
#   2. Converts it to Megatron-LM format using tools/checkpoint/convert.py
#   3. (Optional) Downloads the Gov Report dataset
#   4. Converts the dataset to SFT jsonl format
#
# Download steps are automatically skipped when the target directory/files
# already exist.  Set SKIP_DOWNLOAD=1 to force-skip downloads regardless.
#
# Prerequisites:
#   export MEGATRON_ROOT=/path/to/Megatron-LM-AMD
#
# Usage:
#   bash prepare_data_and_model.sh
#
# Environment variables:
#   MEGATRON_ROOT   - path to Megatron-LM-AMD (REQUIRED)
#   MODEL_NAME      - HF model name (default: meta-llama/Llama-2-70b-hf)
#   MODEL_SIZE      - Megatron model size tag (default: llama2-70B)
#   TP              - target tensor parallel size (default: 8)
#   HF_MODEL_DIR    - directory for HF checkpoint (default: /model)
#   MEGATRON_CKPT   - directory for Megatron checkpoint (default: /results/megatron_ckpt)
#   DATA_DIR        - base data directory (default: /data)
#   SKIP_DOWNLOAD   - set to 1 to skip all download steps (default: auto)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Resolve MEGATRON_ROOT --------------------------------------------------

if [ -z "${MEGATRON_ROOT:-}" ]; then
    _candidate="$(cd "${SCRIPT_DIR}/../../.." && pwd)/Megatron-LM-AMD"
    if [ -d "${_candidate}" ]; then
        MEGATRON_ROOT="${_candidate}"
    else
        echo "ERROR: MEGATRON_ROOT is not set and could not be auto-detected."
        echo "  export MEGATRON_ROOT=/path/to/Megatron-LM-AMD"
        exit 1
    fi
fi

if [ ! -f "${MEGATRON_ROOT}/tools/checkpoint/convert.py" ]; then
    echo "ERROR: MEGATRON_ROOT=${MEGATRON_ROOT} does not look like a Megatron-LM-AMD directory."
    echo "  Expected to find tools/checkpoint/convert.py in that directory."
    exit 1
fi

export MEGATRON_ROOT

MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-2-70b-hf"}
MODEL_SIZE=${MODEL_SIZE:-"llama2-70B"}
TP=${TP:-8}
HF_MODEL_DIR=${HF_MODEL_DIR:-"/model"}
MEGATRON_CKPT=${MEGATRON_CKPT:-"/results/megatron_ckpt"}
DATA_DIR=${DATA_DIR:-"/data"}

echo "============================================================"
echo " LLaMA2 SFT Preparation (Megatron-LM-AMD)"
echo "============================================================"
echo " MEGATRON_ROOT: ${MEGATRON_ROOT}"
echo " Model:        ${MODEL_NAME} (${MODEL_SIZE})"
echo " TP:           ${TP}"
echo " HF dir:       ${HF_MODEL_DIR}"
echo " Megatron:     ${MEGATRON_CKPT}"
echo " Data dir:     ${DATA_DIR}"
echo "============================================================"

# --------------------------------------------------------------------------
# Step 1: Download HuggingFace model (skipped when already present)
# --------------------------------------------------------------------------
echo ""
_model_present=0
[ -f "${HF_MODEL_DIR}/config.json" ] && _model_present=1
if [ "${SKIP_DOWNLOAD:-0}" = "1" ] || [ "${_model_present}" = "1" ]; then
    echo "[Step 1/4] Skipping model download (already present at ${HF_MODEL_DIR})"
else
    echo "[Step 1/4] Downloading HuggingFace model ..."
    python "${SCRIPT_DIR}/download_model.py" \
        --model_name "${MODEL_NAME}" \
        --output_dir "${HF_MODEL_DIR}"
fi

# --------------------------------------------------------------------------
# Step 2: Convert HF checkpoint to Megatron format
# --------------------------------------------------------------------------
echo ""
echo "[Step 2/4] Converting checkpoint to Megatron format ..."
echo "  loader: llama_mistral | model-size: ${MODEL_SIZE} | TP: ${TP}"

python "${MEGATRON_ROOT}/tools/checkpoint/convert.py" \
    --model-type GPT \
    --loader llama_mistral \
    --load-dir "${HF_MODEL_DIR}" \
    --model-size "${MODEL_SIZE}" \
    --checkpoint-type hf \
    --tokenizer-model "${MODEL_NAME}" \
    --saver core \
    --save-dir "${MEGATRON_CKPT}" \
    --target-tensor-parallel-size "${TP}" \
    --target-pipeline-parallel-size 1 \
    --bf16

echo "  Megatron checkpoint saved to: ${MEGATRON_CKPT}"

# --------------------------------------------------------------------------
# Step 3: Download Gov Report dataset (skipped when already present)
# --------------------------------------------------------------------------
echo ""
_data_present=0
{ [ -f "${DATA_DIR}/train.jsonl" ] && [ -f "${DATA_DIR}/validation.jsonl" ]; } && _data_present=1
if [ "${SKIP_DOWNLOAD:-0}" = "1" ] || [ "${_data_present}" = "1" ]; then
    echo "[Step 3/4] Skipping dataset download (train.jsonl / validation.jsonl already present in ${DATA_DIR})"
else
    echo "[Step 3/4] Downloading Gov Report dataset ..."
    python "${SCRIPT_DIR}/download_dataset.py" \
        --output_dir "${DATA_DIR}/gov_report_raw"
fi

# --------------------------------------------------------------------------
# Step 4: Convert dataset to SFT jsonl format (skipped when already done)
# --------------------------------------------------------------------------
echo ""
if [ "${_data_present}" = "1" ]; then
    echo "[Step 4/4] Skipping dataset conversion (train.jsonl / validation.jsonl already present)"
else
    echo "[Step 4/4] Converting dataset to SFT jsonl format ..."
    python "${SCRIPT_DIR}/convert_dataset.py" \
        --input_dir "${DATA_DIR}/gov_report_raw" \
        --output_dir "${DATA_DIR}"
fi

echo ""
echo "============================================================"
echo " Preparation complete!"
echo ""
echo " Checkpoint:  ${MEGATRON_CKPT}"
echo " Train data:  ${DATA_DIR}/train.jsonl"
echo " Valid data:  ${DATA_DIR}/validation.jsonl"
echo ""
echo " To start fine-tuning:"
echo "   CKPT_DIR=${MEGATRON_CKPT} \\"
echo "   TRAIN_DATA=${DATA_DIR}/train.jsonl \\"
echo "   VALID_DATA=${DATA_DIR}/validation.jsonl \\"
echo "   bash $(dirname ${SCRIPT_DIR})/run_finetune.sh"
echo "============================================================"
