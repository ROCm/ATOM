#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# End-to-end data and model preparation for LLaMA 3.1 pretraining.
#
# This script:
#   1. (Optional) Downloads the LLaMA 3.1 tokenizer/model from HuggingFace
#   2. (Megatron only) Converts the HF checkpoint to Megatron-LM format
#   3. (Optional) Downloads the C4 pretraining dataset
#   4. Converts/validates the dataset to jsonl format
#
# Download steps are automatically skipped when the target directory/files
# already exist.  Set SKIP_DOWNLOAD=1 to force-skip downloads regardless.
#
# Usage:
#   bash prepare_data_and_model.sh
#
# Environment variables:
#   BACKEND           - training backend: megatron or fsdp (default: megatron)
#   MEGATRON_ROOT     - path to Megatron-LM-AMD (required for megatron backend)
#   MODEL_NAME        - HF model name (default: meta-llama/Llama-3.1-8B)
#   MODEL_SIZE        - Megatron model size tag (default: llama3.1-8B)
#   TP                - target tensor parallel size (default: 1)
#   HF_MODEL_DIR      - directory for HF checkpoint (default: /model)
#   MEGATRON_CKPT     - directory for Megatron checkpoint (default: /results/megatron_ckpt)
#   DATA_DIR          - base data directory (default: /data)
#   MAX_TRAIN_SAMPLES - max training samples to download (default: 0 = all)
#   MAX_VAL_SAMPLES   - max validation samples to download (default: 10000)
#   SKIP_DOWNLOAD     - set to 1 to force-skip all download steps (default: auto)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND=${BACKEND:-"megatron"}
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.1-8B"}
MODEL_SIZE=${MODEL_SIZE:-"llama3.1-8B"}
TP=${TP:-1}
HF_MODEL_DIR=${HF_MODEL_DIR:-"/model"}
MEGATRON_CKPT=${MEGATRON_CKPT:-"/results/megatron_ckpt"}
DATA_DIR=${DATA_DIR:-"/data"}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-10000}

# ---- Resolve MEGATRON_ROOT (only needed for megatron backend) ---------------

if [ "${BACKEND}" = "megatron" ]; then
    if [ -z "${MEGATRON_ROOT:-}" ]; then
        _candidate="$(cd "${SCRIPT_DIR}/../../.." && pwd)/Megatron-LM-AMD"
        if [ -d "${_candidate}" ]; then
            MEGATRON_ROOT="${_candidate}"
        else
            echo "WARNING: MEGATRON_ROOT is not set and could not be auto-detected."
            echo "  Checkpoint conversion will be skipped."
            echo "  Set: export MEGATRON_ROOT=/path/to/Megatron-LM-AMD"
            MEGATRON_ROOT=""
        fi
    fi
fi

echo "============================================================"
echo " LLaMA 3.1 Pretraining Preparation"
echo "============================================================"
echo " Backend:      ${BACKEND}"
echo " Model:        ${MODEL_NAME} (${MODEL_SIZE})"
echo " TP:           ${TP}"
echo " HF dir:       ${HF_MODEL_DIR}"
[ "${BACKEND}" = "megatron" ] && echo " Megatron:     ${MEGATRON_CKPT}"
echo " Data dir:     ${DATA_DIR}"
echo " Train cap:    ${MAX_TRAIN_SAMPLES} (0=all)"
echo " Val cap:      ${MAX_VAL_SAMPLES} (0=all)"
echo "============================================================"

# --------------------------------------------------------------------------
# Step 1: Download HuggingFace model / tokenizer (skipped when already present)
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
# Step 2: Convert HF checkpoint to Megatron format (megatron backend only)
# --------------------------------------------------------------------------
if [ "${BACKEND}" = "megatron" ] && [ -n "${MEGATRON_ROOT:-}" ]; then
    if [ -f "${MEGATRON_ROOT}/tools/checkpoint/convert.py" ]; then
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
    else
        echo ""
        echo "[Step 2/4] Skipping Megatron conversion (convert.py not found)"
    fi
else
    echo ""
    echo "[Step 2/4] Skipping Megatron conversion (backend=${BACKEND})"
fi

# --------------------------------------------------------------------------
# Step 3: Download C4 pretraining dataset (skipped when already present)
# --------------------------------------------------------------------------
echo ""
_data_present=0
[ -f "${DATA_DIR}/train.jsonl" ] && _data_present=1
if [ "${SKIP_DOWNLOAD:-0}" = "1" ] || [ "${_data_present}" = "1" ]; then
    echo "[Step 3/4] Skipping dataset download (train.jsonl already present in ${DATA_DIR})"
else
    echo "[Step 3/4] Downloading C4 pretraining dataset ..."
    DOWNLOAD_ARGS="--output_dir ${DATA_DIR}/c4_raw"
    [ "${MAX_TRAIN_SAMPLES}" -gt 0 ] 2>/dev/null && DOWNLOAD_ARGS+=" --max_train_samples ${MAX_TRAIN_SAMPLES}"
    [ "${MAX_VAL_SAMPLES}" -gt 0 ] 2>/dev/null && DOWNLOAD_ARGS+=" --max_val_samples ${MAX_VAL_SAMPLES}"
    python "${SCRIPT_DIR}/download_dataset.py" ${DOWNLOAD_ARGS}
fi

# --------------------------------------------------------------------------
# Step 4: Convert / validate dataset to jsonl format (skipped when already done)
# --------------------------------------------------------------------------
echo ""
if [ "${_data_present}" = "1" ]; then
    echo "[Step 4/4] Skipping dataset conversion (train.jsonl already present)"
else
    echo "[Step 4/4] Converting dataset to pretraining jsonl format ..."
    CONVERT_ARGS="--input_dir ${DATA_DIR}/c4_raw --output_dir ${DATA_DIR} --input_format jsonl"
    [ "${MAX_TRAIN_SAMPLES}" -gt 0 ] 2>/dev/null && CONVERT_ARGS+=" --max_train_samples ${MAX_TRAIN_SAMPLES}"
    [ "${MAX_VAL_SAMPLES}" -gt 0 ] 2>/dev/null && CONVERT_ARGS+=" --max_val_samples ${MAX_VAL_SAMPLES}"
    python "${SCRIPT_DIR}/convert_dataset.py" ${CONVERT_ARGS}
fi

echo ""
echo "============================================================"
echo " Preparation complete!"
echo ""
echo " Model/Tokenizer: ${HF_MODEL_DIR}"
[ "${BACKEND}" = "megatron" ] && [ -n "${MEGATRON_ROOT:-}" ] && \
    echo " Megatron ckpt:   ${MEGATRON_CKPT}"
echo " Train data:      ${DATA_DIR}/train.jsonl"
echo " Valid data:       ${DATA_DIR}/validation.jsonl"
echo ""
echo " To start pretraining:"
if [ "${BACKEND}" = "megatron" ]; then
    echo "   TOKENIZER=${HF_MODEL_DIR} \\"
    echo "   TRAIN_DATA=${DATA_DIR}/train.jsonl \\"
    echo "   VALID_DATA=${DATA_DIR}/validation.jsonl \\"
    echo "   BACKEND=megatron bash $(dirname ${SCRIPT_DIR})/run_pretrain.sh"
else
    echo "   TOKENIZER=${HF_MODEL_DIR} \\"
    echo "   TRAIN_DATA=${DATA_DIR}/train.jsonl \\"
    echo "   VALID_DATA=${DATA_DIR}/validation.jsonl \\"
    echo "   BACKEND=fsdp bash $(dirname ${SCRIPT_DIR})/run_pretrain.sh"
fi
echo "============================================================"
