#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Lumen LLaMA2-70B LoRA finetune — Docker launcher.
#
# Uses MI300X reference download scripts (regisss/ public repos) for
# model and dataset, then runs Lumen's Megatron backend aligned to
# MI300X_EPYC_9575F configuration.
#
# Usage:
#   export HF_TOKEN=hf_xxxxx
#   bash run_lumen_finetune.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_SOURCE="${LUMEN_SOURCE:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
IMAGE="${IMAGE:-lumen_unit_test:latest}"
CONT_NAME="${CONT_NAME:-lumen_llama2_finetune}"
LOG_DIR="/data1/lumen/results"

echo "============================================================"
echo " Lumen LLaMA2-70B LoRA Finetune (MI300X)"
echo "============================================================"
echo "  Image:     ${IMAGE}"
echo "  Data root: /data1/lumen"
echo "  Log dir:   ${LOG_DIR}"
echo "============================================================"

docker container rm -f "${CONT_NAME}" 2>/dev/null || true

docker run --rm --init \
    --name="${CONT_NAME}" \
    --net=host --uts=host \
    --ipc=host --device /dev/dri --device /dev/kfd \
    --security-opt=seccomp=unconfined \
    -v /data1:/data1 \
    -v "${LUMEN_SOURCE}:/workspace/Lumen" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HSA_ENABLE_SDMA=0 \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_SOCKET_IFNAME=lo \
    -e NCCL_DEBUG=WARN \
    -e TORCHDYNAMO_DISABLE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    "${IMAGE}" \
    bash -c '
set -euo pipefail

DATA_ROOT=/data1/lumen
LUMEN_DIR=/workspace/Lumen/examples/llama2
source "${LUMEN_DIR}/../scripts/ensure_tokenizer.sh"
example_tokenizer llama2 "${LUMEN_DIR}/tokenizer"
LOG_DIR="${DATA_ROOT}/results"

########################################################################
# Phase 1: Install dependencies
########################################################################
echo "============================================================"
echo " Phase 1: Install dependencies"
echo "============================================================"
pip install -q huggingface-hub==0.30.0 pandas pyarrow sentencepiece transformers>=4.43.0 peft safetensors 2>&1 | tail -5

########################################################################
# Phase 2: Download model and dataset using MI300X reference scripts
########################################################################
echo ""
echo "============================================================"
echo " Phase 2: Download model + dataset (MI300X reference scripts)"
echo "============================================================"

mkdir -p ${DATA_ROOT}/{model,data,megatron_ckpt,results/checkpoints}

# --- Download the fused-QKV model (regisss/llama2-70b-fused-qkv-mlperf) ---
if ls ${DATA_ROOT}/model/*.safetensors 1>/dev/null 2>&1; then
    echo "[Model] Already downloaded at ${DATA_ROOT}/model"
else
    echo "[Model] Downloading regisss/llama2-70b-fused-qkv-mlperf ..."
    python -c "
import argparse, subprocess, hashlib, os
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download

model_dir = \"${DATA_ROOT}/model\"

snapshot_download(
    \"regisss/llama2-70b-fused-qkv-mlperf\",
    revision=\"647cb0c8858ddefd10231a20ddfa68e4eb5e850e\",
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    max_workers=8,
    allow_patterns=[\"*.safetensors\", \"*.json\", \"*.py\"],
)
subprocess.run(
    f\"find {model_dir} -mindepth 1 -type d -name .* -exec rm -rf {{}} +\",
    shell=True, executable=\"/bin/bash\",
)
print(f\"Model downloaded to {model_dir}\")
"
fi

# --- Download the pre-processed dataset ---
if ls ${DATA_ROOT}/data/*.parquet 1>/dev/null 2>&1 || ls ${DATA_ROOT}/data/*.npy 1>/dev/null 2>&1; then
    echo "[Dataset] Already downloaded at ${DATA_ROOT}/data"
else
    echo "[Dataset] Downloading regisss/scrolls_gov_report_preprocessed_mlperf_2 ..."
    python -c "
import subprocess
from huggingface_hub import snapshot_download

data_dir = \"${DATA_ROOT}/data\"

snapshot_download(
    \"regisss/scrolls_gov_report_preprocessed_mlperf_2\",
    revision=\"21ff1233ee3e87bc780ab719c755170148aba1cb\",
    allow_patterns=\"*.parquet\",
    local_dir=data_dir,
    local_dir_use_symlinks=False,
    max_workers=16,
    repo_type=\"dataset\",
)
subprocess.run(
    f\"mv {data_dir}/data/* {data_dir}/ && find {data_dir} -mindepth 1 ! -name *.parquet -exec rm -rf {{}} +\",
    shell=True, executable=\"/bin/bash\",
)
print(f\"Dataset downloaded to {data_dir}\")
"
fi

########################################################################
# Phase 3: Convert dataset parquet → .npy (packed sequence format)
########################################################################
echo ""
echo "============================================================"
echo " Phase 3: Convert dataset (parquet → npy)"
echo "============================================================"

if [ -f "${DATA_ROOT}/data/train.npy" ] && [ -f "${DATA_ROOT}/data/validation.npy" ]; then
    echo "[Dataset] .npy files already exist. Skipping conversion."
else
    echo "[Dataset] Converting parquet → npy ..."
    python -c "
import numpy as np
import pandas as pd

data_dir = \"${DATA_ROOT}/data\"

def convert(split):
    df = pd.read_parquet(f\"{data_dir}/{split}-00000-of-00001.parquet\")
    transformed = df.apply(
        lambda row: {
            \"input_ids\": row[\"input_ids\"],
            \"loss_mask\": [int(x != -100) for x in row[\"labels\"]],
            \"seq_start_id\": [0],
        },
        axis=1,
    ).tolist()
    np.save(f\"{data_dir}/{split}\", transformed)
    print(f\"  {split}: {len(transformed)} samples → {data_dir}/{split}.npy\")

convert(\"train\")
convert(\"validation\")
print(\"Dataset conversion complete.\")
"
fi

########################################################################
# Phase 4: Convert model to Megatron-LM-AMD format
########################################################################
echo ""
echo "============================================================"
echo " Phase 4: Convert model (HF fused-QKV → Megatron TP=1)"
echo "============================================================"

MEGATRON_ROOT=""
for _c in /workspace/megatron_lm /workspace/Megatron-LM; do
    [ -f "${_c}/tools/checkpoint/convert.py" ] && MEGATRON_ROOT="${_c}" && break
done
if [ -z "${MEGATRON_ROOT}" ]; then
    echo "ERROR: Cannot find Megatron-LM-AMD"
    exit 1
fi
echo "  MEGATRON_ROOT: ${MEGATRON_ROOT}"

# Fix Megatron bug: FusedLayerNorm does not support RMSNorm (SOURCE registry)
LUMEN_ROOT="/workspace/Lumen"
# shellcheck source=examples/llama2/scripts/apply_megatron_source_patches.sh
source "${LUMEN_ROOT}/examples/llama2/scripts/apply_megatron_source_patches.sh"
apply_lumen_megatron_source_patches "${MEGATRON_ROOT}" llama

if ls -d "${DATA_ROOT}/megatron_ckpt/iter_"* 1>/dev/null 2>&1 || [ -d "${DATA_ROOT}/megatron_ckpt/release" ]; then
    echo "[Checkpoint] Megatron checkpoint already exists. Skipping."
else
    # Step 4a: Split fused-QKV model into standard LLaMA format
    STANDARD_DIR="${DATA_ROOT}/model-standard"
    if [ -f "${STANDARD_DIR}/config.json" ] && ls ${STANDARD_DIR}/*.safetensors 1>/dev/null 2>&1; then
        echo "[Split] Standard LLaMA model already exists at ${STANDARD_DIR}"
    else
        echo "[Split] Splitting fused qkv_proj into separate q/k/v_proj ..."
        python "${LUMEN_DIR}/scripts/split_fused_qkv.py" \
            --src "${DATA_ROOT}/model" \
            --dst "${STANDARD_DIR}"
    fi

    # Step 4b: Convert standard LLaMA → Megatron
    echo "[Checkpoint] Converting standard LLaMA → Megatron (TP=1) ..."
    TOKENIZER_PATH="${LUMEN_DIR}/tokenizer"
    echo "  Using tokenizer: ${TOKENIZER_PATH}"

    MEGATRON_ROOT="${MEGATRON_ROOT}" python "${LUMEN_DIR}/scripts/convert_to_megatron.py" \
        --model-type GPT \
        --loader llama_mistral \
        --load-dir "${STANDARD_DIR}" \
        --model-size "llama2-70B" \
        --checkpoint-type hf \
        --tokenizer-model "${TOKENIZER_PATH}" \
        --saver core \
        --save-dir "${DATA_ROOT}/megatron_ckpt" \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1 \
        --saver-transformer-impl local \
        --bf16
    echo "[Checkpoint] Conversion complete: ${DATA_ROOT}/megatron_ckpt"
fi

########################################################################
# Phase 5: Run Lumen finetune (Megatron backend, MI300X aligned)
########################################################################
echo ""
echo "============================================================"
echo " Phase 5: Run Lumen finetune (Megatron, MI300X config)"
echo "============================================================"

export CKPT_DIR="${DATA_ROOT}/megatron_ckpt"
export TRAIN_DATA="${DATA_ROOT}/data/train.npy"
export VALID_DATA="${DATA_ROOT}/data/validation.npy"
export SAVE_DIR="${DATA_ROOT}/results/checkpoints"
export TOKENIZER="${LUMEN_DIR}/tokenizer"
export CONFIG="${LUMEN_DIR}/config_MI300X_1x8x1.sh"

echo "  CKPT_DIR:   ${CKPT_DIR}"
echo "  TRAIN_DATA: ${TRAIN_DATA}"
echo "  VALID_DATA: ${VALID_DATA}"
echo "  CONFIG:     ${CONFIG}"

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

bash "${LUMEN_DIR}/run_finetune.sh" 2>&1 | tee "${LOG_DIR}/lumen_llama2_mi300x_finetune.log"

echo ""
echo "============================================================"
echo " Lumen finetune complete!"
echo " Log: ${LOG_DIR}/lumen_llama2_mi300x_finetune.log"
echo "============================================================"
'
