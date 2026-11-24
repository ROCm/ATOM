#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

#
# LLaMA2 LoRA SFT — 8 GPUs, PyTorch FSDP backend (generic launcher).
#
# Model size and precision are chosen by the config (CONFIG=) and, for the 7B
# config, the MODE env var (bf16 / fp8_delayed / fp8_blockwise). Launches the
# lumen/llama2 container, overlays the host Lumen package and examples (keeping
# the image's compiled third_party AITER/mori), mounts the HF model + dataset,
# and runs run_finetune.sh.
#
# Overridable env: CONFIG, MODE, HOST_MODEL, HOST_DATA, HOST_RESULTS, IMAGE,
# CONTAINER_NAME.
#
# Examples:
#   HOST_MODEL=/path/to/70b HOST_DATA=/path/to/npy \
#     bash run_fsdp_lora.sh                                # 70B bf16 (default config)
#   HOST_MODEL=/path/to/7b HOST_DATA=/path/to/npy HOST_RESULTS=/path/out \
#     CONFIG=config_MI308X_fsdp_lora_7b.sh MODE=fp8_delayed \
#     bash run_fsdp_lora.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_LUMEN="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ATOM_ROOT="${ATOM_ROOT:-$(cd "${HOST_LUMEN}/../.." && pwd)}"

# ---- Host paths (model / data / results) — overridable via env --------------
HOST_MODEL="${HOST_MODEL:?Set HOST_MODEL to a HuggingFace model directory}"
HOST_DATA="${HOST_DATA:?Set HOST_DATA to the dataset directory mounted at /data}"
HOST_RESULTS="${HOST_RESULTS:-${SCRIPT_DIR}/results/fsdp_lora}"
CONFIG="${CONFIG:-config_MI308X_fsdp_lora_70b.sh}"
IMAGE="${IMAGE:-lumen/llama2:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-lumen_fsdp_lora}"

mkdir -p "${HOST_RESULTS}"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run --rm --init \
    --name "${CONTAINER_NAME}" \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    --security-opt=seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    ${AITER_SRC:+-v "${AITER_SRC}:/opt/aiter"} \
    ${AITER_JIT_DIR:+-v "${AITER_JIT_DIR}:/aiter-jit"} \
    ${AITER_JIT_DIR:+-e AITER_JIT_DIR=/aiter-jit} \
    -v "${ATOM_ROOT}:/workspace/ATOM" \
    -v "${HOST_LUMEN}/examples:/workspace/Lumen/examples" \
    -v "${HOST_MODEL}:/model-hf:ro" \
    -v "${HOST_DATA}:/data:ro" \
    -v "${HOST_RESULTS}:/results" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e TOKENIZERS_PARALLELISM=false \
    ${NCCL_IB_DISABLE:+-e NCCL_IB_DISABLE="${NCCL_IB_DISABLE}"} \
    ${NCCL_SOCKET_IFNAME:+-e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"} \
    ${NCCL_DEBUG:+-e NCCL_DEBUG="${NCCL_DEBUG}"} \
    ${LUMEN_DEBUG_FSDP:+-e LUMEN_DEBUG_FSDP="${LUMEN_DEBUG_FSDP}"} \
    -e PYTORCH_TUNABLEOP_ENABLED="${PYTORCH_TUNABLEOP_ENABLED:-0}" \
    -e RUN_CONFIG="${CONFIG}" \
    -e MODE="${MODE:-}" \
    -e TRAIN_STEPS="${TRAIN_STEPS:-}" \
    -e SEQ_LEN="${SEQ_LEN:-}" \
    -e MBS="${MBS:-}" \
    -e GRAD_ACCUM="${GRAD_ACCUM:-}" \
    -e LORA_RANK="${LORA_RANK:-}" \
    -e NUM_WORKERS="${NUM_WORKERS:-}" \
    -e TRAIN_DATA="${TRAIN_DATA:-}" \
    -e VALID_DATA="${VALID_DATA:-}" \
    -e LUMEN_BPRESHUFFLE_ONFLY="${LUMEN_BPRESHUFFLE_ONFLY:-}" \
    -e LUMEN_SKIP_FROZEN_WGRAD="${LUMEN_SKIP_FROZEN_WGRAD:-}" \
    -e FSDP_VERSION="${FSDP_VERSION:-}" \
    -e LUMEN_FSDP_AUTO_WRAP="${LUMEN_FSDP_AUTO_WRAP:-}" \
    -e NO_GRADIENT_CHECKPOINTING="${NO_GRADIENT_CHECKPOINTING:-}" \
    -e FSDP_FP8_PARAM_STORAGE="${FSDP_FP8_PARAM_STORAGE:-}" \
    -e EVAL_EVERY="${EVAL_EVERY:-}" \
    -e EVAL_INTERVAL="${EVAL_INTERVAL:-}" \
    -e VAL_SAMPLES="${VAL_SAMPLES:-}" \
    -e VAL_LOSS_TARGET="${VAL_LOSS_TARGET:-}" \
    -e LUMEN_PROF_START="${LUMEN_PROF_START:-}" \
    -e LUMEN_PROF_END="${LUMEN_PROF_END:-}" \
    -e LUMEN_PROF_OUTPUT="${LUMEN_PROF_OUTPUT:-}" \
    -e LUMEN_PROF_TRACE="${LUMEN_PROF_TRACE:-}" \
    -e SAVE_INTERVAL="${SAVE_INTERVAL:-}" \
    -e SAVE_CKPT="${SAVE_CKPT:-}" \
    -e SHARDING="${SHARDING:-}" \
    -e LUMEN_ROCPROF="${LUMEN_ROCPROF:-}" \
    -e LUMEN_ROCPROF_OPTS="${LUMEN_ROCPROF_OPTS:-}" \
    -e LUMEN_ROCPROF_DIR="${LUMEN_ROCPROF_DIR:-}" \
    -e LUMEN_ROCPROF_NAME="${LUMEN_ROCPROF_NAME:-}" \
    "${IMAGE}" \
    bash -c '
set -euo pipefail
# Dockerfile.lumen installs AtomRL editable at this exact bind-mounted path.
# Do not reinstall the root ATOM project here: its package discovery does not
# own the native Lumen extension destination.
cd /workspace/ATOM
test -d atomrl || {
    echo "ERROR: AtomRL checkout is not mounted at /workspace/ATOM/atomrl" >&2
    find /workspace -maxdepth 2 -type d -print >&2
    exit 1
}
cd atomrl
cd /workspace/Lumen/examples/llama2
CONFIG="${RUN_CONFIG}" BACKEND=fsdp bash run_finetune.sh 2>&1 \
    | tee /results/train.log
'
