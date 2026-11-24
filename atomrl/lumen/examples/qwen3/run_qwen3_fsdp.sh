#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

#
# Qwen3-8B LoRA SFT — 8 GPUs, PyTorch FSDP + Lumen FP8 blockwise2d.
#
# Launches the lumen/llama2 container, overlays the host Lumen package and
# examples (keeping the image's compiled third_party AITER/mori), mounts the HF
# model + alpaca jsonl dataset, and runs train_qwen3_fsdp.py.
#
# Overridable env: HOST_MODEL, HOST_DATA, HOST_RESULTS, TRAIN_FILE, VAL_FILE,
# SEQ_LENGTH, MAX_STEPS, EVAL_INTERVAL, IMAGE, CONTAINER_NAME.
#
# Example:
#   HOST_MODEL=/path/to/Qwen3-8B HOST_DATA=/path/to/alpaca_jsons \
#     bash run_qwen3_fsdp.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_LUMEN="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ATOM_ROOT="${ATOM_ROOT:-$(cd "${HOST_LUMEN}/../.." && pwd)}"

# ---- Host paths (model / data / results) — overridable via env --------------
HOST_MODEL="${HOST_MODEL:?Set HOST_MODEL to a Qwen3-8B directory}"
HOST_DATA="${HOST_DATA:?Set HOST_DATA to a directory of alpaca jsonl}"
HOST_RESULTS="${HOST_RESULTS:-${SCRIPT_DIR}/results}"
TRAIN_FILE="${TRAIN_FILE:-alpaca_zh-train-general.jsonl}"
VAL_FILE="${VAL_FILE:-alpaca_zh-valid-general.jsonl}"
MODE="${MODE:-fp8_blockwise2d}"        # bf16 | fp8_blockwise2d
if [[ "${MODE}" == "mxfp4" ]]; then
    echo "ERROR: this launcher is BF16 / FP8 blockwise2d. MXFP4 LoRA SFT is run_qwen3_fsdp_mxfp4_sft.sh; full-parameter pretraining is run_qwen3_fsdp_mxfp4_pretrain.sh." >&2
    exit 2
fi
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
MAX_STEPS="${MAX_STEPS:-200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
IMAGE="${IMAGE:-lumen/llama2:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-lumen_qwen3_fsdp}"

# Optional: use the in-repo tuned a8w8_blockscale CSV (gfx942/MI308X, 80 CU) to
# override the image's default kernels for the Qwen3 transformer-layer shapes.
# USE_TUNED_GEMM=1 enables it. Default off — the gain is small (~3%; the default
# kernel is already near-optimal) and it only matches gfx942/cu_num=80/M=2048;
# on other hardware it would drop the image's gfx950 tuned configs, so keep off
# unless on MI308X. The CSV lives under the already-mounted examples/ dir.
USE_TUNED_GEMM="${USE_TUNED_GEMM:-0}"
TUNED_GEMM_CSV="${TUNED_GEMM_CSV:-/workspace/Lumen/examples/qwen3/configs/a8w8_blockscale_tuned_gemm.csv}"
EXTRA_DOCKER_ARGS=()
if [[ "${USE_TUNED_GEMM}" != "0" ]]; then
    EXTRA_DOCKER_ARGS+=( -e "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=${TUNED_GEMM_CSV}" )
    EXTRA_DOCKER_ARGS+=( -e "AITER_LOG_TUNED_CONFIG=${AITER_LOG_TUNED_CONFIG:-1}" )
fi
# RCCL all-gather tuning: only pass a var when the caller set it (avoid empty-string).
for _v in NCCL_MIN_P2P_NCHANNELS NCCL_MIN_CTAS NCCL_NCHANNELS_PER_NET_PEER NCCL_NVLS_ENABLE NCCL_DEBUG; do
    [[ -n "${!_v:-}" ]] && EXTRA_DOCKER_ARGS+=( -e "${_v}=${!_v}" )
done

mkdir -p "${HOST_RESULTS}"

# Same fetch as the Megatron cases: examples/qwen3/tokenizer, not the model dir.
source "${SCRIPT_DIR}/../scripts/ensure_tokenizer.sh"
example_tokenizer qwen3 "${SCRIPT_DIR}/tokenizer" || exit 1

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run --rm --init \
    ${EXTRA_DOCKER_ARGS[@]+"${EXTRA_DOCKER_ARGS[@]}"} \
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
    -v "${HOST_MODEL}:/model-qwen3:ro" \
    -v "${HOST_DATA}:/data:ro" \
    -v "${HOST_RESULTS}:/results" \
    -e TOKENIZER_PATH=/workspace/Lumen/examples/qwen3/tokenizer \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e PYTORCH_TUNABLEOP_ENABLED="${PYTORCH_TUNABLEOP_ENABLED:-0}" \
    -e TRAIN_FILE="${TRAIN_FILE}" \
    -e VAL_FILE="${VAL_FILE}" \
    -e MODE="${MODE}" \
    -e LUMEN_PROF_START="${LUMEN_PROF_START:-}" \
    -e LUMEN_PROF_END="${LUMEN_PROF_END:-}" \
    -e LUMEN_PROF_OUTPUT="${LUMEN_PROF_OUTPUT:-}" \
    -e LUMEN_PROF_TRACE="${LUMEN_PROF_TRACE:-}" \
    -e LUMEN_COPY_TRACE="${LUMEN_COPY_TRACE:-}" \
    -e LUMEN_PROF_SHAPES="${LUMEN_PROF_SHAPES:-}" \
    -e SEQ_LENGTH="${SEQ_LENGTH}" \
    -e MBS="${MBS:-1}" \
    -e GRAD_ACCUM="${GRAD_ACCUM:-1}" \
    -e MAX_STEPS="${MAX_STEPS}" \
    -e EVAL_INTERVAL="${EVAL_INTERVAL}" \
    -e CACHE_FROZEN_WEIGHT="${CACHE_FROZEN_WEIGHT:-}" \
    -e BPRESHUFFLE="${BPRESHUFFLE:-}" \
    -e SHARDING="${SHARDING:-full_shard}" \
    -e FP8_SCALING="${FP8_SCALING:-blockwise2d}" \
    -e GRAD_CKPT="${GRAD_CKPT:-1}" \
    -e LIMIT_ALL_GATHERS="${LIMIT_ALL_GATHERS:-1}" \
    -e FORWARD_PREFETCH="${FORWARD_PREFETCH:-}" \
    -e FSDP_VERSION="${FSDP_VERSION:-}" \
    -e FSDP_FP8_PARAM_STORAGE="${FSDP_FP8_PARAM_STORAGE:-}" \
    -e LUMEN_BPRESHUFFLE_ONFLY="${LUMEN_BPRESHUFFLE_ONFLY:-}" \
    -e LUMEN_SKIP_FROZEN_WGRAD="${LUMEN_SKIP_FROZEN_WGRAD:-}" \
    -e AITER_ATTN="${AITER_ATTN:-}" \
    -e LUMEN_NORM="${LUMEN_NORM:-}" \
    -e FUSE_ROPE="${FUSE_ROPE:-}" \
    "${IMAGE}" \
    bash -c '
set -euo pipefail
# Dockerfile.lumen already installs AtomRL editable at this bind-mounted path.
# Reinstalling the root ATOM project misplaces the native Lumen extension.
cd /workspace/ATOM/atomrl
cd /workspace/Lumen/examples/qwen3
EXTRA=""
[[ -n "${CACHE_FROZEN_WEIGHT}" ]] && EXTRA="${EXTRA} --cache-frozen-weight"
[[ -n "${BPRESHUFFLE}" ]] && EXTRA="${EXTRA} --bpreshuffle"
[[ "${GRAD_CKPT}" == "0" ]] && EXTRA="${EXTRA} --no-grad-checkpointing"
[[ "${LIMIT_ALL_GATHERS}" == "0" ]] && EXTRA="${EXTRA} --no-limit-all-gathers"
[[ -n "${FORWARD_PREFETCH}" ]] && EXTRA="${EXTRA} --forward-prefetch"
[[ -n "${FSDP_VERSION}" ]] && EXTRA="${EXTRA} --fsdp-version ${FSDP_VERSION}"
[[ -n "${FSDP_FP8_PARAM_STORAGE}" ]] && EXTRA="${EXTRA} --fsdp-fp8-param-storage"
[[ -n "${AITER_ATTN}" ]] && EXTRA="${EXTRA} --aiter-attn"
[[ -n "${LUMEN_NORM}" ]] && EXTRA="${EXTRA} --lumen-norm"
[[ -n "${FUSE_ROPE}" ]] && EXTRA="${EXTRA} --fuse-rope"
torchrun --nproc_per_node=8 train_qwen3_fsdp.py \
    --model-name-or-path /model-qwen3 \
    --tokenizer-name-or-path "${TOKENIZER_PATH}" \
    --train-data-path "/data/${TRAIN_FILE}" \
    --val-data-path "/data/${VAL_FILE}" \
    --mode "${MODE}" ${EXTRA} --sharding "${SHARDING}" --fp8-scaling "${FP8_SCALING}" \
    --seq-length "${SEQ_LENGTH}" --micro-batch-size "${MBS}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" --max-steps "${MAX_STEPS}" \
    --eval-interval "${EVAL_INTERVAL}" --seed 1234 2>&1 \
    | tee "/results/train_${MODE}.log"
'
