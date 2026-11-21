#!/usr/bin/env bash
# run_dsv4_4layer_profile.sh — capture torch.profiler on DSV4 4-layer Megatron pretrain
# and write operator breakdown xlsx under examples/dsv4/results/.
#
# Usage:
#   bash examples/dsv4/run_dsv4_4layer_profile.sh
#
# Overridable:
#   LUMEN_PROF_START=3 LUMEN_PROF_END=5   profile window (Megatron steps)
#   LUMEN_PROF_XLSX=...                   output xlsx path
#   LUMEN_PROF_TRACE=/path/trace.json     optional chrome trace
#   LUMEN_PROF_SHAPES=1                   record input shapes in xlsx
#
# Outputs (default):
#   examples/dsv4/results/dsv4_4layer_profile.txt
#   examples/dsv4/results/dsv4_4layer_profile.json
#   examples/dsv4/results/dsv4_4layer_operator_breakdown.xlsx

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
mkdir -p "${RESULTS_DIR}"

LOGFILE="${LOG_DIR}/lumen_dsv4_4layer_profile_$(date +%Y%m%d_%H%M%S).log"

IMAGE="${IMAGE:-lumen/dsv4-lumen:mi308x}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-FP8-4layer}"
DSV4_HC_MULT="${DSV4_HC_MULT:-4}"

# shellcheck source=examples/dsv4/dsv4_4layer_megatron_args.sh
source "${SCRIPT_DIR}/dsv4_4layer_megatron_args.sh"

LUMEN_RUNTIME_DIR="${LUMEN_RUNTIME_DIR:-${LOG_DIR}/lumen-dsv4-runtime}"
CONTAINER_TMPDIR="${CONTAINER_TMPDIR:-/dev/shm/lumen-dsv4-tmp}"
mkdir -p "${LUMEN_RUNTIME_DIR}"

V4_SPARSE_MLA_BACKEND="${V4_SPARSE_MLA_BACKEND:-triton}"
V4_INDEXER_IMPL="${V4_INDEXER_IMPL:-tilelang}"
V4_INDEXER_BLOCK_N="${V4_INDEXER_BLOCK_N:-64}"
V4_INDEXER_NUM_STAGES="${V4_INDEXER_NUM_STAGES:-1}"
SKIP_PREPARE="${SKIP_PREPARE:-1}"
LOAD_CKPT="${LOAD_CKPT:-1}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
LUMEN_PROF_START="${LUMEN_PROF_START:-3}"
LUMEN_PROF_END="${LUMEN_PROF_END:-5}"
LUMEN_PROF_STOP_AFTER="${LUMEN_PROF_STOP_AFTER:-${LUMEN_PROF_END}}"

USE_BOOTSTRAP=0
BOOTSTRAP_MOUNT="${BOOTSTRAP_DIR}"
if [[ "${IMAGE}" == "lumen/tests:latest" || "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
    USE_BOOTSTRAP=1
    if [[ "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
        BOOTSTRAP_MOUNT=""
    elif [[ ! -f "${BOOTSTRAP_DIR}/.ready" ]]; then
        echo "[prepare] bootstrap missing — running prepare_bootstrap.sh"
        bash "${SCRIPT_DIR}/prepare_bootstrap.sh"
    fi
fi

if ! docker image inspect "${IMAGE}" &>/dev/null; then
    if [[ "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
        bash "${SCRIPT_DIR}/build_dsv4_lumen_image.sh"
    else
        echo "[ERROR] Image not found: ${IMAGE}"
        exit 1
    fi
fi

if ! ls /dev/kfd &>/dev/null; then
    echo "[ERROR] /dev/kfd not found — ROCm device not accessible."
    exit 1
fi

mkdir -p "${MODEL_DIR}" "${LOG_DIR}" "${MODEL_DIR}/miopen-cache" "${TVM_CACHE_DIR}"

echo "════════════════════════════════════════════════"
echo "  DSV4 4-layer Megatron profiler"
echo "  Image     : ${IMAGE}"
echo "  Profile   : steps ${LUMEN_PROF_START}-${LUMEN_PROF_END} (stop after ${LUMEN_PROF_STOP_AFTER})"
echo "  Batch     : GBS=${GBS} MBS=${MBS} seq_len=${SEQ_LEN}"
echo "  Kernels   : MLA=${V4_SPARSE_MLA_BACKEND} MHC=aiter"
echo "  Results   : ${RESULTS_DIR}"
echo "  Log       : ${LOGFILE}"
echo "════════════════════════════════════════════════"

DOCKER_MOUNTS=(
    -v "${LUMEN_DIR}:/workspace/Lumen"
    -v "${AITER_DIR}:/workspace/aiter"
    -v "${MODEL_DIR}:/root/models"
    -v "${MODEL_DIR}/miopen-cache:/root/.config/miopen"
    -v "${TVM_CACHE_DIR}:/root/.cache/tvm-ffi"
    -v "${LUMEN_RUNTIME_DIR}:/opt/dsv4-runtime"
    -v "${RESULTS_DIR}:/workspace/Lumen/examples/dsv4/results"
)
if [[ -d "${NFS_ROOT}" ]]; then
    DOCKER_MOUNTS+=(-v "${NFS_ROOT}:${NFS_ROOT}")
fi
if [[ -d "${MILES_DIR}" ]]; then
    DOCKER_MOUNTS+=(-v "${MILES_DIR}:/workspace/miles")
fi
if [[ "${USE_BOOTSTRAP}" -eq 1 && -n "${BOOTSTRAP_MOUNT}" ]]; then
    DOCKER_MOUNTS+=(-v "${BOOTSTRAP_MOUNT}:/bootstrap:ro")
fi

DOCKER_ENV=(
    -e LUMEN_DIR=/workspace/Lumen
    -e AITER_DIR=/workspace/aiter
    -e LUMEN_DSV4_PRETRAIN=1
    -e MODEL_DIR=/root/models
    -e MODEL_NAME="${MODEL_NAME}"
    -e TRAIN_ITERS="${TRAIN_ITERS}"
    -e GBS="${GBS}"
    -e MBS="${MBS}"
    -e SEQ_LEN="${SEQ_LEN}"
    -e SKIP_PREPARE="${SKIP_PREPARE}"
    -e LOAD_CKPT="${LOAD_CKPT}"
    -e EVAL_ITERS=0
    -e DSV4_HC_MULT="${DSV4_HC_MULT}"
    -e V4_SPARSE_MLA_BACKEND="${V4_SPARSE_MLA_BACKEND}"
    -e V4_INDEXER_IMPL="${V4_INDEXER_IMPL}"
    -e V4_INDEXER_BLOCK_N="${V4_INDEXER_BLOCK_N}"
    -e V4_INDEXER_NUM_STAGES="${V4_INDEXER_NUM_STAGES}"
    -e DSV4_ENABLE_RECOMPUTE="${DSV4_ENABLE_RECOMPUTE:-1}"
    -e LUMEN_PROF_START="${LUMEN_PROF_START}"
    -e LUMEN_PROF_END="${LUMEN_PROF_END}"
    -e LUMEN_PROF_STOP_AFTER="${LUMEN_PROF_STOP_AFTER}"
    -e LUMEN_PROF_OUTPUT="/workspace/Lumen/examples/dsv4/results/dsv4_4layer_profile.txt"
    -e LUMEN_PROF_XLSX="/workspace/Lumen/examples/dsv4/results/dsv4_4layer_operator_breakdown.xlsx"
    -e LUMEN_PROF_SHAPES="${LUMEN_PROF_SHAPES:-0}"
    -e LUMEN_PROF_TRACE="${LUMEN_PROF_TRACE:-}"
    -e RESULTS_DIR=/workspace/Lumen/examples/dsv4/results
)
if [[ -d "${MILES_DIR}" ]]; then
    DOCKER_ENV+=(-e MILES_DIR=/workspace/miles)
fi
DOCKER_ENV+=(
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    -e CUDA_DEVICE_MAX_CONNECTIONS=1
    -e NCCL_NVLS_ENABLE=0
    -e RCCL_MSCCL_ENABLE=0
    -e HSA_FORCE_FINE_GRAIN_PCIE=1
    -e HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
    -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens14np0}"
    -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-ens14np0}}"
    -e MEGATRON_PATH="${MEGATRON_PATH}"
    -e TORCHDYNAMO_DISABLE=1
    -e TL_DISABLE_OUT_OF_BOUND_WARNING="${TL_DISABLE_OUT_OF_BOUND_WARNING:-1}"
    -e TMPDIR="${CONTAINER_TMPDIR}"
)
if [[ "${USE_BOOTSTRAP}" -eq 1 && -n "${BOOTSTRAP_MOUNT}" ]]; then
    DOCKER_ENV+=(-e BOOTSTRAP_DIR=/bootstrap)
elif [[ "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
    DOCKER_ENV+=(-e BOOTSTRAP_DIR=/opt/dsv4-bootstrap -e WRITABLE_ROOT=/opt/dsv4-runtime)
fi

docker rm -f lumen-dsv4-profile 2>/dev/null || true

docker run --rm \
    --name lumen-dsv4-profile \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    --ipc=host \
    --network=host \
    --shm-size=128g \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1 \
    "${DOCKER_MOUNTS[@]}" \
    "${DOCKER_ENV[@]}" \
    "${IMAGE}" \
    bash /workspace/Lumen/examples/dsv4/run_dsv4_4layer_profile_inner.sh \
    2>&1 | tee "${LOGFILE}"

echo ""
echo "Profile log : ${LOGFILE}"
echo "Profile txt : ${RESULTS_DIR}/dsv4_4layer_profile.txt"
echo "Operator xlsx: ${RESULTS_DIR}/dsv4_4layer_operator_breakdown.xlsx"
