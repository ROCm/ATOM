#!/usr/bin/env bash
# run_dsv4_flash_pretrain.sh — DSV4 Flash full model (43L) 2×8 MI300X/MI308X pretrain smoke.
#
# Native Megatron + Lumen get_dsv4_spec (no Miles train.py / Ray).
# Run on EACH node with NODE_RANK=0 (head) and NODE_RANK=1 (worker).
#
# Example (head):
#   NODE_RANK=0 MASTER_ADDR=<head-ip> DATA_ROOT=/nfs/data/$USER \
#   SKIP_PREPARE=1 LOAD_CKPT=0 TRAIN_ITERS=10 \
#     bash examples/dsv4/run_dsv4_flash_pretrain.sh
#
# Example (worker):
#   NODE_RANK=1 MASTER_ADDR=<head-ip> DATA_ROOT=/nfs/data/$USER \
#   SKIP_PREPARE=1 LOAD_CKPT=0 TRAIN_ITERS=10 \
#     bash examples/dsv4/run_dsv4_flash_pretrain.sh
#
# For GRPO full finetune (recommended), use run_dsv4_flash_finetune.sh instead.
# For 4-layer single-node finetune, use run_dsv4_4layer_finetune.sh instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/dsv4/dsv4_paths.sh
source "${SCRIPT_DIR}/dsv4_paths.sh"

IMAGE="${IMAGE:-lumen/dsv4-lumen:mi308x}"

NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to head node IP}"
MASTER_PORT="${MASTER_PORT:-29500}"

MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-FP8}"
DSV4_HC_MULT="${DSV4_HC_MULT:-4}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
LOAD_CKPT="${LOAD_CKPT:-0}"
EVAL_ITERS="${EVAL_ITERS:-1}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0.75}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-180}"

# shellcheck source=examples/dsv4/dsv4_flash_megatron_args.sh
source "${SCRIPT_DIR}/dsv4_flash_megatron_args.sh"

V4_SPARSE_MLA_BACKEND="${V4_SPARSE_MLA_BACKEND:-triton}"
V4_INDEXER_IMPL="${V4_INDEXER_IMPL:-tilelang}"
V4_INDEXER_BLOCK_N="${V4_INDEXER_BLOCK_N:-64}"
V4_INDEXER_NUM_STAGES="${V4_INDEXER_NUM_STAGES:-1}"
LUMEN_DSV4_LINEAR_FP8="${LUMEN_DSV4_LINEAR_FP8:-0}"

TORCH_DIST="${MODEL_DIR}/${MODEL_NAME}_torch_dist"
LOGFILE="${LOG_DIR}/lumen_dsv4_flash_pretrain_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

USE_MILES_IMAGE=0
if [[ "${IMAGE}" == miles-dsv4-mi300x* || "${IMAGE}" == rlsys/miles* ]]; then
    USE_MILES_IMAGE=1
fi

USE_BOOTSTRAP=0
BOOTSTRAP_MOUNT="${BOOTSTRAP_DIR}"
if [[ "${USE_MILES_IMAGE}" -eq 0 && ( "${IMAGE}" == "lumen/tests:latest" || "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ) ]]; then
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

if [[ "${NNODES}" -gt 1 ]]; then
    # shellcheck source=examples/dsv4/preflight_dsv4_flash_multinode.sh
    source "${SCRIPT_DIR}/preflight_dsv4_flash_multinode.sh"
    preflight_dsv4_multinode
fi

echo "════════════════════════════════════════════════"
echo "  Lumen DSV4 Flash FULL pretrain smoke (16 GPU)"
echo "  Image     : ${IMAGE}"
echo "  Workspace : ${WORKSPACE_ROOT}  (data: ${DATA_ROOT})"
echo "  Nodes     : ${NNODES}×${NPROC_PER_NODE}  node_rank=${NODE_RANK}"
echo "  Master    : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Parallel  : TP4 PP4 EP4 (11+11+11+10)"
echo "  Steps     : ${TRAIN_ITERS}"
echo "  Batch     : GBS=${GBS} MBS=${MBS} seq_len=${SEQ_LEN}"
echo "  Ckpt load : LOAD_CKPT=${LOAD_CKPT}"
echo "  Preflight : PREFLIGHT_ID=${PREFLIGHT_ID:-n/a}"
echo "  Kernels   : MLA=${V4_SPARSE_MLA_BACKEND} MHC=aiter"
echo "  HC mult   : ${DSV4_HC_MULT}"
echo "  Optimizer : CPU offload fraction=${OPTIMIZER_OFFLOAD_FRACTION}"
echo "  Ckpt path : ${TORCH_DIST} (used only when LOAD_CKPT=1)"
echo "  Log       : ${LOGFILE}"
echo "════════════════════════════════════════════════"

DOCKER_MOUNTS=(
    -v "${LUMEN_DIR}:/workspace/Lumen"
    -v "${AITER_DIR}:/workspace/aiter"
    -v "${MODEL_DIR}:/root/models"
    -v "${MODEL_DIR}/miopen-cache:/root/.config/miopen"
    -v "${TVM_CACHE_DIR}:/root/.cache/tvm-ffi"
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
if [[ -d /dev/infiniband ]]; then
    DOCKER_MOUNTS+=(--volume /dev/infiniband:/dev/infiniband)
fi

DOCKER_ENV=(
    -e LUMEN_DIR=/workspace/Lumen
    -e AITER_DIR=/workspace/aiter
    -e LUMEN_DSV4_PRETRAIN=1
    -e MODEL_DIR=/root/models
    -e MODEL_NAME="${MODEL_NAME}"
    -e TRAIN_ITERS="${TRAIN_ITERS}"
    -e PRETRAIN_LR="${PRETRAIN_LR:-1e-6}"
    -e PRETRAIN_MIN_LR="${PRETRAIN_MIN_LR:-${PRETRAIN_LR:-1e-6}}"
    -e GBS="${GBS}"
    -e MBS="${MBS}"
    -e SEQ_LEN="${SEQ_LEN}"
    -e SKIP_PREPARE="${SKIP_PREPARE}"
    -e LOAD_CKPT="${LOAD_CKPT}"
    -e EVAL_ITERS="${EVAL_ITERS}"
    -e DSV4_HC_MULT="${DSV4_HC_MULT}"
    -e NNODES="${NNODES}"
    -e NPROC_PER_NODE="${NPROC_PER_NODE}"
    -e NODE_RANK="${NODE_RANK}"
    -e MASTER_ADDR="${MASTER_ADDR}"
    -e MASTER_PORT="${MASTER_PORT}"
    -e OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION}"
    -e EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-bf16}"
    -e EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-bf16}"
    -e DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES}"
    -e V4_SPARSE_MLA_BACKEND="${V4_SPARSE_MLA_BACKEND}"
    -e V4_INDEXER_IMPL="${V4_INDEXER_IMPL}"
    -e V4_INDEXER_BLOCK_N="${V4_INDEXER_BLOCK_N}"
    -e V4_INDEXER_NUM_STAGES="${V4_INDEXER_NUM_STAGES}"
    -e LUMEN_DSV4_LINEAR_FP8="${LUMEN_DSV4_LINEAR_FP8}"
    -e DSV4_ENABLE_RECOMPUTE="${DSV4_ENABLE_RECOMPUTE:-1}"
    -e HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    -e CUDA_DEVICE_MAX_CONNECTIONS=1
    -e NCCL_NVLS_ENABLE=0
    -e RCCL_MSCCL_ENABLE=0
    -e HSA_FORCE_FINE_GRAIN_PCIE=1
    -e TORCHDYNAMO_DISABLE=1
    -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens14np0}"
    -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-ens14np0}}"
    -e NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
    -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    -e NCCL_IB_GDR_LEVEL="${NCCL_IB_GDR_LEVEL:-0}"
    -e NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-LOC}"
    -e MEGATRON_NO_BATCH_P2P_COMM="${MEGATRON_NO_BATCH_P2P_COMM:-1}"
    -e MEGATRON_PATH="${MEGATRON_PATH}"
)
if [[ -d "${MILES_DIR}" ]]; then
    DOCKER_ENV+=(-e MILES_DIR=/workspace/miles)
fi
if [[ "${USE_MILES_IMAGE}" -eq 1 ]]; then
    DOCKER_ENV+=(
        -e RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
        -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
    )
fi
if [[ "${USE_BOOTSTRAP}" -eq 1 && -n "${BOOTSTRAP_MOUNT}" ]]; then
    DOCKER_ENV+=(-e BOOTSTRAP_DIR=/bootstrap)
elif [[ "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
    DOCKER_ENV+=(-e BOOTSTRAP_DIR=/opt/dsv4-bootstrap -e WRITABLE_ROOT=/opt/dsv4-runtime)
fi

CONTAINER_NAME="lumen-dsv4-full-node${NODE_RANK}"
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

DOCKER_RUN=(
    docker run --rm
    --name "${CONTAINER_NAME}"
    --device /dev/kfd
    --device /dev/dri
    --group-add video
    --ipc=host
    --network=host
    --shm-size=128g
    --cap-add=SYS_PTRACE
    --security-opt seccomp=unconfined
    --ulimit memlock=-1
    --ulimit nofile=65536:524288
)
if [[ "${USE_MILES_IMAGE}" -eq 1 ]]; then
    DOCKER_RUN+=(--privileged)
else
    DOCKER_RUN+=(--group-add render)
fi

"${DOCKER_RUN[@]}" \
    "${DOCKER_MOUNTS[@]}" \
    "${DOCKER_ENV[@]}" \
    "${IMAGE}" \
    bash /workspace/Lumen/examples/dsv4/run_dsv4_flash_pretrain_inner.sh \
    2>&1 | tee "${LOGFILE}"

echo ""
echo "Log saved to: ${LOGFILE}"
