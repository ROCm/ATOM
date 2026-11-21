#!/usr/bin/env bash
# Shared Docker/bootstrap helpers for run_dsv4.sh (source, do not execute).

dsv4_docker_bootstrap_setup() {
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
}

dsv4_docker_ensure_image() {
    if docker image inspect "${IMAGE}" &>/dev/null; then
        return 0
    fi
    if [[ "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
        bash "${SCRIPT_DIR}/build_dsv4_lumen_image.sh"
    else
        echo "[ERROR] Image not found: ${IMAGE}"
        exit 1
    fi
}

dsv4_docker_check_rocm() {
    if ! ls /dev/kfd &>/dev/null; then
        echo "[ERROR] /dev/kfd not found — ROCm device not accessible."
        exit 1
    fi
}

dsv4_docker_mkdirs() {
    mkdir -p "${MODEL_DIR}" "${LOG_DIR}" "${MODEL_DIR}/miopen-cache" "${TVM_CACHE_DIR}" \
        "${DATA_DIR}" "${PIP_CACHE_DIR}"
}

# RDMA/IB character devices for RCCL NET/IB (bind-mounting /dev/infiniband alone is insufficient).
dsv4_docker_append_infiniband_devices() {
    DOCKER_DEVICES=("${DOCKER_DEVICES[@]}")
    if [[ ! -d /dev/infiniband ]]; then
        return 0
    fi
    if [[ -e /dev/infiniband/rdma_cm ]]; then
        DOCKER_DEVICES+=(--device "/dev/infiniband/rdma_cm:/dev/infiniband/rdma_cm")
    fi
    local uverb=""
    for uverb in /dev/infiniband/uverbs*; do
        [[ -e "${uverb}" ]] || continue
        DOCKER_DEVICES+=(--device "${uverb}:${uverb}")
    done
}

dsv4_docker_build_mounts() {
    if [[ ! -d "${AITER_DIR}/aiter" ]]; then
        echo "[ERROR] AITER_DIR does not contain the aiter package: ${AITER_DIR}"
        echo "        Set AITER_DIR to the actively developed AIter checkout."
        exit 1
    fi
    DOCKER_DEVICES=()
    DOCKER_MOUNTS=(
        -v "${LUMEN_DIR}:/workspace/Lumen"
        -v "${AITER_DIR}:/workspace/aiter"
        -v "${MODEL_DIR}:/root/models"
        -v "${DATA_DIR}:/root/datasets"
        -v "${MODEL_DIR}/miopen-cache:/root/.config/miopen"
        -v "${TVM_CACHE_DIR}:/root/.cache/tvm-ffi"
        -v "${PIP_CACHE_DIR}:/root/.cache/pip"
    )
    if [[ -d "${DATA_ROOT}" ]]; then
        DOCKER_MOUNTS+=(-v "${DATA_ROOT}:${DATA_ROOT}")
    elif [[ "${DSV4_PROFILE}" == "flash" && -d "${NFS_ROOT}" ]]; then
        DOCKER_MOUNTS+=(-v "${NFS_ROOT}:${NFS_ROOT}")
    fi
    if [[ -d "${MILES_DIR:-}" ]]; then
        DOCKER_MOUNTS+=(-v "${MILES_DIR}:/workspace/miles")
    fi
    if [[ "${USE_BOOTSTRAP}" -eq 1 && -n "${BOOTSTRAP_MOUNT}" ]]; then
        DOCKER_MOUNTS+=(-v "${BOOTSTRAP_MOUNT}:/bootstrap:ro")
    fi
    if [[ "${DSV4_PROFILE}" == "flash" && "${NCCL_IB_DISABLE:-0}" != "1" && -d /dev/infiniband ]]; then
        DOCKER_MOUNTS+=(-v /dev/infiniband:/dev/infiniband)
        dsv4_docker_append_infiniband_devices
    fi
}

dsv4_docker_append_bootstrap_env() {
    if [[ "${USE_BOOTSTRAP}" -eq 1 && -n "${BOOTSTRAP_MOUNT}" ]]; then
        DOCKER_ENV+=(-e BOOTSTRAP_DIR=/bootstrap)
    elif [[ "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
        DOCKER_ENV+=(
            -e BOOTSTRAP_DIR=/opt/dsv4-bootstrap
            -e WRITABLE_ROOT=/opt/dsv4-runtime
            -e TILELANG_DIR=/opt/dsv4-runtime/tilelang
        )
    fi
}

dsv4_docker_append_kernel_env() {
    DOCKER_ENV+=(
        -e AITER_DIR=/workspace/aiter
        -e V4_SPARSE_MLA_BACKEND="${V4_SPARSE_MLA_BACKEND}"
        -e V4_INDEXER_IMPL="${V4_INDEXER_IMPL}"
        -e V4_INDEXER_BLOCK_N="${V4_INDEXER_BLOCK_N}"
        -e V4_INDEXER_NUM_STAGES="${V4_INDEXER_NUM_STAGES}"
        -e DSV4_ENABLE_RECOMPUTE="${DSV4_ENABLE_RECOMPUTE:-1}"
    )
    if [[ -d "${TILELANG_DIR}" ]]; then
        DOCKER_ENV+=(-e TILELANG_DIR="${TILELANG_DIR}")
    fi
    if [[ -d "${PYTHON_EXTRAS_DIR}" ]]; then
        DOCKER_ENV+=(-e PYTHON_EXTRAS_DIR="${PYTHON_EXTRAS_DIR}")
    fi
}

dsv4_docker_append_rocm_env() {
    DOCKER_ENV+=(
        -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        -e CUDA_DEVICE_MAX_CONNECTIONS=1
        -e NCCL_NVLS_ENABLE=0
        -e RCCL_MSCCL_ENABLE=0
        -e HSA_FORCE_FINE_GRAIN_PCIE=1
        -e TORCHDYNAMO_DISABLE=1
    )
    if [[ -n "${NCCL_IB_GDR_LEVEL:-}" ]]; then
        DOCKER_ENV+=(-e NCCL_IB_GDR_LEVEL="${NCCL_IB_GDR_LEVEL}")
    fi
    if [[ -n "${NCCL_NET_GDR_LEVEL:-}" ]]; then
        DOCKER_ENV+=(-e NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL}")
    fi
}

# Pass GEMM routing flags; remap host LUMEN_DIR CSV paths to container /workspace/Lumen.
dsv4_docker_append_gemm_env() {
    DOCKER_ENV+=(
        -e "LUMEN_DSV4_GEMM_BF16=${LUMEN_DSV4_GEMM_BF16:-1}"
        -e "AITER_LOG_TUNED_CONFIG=${AITER_LOG_TUNED_CONFIG:-0}"
        -e "AITER_LOG_MORE=${AITER_LOG_MORE:-0}"
    )
    if [[ -z "${AITER_CONFIG_GEMM_BF16:-}" ]]; then
        return 0
    fi
    local _host_csv="${AITER_CONFIG_GEMM_BF16}"
    local _container_csv="${_host_csv}"
    if [[ "${_host_csv}" == "${LUMEN_DIR}/"* ]]; then
        _container_csv="/workspace/Lumen/${_host_csv#"${LUMEN_DIR}/"}"
    fi
    DOCKER_ENV+=(-e "AITER_CONFIG_GEMM_BF16=${_container_csv}")
}

dsv4_docker_append_multinode_env() {
    DOCKER_ENV+=(
        -e NNODES="${NNODES}"
        -e NPROC_PER_NODE="${NPROC_PER_NODE}"
        -e NODE_RANK="${NODE_RANK}"
        -e MASTER_ADDR="${MASTER_ADDR}"
        -e MASTER_PORT="${MASTER_PORT}"
        -e OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION}"
        -e DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES}"
        -e HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
        -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens14np0}"
        -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens14np0}"
        -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
        -e NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
        -e NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
        -e NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
        -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
        -e MEGATRON_NO_BATCH_P2P_COMM="${MEGATRON_NO_BATCH_P2P_COMM:-1}"
        -e MEGATRON_PATH="${MEGATRON_PATH}"
        -e TP="${TP:-4}"
        -e PP="${PP:-4}"
        -e EP="${EP:-4}"
        -e ETP="${ETP:-1}"
        -e DECODER_FIRST_PP_LAYERS="${DECODER_FIRST_PP_LAYERS:-11}"
        -e DECODER_LAST_PP_LAYERS="${DECODER_LAST_PP_LAYERS:-10}"
    )
}

dsv4_docker_run_opts_single() {
    DOCKER_RUN=(
        docker run --rm
        --name "${CONTAINER_NAME}"
        --device /dev/kfd
        --device /dev/dri
        --group-add video
        --group-add render
        --ipc=host
        --network=host
        --shm-size=128g
        --cap-add=SYS_PTRACE
        --cap-add=IPC_LOCK
        --security-opt seccomp=unconfined
        --ulimit memlock=-1
    )
}

dsv4_docker_run_opts_multinode() {
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
        --cap-add=IPC_LOCK
        --security-opt seccomp=unconfined
        --ulimit memlock=-1
        --ulimit nofile=65536:524288
        --group-add render
    )
}
