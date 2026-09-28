#!/usr/bin/env bash
set -euo pipefail

# Start the atomesh Docker container with RDMA NIC auto-detection.
#
# Optional env (with defaults):
#   CONTAINER=atom_mesh
#   DOCKER_IMAGE=rocm/atom-dev:latest
#   MORI_NIC_TYPE=<auto>           # override NIC detection (bnxt|ionic|mlx5)
#
# Available images:
#   rocm/atom-dev:latest          # ATOM native backend (default)
#   rocm/atom-dev:vllm-latest     # vLLM backend
#   rocm/atom-dev:sglang-latest   # SGLang backend

CONTAINER="${CONTAINER:-atom_mesh}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/atom-dev:latest}"

# ======================== RDMA NIC helpers ========================

detect_nic_type() {
    if [[ -n "${MORI_NIC_TYPE:-}" ]]; then
        echo "$MORI_NIC_TYPE"
        return
    fi
    local bnxt=0 mlx5=0 ionic=0
    if [[ -d /sys/class/infiniband ]]; then
        for dev in /sys/class/infiniband/*; do
            local name
            name=$(basename "$dev")
            case "$name" in
                bnxt_re*) ((bnxt++)) ;;
                mlx5*)    ((mlx5++)) ;;
                ionic*)   ((ionic++)) ;;
                *)
                    local drv
                    drv=$(readlink -f "$dev/device/driver" 2>/dev/null || true)
                    drv=$(basename "$drv" 2>/dev/null || true)
                    case "$drv" in
                        bnxt*)  ((bnxt++)) ;;
                        mlx5*)  ((mlx5++)) ;;
                        ionic*) ((ionic++)) ;;
                    esac
                    ;;
            esac
        done
    fi
    if (( bnxt >= mlx5 && bnxt >= ionic && bnxt > 0 )); then
        echo "bnxt"
    elif (( ionic >= mlx5 && ionic > 0 )); then
        echo "ionic"
    else
        echo "mlx5"
    fi
}

find_host_ibverbs() {
    local candidates=(
        /usr/lib64/libibverbs.so.1
        /lib/x86_64-linux-gnu/libibverbs.so.1
        /usr/lib/x86_64-linux-gnu/libibverbs.so.1
    )
    for c in "${candidates[@]}"; do
        local resolved
        resolved=$(readlink -f "$c" 2>/dev/null || true)
        if [[ -f "$resolved" ]]; then
            echo "$resolved"
            return
        fi
    done
}

nic_mount_flags() {
    local nic_type="$1"
    local flags=()
    case "$nic_type" in
        bnxt)
            local host_ibverbs
            host_ibverbs=$(find_host_ibverbs)
            if [[ -n "$host_ibverbs" ]]; then
                flags+=(-v "$host_ibverbs:/lib/x86_64-linux-gnu/libibverbs.so.1")
            fi
            for lib in /usr/local/lib/libbnxt_re-rdmav*.so; do
                [[ -f "$lib" ]] && flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/libibverbs/$(basename "$lib")")
            done
            for lib in /usr/local/lib/libbnxt_re.so; do
                [[ -f "$lib" ]] && flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/$(basename "$lib")")
            done
            [[ -d /etc/libibverbs.d ]] && flags+=(-v /etc/libibverbs.d:/etc/libibverbs.d:ro)
            ;;
        ionic)
            local host_ibverbs
            host_ibverbs=$(find_host_ibverbs)
            if [[ -n "$host_ibverbs" ]]; then
                flags+=(-v "$host_ibverbs:/lib/x86_64-linux-gnu/libibverbs.so.1")
            fi
            local ionic_dirs=(/usr/local/lib /usr/lib/x86_64-linux-gnu)
            for dir in "${ionic_dirs[@]}"; do
                for lib in "$dir"/libionic*.so; do
                    if [[ -f "$lib" ]]; then
                        local real; real=$(readlink -f "$lib")
                        [[ -f "$real" ]] && flags+=(-v "$real:$real")
                        flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/$(basename "$lib")")
                    fi
                done
            done
            local provider_dir=/usr/lib/x86_64-linux-gnu/libibverbs
            if [[ -d "$provider_dir" ]]; then
                for lib in "$provider_dir"/libionic-rdmav*.so; do
                    [[ -f "$lib" ]] && flags+=(-v "$lib:$lib")
                done
            fi
            [[ -d /etc/libibverbs.d ]] && flags+=(-v /etc/libibverbs.d:/etc/libibverbs.d:ro)
            ;;
        mlx5) ;;
    esac
    echo "${flags[@]}"
}

# ======================== detect NIC & build mounts ========================

NIC_TYPE=$(detect_nic_type)
echo "[docker] NIC type detected: ${NIC_TYPE}"
read -ra NIC_MOUNTS <<< "$(nic_mount_flags "${NIC_TYPE}")"
if [[ ${#NIC_MOUNTS[@]} -gt 0 ]]; then
    echo "[docker] RDMA mounts: ${NIC_MOUNTS[*]}"
else
    echo "[docker] no out-of-tree RDMA mounts needed"
fi

# ======================== start container ========================

echo "[docker] starting container=${CONTAINER} image=${DOCKER_IMAGE}"

docker rm -f "${CONTAINER}" 2>/dev/null || true

docker run -d --name "${CONTAINER}" \
    --network host --ipc host --privileged \
    --device /dev/kfd --device /dev/dri \
    --device /dev/infiniband \
    --group-add video \
    --cap-add IPC_LOCK --cap-add NET_ADMIN \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:524288 \
    --shm-size 128G \
    -v /mnt:/mnt \
    -v /data:/data \
    -v /it-share:/it-share \
    "${NIC_MOUNTS[@]}" \
    "${DOCKER_IMAGE}" sleep infinity

docker exec "${CONTAINER}" bash -c '
    sysctl -w net.core.somaxconn=4096 2>/dev/null || true
    sysctl -w net.ipv4.tcp_max_syn_backlog=4096 2>/dev/null || true
'

echo "[docker] container ${CONTAINER} started (TCP backlog tuned)"
