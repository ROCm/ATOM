#!/bin/bash
set -euo pipefail

# ci_run.sh — Launch a container with automatic NIC detection and
# bind-mount of out-of-tree RDMA userspace libraries.
#
# Vendored from ROCm/mori (docker/ci_run.sh) so ATOM's PD-disagg CI doesn't
# silently fall back to TCP on hosts whose NICs aren't named bnxt_re*/mlx5*/
# ionic* (e.g. udev-renamed to rdma0..rdmaN or rocep*s0). The inline detection
# that used to live in atom_pd_test.sh and atom-benchmark.yaml only matched by
# device name, missed the driver-symlink fallback, and as a result NIC_TYPE
# silently stayed at the default "mlx5" → no NIC_MOUNTS → libibverbs in the
# container had no userspace provider → `ibv_devices` empty → mooncake fell
# back to TCP → 170+ "TcpTransport ... connect: Connection timed out" per
# nightly and PD throughput dropped to single-digit %.
#
# Keep this in sync with https://github.com/ROCm/mori/blob/main/docker/ci_run.sh
#
# Usage: ci_run.sh [docker-run-args...] IMAGE [cmd...]
#   e.g.: ./ci_run.sh --name atom_pd_prefill rocm/atom-dev:latest
#         ./ci_run.sh --name atom_pd_prefill -v /models:/models IMAGE bash
#
# Environment:
#   MORI_NIC_TYPE        — Override auto-detection (mlx5 | bnxt | ionic)
#   CONTAINER_RUNTIME    — Override runtime (docker | podman); auto-detected

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── NIC detection ────────────────────────────────────────────────────────────

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
                        bnxt*) ((bnxt++)) ;;
                        mlx5*) ((mlx5++)) ;;
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

# ── Build bind-mount flags for OOT RDMA libs ────────────────────────────────

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
                if [[ -f "$lib" ]]; then
                    flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/libibverbs/$(basename "$lib")")
                fi
            done
            for lib in /usr/local/lib/libbnxt_re.so; do
                if [[ -f "$lib" ]]; then
                    flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/$(basename "$lib")")
                fi
            done
            if [[ -d /etc/libibverbs.d ]]; then
                flags+=(-v /etc/libibverbs.d:/etc/libibverbs.d:ro)
            fi
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
                        local real
                        real=$(readlink -f "$lib")
                        if [[ -f "$real" ]]; then
                            flags+=(-v "$real:$real")
                        fi
                        flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/$(basename "$lib")")
                    fi
                done
            done
            local provider_dir=/usr/lib/x86_64-linux-gnu/libibverbs
            if [[ -d "$provider_dir" ]]; then
                for lib in "$provider_dir"/libionic-rdmav*.so; do
                    if [[ -f "$lib" ]]; then
                        flags+=(-v "$lib:$lib")
                    fi
                done
            fi
            if [[ -d /etc/libibverbs.d ]]; then
                flags+=(-v /etc/libibverbs.d:/etc/libibverbs.d:ro)
            fi
            ;;
        mlx5)
            ;;
    esac

    echo "${flags[@]}"
}

# ── Container runtime detection ───────────────────────────────────────────────

detect_runtime() {
    if [[ -n "${CONTAINER_RUNTIME:-}" ]]; then
        echo "$CONTAINER_RUNTIME"
        return
    fi
    if docker info &>/dev/null; then
        echo "docker"
    elif command -v podman &>/dev/null; then
        echo "podman"
    elif command -v docker &>/dev/null; then
        echo "docker"
    else
        echo "docker"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────

RUNTIME=$(detect_runtime)
NIC_TYPE=$(detect_nic_type)
echo "[ci_run] Runtime: $RUNTIME | NIC type: $NIC_TYPE"

read -ra NIC_MOUNTS <<< "$(nic_mount_flags "$NIC_TYPE")"

EXTRA_ARGS=()
if [[ "$RUNTIME" == "podman" ]]; then
    EXTRA_ARGS+=(--security-opt label=disable)
else
    EXTRA_ARGS+=(--ulimit nproc=100000:100000 --pids-limit=-1)
fi

exec "$RUNTIME" run \
    --group-add video \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    -d --ipc=host --privileged \
    "${EXTRA_ARGS[@]}" \
    "${NIC_MOUNTS[@]}" \
    "$@"
