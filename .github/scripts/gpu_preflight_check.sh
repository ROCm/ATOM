#!/usr/bin/env bash
# Preflight GPU state before ATOM starts touching distributed/RCCL paths.
#
# This intentionally avoids importing ATOM or aiter. It records the visible GPU
# process/memory state, then performs a minimal torch allocation on every visible
# HIP device. Leftover docker cleanup is opt-in via GPU_PREFLIGHT_KILL_DOCKER=1
# (vLLM/SGLang plugin CI). After that, GPU_PREFLIGHT_KILL_KFD=1 (defaults to the
# same as KILL_DOCKER) SIGKILLs leftover host/KFD PIDs that still hold VRAM.
# ATOM native CI leaves both off.

set -euo pipefail

CONTAINER="${1:-}"
ENGINE="${2:-docker}"
GPU_PREFLIGHT_ALLOCATION_MB="${GPU_PREFLIGHT_ALLOCATION_MB:-8}"
GPU_PREFLIGHT_KILL_DOCKER="${GPU_PREFLIGHT_KILL_DOCKER:-0}"
GPU_PREFLIGHT_KILL_KFD="${GPU_PREFLIGHT_KILL_KFD:-${GPU_PREFLIGHT_KILL_DOCKER}}"
GPU_PREFLIGHT_KILL_WAIT_SECONDS="${GPU_PREFLIGHT_KILL_WAIT_SECONDS:-30}"

case "$GPU_PREFLIGHT_ALLOCATION_MB" in
    ''|*[!0-9]*)
        echo "ERROR: GPU_PREFLIGHT_ALLOCATION_MB must be a positive integer, got '${GPU_PREFLIGHT_ALLOCATION_MB}'"
        exit 2
        ;;
esac

if [ "$GPU_PREFLIGHT_ALLOCATION_MB" -le 0 ]; then
    echo "ERROR: GPU_PREFLIGHT_ALLOCATION_MB must be greater than zero"
    exit 2
fi

if [ -n "$CONTAINER" ]; then
    exec_in() { "$ENGINE" exec "$CONTAINER" bash -lc "$1"; }
else
    exec_in() { bash -lc "$1"; }
fi

print_probe() {
    local title="$1"
    local command="$2"

    echo ""
    echo "========== ${title} =========="
    if ! exec_in "$command"; then
        echo "WARNING: ${title} failed"
    fi
}

gpu_vram_in_use_count() {
    local count
    count=$(rocm-smi --showmemuse 2>/dev/null | awk '/VRAM%/ { if ($NF+0 > 0) n++ } END { print n+0 }' || true)
    if [ -z "${count}" ]; then
        echo 0
    else
        echo "${count}"
    fi
}

wait_for_gpu_memory_release() {
    local i
    local used=0
    echo "Waiting for GPU memory to release after leftover cleanup..."
    for i in $(seq 1 "${GPU_PREFLIGHT_KILL_WAIT_SECONDS}"); do
        used=$(gpu_vram_in_use_count)
        if [ "${used}" -eq 0 ]; then
            echo "GPU memory released after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "WARNING: GPU memory still in use after ${GPU_PREFLIGHT_KILL_WAIT_SECONDS}s (used GPUs=${used})"
    return 1
}

# Host PIDs that belong to this preflight / the job's own container. Do not SIGKILL them.
collect_protected_pids() {
    local pid
    echo "$$"
    echo "${PPID}"
    echo "1"
    if [ -z "${CONTAINER}" ] || ! command -v "${ENGINE}" >/dev/null 2>&1; then
        return 0
    fi
    pid=$("${ENGINE}" inspect -f '{{.State.Pid}}' "${CONTAINER}" 2>/dev/null || true)
    if [ -n "${pid}" ] && [ "${pid}" != "0" ]; then
        echo "${pid}"
        # shellcheck disable=SC2009
        ps -eo pid=,ppid= 2>/dev/null | awk -v root="${pid}" '
            { p[$1]=$2 }
            END {
                for (pid in p) {
                    cur=pid
                    while (cur != "" && cur != 0 && cur != 1) {
                        if (cur == root) { print pid; break }
                        cur=p[cur]
                    }
                }
            }
        '
    fi
    "${ENGINE}" top "${CONTAINER}" -eo pid= 2>/dev/null | awk '{ if ($1 ~ /^[0-9]+$/) print $1 }' || true
}

pid_is_protected() {
    local candidate="$1"
    local protected="$2"
    case " ${protected} " in
        *" ${candidate} "*) return 0 ;;
    esac
    return 1
}

# Host-side SIGKILL of processes still holding VRAM after leftover containers
# are gone. rocm-smi --showpids is consulted on the host; docker-exec PIDs from
# the probe are host PIDs and must be killed on the host, not in the container.
list_vram_holder_pids() {
    if ! command -v rocm-smi >/dev/null 2>&1; then
        return 0
    fi
    rocm-smi --showpids 2>/dev/null | awk '
        /^PID:[[:space:]]*[0-9]+/ { print $2; next }
        /^[0-9]+/ {
            vram=$(NF-2)
            if (vram+0 > 0) print $1
        }
    '
}

kill_pid_best_effort() {
    local pid="$1"
    if kill -9 "${pid}" 2>/dev/null; then
        return 0
    fi
    if command -v sudo >/dev/null 2>&1 && sudo -n kill -9 "${pid}" 2>/dev/null; then
        return 0
    fi
    return 1
}

kill_leftover_kfd_pids() {
    local pid
    local killed=0
    local protected
    local holders

    if [ "${GPU_PREFLIGHT_KILL_KFD}" != "1" ]; then
        return 0
    fi

    echo ""
    echo "========== GPU preflight: leftover host/KFD VRAM holders =========="
    if ! command -v rocm-smi >/dev/null 2>&1; then
        echo "rocm-smi not found on host; skip leftover KFD PID cleanup"
        return 0
    fi

    protected=$(collect_protected_pids | awk '{ if ($1 ~ /^[0-9]+$/) print $1 }' | sort -u | tr '\n' ' ')
    echo "Protected PIDs: ${protected}"

    echo "rocm-smi --showpids:"
    rocm-smi --showpids || true
    echo "rocm-smi --showpidgpus:"
    rocm-smi --showpidgpus || true

    if command -v fuser >/dev/null 2>&1; then
        echo "host fuser /dev/kfd /dev/dri/renderD*:"
        fuser -v /dev/kfd /dev/dri/renderD* 2>/dev/null || true
    else
        echo "fuser not found on host; using rocm-smi --showpids only"
    fi

    holders=$(list_vram_holder_pids | sort -u || true)
    if [ -z "${holders}" ]; then
        echo "No host KFD PIDs reporting VRAM use"
        return 0
    fi

    for pid in ${holders}; do
        if pid_is_protected "${pid}" "${protected}"; then
            echo "  skip protected PID ${pid}"
            continue
        fi
        if [ ! -d "/proc/${pid}" ]; then
            echo "  skip gone PID ${pid}"
            continue
        fi
        echo "  kill -9 ${pid} ($(tr '\0' ' ' < "/proc/${pid}/comm" 2>/dev/null || echo unknown))"
        if kill_pid_best_effort "${pid}"; then
            killed=1
        else
            echo "  failed to kill PID ${pid}"
        fi
    done

    if [ "${killed}" -eq 0 ]; then
        echo "No leftover host/KFD VRAM holders to kill"
        return 0
    fi

    wait_for_gpu_memory_release || true
}

kill_leftover_docker() {
    local id
    local name
    local killed=0
    local running

    if [ "${GPU_PREFLIGHT_KILL_DOCKER}" != "1" ]; then
        return 0
    fi

    echo ""
    echo "========== GPU preflight: leftover docker containers =========="
    if ! command -v "${ENGINE}" >/dev/null 2>&1; then
        echo "${ENGINE} not found; skip leftover container cleanup"
        return 0
    fi

    running=$("${ENGINE}" ps --format '{{.ID}} {{.Names}}' 2>/dev/null || true)
    if [ -z "${running}" ]; then
        echo "No running docker containers"
        return 0
    fi

    echo "Running containers:"
    "${ENGINE}" ps --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}' || true

    while read -r id name; do
        if [ -z "${id}" ]; then
            continue
        fi
        if [ -n "${CONTAINER}" ] && { [ "${name}" = "${CONTAINER}" ] || [ "${id}" = "${CONTAINER}" ]; }; then
            echo "  skip current container ${name} (${id})"
            continue
        fi
        echo "  ${ENGINE} kill ${name} (${id})"
        if "${ENGINE}" kill "${id}"; then
            killed=1
        else
            echo "  ${ENGINE} kill failed; trying rm -f ${name} (${id})"
            "${ENGINE}" rm -f "${id}" || true
            killed=1
        fi
    done <<< "${running}"

    if [ "${killed}" -eq 0 ]; then
        echo "No leftover docker containers to kill"
        return 0
    fi

    wait_for_gpu_memory_release || true
}

run_hip_smoke_test() {
    echo ""
    echo "========== GPU preflight: torch HIP allocation smoke test =========="
    exec_in "GPU_PREFLIGHT_ALLOCATION_MB='${GPU_PREFLIGHT_ALLOCATION_MB}' python3 - <<'PY'
import os
import sys
import traceback

keys = [
    'HIP_VISIBLE_DEVICES',
    'CUDA_VISIBLE_DEVICES',
    'ROCR_VISIBLE_DEVICES',
    'LOCAL_RANK',
    'RANK',
    'WORLD_SIZE',
]
for key in keys:
    print(f'{key}={os.environ.get(key)}')

try:
    import torch
except Exception:
    print('torch import failed:')
    traceback.print_exc()
    sys.exit(10)

print(f'torch.version.hip={getattr(torch.version, \"hip\", None)}')
print(f'torch.cuda.is_available={torch.cuda.is_available()}')

try:
    count = torch.cuda.device_count()
    print(f'torch.cuda.device_count={count}')
    if not torch.cuda.is_available() or count <= 0:
        print('ERROR: no available HIP devices for preflight allocation')
        sys.exit(11)

    alloc_mb = int(os.environ.get('GPU_PREFLIGHT_ALLOCATION_MB', '8'))
    alloc_bytes = alloc_mb * 1024 * 1024
    for index in range(count):
        torch.cuda.set_device(index)
        name = torch.cuda.get_device_name(index)
        print(f'device[{index}]={name}; allocating {alloc_mb} MiB')
        tensor = torch.empty(alloc_bytes, dtype=torch.uint8, device=f'cuda:{index}')
        torch.cuda.synchronize()
        print(
            f'device[{index}] allocation ok; '
            f'memory_allocated={torch.cuda.memory_allocated(index)}'
        )
        del tensor
        torch.cuda.empty_cache()

    print('GPU preflight HIP allocation passed on all visible devices')
except Exception:
    print('GPU preflight HIP allocation failed:')
    traceback.print_exc()
    sys.exit(12)
PY"
}

print_probe "GPU preflight: ROCm memory and processes before HIP smoke test" '
    set +e
    command -v rocm-smi >/dev/null 2>&1 || { echo "rocm-smi not found"; exit 127; }
    rocm-smi --showmemuse || true
    rocm-smi --showpids || true
    rocm-smi --showpidgpus || true
'

print_probe "GPU preflight: device file users before HIP smoke test" '
    set +e
    if command -v fuser >/dev/null 2>&1; then
        fuser -v /dev/kfd /dev/dri/renderD* 2>/dev/null || true
    else
        echo "fuser not found"
    fi
'

kill_leftover_docker
kill_leftover_kfd_pids

set +e
run_hip_smoke_test
hip_rc=$?
set -e

if [ "${hip_rc}" -ne 0 ] && { [ "${GPU_PREFLIGHT_KILL_DOCKER}" = "1" ] || [ "${GPU_PREFLIGHT_KILL_KFD}" = "1" ]; }; then
    echo "HIP smoke test failed (rc=${hip_rc}); retrying after leftover docker and KFD cleanup"
    kill_leftover_docker
    kill_leftover_kfd_pids
    set +e
    run_hip_smoke_test
    hip_rc=$?
    set -e
fi

exit "${hip_rc}"
