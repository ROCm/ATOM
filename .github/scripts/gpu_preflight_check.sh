#!/usr/bin/env bash
# Preflight GPU state before ATOM starts touching distributed/RCCL paths.
#
# This intentionally avoids importing ATOM or aiter. It records the visible GPU
# process/memory state, then performs a minimal torch allocation on every visible
# HIP device. Occupant kill is opt-in via GPU_PREFLIGHT_KILL_OCCUPANTS=1 (used by
# vLLM/SGLang plugin CI). ATOM native CI leaves it off and only reports unhealthy
# GPU state.

set -euo pipefail

CONTAINER="${1:-}"
ENGINE="${2:-docker}"
GPU_PREFLIGHT_ALLOCATION_MB="${GPU_PREFLIGHT_ALLOCATION_MB:-8}"
GPU_PREFLIGHT_KILL_OCCUPANTS="${GPU_PREFLIGHT_KILL_OCCUPANTS:-0}"
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

protected_pid_list() {
    local p=$$
    while [ -n "${p}" ] && [ "${p}" != "0" ]; do
        echo "${p}"
        if [ "${p}" = "1" ]; then
            break
        fi
        p=$(ps -o ppid= -p "${p}" 2>/dev/null | tr -d ' ')
    done
}

is_protected_pid() {
    local pid="$1"
    local comm
    if grep -qx "${pid}" <<<"${PROTECTED_PIDS}"; then
        return 0
    fi
    comm=$(ps -o comm= -p "${pid}" 2>/dev/null | awk '{print $1}')
    case "${comm}" in
        systemd|init|dockerd|docker-proxy|containerd|containerd-shim*|slurmd|slurmstepd|srun|sshd|udevd|dbus-daemon)
            return 0
            ;;
    esac
    return 1
}

collect_occupant_pids_from_rocm_smi() {
    command -v rocm-smi >/dev/null 2>&1 || return 0
    {
        rocm-smi --showpids 2>/dev/null | awk '
            $1 ~ /^[0-9]+$/ {
                pid=$1
                gpu=$3
                vram=$4
                if (pid+0 > 1 && (gpu+0 > 0 || vram+0 > 0)) print pid
            }
        '
        rocm-smi --showpidgpus 2>/dev/null | awk '
            /PID [0-9]+ is using [1-9]/ {
                for (i = 1; i <= NF; i++) {
                    if ($i == "PID") pid = $(i + 1)
                    if ($i == "using") n = $(i + 1)
                }
                if (pid + 0 > 1 && n + 0 > 0) print pid
            }
        '
    } | grep -E '^[0-9]+$' | sort -u
}

collect_occupant_pids() {
    local host_pids=""
    local container_pids=""
    host_pids=$(collect_occupant_pids_from_rocm_smi || true)
    if [ -n "${CONTAINER}" ]; then
        container_pids=$(exec_in '
            command -v rocm-smi >/dev/null 2>&1 || exit 0
            {
                rocm-smi --showpids 2>/dev/null | awk '"'"'
                    $1 ~ /^[0-9]+$/ {
                        pid=$1
                        gpu=$3
                        vram=$4
                        if (pid+0 > 1 && (gpu+0 > 0 || vram+0 > 0)) print pid
                    }
                '"'"'
                rocm-smi --showpidgpus 2>/dev/null | awk '"'"'
                    /PID [0-9]+ is using [1-9]/ {
                        for (i = 1; i <= NF; i++) {
                            if ($i == "PID") pid = $(i + 1)
                            if ($i == "using") n = $(i + 1)
                        }
                        if (pid + 0 > 1 && n + 0 > 0) print pid
                    }
                '"'"'
            } | grep -E "^[0-9]+$" | sort -u
        ' || true)
    fi
    printf '%s\n%s\n' "${host_pids}" "${container_pids}" | grep -E '^[0-9]+$' | sort -u
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
    local used
    echo "Waiting for GPU memory to release after killing occupants..."
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

kill_gpu_occupants() {
    local pid
    local occupants
    local killed=0

    if [ "${GPU_PREFLIGHT_KILL_OCCUPANTS}" != "1" ]; then
        echo "GPU occupant kill disabled (GPU_PREFLIGHT_KILL_OCCUPANTS=${GPU_PREFLIGHT_KILL_OCCUPANTS})"
        return 0
    fi

    occupants=$(collect_occupant_pids || true)
    if [ -z "${occupants}" ]; then
        echo "No GPU occupant PIDs found"
        return 0
    fi

    echo ""
    echo "========== GPU preflight: killing leftover GPU occupant PIDs =========="
    echo "Occupant PIDs: ${occupants//$'\n'/ }"
    for pid in ${occupants}; do
        if is_protected_pid "${pid}"; then
            echo "  skip protected PID ${pid} ($(ps -o comm= -p "${pid}" 2>/dev/null | awk '{print $1}'))"
            continue
        fi
        echo "  kill TERM ${pid} ($(ps -o comm= -p "${pid}" 2>/dev/null | tr -s ' ' || echo unknown))"
        kill -TERM "${pid}" 2>/dev/null || true
        killed=1
    done

    if [ "${killed}" -eq 0 ]; then
        echo "No occupant PIDs were eligible to kill"
        return 0
    fi

    sleep 2
    for pid in ${occupants}; do
        if is_protected_pid "${pid}"; then
            continue
        fi
        if kill -0 "${pid}" 2>/dev/null; then
            echo "  kill KILL ${pid}"
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    done

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

PROTECTED_PIDS="$(protected_pid_list)"

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

if [ "${GPU_PREFLIGHT_KILL_OCCUPANTS}" = "1" ]; then
    kill_gpu_occupants
fi

set +e
run_hip_smoke_test
hip_rc=$?
set -e

if [ "${hip_rc}" -ne 0 ] && [ "${GPU_PREFLIGHT_KILL_OCCUPANTS}" = "1" ]; then
    echo "HIP smoke test failed (rc=${hip_rc}); retrying after killing GPU occupants"
    kill_gpu_occupants
    set +e
    run_hip_smoke_test
    hip_rc=$?
    set -e
fi

exit "${hip_rc}"
