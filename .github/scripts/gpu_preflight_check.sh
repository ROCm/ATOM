#!/usr/bin/env bash
# Preflight GPU state before ATOM starts touching distributed/RCCL paths.
#
# This intentionally avoids importing ATOM or aiter. It records the visible GPU
# process/memory state, then performs a minimal torch allocation on every visible
# HIP device. If this fails, the node/container is already unhealthy before ATOM
# participates in the run.

set -euo pipefail

CONTAINER="${1:-}"
ENGINE="${2:-docker}"
GPU_PREFLIGHT_ALLOCATION_MB="${GPU_PREFLIGHT_ALLOCATION_MB:-8}"

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
