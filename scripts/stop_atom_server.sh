#!/bin/bash
# Stop ATOM server and release GPU memory.
# Usage: bash stop_atom_server.sh
#
# Escalates SIGTERM -> SIGKILL on the main entrypoint process AND its
# multiprocessing children, then waits up to 60s for VRAM to drop to 0.
# If VRAM is still held, dumps the GPU PID list and force-kills each one.

set -uo pipefail

echo "=== Stopping ATOM server ==="

# Patterns matching every process the server can spawn AND clients hammering
# it (lm_eval) so the next launch starts from a fully clean state.
PATTERNS=(
    'atom.entrypoints'
    'multiprocessing.spawn'
    'multiprocessing.resource_tracker'
    'atom.model_engine'
    'lm_eval'
)

count_alive() {
    local total=0
    for pat in "${PATTERNS[@]}"; do
        local n
        n=$(pgrep -f "$pat" 2>/dev/null | wc -l)
        total=$((total + n))
    done
    echo "$total"
}

# 1. Graceful SIGTERM round.
if [ "$(count_alive)" -gt 0 ]; then
    echo "Sending SIGTERM to ATOM processes..."
    for pat in "${PATTERNS[@]}"; do
        pkill -TERM -f "$pat" 2>/dev/null || true
    done
    for i in $(seq 1 5); do
        sleep 1
        [ "$(count_alive)" -eq 0 ] && break
    done
fi

# 2. SIGKILL anything still alive — escalate without polite waiting.
if [ "$(count_alive)" -gt 0 ]; then
    echo "SIGTERM did not finish cleanup; escalating to SIGKILL..."
    for pat in "${PATTERNS[@]}"; do
        pkill -KILL -f "$pat" 2>/dev/null || true
    done
    sleep 2
fi

# 3. Final sweep: any leftover that still references the server by name.
if [ "$(count_alive)" -gt 0 ]; then
    echo "Stubborn processes still alive — final SIGKILL pass:"
    for pat in "${PATTERNS[@]}"; do
        pgrep -af "$pat" 2>/dev/null || true
    done
    for pat in "${PATTERNS[@]}"; do
        pkill -KILL -9 -f "$pat" 2>/dev/null || true
    done
    sleep 3
fi

# 4. Wait for GPU memory to release.
echo "Waiting for GPU memory to release..."
for i in $(seq 1 60); do
    USED_GPUS=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk '{print $NF}' | awk '$1 > 0' | wc -l)
    if [ "$USED_GPUS" -eq 0 ]; then
        echo "GPU memory released after ${i}s"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "WARNING: GPU memory still in use after 60s, force-killing GPU PIDs..."
        # Newer rocm-smi exposes per-GPU PIDs via --showpids
        for pid in $(rocm-smi --showpids 2>/dev/null | awk '/^PID/{f=1; next} f && $1 ~ /^[0-9]+$/ {print $1}'); do
            echo "  kill -9 $pid"
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 5
        USED_GPUS=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk '{print $NF}' | awk '$1 > 0' | wc -l)
        if [ "$USED_GPUS" -gt 0 ]; then
            echo "ERROR: GPU memory still held after force-kill. Manual intervention required."
            rocm-smi --showpids 2>&1 | head -40
            exit 1
        fi
    fi
    sleep 1
done

echo "Server stopped."
