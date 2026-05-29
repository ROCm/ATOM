#!/bin/bash
# scripts/gfx1201/setup_aiter_configs.sh
#
# aiter ships ZERO gfx1201 GEMM tuned configs (only gfx1250, gfx950, gfx942
# as of `rocm/atom-dev:latest` digest sha256:b704d9a8...). When a kernel runs
# on gfx1201 and looks up a tuned config keyed by the arch string, the lookup
# misses and aiter's autotuner falls back to a default config that is 30-50%
# slower at our 8B model shapes (verified on Ministral-3-8B: 22 ms TPOT with
# this script vs 32.5 ms without).
#
# gfx1250 (RDNA4 successor) has the closest matrix-instruction profile to
# gfx1201 — its tuned configs are the best off-the-shelf approximation. This
# script symlinks every gfx1250-* config in aiter as gfx1201-*.
#
# This is a SETUP step that runs ONCE per container. Re-run if you re-pull
# the rocm/atom-dev image (the symlinks live in the image overlay).
#
# Usage: bash scripts/gfx1201/setup_aiter_configs.sh

set -euo pipefail

CONFIG_DIR="${AITER_CONFIG_DIR:-/app/aiter-test/aiter/ops/triton/configs/gemm}"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "ERROR: aiter config dir not found at $CONFIG_DIR" >&2
    echo "       Set AITER_CONFIG_DIR if your aiter is installed elsewhere." >&2
    exit 1
fi

cd "$CONFIG_DIR"

count=0
for src in gfx1250-*.json; do
    [ -f "$src" ] || continue
    dst="${src/gfx1250/gfx1201}"
    if [ ! -e "$dst" ]; then
        ln -sf "$src" "$dst"
        count=$((count + 1))
    fi
done

echo "[gfx1201 setup] created $count symlinks in $CONFIG_DIR"
echo "[gfx1201 setup] gfx1201-* config files now: $(ls -1 gfx1201-*.json 2>/dev/null | wc -l)"
