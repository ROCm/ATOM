#!/usr/bin/env bash
# GPU-less, in a container of the base nightly: AOT-compile the FlyDSL kernels of ganyi's C17 decode tables for the
# aiter tree at SHA (from the bundle), starting from SEED_DIR's caches (flydsl_cache/ + flydsl_aot/ of the stage-1 build
# of the same SHA), and tar the resulting flydsl_cache/ + flydsl_aot/ to OUT_TAR (relative to aiter/jit). Runs beside
# stage 1 so the image build only has to unpack it. Usage: c17_aot_prewarm_inner.sh SHA BUNDLE SEED_DIR CSV_DIR OUT_TAR JOBS
set -euo pipefail
SHA=$1 BUNDLE=$2 SEED=$3 CSV=$4 OUT=$5 JOBS=${6:-32}
git config --global --add safe.directory '*'
cd /app/aiter-test
git fetch -q "${BUNDLE}" "+refs/heads/*:refs/remotes/agx/*"
git checkout -q -B m31-agentx "${SHA}"
git clean -ffdxq
mkdir -p /tmp/pw && cp -a "${SEED}/flydsl_cache" /tmp/pw/flydsl_cache && cp -a "${SEED}/flydsl_aot" /tmp/pw/flydsl_aot
n0=$(find /tmp/pw/flydsl_cache -name '*.pkl' | wc -l)
t0=$(date +%s)
rc=0
AITER_AOT_IMPORT=1 GPU_ARCHS=gfx950 AITER_MAX_JOBS=${JOBS} FLYDSL_RUNTIME_CACHE_DIR=/tmp/pw/flydsl_cache AITER_FLYDSL_AOT_CACHE_DIR=/tmp/pw/flydsl_aot \
  AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE=${CSV}/a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm_m31c17.csv \
  AITER_CONFIG_GEMM_BF16=${CSV}/bf16_tuned_gemm_m31c17.csv \
  nice -n 19 python3 -m aiter.aot.flydsl.gemm --csv ${CSV}/a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm_m31c17.csv ${CSV}/bf16_tuned_gemm_m31c17.csv \
  > /tmp/pw/aot.log 2>&1 || rc=$?
grep -E "replay gfx950|\[FAIL\]|BISECT|UNREACHED\] [0-9]+ x|Total time|All replays" /tmp/pw/aot.log | cut -c1-260
echo "[agx] c17 AOT rc=${rc}; flydsl_cache entries ${n0} -> $(find /tmp/pw/flydsl_cache -name '*.pkl' | wc -l) in $(( $(date +%s) - t0 )) s"
find /tmp/pw -name '*.tmp' -delete
chmod -R a+rX /tmp/pw
cp /tmp/pw/aot.log /tmp/pw/flydsl_cache/agx_c17_aot.log
tar -C /tmp/pw -cf "${OUT}.part" flydsl_cache flydsl_aot && mv "${OUT}.part" "${OUT}"
echo "[agx] wrote ${OUT} ($(du -sh "${OUT}" | cut -f1))"
