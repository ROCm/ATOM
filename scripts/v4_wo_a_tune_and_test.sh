#!/usr/bin/env bash
# One-shot: tune the FP8 BMM kernel for V4 wo_a shape via aiter's standard
# JSON DB mechanism, then run BMM + einsum + tuned-BMM and print comparison.
#
# Uses aiter's intended config-file flow (no kernel source patching):
#   1. Sweep candidate configs at our shape (B=2, K=4096, N=1024, M={1..1024}).
#   2. Write best-per-bucket JSON to aiter/ops/triton/configs/gemm/.
#   3. Aiter's _get_config(M, N=1024, K=4096) auto-picks the new file on next
#      kernel call.
#
# Usage on GPU node (gfx950 docker, container has working aiter + triton_kernels):
#   bash /shared/amdgpu/home/zufa_yu_qle/ATOM/scripts/v4_wo_a_tune_and_test.sh

set -uo pipefail

REPO=/shared/amdgpu/home/zufa_yu_qle/ATOM
MODEL=/shared/data/amd_int/models/DeepSeek-V4-Pro
COMMON_ARGS="--model $MODEL --kv_cache_dtype fp8 -tp 8 --max-num-seqs 4 --max-num-batched-tokens 1024 --max-model-len 1024 --gpu-memory-utilization 0.85 --enforce-eager --temperature 0.0 --max-tokens 128"

cd "$REPO"

echo "=========================================================================="
echo "[$(date +%H:%M:%S)] Phase 1: detect arch + locate aiter config dir"
echo "=========================================================================="
ARCH=$(python3 -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])")
AITER_DIR=$(python3 -c "import aiter, os; print(os.path.dirname(aiter.__file__))")
TARGET_JSON="$AITER_DIR/ops/triton/configs/gemm/${ARCH}-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=1024-K=4096.json"
echo "  arch:        $ARCH"
echo "  aiter:       $AITER_DIR"
echo "  target JSON: $TARGET_JSON"

# Backup any existing tuned config so we can restore if anything goes sideways
if [ -f "$TARGET_JSON" ]; then
  echo "  (existing tuned JSON found, backing up to .bak)"
  cp "$TARGET_JSON" "$TARGET_JSON.bak"
fi

echo
echo "=========================================================================="
echo "[$(date +%H:%M:%S)] Phase 2: baseline measurement (BMM with default config)"
echo "=========================================================================="
# Make sure no tuned config is in place yet (so this is true baseline)
if [ -f "$TARGET_JSON" ]; then
  mv "$TARGET_JSON" "$TARGET_JSON.tmp_baseline"
fi
rm -rf /root/.cache/atom/* /root/.triton/cache/* 2>/dev/null

ATOM_USE_TRITON_MOE=1 AITER_LOG_LEVEL=WARNING python -m atom.examples.simple_inference $COMMON_ARGS 2>&1 | tee /tmp/v4_bmm_default.log > /dev/null
echo "--- BMM (default config) results ---"
grep "Request [0-9]\+ finished" /tmp/v4_bmm_default.log
BMM_DEFAULT=$(grep "Request 3 finished" /tmp/v4_bmm_default.log | grep -oP "TPOT: \K[0-9.]+" || echo "N/A")

# Restore baseline backup if any
if [ -f "$TARGET_JSON.tmp_baseline" ]; then
  mv "$TARGET_JSON.tmp_baseline" "$TARGET_JSON"
fi

echo
echo "=========================================================================="
echo "[$(date +%H:%M:%S)] Phase 3: einsum baseline (for comparison)"
echo "=========================================================================="
rm -rf /root/.cache/atom/* /root/.triton/cache/* 2>/dev/null
ATOM_USE_TRITON_MOE=1 ATOM_V4_OA_USE_EINSUM=1 AITER_LOG_LEVEL=WARNING python -m atom.examples.simple_inference $COMMON_ARGS 2>&1 | tee /tmp/v4_einsum.log > /dev/null
echo "--- einsum baseline results ---"
grep "Request [0-9]\+ finished" /tmp/v4_einsum.log
EINSUM_TPOT=$(grep "Request 3 finished" /tmp/v4_einsum.log | grep -oP "TPOT: \K[0-9.]+" || echo "N/A")

echo
echo "=========================================================================="
echo "[$(date +%H:%M:%S)] Phase 4: sweep + write tuned JSON via aiter standard mechanism"
echo "=========================================================================="
# --quick mode: ~30 configs/M, covers PR's actual M range (decode + prefill)
python "$REPO/scripts/v4_wo_a_tune.py" --m-list 1 4 8 16 32 64 --quick --write 2>&1 | tee /tmp/v4_tune_sweep.log

if [ ! -f "$TARGET_JSON" ]; then
  echo "ERROR: tune script did not create $TARGET_JSON"
  echo "  check /tmp/v4_tune_sweep.log for sweep failures"
  exit 1
fi
echo "--- tuned JSON created ---"
ls -la "$TARGET_JSON"
python3 -m json.tool "$TARGET_JSON" | head -30

echo
echo "=========================================================================="
echo "[$(date +%H:%M:%S)] Phase 5: BMM with tuned config (aiter auto-picks the new JSON)"
echo "=========================================================================="
rm -rf /root/.cache/atom/* /root/.triton/cache/* 2>/dev/null
ATOM_USE_TRITON_MOE=1 AITER_LOG_LEVEL=WARNING python -m atom.examples.simple_inference $COMMON_ARGS 2>&1 | tee /tmp/v4_bmm_tuned.log > /dev/null
echo "--- BMM (tuned config) results ---"
grep "Request [0-9]\+ finished" /tmp/v4_bmm_tuned.log
BMM_TUNED=$(grep "Request 3 finished" /tmp/v4_bmm_tuned.log | grep -oP "TPOT: \K[0-9.]+" || echo "N/A")

echo
echo "=========================================================================="
echo "[$(date +%H:%M:%S)] Phase 6: three-way comparison summary"
echo "=========================================================================="
printf "%-25s %12s %12s\n" "config" "Req3 TPOT" "vs einsum"
printf "%-25s %12s %12s\n" "------" "---------" "---------"
printf "%-25s %12s %12s\n" "einsum (baseline)"  "${EINSUM_TPOT}s"  "1.00x"
if [ "$BMM_DEFAULT" != "N/A" ] && [ "$EINSUM_TPOT" != "N/A" ]; then
  RATIO=$(python3 -c "print(f'{$BMM_DEFAULT/$EINSUM_TPOT:.3f}x')")
  printf "%-25s %12s %12s\n" "BMM (default config)"  "${BMM_DEFAULT}s"  "$RATIO"
fi
if [ "$BMM_TUNED" != "N/A" ] && [ "$EINSUM_TPOT" != "N/A" ]; then
  RATIO=$(python3 -c "print(f'{$BMM_TUNED/$EINSUM_TPOT:.3f}x')")
  printf "%-25s %12s %12s\n" "BMM (tuned config)"   "${BMM_TUNED}s"   "$RATIO"
fi

echo
echo "Logs: /tmp/v4_bmm_default.log  /tmp/v4_einsum.log  /tmp/v4_bmm_tuned.log  /tmp/v4_tune_sweep.log"
echo "Tuned JSON: $TARGET_JSON  (delete or restore .bak to revert)"
