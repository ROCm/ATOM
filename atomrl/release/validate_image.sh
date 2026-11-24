#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# End-to-end check of a built release image: start it the way the README tells a
# customer to, run a smoke off the code baked into the image, and report the
# health criteria from README §6.
#
#   TAG=atomrl:release-20260902-kernels DATA_ROOT=/path/to/data EX=1 \
#     bash atomrl/release/validate_image.sh
set -uo pipefail

TAG=${TAG:?set TAG}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT}
EX=${EX:-1}
NAME=${NAME:-atomrl-validate}

case "$EX" in
  1) CFG=dapo_qwen3_8b_ray_vllm_smoke.yaml;         MODE=bf16;     XTRA="" ;;
  2) CFG=dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml;     MODE=fp8;      XTRA="" ;;
  3) CFG=dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml;     MODE=fp8;      XTRA="TRAIN_FP8=1" ;;
  4) CFG=dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml;  MODE=atomfp8;  XTRA="TRAIN_FP8=1" ;;
  5) CFG=dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml; MODE=atombf16; XTRA="" ;;
  *) echo "EX=$EX not covered by the image-only path (6/7 need the MoE weights)"; exit 2 ;;
esac

LOG="$DATA_ROOT/logs/validate-ex${EX}.log"

echo "==> (re)starting $NAME from $TAG"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --entrypoint /bin/bash \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -v "$DATA_ROOT":"$DATA_ROOT" -e DATA_ROOT="$DATA_ROOT" \
  "$TAG" -lc 'sleep infinity' >/dev/null

echo "==> idle baseline (README 6.1)"
docker exec "$NAME" bash -lc 'rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM Total Used" \
  | awk "{printf \"    GPU%d %.0f MB\n\", NR-1, \$NF/1048576}"'

echo "==> stack (README 6.2)"
docker exec "$NAME" bash -lc '
python3 -c "
import aiter, atomrl, atomrl.lumen, vllm, flydsl, transformers
print(\"    vllm\", vllm.__version__, \"flydsl\", flydsl.__version__, \"transformers\", transformers.__version__)
print(\"    aiter\", aiter.__file__)
assert aiter.__file__.startswith(\"/app/aiter-test/\"), aiter.__file__
print(\"    source install wins over the wheel: OK\")"'

echo "==> baked kernels"
docker exec "$NAME" bash -lc 'ls /opt/atomrl/aiter-jit/*.so 2>/dev/null | wc -l | xargs -I{} echo "    {} objects"'

echo "==> smoke: example $EX ($MODE $XTRA)"
S=/opt/atomrl/ATOM/atomrl/examples/DAPO/run_dapo.sh
ENVX="export RL_ROOT=/opt/atomrl DATA_ROOT=$DATA_ROOT PYTORCH_CUDA_ALLOC_CONF=;"
START=$(date +%s)
docker exec "$NAME" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/$CFG \
  STEPS=1 MODE=$MODE $XTRA LOG=$LOG bash '$S'" >/dev/null 2>&1
ELAPSED=$(( $(date +%s) - START ))

echo "==> verdict (README 6.3, 6.4)   wall time ${ELAPSED}s"
docker exec "$NAME" bash -lc "
L=$LOG
faults=\$(grep -acE 'Traceback|OutOfMemory|CUDA error|HSA_STATUS' \$L)
fin=\$(grep -ac 'AtomRL finished' \$L)
echo \"    faults=\$faults  finished=\$fin\"
grep -ao 'step=1 actor/entropy.*' \$L | head -1 | tr ' ' '\n' \
  | grep -E '^(rollout_corr/(kl|ppl_ratio)|actor/(entropy|grad_norm)|reward/accuracy)=' \
  | sed 's/^/    /'
[ \"\$faults\" = 0 ] && [ \"\$fin\" != 0 ] && echo '    RESULT: PASS' || echo '    RESULT: FAIL'
"
