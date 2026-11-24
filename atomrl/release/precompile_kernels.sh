#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Bake aiter's JIT kernels into the release image.
#
# `docker build` has no GPU, so the kernels cannot be compiled in the Dockerfile.
# This starts the built image with the GPUs attached, triggers every kernel the
# examples need, and commits the result. Without it the first FP8 run pays the
# compile cost at startup — measured on MI355X:
#   module_gemm_a8w8_blockscale_bpreshuffle_cktile   411 s
#   module_gemm_a8w8_blockscale_cktile               419 s
#   module_gemm_a8w8_blockscale                      278 s
# and the 8 training actors serialise on a build lock while it happens, which
# looks like a hang ("waiting for baton release") but is not.
#
#   TAG=atomrl:release-20260902 bash atomrl/release/precompile_kernels.sh
#
# Coverage note: the synthetic warmup below reaches ~5 of the 16 kernel objects
# the examples end up needing. For a fully warmed image, run an actual example 4
# against the result and commit that container instead — example 4 is the widest
# path (ATOM FP8 rollout + FSDP2 FP8 training) and pulls in the bpreshuffle
# variants this script does not:
#
#   TAG=<tag> DATA_ROOT=... EX=4 NAME=warm bash atomrl/release/validate_image.sh
#   docker exec warm bash -lc 'rm -rf /tmp/ray /tmp/atom_torch_compile_cache \
#     /tmp/torchinductor_root /tmp/aiter_configs /root/.cache'
#   docker commit --change 'ENTRYPOINT ["/opt/atomrl/entrypoint.sh"]' \
#     --change 'CMD ["sleep","infinity"]' \
#     --change 'WORKDIR /opt/atomrl/ATOM/atomrl' \
#     --change 'ENV PYTHONPATH=/opt/atomrl/ATOM/atomrl:/opt/atomrl/aiter:/opt/atomrl/mori/python' \
#     --change 'ENV AITER_JIT_DIR=/opt/atomrl/aiter-jit' warm <tag>
#
# Measured on 8x MI355X, example 4 smoke: 1256 s with 5 objects baked, 447 s with
# all 16.
set -euo pipefail

TAG=${TAG:?set TAG to the image built by build_image.sh}
OUT_TAG=${OUT_TAG:-${TAG}-kernels}
NAME=atomrl-precompile-$$

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> starting $TAG with GPUs"
docker run -d --name "$NAME" --entrypoint /bin/bash \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  "$TAG" -lc 'sleep infinity' >/dev/null

echo "==> compiling kernels (~25 min)"
docker exec "$NAME" bash -lc '
set -e
export PYTHONPATH="$RL_ROOT/ATOM/atomrl:$AITER_DIR:$MORI_DIR/python:${PYTHONPATH:-}"
export AITER_JIT_DIR=/opt/atomrl/aiter-jit
mkdir -p "$AITER_JIT_DIR"
python3 - <<PY
import traceback

import torch

# aiter.dtypes.fp8 resolves to e4m3fn on gfx950 and e4m3fnuz on gfx942.
# Hardcoding either one fails on the other arch with
# "AssertionError: Unsupported dtype".
from aiter import dtypes

FP8 = dtypes.fp8
print("fp8 dtype for this arch:", FP8, flush=True)


def warm(name, fn):
    """Each kernel family is independent; a probe that cannot find a matching
    tuned shape has still done the JIT build, which is the point."""
    print(f"warming {name}", flush=True)
    try:
        fn()
        torch.cuda.synchronize()
    except Exception:
        print(f"  ({name} probe raised; kernels are still built)", flush=True)
        traceback.print_exc()


def _rmsnorm():
    from aiter import rmsnorm2d_fwd, rmsnorm2d_fwd_with_add
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
    rmsnorm2d_fwd(x, w, 1e-6)
    rmsnorm2d_fwd(x, w, 1e-6, use_model_sensitive_rmsnorm=1)
    out, res_out = torch.empty_like(x), torch.empty_like(x)
    rmsnorm2d_fwd_with_add(out, x, torch.randn_like(x), res_out, w, 1e-6,
                           use_model_sensitive_rmsnorm=1)


def _blockscale():
    from aiter import QuantType, get_hip_quant
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale
    q = get_hip_quant(QuantType.per_1x128)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    aq, asc = q(a, quant_dtype=FP8)
    # Weights carry one scale per 128x128 block, (N/128, K/128); a per-1x128
    # activation quant gives (N, K/128), which the FlyDSL kernel rejects.
    bq = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16).to(FP8)
    bsc = torch.rand(4096 // 128, 4096 // 128, device="cuda", dtype=torch.float32)
    gemm_a8w8_blockscale(aq, bq, asc, bsc, dtype=torch.bfloat16)


def _cross_entropy():
    from aiter.ops.triton.cross_entropy import (
        cross_entropy_forward,
        cross_entropy_forward_chunked,
    )
    logits = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float32)
    tgt = torch.randint(0, 4096, (2, 128), device="cuda")
    cross_entropy_forward(logits.clone(), tgt, 0.0, False, None, -100)
    cross_entropy_forward_chunked(logits.clone(), tgt, 0.0, False, None, -100, 64)


def _quant():
    from aiter import QuantType, get_hip_quant
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for qt in (QuantType.per_Token, QuantType.per_1x128):
        get_hip_quant(qt)(x, quant_dtype=FP8)


warm("RMSNorm (plain + model-sensitive)", _rmsnorm)
warm("quant (per-token, per-1x128)", _quant)
warm("FP8 blockscale GEMM (the slow ones)", _blockscale)
warm("cross entropy (plain + chunked)", _cross_entropy)
print("PRECOMPILE_DONE", flush=True)
PY
'

echo "==> committing to $OUT_TAG"
# The precompile container is started with --entrypoint /bin/bash so it can just
# sleep. Plain `docker commit` would bake that override in, and the resulting
# image would answer `docker run IMAGE bash -lc ...` with
# "/usr/bin/bash: cannot execute binary file". Restore the real entrypoint.
docker commit \
  --change 'ENTRYPOINT ["/opt/atomrl/entrypoint.sh"]' \
  --change 'CMD ["sleep", "infinity"]' \
  --change 'WORKDIR /opt/atomrl/ATOM/atomrl' \
  "$NAME" "$OUT_TAG" >/dev/null
docker exec "$NAME" bash -lc 'ls /opt/atomrl/aiter-jit/*.so 2>/dev/null | wc -l' \
  | xargs -I{} echo "    baked {} kernel objects"
echo "==> done: $OUT_TAG"
