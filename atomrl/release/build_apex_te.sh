#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Source-build ROCm Apex and TransformerEngine at the pinned revisions.
# Runs inside the image build; no GPU required, only the target arch.
#
# TE is the load-bearing one: without it megatron's TE layer-spec constructor
# returns None and example 7 dies in init_model with a bare
# "TypeError: 'NoneType' object is not callable" that never mentions TE.
# Apex missing is only a warning (falls back to Torch Norm).
#
# NEVER `pip install transformer_engine` from PyPI — that is the NVIDIA build and
# fails at import with undefined symbols.
set -eux

ARCH=${PYTORCH_ROCM_ARCH:-gfx950}
# Apex vendors aiter, whose chip_info shells out to rocminfo to detect the arch.
# There is no GPU during `docker build`, so rocminfo exits 1 and the build dies
# with "Get GPU arch from rocminfo failed". GPU_ARCHS is the documented override
# and makes the detection path a no-op.
export GPU_ARCHS=${GPU_ARCHS:-$ARCH}
APEX_SHA=${APEX_SHA:-daed85255d51476425080e7e6203f0bee6d7e4cc}
TE_SHA=${TE_SHA:-6e541a10419a6e31bdc98b1516db04eb81a463b6}
SRC=${SRC:-/tmp/megatron_src}
export TMPDIR=${TMPDIR:-/tmp/build}
mkdir -p "$SRC" "$TMPDIR"

git config --global --add safe.directory '*'

# ---- Apex ----
git clone https://github.com/ROCm/apex.git "$SRC/apex"
git -C "$SRC/apex" checkout "$APEX_SHA"
git -C "$SRC/apex" submodule update --init --recursive --jobs 16
cd "$SRC/apex"
PYTORCH_ROCM_ARCH="$ARCH" MAX_JOBS=${MAX_JOBS:-48} python3 setup.py install --cpp_ext --cuda_ext
python3 -c "import apex; print('apex ok')"

# ---- TransformerEngine ----
git clone https://github.com/ROCm/TransformerEngine.git "$SRC/te"
git -C "$SRC/te" checkout "$TE_SHA"
git -C "$SRC/te" submodule update --init --recursive --jobs 16
# ROCm LLVM 24 rejects quoted wildcard exports when no matching typeinfo
# symbols are emitted. Removing these two no-op patterns leaves all concrete
# QoLA exports unchanged.
sed -i \
    '/"typeinfo for qola::\*";/d; /"typeinfo name for qola::\*";/d' \
    "$SRC/te/3rdparty/QoLA/qola/cpp_itfs/qola_exports.lds"
cd "$SRC/te"
export NVTE_FRAMEWORK=pytorch NVTE_USE_ROCM=1
export NVTE_ROCM_ARCH="$ARCH" PYTORCH_ROCM_ARCH="$ARCH"
export NVTE_FUSED_ATTN=1 NVTE_FUSED_ATTN_CK=1 NVTE_FUSED_ATTN_AOTRITON=1
export MAX_JOBS=${MAX_JOBS:-48}
# ROCm 7.2's hipcc -v exits 1 when given no input file and the CK-JIT compiler
# ABI probe reads that as "compiler unusable". The probe is not load-bearing.
export TORCH_DONT_CHECK_COMPILER_ABI=1
# Do not let TransformerEngine's upstream metadata replace the ROCm torch
# supplied by the base image with a CUDA wheel.
python3 -m pip install -v . --no-build-isolation --no-deps

python3 -c "import transformer_engine; import transformer_engine.pytorch; print('TE', transformer_engine.__version__)"

# The import above succeeds even when TE is broken; this is the check that
# actually distinguishes a usable install.
python3 - <<'PY'
import transformer_engine.pytorch
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec as spec,
)
print("megatron TE layer spec OK:", type(spec()).__name__)
PY

rm -rf "$SRC"
