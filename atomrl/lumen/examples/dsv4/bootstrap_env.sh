#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Runtime env setup inside lumen/tests:latest for DSV4 smoke.
set -euo pipefail

BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-/bootstrap}"
LUMEN_DIR="${LUMEN_DIR:-/workspace/Lumen}"
ATOMRL_DIR="${ATOMRL_DIR:-/workspace/ATOM/atomrl}"
AITER_DIR="${AITER_DIR:-/workspace/aiter}"
WRITABLE_ROOT="${WRITABLE_ROOT:-/tmp/lumen-dsv4-runtime}"
LUMEN_DSV4_NATIVE_FINETUNE="${LUMEN_DSV4_NATIVE_FINETUNE:-0}"
LUMEN_DSV4_PRETRAIN="${LUMEN_DSV4_PRETRAIN:-0}"
MILES_DIR="${MILES_DIR:-/workspace/miles}"

_dsv4_megatron_only=0
if [[ "${LUMEN_DSV4_NATIVE_FINETUNE}" == "1" || "${LUMEN_DSV4_PRETRAIN}" == "1" ]]; then
    _dsv4_megatron_only=1
fi

export MEGATRON_PATH="${MEGATRON_PATH:-${BOOTSTRAP_DIR}/Megatron-LM}"

mkdir -p "${WRITABLE_ROOT}"

SITE_PKGS="${SITE_PKGS:-${WRITABLE_ROOT}/site-packages}"
if [[ ! -d "${SITE_PKGS}" ]]; then
    echo "[bootstrap_env] copying site-packages -> ${SITE_PKGS} (no TE)"
    rm -rf "${SITE_PKGS}"
    mkdir -p "${SITE_PKGS}"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a \
            --exclude='transformer_engine' \
            --exclude='transformer_engine*.dist-info' \
            --exclude='transformer_engine*.egg-info' \
            "${BOOTSTRAP_DIR}/site-packages/." "${SITE_PKGS}/"
    else
        cp -a "${BOOTSTRAP_DIR}/site-packages/." "${SITE_PKGS}/"
        rm -rf "${SITE_PKGS}/transformer_engine" \
            "${SITE_PKGS}"/transformer_engine*.dist-info \
            "${SITE_PKGS}"/transformer_engine*.egg-info 2>/dev/null || true
    fi
fi
export SITE_PKGS

if [[ ! -f "${SITE_PKGS}/lumen_dsv4_bootstrap.pth" ]]; then
    cat > "${SITE_PKGS}/lumen_dsv4_bootstrap.pth" <<'PY'
import os

import megatron.core.jit as _jit
_jit.disable_jit_fuser()
PY
fi

if [[ "${_dsv4_megatron_only}" == "1" ]]; then
    echo "[bootstrap_env] megatron-only path (native GRPO finetune)"
    if [[ -d "${MILES_DIR}" ]] && ! python -c "import miles" 2>/dev/null; then
        echo "[bootstrap_env] installing miles (no-deps, for Megatron router hook) ..."
        pip install -e "${MILES_DIR}" --no-deps -q
    fi
    if [[ -d "${MILES_DIR}" ]] && ! python -c "import backports.strenum" 2>/dev/null; then
        echo "[bootstrap_env] installing backports.strenum for Miles convert helper ..."
        pip install backports.strenum -q
    fi
else
    echo "[bootstrap_env] WARN: set LUMEN_DSV4_NATIVE_FINETUNE=1"
fi

_dsv4_prepend_pythonpath() {
    local dir="$1"
    if [[ -n "${dir}" && -d "${dir}" ]]; then
        export PYTHONPATH="${dir}:${PYTHONPATH:-}"
    fi
}

# FP8 QAT uses tile_kernels.quant (TileLang JIT) from the runtime environment.
_dsv4_resolve_tilelang_dir() {
    if [[ -n "${TILELANG_DIR:-}" && -d "${TILELANG_DIR}/tilelang" ]]; then
        return
    fi
    for candidate in \
        "${TILELANG_DIR:-}" \
        "${DATA_ROOT:-}/tilelang" \
        "${BOOTSTRAP_DIR}/tilelang" \
        "${WRITABLE_ROOT}/tilelang" \
        "/opt/dsv4-runtime/tilelang" \
        "/opt/dsv4-bootstrap/tilelang"; do
        if [[ -n "${candidate}" && -d "${candidate}/tilelang" ]]; then
            TILELANG_DIR="${candidate}"
            export TILELANG_DIR
            echo "[bootstrap_env] TILELANG_DIR=${TILELANG_DIR}"
            return
        fi
    done
}
_dsv4_resolve_tilelang_dir
_dsv4_prepend_pythonpath "${PYTHON_EXTRAS_DIR:-}"
_dsv4_prepend_pythonpath "${TILELANG_DIR:-}"

export PYTHONPATH="${ATOMRL_DIR}:${AITER_DIR}:${MEGATRON_PATH}:${SITE_PKGS}:${PYTHONPATH:-}"
if [[ "${_dsv4_megatron_only}" == "1" && -d "${MILES_DIR}" ]]; then
    export PYTHONPATH="${MILES_DIR}:${PYTHONPATH}"
fi

NATIVE_LIBS="${BOOTSTRAP_DIR}/native-libs"
if [[ -d "${NATIVE_LIBS}" ]]; then
    export LD_LIBRARY_PATH="${NATIVE_LIBS}:${LD_LIBRARY_PATH:-}"
fi

if ! ldconfig -p 2>/dev/null | grep -q 'libglog.so.0'; then
    echo "[bootstrap_env] installing libgoogle-glog0v5 (Megatron fused CE)..."
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgoogle-glog0v5 >/dev/null
fi

echo "[bootstrap_env] MEGATRON_PATH=${MEGATRON_PATH} (ROCm rocm_dev @ ${MEGATRON_ROCM_REF:-see dsv4_paths.sh})"

if [[ "${LUMEN_DSV4_MOE_MORI:-0}" == "1" ]]; then
    MORI_DIR="${MORI_DIR:-/opt/mori}"
    MORI_PKG="${MORI_DIR}/python/mori"
    MORI_CACHE=""
    for candidate in \
        "${WRITABLE_ROOT}/mori-libs" \
        "${BOOTSTRAP_DIR}/mori-libs" \
        "/opt/dsv4-runtime/mori-libs"; do
        if [[ -f "${candidate}/libmori_pybinds.so" ]]; then
            MORI_CACHE="${candidate}"
            break
        fi
    done
    if [[ ! -f "${MORI_PKG}/libmori_pybinds.so" && -n "${MORI_CACHE}" ]]; then
        echo "[bootstrap_env] restoring mori libs from ${MORI_CACHE}"
        cp -a "${MORI_CACHE}/." "${MORI_PKG}/"
    fi
    if ! python -c "import mori.cpp" 2>/dev/null; then
        echo "[bootstrap_env] building mori for LUMEN_DSV4_MOE_MORI..."
        git config --global --add safe.directory "${MORI_DIR}" || true
        cd "${MORI_DIR}"
        pip install setuptools-scm -q
        SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MORI=0.1.0 pip install -e . -q
    fi
fi

python - <<'PY'
import importlib
import os

required = ["megatron.core"]
if os.environ.get("LUMEN_DSV4_LINEAR_FP8", "0") == "1":
    required.append("tile_kernels")  # QAT-only dependency; MHC uses AIter.
if os.environ.get("LUMEN_DSV4_NATIVE_FINETUNE", "0") == "1" or os.environ.get("LUMEN_DSV4_PRETRAIN", "0") == "1":
    if os.path.isdir(os.environ.get("MILES_DIR", "/workspace/miles")):
        required.append("miles")
_indexer_impl = os.environ.get("V4_INDEXER_IMPL", "aiter").lower()
if _indexer_impl == "aiter":
    try:
        import aiter.ops.triton.attention.dsv4_indexer  # noqa: F401

        print("[bootstrap_env] OK aiter dsv4_indexer (V4_INDEXER_IMPL=aiter)")
    except Exception as e:
        print(f"[bootstrap_env] FAIL aiter dsv4_indexer: {e}")
        raise SystemExit(1)
elif _indexer_impl == "tilelang":
    try:
        import tilelang  # noqa: F401

        print("[bootstrap_env] OK tilelang (V4_INDEXER_IMPL=tilelang)")
    except Exception as e:
        print(f"[bootstrap_env] FAIL tilelang: {e}")
        raise SystemExit(1)
else:
    print(f"[bootstrap_env] FAIL V4_INDEXER_IMPL={_indexer_impl} (use aiter or tilelang)")
    raise SystemExit(1)
optional = ["transformer_engine", "fast_hadamard_transform"]
if os.environ.get("LUMEN_DSV4_MOE_MORI", "0") == "1":
    required.append("mori")

for m in required:
    try:
        mod = importlib.import_module(m)
        print(f"[bootstrap_env] OK {m}: {getattr(mod, '__file__', mod)}")
    except Exception as e:
        print(f"[bootstrap_env] FAIL {m}: {e}")
        raise SystemExit(1)

for m in optional:
    try:
        mod = importlib.import_module(m)
        print(f"[bootstrap_env] OK {m} (optional): {getattr(mod, '__file__', mod)}")
    except Exception as e:
        print(f"[bootstrap_env] SKIP {m} (optional): {e}")

if os.environ.get("V4_SPARSE_MLA_BACKEND", "triton").lower() == "triton":
    try:
        from aiter.ops.triton.attention.sparse_mla_dsv4_train import sparse_mla_dsv4_train
        from atomrl.lumen.kernels.dsv4.sparse_mla.triton_sparse_mla import sparse_attn_triton

        print(f"[bootstrap_env] OK sparse MLA triton: {sparse_mla_dsv4_train.__name__} -> {sparse_attn_triton.__name__}")
    except Exception as e:
        print(f"[bootstrap_env] FAIL sparse MLA triton: {e}")
        raise SystemExit(1)

try:
    from aiter.ops.triton.fusions.mhc import mhc_head_dsv4, mhc_post_dsv4, mhc_pre_dsv4

    print(
        "[bootstrap_env] OK AIter DSV4 MHC: "
        f"{mhc_pre_dsv4.__name__}, {mhc_post_dsv4.__name__}, {mhc_head_dsv4.__name__}"
    )
except Exception as e:
    print(f"[bootstrap_env] FAIL AIter DSV4 MHC: {e}")
    raise SystemExit(1)

_gemm_bf16 = os.environ.get("LUMEN_DSV4_GEMM_BF16", "1")
_gemm_csv = os.environ.get("AITER_CONFIG_GEMM_BF16", "")
print(f"[bootstrap_env] GEMM BF16: LUMEN_DSV4_GEMM_BF16={_gemm_bf16} AITER_CONFIG_GEMM_BF16={_gemm_csv}")
PY
