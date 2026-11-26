#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Shared in-container bootstrap for DSV4 GRPO finetune (source, do not execute).

setup_dsv4_container_env() {
    if [[ -d /opt/dsv4-bootstrap && -f /opt/dsv4-bootstrap/.ready ]]; then
        unset MEGATRON_PATH PYTHONPATH
        export BOOTSTRAP_DIR=/opt/dsv4-bootstrap
        # shellcheck source=examples/dsv4/bootstrap_env.sh
        source examples/dsv4/bootstrap_env.sh
        export MEGATRON_PATH=/opt/dsv4-bootstrap/Megatron-LM
    elif [[ -d /bootstrap && -f /bootstrap/.ready ]]; then
        unset MEGATRON_PATH PYTHONPATH
        export BOOTSTRAP_DIR=/bootstrap
        # shellcheck source=examples/dsv4/bootstrap_env.sh
        source examples/dsv4/bootstrap_env.sh
        export MEGATRON_PATH=/bootstrap/Megatron-LM
    else
        export MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
    fi

    export AITER_DIR="${AITER_DIR:-/workspace/aiter}"
    if ! PYTHONPATH="${AITER_DIR}:${PYTHONPATH:-}" python3 - <<'PY'
import os
from importlib import import_module
from pathlib import Path

mhc = import_module("aiter.ops.triton.fusions.mhc")

required = ("mhc_pre_dsv4", "mhc_post_dsv4", "mhc_head_dsv4")
missing = [name for name in required if not callable(getattr(mhc, name, None))]
if missing:
    raise SystemExit(
        "missing required AIter DSV4 MHC APIs: " + ", ".join(missing)
    )

aiter_dir = Path(os.environ["AITER_DIR"]).resolve()
module_path = Path(mhc.__file__).resolve()
if aiter_dir not in module_path.parents:
    raise SystemExit(
        f"AIter DSV4 MHC resolved from {module_path}, expected checkout {aiter_dir}"
    )
print(f"[setup] OK AIter DSV4 MHC APIs: {module_path}")
PY
    then
        echo "[setup][ERROR] missing required AIter DSV4 MHC APIs in ${AITER_DIR}" >&2
        return 1
    fi

    if [[ -f examples/dsv4/patch_megatron_source.py && -d "${MEGATRON_PATH}" ]]; then
        echo "[setup] ensuring ROCm Megatron SOURCE patches on ${MEGATRON_PATH}"
        # Idempotent SOURCE patches; list with: patch_megatron_source.py --list --tag dsv4
        PYTHONPATH="/workspace/ATOM/atomrl:${PYTHONPATH:-}" \
            python3 examples/dsv4/patch_megatron_source.py "${MEGATRON_PATH}"
    fi

    local datasets_dir="${MEGATRON_PATH}/megatron/core/datasets"
    if [[ -d "${datasets_dir}" ]] && ! compgen -G "${datasets_dir}/helpers_cpp*.so" >/dev/null; then
        echo "[setup] building Megatron helpers_cpp in ${datasets_dir} ..."
        make -C "${datasets_dir}" -j"$(nproc)"
    fi
}
