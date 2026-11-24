#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Detect gfx942 GEMM cu_num for tune merge (source, do not execute).
#
# Priority: CU_NUM env > TARGET_CU env > docker aiter get_cu_num() > rocminfo.
# Override: CU_NUM=304 or TARGET_CU=304

dsv4_detect_gemm_cu() {
    local _cu="" _gfx="" _image="${IMAGE:-lumen/dsv4-lumen:mi308x}"

    if [[ -n "${CU_NUM:-}" && "${CU_NUM}" != "0" ]]; then
        echo "${CU_NUM}"
        return 0
    fi
    if [[ -n "${TARGET_CU:-}" ]]; then
        echo "${TARGET_CU}"
        return 0
    fi

    if [[ -e /dev/kfd ]] && docker image inspect "${_image}" &>/dev/null; then
        _cu="$(docker run --rm \
            --device /dev/kfd \
            --device /dev/dri \
            --group-add video \
            --ipc=host \
            -v "${LUMEN_DIR:-/workspace/Lumen}:/workspace/Lumen" \
            -e AITER_ROOT=/opt/aiter \
            -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
            "${_image}" \
            python3 -c "from aiter.jit.utils.chip_info import get_cu_num; print(get_cu_num())" \
            2>/dev/null | tail -1)" || true
        if [[ "${_cu}" =~ ^[0-9]+$ ]]; then
            echo "${_cu}"
            return 0
        fi
    fi

    if command -v rocminfo &>/dev/null; then
        _cu="$(rocminfo 2>/dev/null | awk '
            /Device Type/ && /GPU/ { in_gpu=1; next }
            in_gpu && /Compute Unit/ {
                gsub(/[^0-9]/, "", $NF);
                print $NF;
                exit
            }
        ')" || true
        if [[ "${_cu}" =~ ^[0-9]+$ ]]; then
            echo "${_cu}"
            return 0
        fi
    fi

    echo "[dsv4_gemm_detect_cu] ERROR: cannot detect cu_num (set CU_NUM or TARGET_CU)" >&2
    return 1
}

dsv4_gemm_overlay_path() {
    local _cu="${1:?cu_num}"
    local _dir="${GEMM_TUNE_DIR:-${LUMEN_DIR}/examples/dsv4/.gemm_tune}"
    if [[ -n "${TUNE_OUTPUT:-}" ]]; then
        echo "${TUNE_OUTPUT}"
    else
        echo "${_dir}/dsv4_bf16_tuned_gfx942_cu${_cu}.csv"
    fi
}

dsv4_gemm_tune_input_container() {
    local _input="${1:?tune_input_host_path}"
    if [[ "${_input}" == "${LUMEN_DIR}/"* ]]; then
        echo "/workspace/Lumen/${_input#"${LUMEN_DIR}/"}"
    else
        echo "${_input}"
    fi
}
