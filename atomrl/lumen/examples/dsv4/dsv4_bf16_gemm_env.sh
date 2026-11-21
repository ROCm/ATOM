#!/usr/bin/env bash
# DSV4 BF16 tuned GEMM defaults — sourced from dsv4_paths.sh.
#
# CSV strategy: shared gfx942 runtime file configs/dsv4_bf16_tuned_gemm_mi308x.csv
# holds BOTH MI308X (cu_num=80) and MI300X (cu_num=304) rows. AITER lookup uses
# runtime get_cu_num(), so platforms never conflict. Tune intermediates under
# .gemm_tune/ (gitignored):
#   bash examples/dsv4/run_tune_dsv4_bf16_gemm.sh          # auto cu_num
#   bash examples/dsv4/run_tune_dsv4_bf16_gemm_flash2node.sh
#
# Optional override: CU_NUM=304 bash examples/dsv4/run_tune_dsv4_bf16_gemm.sh
#
# Env:
#   LUMEN_DSV4_GEMM_BF16   1 (default) | 0 — route BF16 linear through gemm_bf16
#   AITER_CONFIG_GEMM_BF16 path to tuned CSV (host path; remapped in Docker)

: "${LUMEN_DSV4_GEMM_BF16:=1}"
export LUMEN_DSV4_GEMM_BF16

if [[ -z "${SCRIPT_DIR:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [[ -z "${AITER_CONFIG_GEMM_BF16:-}" ]]; then
    _dsv4_gemm_csv="${SCRIPT_DIR}/configs/dsv4_bf16_tuned_gemm_mi308x.csv"
    if [[ -f "${_dsv4_gemm_csv}" ]]; then
        AITER_CONFIG_GEMM_BF16="${_dsv4_gemm_csv}"
    fi
fi
export AITER_CONFIG_GEMM_BF16

# Avoid per-GEMM CSV lookup logging overhead during training.
: "${AITER_LOG_TUNED_CONFIG:=0}"
: "${AITER_LOG_MORE:=0}"
export AITER_LOG_TUNED_CONFIG AITER_LOG_MORE

unset _dsv4_gemm_csv

# Launch banner helper (host paths).
dsv4_print_gemm_env() {
    local _csv="${AITER_CONFIG_GEMM_BF16:-}"
    if [[ -n "${_csv}" ]]; then
        echo "  GEMM BF16  : LUMEN_DSV4_GEMM_BF16=${LUMEN_DSV4_GEMM_BF16} CSV=${_csv}"
    else
        echo "  GEMM BF16  : LUMEN_DSV4_GEMM_BF16=${LUMEN_DSV4_GEMM_BF16} (no tuned CSV — AITER default)"
    fi
}
