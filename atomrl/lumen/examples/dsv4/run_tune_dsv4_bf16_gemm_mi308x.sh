#!/usr/bin/env bash
# Backward-compatible alias — auto-detects cu_num (MI308X/MI300X).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_tune_dsv4_bf16_gemm.sh" "$@"
