#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Backward-compatible alias — auto-detects cu_num (MI308X/MI300X).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_tune_dsv4_bf16_gemm.sh" "$@"
