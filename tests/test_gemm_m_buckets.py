# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The prefill GEMM warmup must reach every AITER tuned-GEMM row once.

AITER rounds a GEMM's M up to a fixed bucket before its tuned-config lookup
and JIT-compiles the chosen kernel on first use. A warmup size that misses a
bucket leaves a 10-30 s stall for serving; two sizes in one bucket only cost
startup time.
"""

import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_engine.model_runner import gemm_m_buckets


def _get_padded_m_gl0(m: int) -> int:
    # AITER csrc/py_itfs_cu/gemm_common.cu, getPaddedM(M, N, K, gl=0).
    if m <= 256:
        return (m + 15) // 16 * 16
    if m <= 1024:
        return (m + 31) // 32 * 32
    if m <= 4096:
        return (m + 63) // 64 * 64
    return (m + 127) // 128 * 128


@pytest.mark.parametrize("max_m", [1, 10, 16, 255, 1000, 4096, 5000, 8192, 16384])
def test_buckets_reach_every_padded_m_once(max_m):
    buckets = gemm_m_buckets(max_m)
    assert buckets == sorted(set(buckets))
    assert buckets[-1] == max_m
    # Below 16 the tables key exact rows at 1, 2, 4, 8; from 16 up every
    # bucket but the closing max_m is its own padded M.
    assert [b for b in buckets if b < 16 and b != max_m] == [
        m for m in (1, 2, 4, 8) if m < max_m
    ]
    assert all(_get_padded_m_gl0(b) == b for b in buckets[:-1] if b >= 16)
    reachable = {_get_padded_m_gl0(m) for m in range(16, max_m + 1)}
    assert reachable <= {_get_padded_m_gl0(b) for b in buckets if b >= 16}


def test_bucket_count_at_8192():
    # 1/2/4/8, then 16 below 256, 24 to 1024, 48 to 4096, 32 to 8192.
    assert len(gemm_m_buckets(8192)) == 124
