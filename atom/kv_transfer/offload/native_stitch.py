# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Optional native CPU stitch for LMCache chunk buffers.

This is deliberately a small C++ CPU extension, not a HIP op: current profiles
show H2D at tens of milliseconds, while Python/Torch host repacking takes more
than a second for MiniMax-M2.5 32K.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.cpp_extension import load


_EXT = None


def _load_ext():
    global _EXT
    if _EXT is None:
        src = Path(__file__).with_name("native_stitch.cpp")
        _EXT = load(
            name="atom_lmcache_native_stitch",
            sources=[str(src)],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXT


def load_extension() -> None:
    _load_ext()


def stitch_chunk_buffers(dst, chunk_buffers, chunk_block_counts, seg_block_bytes) -> None:
    _load_ext().stitch_chunk_buffers(
        dst,
        chunk_buffers,
        [int(x) for x in chunk_block_counts],
        [int(x) for x in seg_block_bytes],
    )
