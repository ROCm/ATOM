# Derived from SGLang (https://github.com/sgl-project/sglang), Apache-2.0; modified for ATOM.
# Copyright 2025 XunhaoLai. All rights reserved.

from collections import deque

import torch

_tma_keep_alive_buf = deque(maxlen=200)


def robust_allocator(size: int, alignment: int, stream: int | None = None):
    """Allocator for Triton TMA descriptors.

    We keep reference in deque to prevent GC from collecting the buffer.
    """
    tensor = torch.empty(size, device="cuda", dtype=torch.uint8)
    _tma_keep_alive_buf.append(tensor)
    return tensor
