# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen3-Next / Qwen3.5 attention output gate (and a row selection) in the
attention's own epilogue (ATOM_MTP_ATTN_GATE_IN_REDUCE).

The attention ends in aiter's unified-attention `reduce_segments` (decode, 3D
split-KV), and the layer then runs `o * sigmoid(gate)` as a kernel of its own
(`_gated_pointwise_kernel`), which reads and writes the whole attention output
once more. aiter's `unified_attention(output_gate=, output_rows=)` folds the gate
into the reduce's store, bitwise the separate kernel, and can keep only some rows
(the MTP drafter's step 0 reads one row a sequence after its attention).

The attention runs behind a custom op whose arguments are fixed, so the layer
hands the gate to it as a request in scope around the call: the unified-attention
decode path takes it and marks it applied; any other path leaves it, and the layer
applies the gate (and the selection) itself. A Python-side request is only seen
when the attention runs eagerly (or is captured into a CUDA graph) -- the MTP
drafter, which is not torch.compiled.
"""

from __future__ import annotations

import functools
from typing import Self

from torch import Tensor


class AttentionOutputEpilogue:
    """`out = attention(...) * sigmoid(gate)`, of rows `rows` only when given
    (`out[i]` is token `rows[i]`), for the next attention call in its scope.

    `gate` is `[tokens, heads * head_dim]` (may be a strided column slice).
    `applied` tells the caller whether the attention did it."""

    def __init__(self, gate: Tensor | None, rows: Tensor | None = None):
        self.gate = gate
        self.rows = rows
        self.applied = False
        self._prev = None

    def __enter__(self) -> Self:
        global _CURRENT
        self._prev, _CURRENT = _CURRENT, self
        return self

    def __exit__(self, *exc) -> None:
        global _CURRENT
        _CURRENT = self._prev


_CURRENT: AttentionOutputEpilogue | None = None


def current_attention_output_epilogue() -> AttentionOutputEpilogue | None:
    """The request in scope that no attention has applied yet, or None."""
    req = _CURRENT
    return None if req is None or req.applied else req


@functools.lru_cache(maxsize=1)
def unified_attention_takes_output_epilogue() -> bool:
    """Whether aiter's unified_attention has `output_gate` / `output_rows`."""
    import inspect

    from aiter.ops.triton.unified_attention import unified_attention

    params = inspect.signature(unified_attention).parameters
    return "output_gate" in params and "output_rows" in params


@functools.lru_cache(maxsize=1)
def unified_attention_takes_split_reduce() -> bool:
    """Whether aiter's unified_attention has `split_reduce`."""
    import inspect

    from aiter.ops.triton.unified_attention import unified_attention

    return "split_reduce" in inspect.signature(unified_attention).parameters


@functools.lru_cache(maxsize=1)
def unified_attention_takes_split_reduce_oneshot() -> bool:
    """Whether aiter's unified_attention has `split_reduce_oneshot`."""
    import inspect

    from aiter.ops.triton.unified_attention import unified_attention

    return "split_reduce_oneshot" in inspect.signature(unified_attention).parameters
