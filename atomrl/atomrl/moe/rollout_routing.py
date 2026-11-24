# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Rollout Routing Replay (R3): train on the experts the rollout actually used.

A routed-expert model picks top-k of N experts by comparing router logits, which
is a *discrete* decision. Any numerical disagreement between the training forward
and the rollout engine can flip which experts a token is sent to, and a flipped
expert moves that token's log-prob by ~0.1 rather than ~1e-3. Measured on
Qwen3-30B-A3B with identical weights: 7.6% of tokens exceed |delta| 0.1 and carry
57% of the total deviation, while only ~4% of tokens route identically through all
48 layers.

Raising the router to fp32 does not fix this -- it removes BF16 ties, but the
flips are driven by the router's *input*, which differs at BF16 scale at every
layer because the two stacks run different attention and MLP kernels. Replaying
the rollout's own selections is what takes the flip rate to zero.

The replay substitutes only the discrete choice. Weights still come from
``gather`` on the live probabilities, so gradients flow to the router exactly as
before; overwriting the logits instead (as a naive implementation would) turns
the router into a constant and severs them.

Lookup is by LAYER INDEX, not by call order. HF gradient checkpointing recomputes
segments during backward in reverse layer order, so a cursor that advances per
call would read another layer's routing on the backward pass.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
from typing_extensions import Self

_LAYER_RE = re.compile(r"layers\.(\d+)\.")

# Set for the duration of one packed micro-batch's forward AND backward, so the
# checkpointing recompute sees the same routing the forward used.
#
# A module global, NOT threading.local(): non-reentrant checkpointing recomputes
# the forward on an autograd worker thread where thread-local state is invisible,
# so the recompute would silently skip the replay and build a different graph
# ("a different number of tensors was saved during the original forward and
# recomputation"). Safe because training is single-threaded per process. This is
# the same reasoning as the packing context in engine/training/packing.py.
_routing: _Routing | None = None


def _current() -> _Routing | None:
    return _routing


class _Routing:
    __slots__ = ("indices", "valid")

    def __init__(self, indices: torch.Tensor, valid: torch.Tensor) -> None:
        self.indices = indices  # [total_tokens, n_layers, top_k] long
        self.valid = valid  # [total_tokens] bool

    def for_layer(self, layer_idx: int, n_tokens: int):
        if layer_idx is None or layer_idx >= self.indices.shape[1]:
            return None, None
        if self.indices.shape[0] != n_tokens:
            raise RuntimeError(
                "R3 routing/token count mismatch: routing has "
                f"{self.indices.shape[0]} rows but the router saw {n_tokens} tokens. "
                "Replaying misaligned routing would corrupt training silently."
            )
        return self.indices[:, layer_idx], self.valid


class RoutingReplayContext:
    """Install packed rollout routing for one micro-batch's forward+backward."""

    def __init__(
        self, indices: torch.Tensor | None, valid: torch.Tensor | None
    ) -> None:
        self._new = _Routing(indices, valid) if indices is not None else None
        self._prev = None

    def __enter__(self) -> Self:
        global _routing
        self._prev = _routing
        _routing = self._new
        return self

    def __exit__(self, *exc) -> None:
        global _routing
        _routing = self._prev


def layer_index_of(name: str) -> int | None:
    """Layer number out of a module path like ``model.layers.7.mlp.gate``."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def apply_replay(
    router_probs: torch.Tensor, indices: torch.Tensor, layer_idx: int | None
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Swap in the rollout's expert ids for this layer; regather the weights.

    Returns ``(top_value, indices)`` or ``None`` when no routing is installed.
    Called from the patched router forward in ``router_precision``.
    """
    cur = _current()
    if cur is None:
        return None
    injected, valid = cur.for_layer(layer_idx, router_probs.shape[0])
    if injected is None:
        return None
    # Rows without rollout routing (the final token of each sequence, which the
    # engine never ran a forward for) keep the freshly computed selection.
    merged = torch.where(valid.unsqueeze(1), injected, indices)
    return router_probs.gather(1, merged), merged


def pack_rollout_routing(
    rows: list[Any],
    seq_lens: torch.Tensor,
    n_layers: int,
    top_k: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Flatten per-sequence rollout routing into the packed token order.

    ``rows[i]`` is the engine's ``[L_i - 1, n_layers, top_k]`` uint8 array for
    sequence ``i`` (row t is the forward at position t). ``pack_sequences``
    concatenates each sequence's ``L_i`` real tokens in row order, so this lays
    the routing out the same way and marks the one uncovered position per
    sequence -- the last token -- invalid.
    """
    lens = [int(v) for v in seq_lens.tolist()]
    total = sum(lens)
    if total == 0 or not rows:
        return None, None

    indices = torch.zeros((total, n_layers, top_k), dtype=torch.long)
    valid = torch.zeros(total, dtype=torch.bool)
    off = 0
    for i, sl in enumerate(lens):
        r = rows[i] if i < len(rows) else None
        if r is not None and sl > 1:
            n = min(int(np.shape(r)[0]), sl - 1)
            if n > 0:
                block = torch.from_numpy(np.ascontiguousarray(r[:n])).long()
                if block.shape[1:] != (n_layers, top_k):
                    raise RuntimeError(
                        f"R3 routing for row {i} has shape {tuple(block.shape)}, "
                        f"expected (*, {n_layers}, {top_k})"
                    )
                indices[off : off + n] = block
                valid[off : off + n] = True
        off += sl
    return indices.to(device), valid.to(device)
