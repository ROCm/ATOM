# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The drafter's aux hidden-state buffers, for the models whose mono path
has to fill them itself.

Shared because the staleness is: a drafter taps the target's residual stream
with forward hooks on decoder layers, and a mono step does not call those
layers through ``nn.Module.__call__``, so the hooks never fire and the drafter
drafts from whatever the last non-mono step left behind. Every mono model that
runs under a drafter has to write these buffers itself -- V4.1 from its
kernels, K3 from its runner's per-layer seam."""

from dataclasses import dataclass

import torch

from atom.mono.runtime.consensus import MonoUnsupported


@dataclass(frozen=True)
class AuxTaps:
    """The DSpark drafter's aux hidden-state buffers, one per tapped layer.

    On a production step the drafter's own hooks fill them from target layers
    ``layer_ids``; what each one holds is that target's business (V4.1 the mean
    over the hc copies, K3 the settled ``prefix_sum`` its layer reconstructs).
    A mono step bypasses those modules, so it owes the same tensor here.

    A ``-1`` entry taps the embedding rather than a decoder layer. That module
    IS still called on a mono step, so its hook fires and ``write`` simply
    never matches it.
    """

    layer_ids: tuple[int, ...]
    buffers: tuple[torch.Tensor, ...]

    def write(self, layer_id: int, tap: torch.Tensor | None) -> None:
        """Record ``tap`` as layer ``layer_id``'s aux hidden state, if it is
        tapped at all.

        ``tap`` is ``None`` where the layer has nothing settled to report; the
        drafter's own hook treats that the same way, by leaving the buffer as
        it was. Rows start at 0 like the hook's (its only offset is the TBO
        one, and every mono path refuses TBO), and the write is in place so a
        captured graph keeps replaying into the same buffer."""
        if tap is None or layer_id not in self.layer_ids:
            return
        self.buffers[self.layer_ids.index(layer_id)][: tap.shape[0]].copy_(tap)


def bind_aux_taps(drafter, hidden_size: int) -> AuxTaps:
    # the layer ids the drafter armed its hooks from (DSparkProposer._aux_capture_spec)
    draft_config = drafter.speculative_config.draft_model_hf_config
    layer_ids = tuple(
        int(i) for i in getattr(draft_config, "dspark_target_layer_ids", ())
    )
    buffers = tuple(getattr(drafter, "_aux_buffers", ()))
    if not buffers:
        raise MonoUnsupported("the drafter captures no aux hidden states")
    if len(layer_ids) != len(buffers):
        raise MonoUnsupported(
            f"{len(buffers)} aux buffers for target layers {layer_ids}"
        )
    for buf in buffers:
        if buf.dim() != 2 or buf.shape[1] != hidden_size or not buf.is_contiguous():
            raise MonoUnsupported(f"aux buffer {tuple(buf.shape)} {buf.stride()}")
        if buf.dtype != torch.bfloat16:
            raise MonoUnsupported(f"aux buffer {buf.dtype}")
    return AuxTaps(layer_ids, buffers)
