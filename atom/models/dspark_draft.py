# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared surface for the DSpark block drafters.

Every DSpark flavor -- the inline DeepSeek-V4 draft that ships inside the target
checkpoint (``deepseek_v4_dspark.py``), the standalone Kimi-K3 draft
(``kimi_k3_dspark.py``), whatever lands next -- owes ``DSparkProposer`` the same
two things: absorb the target's context into the draft's own KV, then draft a
whole block in one backbone pass.

Absorbing the context has the same shape in every flavor, and that shape is the
only thing implemented here:

    project the concatenated aux hidden states ONCE, then hand that single
    tensor to every layer, each of which recomputes KV from it using its OWN
    weights (the q half of the fused projection is computed and dropped).

Below that line the flavors diverge at every step, so the per-layer write stays
in the subclass. V4 writes a 512-wide MQA row into a private rolling window
addressed by absolute position, with the RMSNorm covering the RoPE lanes and
fp8 QAT on the rest; K3 writes a 576-wide MLA latent into a paged sibling pool
addressed by slot mapping, norms only the compressed half and stays bf16. The
steps line up one for one and not one of them is the same computation -- which
is why the seam is here, above the per-layer write, and not inside it.
"""

import torch
from torch import nn


class DSparkDraftModel(nn.Module):
    """What a DSpark draft model must provide to ``DSparkProposer``."""

    def write_context_kv(
        self,
        aux_concat: torch.Tensor,  # [N, target_hidden * num_target_layers]
        positions: torch.Tensor,  # [N] absolute positions
    ) -> None:
        """Absorb one target forward's hidden states into this draft's KV.

        Where those rows land -- a private rolling window, a paged pool -- and
        how they are addressed is the layer's business, not this method's. A
        layer reads its own addressing (``cu_seqlens_q``, ``slot_mapping``, ...)
        off the live forward context rather than taking it as an argument,
        because the two flavors need different pieces of the same metadata.

        That makes the call site load-bearing: this must run while
        ``attn_metadata`` still describes the TARGET forward that produced
        ``aux_concat``. A drafter that has already retargeted the metadata to
        its own block pass (see ``DSparkProposer._propose_with_draft``) is
        past the point where these rows can be placed.
        """
        from atom.utils.forward_context import get_forward_context

        # warmup_model() runs at the end of ModelRunner.__init__, BEFORE
        # allocate_kv_cache(), so on a dummy run there is nothing to write into:
        # every layer's cache is still the empty init tensor and the store would
        # abort inside the cache kernels. Skipping is safe -- warmup discards
        # the draft's output. The block forward needs no such guard; it is also
        # the memory-profiling pass, so it must still run.
        if get_forward_context().context.is_dummy_run:
            return
        ctx_hidden = self.project_context(aux_concat)
        for layer in self.context_layers:
            layer.write_context_kv(ctx_hidden, positions)

    def project_context(self, aux_concat: torch.Tensor) -> torch.Tensor:
        """Fuse the concatenated target aux hidden states into one [N, hidden]
        context tensor, shared by every layer."""
        raise NotImplementedError

    @property
    def context_layers(self):
        """The layers that hold context KV, in order."""
        raise NotImplementedError

    def forward_spec(
        self,
        input_ids: torch.Tensor,  # [B] verified anchor token per request
        positions: torch.Tensor,  # [B*T] block absolute positions
        num_draft: "int | None" = None,
    ):
        """One DSpark block: parallel backbone pass + sequential Markov
        sampling. Returns ``(draft_token_ids [B, T], confidence | None)``."""
        raise NotImplementedError
