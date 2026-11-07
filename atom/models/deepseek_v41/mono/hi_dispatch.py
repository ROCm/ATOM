# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which V4.1 forwards the hi hook serves (``ATOM_DSV41_MONO_HI=1``), and its installation.

``install_hi`` is ``dispatch.install_mono_decode``'s hi branch: it builds the hook (``hi_runner``) and
returns the model wrapped as ``HiDecodeModel`` (and the DSpark draft model as ``HiDraftModel``), the
classes of ``dispatch`` with the hook's routing on top; or None, and the deployment takes ATOM's mono decode
unchanged. Nothing here is imported without the flag.

- ``HiDecodeModel``: the forwards the hook serves (``HiLayers.serves_step``) run every decoder layer on
  hi (``HiDecodeModel._hi_forward``); every other forward takes ATOM's path as before (a decode step ATOM's K1 / K2,
  their runner created on the first such step only) and invalidates the hook's incremental import for its requests. Its
  ``verify_head`` (``round.verify_head.HiVerifyHead``, ``ModelRunner``'s model-side verify head): a served verify
  step's ids come from the head inside the forward instead of the logits.
- ``HiDraftModel``: the DSpark draft's context-KV write and its block draft on the hi path on the steps the hook
  served (``context_kv_served``, ``block_backbone``), ATOM's draft otherwise.
- the host work of the decode round at the engine's explicit hooks (``round.HiRound``: the runner's through
  ``HiDecodeModel.decode_round``, the V4.1 builder's, the DSpark proposer's), the token embedding's shards read in
  place across the ranks (``embed_full``) and ATOM's Engram staging skipped on served steps (``HiLayers.use_host_tables``).
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from atom.models.deepseek_v41.mono import hi_runner
from atom.models.deepseek_v41.mono.round import embed_full
from atom.models.deepseek_v41.mono import round as hi_round
from atom.models.deepseek_v41.mono.dispatch import (
    MonoDecodeModel,
    MonoDraftModel,
    step_supported,
)
from atom.mono.runtime.consensus import MonoUnsupported
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

# ``HiDraftModel.block_backbone``'s result when the block draft ran on the hi path (``head_and_sample``)
_HI_DRAFT = object()


class HiDecodeModel(MonoDecodeModel):
    """``MonoDecodeModel`` with the hi hook (module docstring): a decode step the hook serves runs every decoder layer
    on the hi path (``_hi_forward``); any other decode step takes ATOM's K1 / K2 (``V41MonoDecodeRunner``, created on
    the first such step only: none at the served deployment, whose decode steps are all hi widths), and every other
    forward ATOM's model."""

    def __init__(
        self, model, drafter, max_model_len, index_fp4, dp_ranks, hook
    ) -> None:
        super().__init__(model, drafter, max_model_len, index_fp4, dp_ranks)
        self._hi = hook

    def forward(self, input_ids, positions, inputs_embeds=None):
        rows = input_ids.numel()
        supported = step_supported(input_ids, inputs_embeds, self._dp_ranks)
        hook = self._hi
        hook.head_rows = 0  # set again by a step whose verify head ran
        fwd = get_forward_context()
        step = getattr(fwd.attn_metadata, "step", None)
        if supported and hook.serves_step(rows, step):
            return self._hi_forward(input_ids, positions)
        # it writes its own requests' caches: their request slots import in full next
        ctx = fwd.context
        dummy = ctx is None or bool(getattr(ctx, "is_dummy_run", False))
        hook.invalidate(step, dummy=dummy)
        if hook.lean_host and getattr(step, "lean_host", False):
            raise RuntimeError(
                "V4.1 mono: a lean-host step is not served by the hi hook: "
                "its metadata has no plans / indptrs"
            )
        if supported and self._mono.ready(rows):
            return self._mono.runner.forward(input_ids, positions)
        if inputs_embeds is None:
            return self.model(input_ids, positions)
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def _hi_forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """A step the hook serves: every decoder layer on the hi path (one launch a layer, the bridge and the verify
        head: ``hi_runner``)."""
        hook, model = self._hi, self.model
        metadata = get_forward_context().attn_metadata
        step = metadata.step
        rows = input_ids.numel()
        cache = metadata.cache
        if cache.packed:
            raise MonoUnsupported("packed KV cache")
        hook.served_widths.add(rows)
        # Every layer runs on the hi path: ATOM's mailboxes and K1 / K2 step metadata have no reader on this step
        # (``lean_host``). The begin-step looks the rows up itself in the embedding's shards; the forward's boundary op,
        # which reads only the rows' count, gets a shape-only stand-in.
        model.begin_forward(hook.no_embedding[:, :rows], None)
        # Engram staging's side-stream all-gather spins on CUs until every rank joins it: beside a kernel that needs
        # every CU resident it deadlocks, so it ends before the first one
        rows_staged = metadata.engram_embeddings
        if getattr(rows_staged, "stage", None) is not None:
            rows_staged.join()
        # the embedding streams in layer 0's input, the cache import, the Engram token-only halves
        hook.begin_step(cache, step, rows, staged=rows_staged, input_ids=input_ids)
        for spec in model.topology:
            hook.layer(spec.layer_id, rows)
        # the cache export, the DSpark taps, ATOM's collapse into this forward's output rows; the draft's context
        # write and the post-verify
        hook.end_step(cache, step, rows)
        hidden = hook.collapse_view(rows).view(1, rows, -1)
        # (every rank holds every row: the hi runtime's layers are not token parallel)
        model.end_forward(hidden, None)
        return hidden.squeeze(0)

    @property
    def decode_round(self):
        """The decode round's host work at ``ModelRunner``'s hooks (``round.HiRound``)."""
        return self._hi.round

    @property
    def verify_head(self):
        """The model-side verify head ``ModelRunner`` takes a served verify's ids from (``round.verify_head``)."""
        return self._hi.verify_head


class HiDraftModel(MonoDraftModel):
    """``MonoDraftModel`` with the hi hook's draft (module docstring): on a forward the hook served,
    ``write_context_kv`` is a no-op (the hook's ``CTX`` launch wrote the rows inside the target graph,
    ``context_kv_served``), and the block draft of up to ``HiLayers.dr_bmax`` requests runs on the hi path.
    """

    def __init__(self, draft: nn.Module, dp_ranks: int, hook) -> None:
        super().__init__(draft, dp_ranks)
        self.hi = hook

    def context_kv_served(self) -> bool:
        """Whether the forward that just ran wrote the draft's context rows on the hi path (``compute_draft_kv`` then
        skips the taps' concat and the write)."""
        step = getattr(get_forward_context().attn_metadata, "step", None)
        return self.hi.ctx_served(step)

    def write_context_kv(self, aux_concat, positions):
        if self.context_kv_served():
            return None
        return self.model.write_context_kv(aux_concat, positions)

    def block_backbone(self, input_ids, positions, num_draft):
        hook = self.hi
        width = self.model.block_size if num_draft is None else num_draft
        requests = input_ids.numel()
        if hook.draft_serves(requests, width):
            # the block draft on the hi path: its ids are ``head_and_sample``'s
            metadata = get_forward_context().attn_metadata
            hook.draft_run(
                metadata.cache,
                input_ids,
                positions,
                metadata.state_slot_out[:requests],
                self.model._noise_embedding(),
            )
            return _HI_DRAFT
        return super().block_backbone(input_ids, positions, num_draft)

    def head_and_sample(self, out, anchor_ids, num_draft):
        if out is _HI_DRAFT:
            # a row of ids a request (``draft_run``)
            requests = self.hi.dr_requests
            return self.hi.mk_ids[:requests].view(requests, -1), None
        return self.model.head_and_sample(out, anchor_ids, num_draft)


def install_hi(
    model: nn.Module, atom_config, drafter, metadata_builder, dp_ranks: int
) -> nn.Module | None:
    """``model`` wrapped for the hi hook (and ``drafter``'s draft model for its draft), on a deployment the hi path
    serves (``hi_gate.decide``). Collective (the hook's peer groups): every rank calls it at the same point.
    """
    engram = metadata_builder.engram
    # the hi runtime's layers for every layer: weights (packed from the target's and the draft's loaded tensors), kernels
    # and peer groups bound here, before the KV cache is sized (the Engram table lookup's hash constants from ATOM's own
    # hash mapping)
    hook = hi_runner.install(model, atom_config, engram.mapping, drafter.model)
    runner = metadata_builder.model_runner
    hook.runner = runner
    # the token embedding's shards read in place from the rank holding a row (no all-reduce a lookup, no copy of
    # the table): the begin-step's and the draft's begin kernel's lookups
    hook.embed = embed_full.install(model, hook._cpu_group)
    if hook.eng:
        # the Engram lookup from ATOM's host shards, its staging skipped on served steps
        hook.use_host_tables(engram.host, engram.host.overlap)
    # the draft's context write and block draft on the hi path
    drafter.model = HiDraftModel(drafter.model, dp_ranks, hook)
    hook.bind_draft(drafter)
    # the round's host work at the engine's decode-round hooks: lean host prep on the steps it serves, the device-side
    # round advance on top of it, the verdict / Engram commit /
    # DSpark anchors inside the target graph, the packed publications reading their pinned arenas directly
    hi_round.install(hook, metadata_builder, runner, drafter)
    return HiDecodeModel(
        model,
        drafter,
        atom_config.max_model_len,
        atom_config.index_cache_dtype == "fp4",
        dp_ranks,
        hook,
    )
