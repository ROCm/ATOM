# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which V4.1 forwards the hi hook serves (``ATOM_DSV41_MONO_HI``), and its installation.

``install_hi`` is ``dispatch.install_mono_decode``'s hi branch: it builds the hook (``hi_runner``) and
returns the model wrapped as ``HiDecodeModel`` (and the DSpark draft model as ``HiDraftModel``), the
classes of ``dispatch`` with the hook's routing on top; a loaded model it does not serve raises
``MonoUnsupported``, and under "auto" the deployment takes ATOM's mono decode unchanged. Nothing here is imported
unless ``hi_gate.decide`` says yes.

- ``HiDecodeModel``: the forwards the builder's routing decision gives the hook (``AttentionMetaData.served_by_round``,
  ``round.HiRound.service``) run every decoder layer on hi (``HiDecodeModel._hi_forward``); every other forward takes
  ATOM's path as before (a decode step ATOM's K1 / K2, their runner created on the first such step only) and
  invalidates the hook's incremental import for its requests. Its head output (``compute_head``): a served verify
  step's output ids and verdict from the head and the post-verify inside the forward, which the runner takes instead
  of the logits where they stand for the step.
- ``HiDraftModel``: the DSpark draft's block draft on the hi path (``block_backbone``); its context-KV write on a
  served step is the target forward's (the proposer skips ATOM's on such a step).
- the decode round (``round.HiRound``) for V4.1's metadata builder (``use_decode_round``): the routing decision, the
  device round, the commit; ATOM's Engram staging skipped on served steps (``HiLayers.use_host_tables``).
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from atom.models.deepseek_v41.mono import hi_runner
from atom.models.deepseek_v41.mono import round as hi_round
from atom.models.deepseek_v41.mono.dispatch import (
    MonoDecodeModel,
    MonoDraftModel,
    step_supported,
)
from atom.mono.runtime.consensus import MonoUnsupported, tp_uniform_local_validation
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
        # set again by a step whose post-verify ran (``compute_head``)
        hook.verified = 0
        fwd = get_forward_context()
        metadata = fwd.attn_metadata
        if metadata is not None and metadata.served_by_round:
            # the builder laid the step out for the hi path (Engram staging and the DSpark proposer follow the same
            # decision): its forward must be one the hi layers run
            if not supported:
                raise RuntimeError(
                    "V4.1 mono hi: a step routed to the hi path whose forward is not a mono decode step "
                    f"({rows} rows, inputs_embeds {inputs_embeds is not None}, ubatches "
                    f"{fwd.ubatch_slices is not None})"
                )
            return self._hi_forward(input_ids, positions)
        # it writes its own requests' caches: their request slots import in full next
        step = getattr(metadata, "step", None)
        ctx = fwd.context
        dummy = ctx is None or bool(getattr(ctx, "is_dummy_run", False))
        hook.invalidate(step, dummy=dummy)
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
        # (``lean_host``). The begin-step expands ATOM's embedding rows into layer 0's streams.
        hidden_in = model.embed(input_ids)
        model.begin_forward(hidden_in.view(1, rows, -1), None)
        # Engram staging's side-stream all-gather spins on CUs until every rank joins it: beside a kernel that needs
        # every CU resident it deadlocks, so it ends before the first one
        rows_staged = metadata.engram_embeddings
        if getattr(rows_staged, "stage", None) is not None:
            rows_staged.join()
        # the embedding streams in layer 0's input, the cache import, the Engram token-only halves
        hook.begin_step(
            cache,
            step,
            rows,
            hidden_in.view(rows, -1),
            staged=rows_staged,
            input_ids=input_ids,
        )
        for spec in model.topology:
            hook.layer(spec.layer_id, rows)
        # the cache export, the DSpark taps, ATOM's collapse into this forward's output rows; the draft's context
        # write and the post-verify
        hook.end_step(cache, step, rows)
        hidden = hook.collapse_view(rows).view(1, rows, -1)
        # (every rank holds every row: the hi runtime's layers are not token parallel)
        model.end_forward(hidden, None)
        return hidden.squeeze(0)

    def compute_head(self, hidden_states):
        """The head output of the forward that just ran (``ModelRunner._head_output``, recorded with a captured one):
        on a step the hook served, its in-graph verify's output ids and verdict (``round.post_verify``); else None
        (the runner computes the logits)."""
        return self._hi.head_output(hidden_states.shape[0])


class HiDraftModel(MonoDraftModel):
    """``MonoDraftModel`` with the hi hook's block draft (module docstring): up to ``HiLayers.dr_bmax`` requests run on
    the hi path. (The context-KV write of a served step is the hook's ``CTX`` launch, inside the target graph.)
    """

    def __init__(self, draft: nn.Module, dp_ranks: int, hook) -> None:
        super().__init__(draft, dp_ranks)
        self.hi = hook

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
                # ATOM's embedding rows of the anchors (as its own block draft looks them up)
                self.model.embed(input_ids.view(-1)),
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


def _tp() -> tuple:
    """(the TP group's CPU group, its size, this rank)."""
    from aiter.dist.parallel_state import get_tp_group

    tp = get_tp_group()
    return tp.cpu_group, tp.world_size, tp.rank_in_group


def install_hi(
    model: nn.Module, atom_config, drafter, metadata_builder, dp_ranks: int
) -> nn.Module:
    """``model`` wrapped for the hi hook (and ``drafter``'s draft model for its draft), on a deployment the hi path
    serves (``hi_gate.decide``). Collective (the agreement, the hook's peer groups): every rank calls it at the same
    point. First every refusal this rank's loaded model gives (``hi_runner.check``: its weights, Engram host shards),
    before anything collective or allocated, its verdict agreed over the TP
    group (``consensus.tp_uniform_local_validation``): every rank builds, or every rank raises ``MonoUnsupported``
    naming the ranks that refused (``dispatch.install_mono_decode`` falls back under "auto") with nothing built, the
    draft model left as it was."""
    group, world, rank = _tp()
    draft = drafter.model
    # whatever this rank's check raises is its refusal: a rank that alone raised would leave the others waiting in
    # the build's first collective
    refused = None
    try:
        hi_runner.check(model, atom_config, metadata_builder.engram, draft, world, rank)
    except Exception as error:  # noqa: BLE001 - any error is this rank's refusal
        refused = error
    try:
        tp_uniform_local_validation(
            refused, group=group, world_size=world, context="refused at install"
        )
    except MonoUnsupported as agreed:
        raise agreed from refused
    try:
        return _install_hi(model, atom_config, drafter, metadata_builder, dp_ranks)
    except MonoUnsupported:
        drafter.model = draft
        raise


def _install_hi(
    model, atom_config, drafter, metadata_builder, dp_ranks: int
) -> nn.Module:
    engram = metadata_builder.engram
    # the hi runtime's layers for every layer: weights (packed from the target's and the draft's loaded tensors),
    # kernels and peer groups bound here, before the KV cache is sized (the Engram table lookup's hash constants from
    # ATOM's own hash mapping)
    hook = hi_runner.install(model, atom_config, engram.mapping, drafter.model)
    hook.runner = metadata_builder.model_runner
    if hook.eng:
        # the Engram lookup from ATOM's host shards, its staging skipped on served steps
        hook.use_host_tables(engram.host, engram.host.overlap)
    # the draft's context write and block draft on the hi path
    drafter.model = HiDraftModel(drafter.model, dp_ranks, hook)
    hook.bind_draft(drafter)
    # the round's host work at the builder's points (``round.HiRound``): the routing decision with lean host prep on the
    # steps it serves, the device-side round advance on top of it, the verdict / Engram commit inside the target graph
    metadata_builder.use_decode_round(hi_round.install(hook, metadata_builder))
    return HiDecodeModel(
        model,
        drafter,
        atom_config.max_model_len,
        atom_config.index_cache_dtype == "fp4",
        dp_ranks,
        hook,
    )
