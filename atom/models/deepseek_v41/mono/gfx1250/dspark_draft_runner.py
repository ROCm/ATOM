# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One TP rank's DSpark draft backbone on the mono kernels.

The draft's three stages are V4.1 blocks with window-only attention: the same
K1 / K2 as the target's layers (``runner``), S = requests x block width rows,
the draft's routing (``dspark_n_routed_experts`` / ``dspark_num_experts_per_tok``)
and no indexer. Two things differ from a target step, both in the step's
integers (``step_meta.write_draft_step_meta``), not in a kernel:

- K1 writes the block's keys into the stage's ring at the block's own positions
  (anchor + 1 ..). The ring holds window + width slots, so they never overwrite
  the window; only the draft reads these rings, and the target's context write
  replaces every position it commits.
- Every row's keys are its request's window up to the anchor and its whole
  block, both ways: one span, the same for every row of the block.

The key order is the paged decode's (window, then block), not the draft's
Triton ``sparse_attn`` indices order, so the attention's sums are not the
original draft's bit for bit.
"""

from copy import copy

import torch

from atom.model_ops.deepseek_v41.dspark import build_block_state
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.models.deepseek_v41.mono.gfx1250.dspark_config import MAX_REQUESTS
from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
from atom.models.deepseek_v41.mono.gfx1250.dspark_runner import (
    V41MonoDecodeRunner,
    _ROW_BUFFERS,
    _rows,
)
from atom.models.deepseek_v41.mono.step_meta import write_draft_step_meta
from atom.models.deepseek_v41.mono.gfx1250.dspark_weights import bind_layer
from atom.mono.runtime.consensus import MonoUnsupported
from atom.utils import envs
from atom.utils.forward_context import get_forward_context


class DraftMonoRunner(V41MonoDecodeRunner):
    """The draft (``models.deepseek_v41.dspark.DSparkDraftModel``) backbone of
    up to ``MAX_REQUESTS`` requests, a block of ``width`` rows each, bound (and
    agreed on) like the target's runner."""

    # CT1-M3D: the widest draft step (8 requests x 5 = 40 rows) runs the target's
    # wide 48-row K1/K2 (row-tile mHC, P1A attention, native front/UG/down) with
    # rows 40..47 replicas of row 39; narrower steps keep the original kernels.
    row_tile_mhc = True
    WIDE_FROM = 40
    WIDE_ROWS = 48

    def __init__(self, draft) -> None:
        config = draft.config
        self.width = draft.block_size
        self.mhtjstest_widths = tuple(self.width * n for n in range(1, 9))
        assert max(self.mhtjstest_widths) == self.WIDE_FROM
        self.max_rows = self.WIDE_ROWS
        self._wide_step = False
        self.experts = config.dspark_n_routed_experts
        self.topk = config.dspark_num_experts_per_tok
        self.ug_groups = 1
        # no indexer: the K2a build without it
        self.index_bound_max = 0
        self._bind_agreed(lambda: self._bind_draft(draft))

    def _bind_draft(self, draft) -> None:
        self.model = draft
        self._bind_rank()
        config = copy(draft.config)
        config.n_routed_experts = self.experts
        config.num_experts_per_tok = self.topk
        self.layers = [
            bind_layer(block, config, self.dims, self.moe_tp) for block in draft.mtp
        ]
        for table in (draft.rope.cos_cache, draft.rope.sin_cache):
            if table.dtype != torch.bfloat16 or table.shape[-1] != k1.HALF:
                raise MonoUnsupported(f"draft rope table {table.dtype} {table.shape}")
        self.timelines = {}
        self.timeline = False
        self.check = False
        self.split_k2 = False
        self.debug = envs.ATOM_MONO_DEBUG
        self.selecting = []
        self._allocate(config)
        self.positions = torch.empty(
            self.max_rows, device=self.scratch.device, dtype=torch.int64
        )

    def _layer_keys(self, s: int) -> list:
        # stage 0 starts from the block's embeddings: no owed post to fold
        return [
            k1.AttnPreBuild(
                tokens=s,
                fold=i != 0,
                tp=self.tp,
                timeline=self.timeline,
                diag_off=self.diag_off,
                kv_fp8=self.kv_fp8,
            )
            for i in range(len(self.model.mtp))
        ]

    def _rows_build(self, s: int):
        if s != self.WIDE_FROM:
            # original draft kernels (no row-tile mHC workspace in their ABI)
            self.row_tile_mhc = False
            try:
                return super()._rows_build(s)
            finally:
                del self.row_tile_mhc
        # wide build: native UG's 2x2x2 wave split needs ug_groups 2 (the same
        # outputs as 1, ``fine_ug_below``)
        ug = self.ug_groups
        self.ug_groups = 2
        try:
            return super()._rows_build(self.WIDE_ROWS)
        finally:
            self.ug_groups = ug

    def _timeline_ptr(self, kernel, layer_id) -> int:
        if self._wide_step and kernel in ("k1", "k2"):
            return self.mhc_sq.data_ptr()
        timeline = self.timelines.get(kernel)
        return timeline.ptr(layer_id) if timeline is not None else 0

    def _tok_par(self, s: int) -> bool:
        # the backbone gathers no rows: every K1 and K2 runs on all of them
        return False

    def _rope_positions(self, cache, step) -> torch.Tensor:
        return self.positions

    def backbone(self, anchor_ids, anchor_positions):
        """``DSparkDraftModel.block_backbone`` of ``anchor_ids``' requests at the
        full block width: (the last stage's normed hidden [B width, dim], the
        pre-norm hidden [B, width, dim])."""
        draft = self.model
        metadata = get_forward_context().attn_metadata
        cache = metadata.cache
        if cache.packed:
            raise MonoUnsupported("packed KV cache")
        requests = anchor_ids.numel()
        rows = requests * self.width
        self._use_rows(rows)
        wide = rows == self.WIDE_FROM
        self._wide_step = wide
        step_rows = self.WIDE_ROWS if wide else rows
        meta_ring, meta_keys = self.ring_rel, self.key_meta
        if wide:
            for name, dim in _ROW_BUFFERS:
                setattr(self, name, _rows(self._full[name], dim, step_rows))
        self.mailboxes.begin_step()
        residual, pre_mix = build_block_state(
            draft.embed(anchor_ids),
            draft._noise_embedding(),
            self.width,
            draft.config.hc_mult,
        )
        # the requests' blocks as one step's rows, request after request
        residual = residual.flatten(0, 1).unsqueeze(0)
        pre_mix = pre_mix.flatten(0, 1).unsqueeze(0)
        if wide:
            pad = step_rows - rows

            def replicate(x):
                return torch.cat([x, x[:, -1:].expand(1, pad, *x.shape[2:])], 1)

            residual, pre_mix = replicate(residual), replicate(pre_mix)
        state = SinglePassHCState(residual, pre_mix)
        write_draft_step_meta(
            anchor_positions,
            metadata.state_slot_out,
            cache.num_slots,
            cache.geometry,
            cache.geometry.window(0, cache.num_pages),
            self.positions,
            meta_ring,
            meta_keys,
        )
        if wide:
            # pad rows repeat row rows-1: same position, ring slot and keys, so
            # their KV writes are the same bytes and their routing adds no expert
            self.positions[rows:step_rows].copy_(self.positions[rows - 1 : rows].expand(pad))
            self.ring_rel[rows:].copy_(self.ring_rel[rows - 1 : rows].expand(pad, *self.ring_rel.shape[1:]))
            self.key_meta[rows:].copy_(self.key_meta[rows - 1 : rows].expand(pad, *self.key_meta.shape[1:]))
        for block, weights, key in zip(draft.mtp, self.layers, self.build.keys):
            spec = block.attn.spec
            residual, gates = self._front(
                spec.layer_id, key, weights, state, cache, None, draft.rope
            )
            self._layer_post(
                spec, block, weights, residual, gates, None, cache, draft.rope
            )
            pre, post, comb = self.gates_ffn
            state = SinglePassHCState(self.res_ffn, pre, self.ffn_out, post, comb)
        hidden = state.collapse()
        if wide:
            hidden = hidden[:, :rows]
        hidden = hidden.reshape(requests, self.width, -1)
        self._raise_on_given_up_waits(step_rows)
        return draft.mtp[-1].norm(hidden).flatten(0, 1), hidden
