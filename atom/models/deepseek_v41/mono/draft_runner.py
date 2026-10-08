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
from atom.models.deepseek_v41.mono.config import MAX_REQUESTS
from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
from atom.models.deepseek_v41.mono.runner import V41MonoDecodeRunner
from atom.models.deepseek_v41.mono.step_meta import write_draft_step_meta
from atom.models.deepseek_v41.mono.weights import bind_layer
from atom.mono.runtime.consensus import MonoUnsupported
from atom.utils import envs
from atom.utils.forward_context import get_forward_context


class DraftMonoRunner(V41MonoDecodeRunner):
    """The draft (``models.deepseek_v41.dspark.DSparkDraftModel``) backbone of
    up to ``MAX_REQUESTS`` requests, a block of ``width`` rows each, bound (and
    agreed on) like the target's runner."""

    def __init__(self, draft) -> None:
        config = draft.config
        self.width = draft.block_size
        self.max_rows = self.width * MAX_REQUESTS
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
        self.layers = [bind_layer(block, config, self.tp) for block in draft.mtp]
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
            )
            for i in range(len(self.model.mtp))
        ]

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
        self.mailboxes.begin_step()
        residual, pre_mix = build_block_state(
            draft.embed(anchor_ids),
            draft._noise_embedding(),
            self.width,
            draft.config.hc_mult,
        )
        # the requests' blocks as one step's rows, request after request
        state = SinglePassHCState(
            residual.flatten(0, 1).unsqueeze(0), pre_mix.flatten(0, 1).unsqueeze(0)
        )
        write_draft_step_meta(
            anchor_positions,
            metadata.state_slot_out,
            cache.num_slots,
            cache.geometry,
            cache.geometry.window(0, cache.num_pages),
            self.positions,
            self.ring_rel,
            self.key_meta,
        )
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
        hidden = state.collapse().view(requests, self.width, -1)
        self._raise_on_given_up_waits(rows)
        return draft.mtp[-1].norm(hidden).flatten(0, 1), hidden
