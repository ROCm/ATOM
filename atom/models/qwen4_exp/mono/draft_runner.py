# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen3.8-Flash-Next MTP draft layer on the mono kernels.

ATOM's drafter (``Qwen4ExpMTP``) runs per draft step a front end (the tokens'
embeddings through ``pre_fc_norm_embedding`` and ``fc_embedding``, the
target's HC streams through ``pre_fc_norm_hidden`` and ``fc_hidden``, summed
into four streams), one QSA decoder layer (``mtp.layers.0``) and the head
(the HC mixer and ``lm_head``). Its positions are mRoPE, so it declares no
draft graphs: every op runs eagerly, and the decoder layer's hundreds of
launches leave a speculative step host-bound.

:class:`Qwen4MonoDraftRunner` runs that decoder layer as one mono QSA launch:
the target's build (``build_layer`` is cached) with the draft layer's weights
and caches, its own step counter and mailbox scratch, and layer id 48 (the
target's are 0 .. 47). The front end is one launch too (``front``), op for
op as ATOM's but for its summation order; the head stays ATOM's ops, its
``lm_head`` the target's MXFP4 one (``head``). After a verify of one request
a draft step is

- step 0 of a verify, the target's ``k + 1`` rows of one request: the seq
  build of ``k + 1`` rows, as the target's verify runs;
- a later step (k >= 2), one row: the ``rows=1`` build.

After a verify of n = 2 .. ``max_groups`` requests (``groups``) they are

- step 0, the verify's ``n·S`` rows (``S = k + 1``), never padded: one
  launch of the grouped build (``build_layer(..., group=S)``), past
  ``MAX_ROWS`` rows on a wide scratch, as the target's;
- a later step (k >= 2), one row per request: the ``n``-row decode build, each
  row its request's (``write_group_draft_meta``).

Its rows' cache rows are ATOM's (``qsa_metadata.logical_positions``): the
draft writes token ``t + 1``'s key at the target's row ``t``, rotated at
``t + 1``. The kernels rotate at a row's cache position, so the draft's QSA
table points at RoPE tables one row on (row ``i`` holds position ``i + 1``):
ATOM's draft rotates its queries, keys and compressed keys
(``rope_position_offset``) one ahead of their rows.

The step returns the layer's HC streams ``[rows, hc_count, hidden]``, as the
drafter's forward does. Not bit-identical to ATOM's draft layer: the mono sums
in its own order, as the target's layers do.
"""

import logging
from dataclasses import dataclass

import torch

from atom.utils.forward_context import get_forward_context

from . import weights as W
from .front import DraftFront, draft_front
from .front import refusal as front_refusal
from .group_meta import write_group_draft_meta
from .kernels.config import from_dict
from .kernels.kernel import LAYER_ARGS, STAGES, build_layer
from .kernels.layout import MAX_ROWS as KERNEL_ROWS
from .kernels.layout import scratch_bytes
from .kernels.layout_wide import WIDE_ROWS
from .kernels.stages import qsa as qsa_stage
from .state import bind_qsa_caches
from .step_meta import write_draft_meta

logger = logging.getLogger("atom")

I64, BF16 = torch.int64, torch.bfloat16


@dataclass
class DraftWidth:
    """A ``rows``-row draft step of ``rows // group`` requests (``group`` rows
    each): its launcher and scratch pointer, the layer's input and output
    streams ``[rows, hc_width]``, the rows' cache rows ``pos`` [rows] int32,
    the QSA table ``[rows, qsa.STATE]`` and the kernel arguments
    (``LAYER_ARGS[1:]`` addresses)."""

    rows: int
    group: int
    launch: object
    scratch: int
    h_in: torch.Tensor
    h_out: torch.Tensor
    pos: torch.Tensor
    qsa: torch.Tensor
    args: list


class Qwen4MonoDraftRunner:
    """Binds the draft layer's weights and caches (``MonoUnsupported`` if any
    is not as the kernels need) and runs the draft steps after a verify of 1
    .. ``max_groups`` requests of ``verify_rows`` rows each."""

    def __init__(
        self, draft, atom_config, metadata_builder, verify_rows, max_groups
    ) -> None:
        g = from_dict(atom_config.hf_config.to_dict())
        self.geometry = g
        self.verify_rows = verify_rows
        self.max_groups = max_groups
        # every step 0 of up to ``max_groups`` requests has a grouped build
        assert max_groups * verify_rows <= WIDE_ROWS
        self.predictor = draft.model
        (layer,) = self.predictor.layers
        # the front end in one launch, else ATOM's ops
        why = front_refusal(self.predictor)
        self.front = DraftFront(self.predictor) if why is None else None
        if why is not None:
            logger.info("Qwen3.8 mono draft: front end op for op: %s", why)
        self.layer_id = g.layers
        self.bound = W.bind_layer(layer, self.layer_id, "qsa", g.experts)
        self.caches = bind_qsa_caches(metadata_builder, layer, self.layer_id)
        # one row on: row i holds position i + 1. Where the rotary tables end
        # at max_model_len, the last row repeats the last position: a draft
        # from the last cache row is never verified.
        prm, n = W.parameters(layer), atom_config.max_model_len
        held = prm["self_attn.rotary_emb.cos_cache"].shape[0]
        tables = W.rope(prm, min(n + 1, held))
        self.rope = tuple(torch.cat([t[1:], t[-1:]])[:n] for t in tables)

        dev = self.caches["key"].device
        self.device = dev
        self.dummy = torch.zeros(64, dtype=torch.int32, device=dev)
        scratch = scratch_bytes(KERNEL_ROWS)
        self.scratch = torch.zeros(scratch // 4, dtype=torch.int32, device=dev)
        self.wide_scratch = None
        self.counter = torch.zeros(1, dtype=torch.int32, device=dev)
        # by rows: the steps of one request, and after a verify of several
        # step 0 (n·S rows) and the later steps (n rows); at n = S a later
        # step has the rows of a single request's step 0
        self.single: dict[int, DraftWidth] = {}
        self.step0: dict[int, DraftWidth] = {}
        self.later: dict[int, DraftWidth] = {}

    def prepare(self, rows: int) -> bool:
        """Build (on first use) the draft steps of ``rows`` rows; False if
        mono serves none."""
        s = self.verify_rows
        served = False
        # one request: step 0, the verify's rows; with k >= 2 the later
        # steps' one row
        if rows == s or (rows == 1 and s > 2):
            if rows not in self.single:
                self.single[rows] = self._width(rows, rows, seq=rows > 1)
            served = True
        n, part = divmod(rows, s)
        if not part and 2 <= n <= self.max_groups:
            if rows not in self.step0:
                self.step0[rows] = self._width(rows, s, group=s)
            served = True
        if s > 2 and 2 <= rows <= self.max_groups:
            if rows not in self.later:
                self.later[rows] = self._width(rows, 1)
            served = True
        return served

    def _width(self, rows: int, span: int, **build) -> DraftWidth:
        """``rows`` rows of ``rows // span`` requests: one ``rows``-row
        launch of ``build_layer(..., **build)``."""
        g, dev = self.geometry, self.device
        launch = build_layer(STAGES["qsa"], g.experts, rows=rows, **build)
        h_in = torch.zeros(rows, g.hc_width, dtype=BF16, device=dev)
        h_out = torch.zeros_like(h_in)
        pos = torch.zeros(rows, dtype=torch.int32, device=dev)
        qsa = torch.zeros(rows, len(qsa_stage.STATE), dtype=I64)
        for r in range(rows):
            entries = self.bound.qsa_norms | self.caches
            entries |= {"pos": pos[r], "cos": self.rope[0], "sin": self.rope[1]}
            for k, name in enumerate(qsa_stage.STATE):
                if name in entries:
                    qsa[r, k] = entries[name].data_ptr()
        qsa = qsa.to(dev)
        a = dict(self.bound.args) | {"h_in": h_in, "h_out": h_out, "qsa": qsa}
        args = [a.get(k, self.dummy).data_ptr() for k in LAYER_ARGS[1:]]
        scratch = self._wide_scratch() if rows > KERNEL_ROWS else self.scratch
        logger.info(
            "Qwen3.8 mono draft: %d-row step prepared, %d requests",
            rows,
            rows // span,
        )
        return DraftWidth(
            rows, span, launch, scratch.data_ptr(), h_in, h_out, pos, qsa, args
        )

    def _wide_scratch(self) -> torch.Tensor:
        """The wide builds' scratch, for ``WIDE_ROWS`` rows."""
        if self.wide_scratch is None:
            size = scratch_bytes(WIDE_ROWS)
            self.wide_scratch = torch.zeros(
                size // 4, dtype=torch.int32, device=self.device
            )
        return self.wide_scratch

    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor):
        md = get_forward_context().attn_metadata
        g, p = self.geometry, self.predictor
        ids = input_ids.reshape(-1)
        n = ids.numel()
        single = md.gdn_metadata.num_spec_decodes == 1
        if single:
            w = self.single[n]
        else:
            w = (self.step0 if md.max_seqlen_q == self.verify_rows else self.later)[n]
        draft_front(self.front, p, ids, hidden_states.reshape(n, -1), w.h_in)
        qsa = md.qsa_metadata
        meta = (qsa.logical_positions, qsa.block_tables, w.pos, w.qsa)
        if single:
            write_draft_meta(self.counter, *meta)
        else:
            write_group_draft_meta(self.counter, *meta, w.group)
        stream = torch.cuda.current_stream(self.device)
        counter = self.counter.data_ptr()
        w.launch(counter, self.layer_id, w.scratch, *w.args, stream=stream)
        return w.h_out.view(n, g.hc_count, g.hidden)
