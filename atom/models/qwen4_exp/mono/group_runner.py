# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MTP verify steps of several requests on mono.

A verify step of G groups of ``S = k + 1`` rows (``groups``) runs on one
``G·S``-row set of buffers and tables, one launch per layer of the grouped
build (``build_layer(..., group=S)``) for all the rows, each group's seq
chains restricted to its own rows. ``G·S`` is at most 16 rows
(``layout_wide.WIDE_ROWS``). A launch of more than ``MAX_ROWS`` rows (a wide
build) has a mailbox region per row, so it runs on a scratch of
``WIDE_ROWS`` rows' regions, one for every wide width (as the base scratch
serves every width up to ``MAX_ROWS``: a step's tags are new). One counter
serves them all, advanced once a step (``group_meta``).

The step's words are ``group_meta``'s: each group's state slots, block-table
row and PLE window; a padded group's the runner's null state. After the
layers ``commit_ple_groups`` writes back each live group's PLE window. The
model returns the ``G·S`` rows' HC streams, as a single verify does.

A single-request verify (``S`` rows) is the base runner's.
"""

import logging
from dataclasses import dataclass

import torch

from atom.utils.forward_context import get_forward_context

from .group_meta import commit_ple_groups, write_group_verify_meta
from .kernels.kernel import LAYER_ARGS, STAGES, build_layer
from .kernels.layout import MAX_ROWS as KERNEL_ROWS
from .kernels.layout import scratch_bytes
from .kernels.layout_wide import WIDE_ROWS
from .runner import BF16, Qwen4MonoDecodeRunner, Tables
from .step_meta import PLE_COLS, PLE_CONV_IN

logger = logging.getLogger("atom")


@dataclass
class GroupWidth:
    """A verify step of ``groups`` groups, ``rows`` = groups·S rows: the
    grouped builds' launchers by layer kind, the step's buffers and tables
    (as ``runner.Width``'s), the kernel arguments by layer, the groups' PLE
    windows ``[groups, QKV, 9]``, the rows' snapshots ``[rows, QKV, 9]`` and
    the launches' scratch pointer."""

    rows: int
    groups: int
    launch: dict
    h0: torch.Tensor
    chain: torch.Tensor
    emb: torch.Tensor
    tables: Tables
    args: list
    ple_window: torch.Tensor
    ple_snaps: torch.Tensor
    scratch: int


class Qwen4MonoGroupRunner(Qwen4MonoDecodeRunner):
    """``Qwen4MonoDecodeRunner`` that also runs verify steps of 2 ..
    ``max_groups`` groups."""

    def __init__(self, model, atom_config, metadata_builder, max_groups) -> None:
        super().__init__(model, atom_config, metadata_builder)
        self.max_groups = max_groups
        # every verify of up to ``max_groups`` requests has a grouped build
        assert max_groups * self.verify_rows <= WIDE_ROWS
        self.wide_scratch = None
        self.groups: dict[int, GroupWidth] = {}

    def prepare(self, rows: int) -> bool:
        """``Qwen4MonoDecodeRunner.prepare``, and a verify step of 2 ..
        ``max_groups`` groups."""
        if rows in self.groups:
            return True
        s = self.verify_rows
        groups, part = divmod(rows, s) if s else (0, 0)
        if part or not 2 <= groups <= self.max_groups:
            return super().prepare(rows)
        g, dev = self.geometry, self.device
        launch = {
            k: build_layer(STAGES[k], g.experts, rows=rows, group=s)
            for k in set(self.kinds)
        }
        h0 = torch.zeros(rows, g.hc_width, dtype=BF16, device=dev)
        chain = torch.zeros(g.layers, rows, g.hc_width, dtype=BF16, device=dev)
        emb = torch.zeros(rows, g.ple_dim, dtype=BF16, device=dev)
        snaps = torch.zeros(rows, g.hc_width, PLE_COLS, dtype=BF16, device=dev)
        window = torch.zeros(groups, g.hc_width, PLE_COLS, dtype=BF16, device=dev)
        tables = self._tables(rows, emb, window[0], snaps)
        # each group's PLE rows read the group's window
        for k in range(1, groups):
            tables.ple[k * s : (k + 1) * s, PLE_CONV_IN] = window[k].data_ptr()
        args = []
        for i, b in enumerate(self.bound):
            a = dict(b.args)
            a["h_in"] = h0 if i == 0 else chain[i - 1]
            a["h_out"] = chain[i]
            if b.kind == "qsa":
                a["qsa"] = tables.qsa[self.qsa_layers.index(i)]
            else:
                a["conv_in"] = tables.delta[self.delta_layers.index(i)]
            if b.kind == "delta_ple":
                a["ple"] = tables.ple
            args.append([a.get(n, self.dummy).data_ptr() for n in LAYER_ARGS[1:]])
        scratch = self._wide_scratch() if rows > KERNEL_ROWS else self.scratch
        self.groups[rows] = GroupWidth(
            rows,
            groups,
            launch,
            h0,
            chain,
            emb,
            tables,
            args,
            window,
            snaps,
            scratch.data_ptr(),
        )
        logger.info(
            "Qwen3.8 mono decode: %d-row verify step prepared, %d groups of %d",
            rows,
            groups,
            s,
        )
        return True

    def _wide_scratch(self) -> torch.Tensor:
        """The wide builds' scratch, for ``WIDE_ROWS`` rows."""
        if self.wide_scratch is None:
            size = scratch_bytes(WIDE_ROWS)
            self.wide_scratch = torch.zeros(
                size // 4, dtype=torch.int32, device=self.device
            )
        return self.wide_scratch

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        ids = input_ids.reshape(-1)
        w = self.groups.get(ids.numel())
        if w is None:
            return super().forward(input_ids, positions)
        if positions.dim() == 2:
            positions = positions[0]
        self.embed(ids, w)
        md = get_forward_context().attn_metadata
        gdn = md.gdn_metadata
        write_group_verify_meta(
            self.counter,
            positions,
            gdn.spec_state_indices_tensor,
            gdn.num_accepted_tokens,
            md.qsa_metadata.block_tables,
            w.tables,
            self.state.ple_conv,
            w.ple_window,
            self.nulls,
        )
        stream = torch.cuda.current_stream(self.device)
        counter = self.counter.data_ptr()
        for i, kind in enumerate(self.kinds):
            w.launch[kind](counter, i, w.scratch, *w.args[i], stream=stream)
        commit_ple_groups(
            self.state.ple_conv, w.ple_snaps, gdn.spec_state_indices_tensor
        )
        g = self.geometry
        return w.chain[-1].view(w.rows, g.hc_count, g.hidden)
