# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""A sliding-window draft's KV kept as per-request state: a ring of blocks a slot,
logical block j of slot s at entry `1 + s * ring_blocks + j % ring_blocks`
(entry 0 is the null page), read through a table shaped like the target's."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np

# Draft architectures whose block pass reads its window from a known page:
# the MiniMax-M3.1 draft starts a row's keys on a 16-token ASM page.
_WINDOW_READ_PAGES = {"DSparkMiniMaxDraftModel": 16}


@dataclass(frozen=True)
class DraftWindow:
    """The ring a sliding-window draft keeps per request."""

    window: int  # rows a block-pass row reads back, its own included
    read_page: int  # a row's read starts on this page, up to page - 1 earlier
    lookahead: int  # rows past the anchor a step writes (block, verify)
    block_size: int  # tokens per ring entry, the scheduler's block
    anchor_margin: int  # how far a prompt-end anchor may sit behind its end

    @property
    def rebuild_tokens(self) -> int:
        """Tokens before a request's end from which a forward rewrites every
        row a reader takes (`StateTransfer.rebuild_tokens`)."""
        return self.window + self.read_page + self.lookahead

    @property
    def ring_blocks(self) -> int:
        """Entries a request's ring holds: what a reader takes, plus the most
        an anchor trails the forward it is filed from."""
        return ceil((self.rebuild_tokens + self.anchor_margin) / self.block_size)

    @property
    def ring_tokens(self) -> int:
        return self.ring_blocks * self.block_size

    @property
    def trailing_slack(self) -> int:
        """`StateTransfer.trailing_slack`: rows the ring holds past a boundary
        a reader resuming there still needs."""
        return self.ring_tokens - self.rebuild_tokens

    def ring_table(self, slots: np.ndarray, cols: int) -> np.ndarray:
        """Each slot's ring entry for logical blocks `[0, cols)`; slot -1 (a
        pad row) reads the null entry 0."""
        slots = np.asarray(slots, dtype=np.int64)
        table = (
            1
            + slots[:, None] * self.ring_blocks
            + (np.arange(cols, dtype=np.int64) % self.ring_blocks)
        )
        table[slots < 0] = 0
        return table.astype(np.int32)

    def context_slots(
        self,
        positions: np.ndarray,
        row_seq: np.ndarray,
        seq_ends: np.ndarray,
        ring_table: np.ndarray,
    ) -> np.ndarray:
        """The ring slot of each target row's context K/V, -1 for none: a row below
        its request's last `ring_tokens` would share a slot with a later one."""
        positions = np.asarray(positions, dtype=np.int64)
        row_seq = np.asarray(row_seq, dtype=np.int64)
        out = np.full(positions.shape, -1, dtype=np.int64)
        real = row_seq >= 0
        seq = row_seq[real]
        pos = positions[real]
        blocks = ring_table[seq, pos // self.block_size].astype(np.int64)
        slots = blocks * self.block_size + pos % self.block_size
        keep = pos >= np.asarray(seq_ends, dtype=np.int64)[seq] - self.ring_tokens
        out[real] = np.where(keep, slots, -1)
        return out


def draft_window(config: Any) -> DraftWindow | None:
    """`config`'s draft window when its draft keeps a sliding window as
    per-request state; None otherwise. Config only, so the API process can
    ask before any runner exists."""
    from atom.plugin import is_plugin_mode

    spec = getattr(config, "speculative_config", None)
    if is_plugin_mode() or spec is None:
        return None
    # The draft's architecture is the gate: every one in the table is DSpark.
    draft = getattr(spec, "draft_model_hf_config", None)
    architectures = getattr(draft, "architectures", None) or ()
    page = _WINDOW_READ_PAGES.get(architectures[0] if architectures else None)
    window = int(getattr(draft, "sliding_window", None) or 0)
    if page is None or window <= 0:
        return None
    from atom.utils import envs

    block = int(config.kv_cache_block_size)
    return DraftWindow(
        window=window,
        read_page=page,
        lookahead=2 * int(spec.num_speculative_tokens),
        block_size=block,
        anchor_margin=2 * block + envs.ATOM_STATE_CHECKPOINT_END_MARGIN,
    )


__all__ = ["DraftWindow", "draft_window"]
