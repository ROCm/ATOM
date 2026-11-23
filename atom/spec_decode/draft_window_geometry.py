# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Which draft rows a sliding-window DSpark draft riding the target's blocks reads,
for the prefix-cache gates on `Block.draft_rows_missing`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atom.model_ops.minimax_m3.geometry import ASM_PAGE_SIZE

# Draft architectures whose block pass reads through a known page: the M3 draft
# starts a row's keys on an ASM page.
_DRAFT_READ_PAGES = {"DSparkMiniMaxDraftModel": ASM_PAGE_SIZE}


@dataclass(frozen=True)
class DraftReadWindow:
    """A block-pass row at `p` reads keys from the `read_page` page holding
    `p - window + 1` (`_window_rows` in the MiniMax-M3 draft)."""

    window: int
    read_page: int

    def first_read_token(self, num_tokens: int) -> int:
        """Lowest key a block pass whose drafts are used reads for a
        `num_tokens`-token sequence: its rows start at `num_tokens`."""
        start = max(0, num_tokens + 1 - self.window)
        return start - start % self.read_page


def draft_read_window(config: Any) -> DraftReadWindow | None:
    """`config`'s draft read window when its KV rides the target's blocks and
    only a sliding window of it is read; None otherwise."""
    spec = config.speculative_config
    if spec is None or not spec.use_dspark():
        return None
    draft = spec.draft_model_hf_config
    page = _DRAFT_READ_PAGES.get(draft.architectures[0])
    window = getattr(draft, "sliding_window", None) or 0
    if page is None or window <= 0:
        return None
    return DraftReadWindow(window=window, read_page=page)


__all__ = ["DraftReadWindow", "draft_read_window"]
