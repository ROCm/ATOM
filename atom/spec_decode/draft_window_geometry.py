# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which draft rows a sliding-window DSpark draft riding the target's blocks reads.

Pure arithmetic: the prefix caches and the Mooncake Store's split layout use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# The page the MiniMax-M3 draft's per-row decode starts a row on (ASM_PAGE_SIZE).
M3_DRAFT_READ_PAGE = 16
# Draft architectures whose block pass reads through a known page.
_DRAFT_READ_PAGES = {"DSparkMiniMaxDraftModel": M3_DRAFT_READ_PAGE}


@dataclass(frozen=True)
class DraftReadWindow:
    """A block-pass row at `p` reads keys from the `read_page` page holding
    `p - window + 1` (`_window_rows` in the MiniMax-M3 draft)."""

    window: int
    read_page: int

    def __post_init__(self) -> None:
        if self.window <= 0 or self.read_page <= 0:
            raise ValueError(
                f"a draft read window needs a positive window and page, got "
                f"window={self.window} read_page={self.read_page}"
            )

    def first_read_token(self, num_tokens: int) -> int:
        """Lowest key any block pass of a `num_tokens`-token sequence reads:
        its rows sit at `num_tokens` or later, past an anchor that never retreats."""
        start = max(0, int(num_tokens) + 1 - self.window)
        return start - start % self.read_page


def draft_read_window(config: Any) -> DraftReadWindow | None:
    """`config`'s draft read window when its KV rides the target's blocks and
    only a sliding window of it is read; None otherwise."""
    spec = getattr(config, "speculative_config", None)
    draft = getattr(spec, "draft_model_hf_config", None)
    if draft is None or getattr(spec, "method", None) != "dspark":
        return None
    arch = (getattr(draft, "architectures", None) or [""])[0]
    page = _DRAFT_READ_PAGES.get(arch)
    window = int(getattr(draft, "sliding_window", 0) or 0)
    if page is None or window <= 0:
        return None
    # Local: the draft pool's predicate reads the attention selector's tables.
    from atom.spec_decode.draft_kv import draft_owns_kv_pool

    if not draft_owns_kv_pool(config):
        return None
    return DraftReadWindow(window=window, read_page=page)


__all__ = ["M3_DRAFT_READ_PAGE", "DraftReadWindow", "draft_read_window"]
