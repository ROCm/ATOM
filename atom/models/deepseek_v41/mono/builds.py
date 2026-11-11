# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What a V4.1 mono width was built as.

One record a width (``WidthBuilds``, keyed by the step's row count), holding
the launchers and -- the reason it is a record and not a row count -- the K2
build key they came from. A step's shape is not a function of its rows alone,
so ``runner`` reads it off the build in hand rather than deriving it again.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RowsBuild:
    """The kernels of an S-row step: K1's build a layer and their launchers,
    and K2 (or, with ``check``, K2a and K2b apart).

    ``post_key`` is the K2 build key the launchers came from: the one place a
    step's ``tok_par`` and its K2 scratch layout are recorded, so the forward
    and the debug paths ask the build what it is instead of recomputing it
    from the row count and hoping the two answers agree."""

    keys: list
    launchers: dict
    layer_post: object = None
    attn_post: object = None
    moe: object = None
    post_key: object = None

    @property
    def tok_par(self) -> bool:
        """Whether this build's kernels run token parallel."""
        return bool(self.post_key is not None and self.post_key.tok_par)
