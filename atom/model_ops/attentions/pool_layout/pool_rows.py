# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which row of a pool each module gets, and how many there are to buy.

One walk answers both. Sizing asks how many rows of each kind a builder needs
and binding asks which row a module holds, and they read the same assignment
rather than being two walks that agree by convention: the runner's bind walk
offers a draft's modules to the target builder whenever a sibling builder
declines them, and a counter following that walk would quietly number them into
rows nobody bought. Here that is a `KeyError`.

A mixin rather than a base class because the two things that need it are not
kin -- an attention metadata builder, and a draft's own KV pool, which is not
an attention builder and whose package promises to import no attention at all.
"""

from __future__ import annotations

from typing import NamedTuple


class KvGeometry(NamedTuple):
    """A KV row space, keyed by what one of its rows holds.

    A type rather than a bare tuple so that "which row spaces are pools" is a
    question with an answer -- a hybrid adds named row spaces of its own (the
    indexer's keys, a linear-attention slot), and telling them apart by
    exclusion means every new one has to be added to a list it does not know
    about.
    """

    num_kv_heads: int
    head_dim: int

    def __str__(self) -> str:
        """What this row space is called, wherever one is named -- the startup
        log, and the `semantic_role` its transfer regions carry. Both are for a
        person: the default repr would print the class and field names."""
        return f"h{self.num_kv_heads}d{self.head_dim}"


class PoolRowsMixin:
    """`self.model_runner`'s modules, assigned to this builder's rows."""

    # Unassigned until something asks. On the class so a builder needs no
    # constructor of its own to be askable -- several are built by
    # `object.__new__` on paths that only want the geometry.
    _pool_rows: dict | None = None

    def _pooled_models(self) -> list:
        """The models whose modules this builder's pools hold rows for."""
        runner = self.model_runner
        models = [runner.model]
        if runner.draft_shares_kv_pool():
            models.append(runner.drafter.model)
        return models

    def _module_kinds(self, module) -> tuple:
        """The row spaces this builder gives `module` a row in, if any.

        A row space is one numbering, not one pool: MiniMax-M3's indexer keys
        are a second row space over the same modules as its KV, and a hybrid's
        linear-attention state is a third the same builder hands out. Empty for
        a module it does not cache -- an `nn.LayerNorm`, another backend's
        attention -- which is also what keeps it out of the walk.
        """
        return ()

    @property
    def pool_rows(self) -> dict:
        """`{kind: {module: row}}`, in the order the modules are bound.

        Reading the layers off the modules is what turns every count that used
        to be derived from config -- which layers of a hybrid are full
        attention, where a draft's stack starts, what this PP stage holds --
        into a property of what was actually built.
        """
        if self._pool_rows is None:
            rows: dict = {}
            for model in self._pooled_models():
                for module in model.modules():
                    for kind in self._module_kinds(module):
                        in_kind = rows.setdefault(kind, {})
                        in_kind[module] = len(in_kind)
            self._pool_rows = rows
        return self._pool_rows

    def row_counts(self) -> dict:
        """`{kind: rows}` -- what sizing charges for, from the same walk."""
        return {kind: len(in_kind) for kind, in_kind in self.pool_rows.items()}

    def invalidate_pool_rows(self) -> None:
        """Drop the assignment, before a bind walk that may not match it.

        The runner's to call, because the model tree is its: a pool is re-bound
        on a P/D import and again on a rollout wake, and neither promises the
        same modules.
        """
        self._pool_rows = None
