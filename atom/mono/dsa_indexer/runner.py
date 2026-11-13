# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A model's DSA indexers on the mono indexer kernel: their binding, the step
widths the kernel serves, what they take of a step's scratch, plain words and
peer buffer, each width's launch, and their routing (``bind.IndexerHook``).
The buffers are a host runner's when the model has mono kernels of its own
(Kimi-K3's, past its regions), else the indexers' own (``IndexerRunner.alone``).
"""

import logging
import weakref
from dataclasses import replace

from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

from atom.mono.dsa_indexer import kernel as kix
from atom.mono.dsa_indexer.bind import IndexerHook, bind_indexers, logits_plane
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes
from atom.mono.runtime.uncached import uncached_bytes
from atom.mono.runtime.widths import WidthBuilds

logger = logging.getLogger("atom")

# a model whose indexers are its only mono launches: one tag a layer from here
TAG0 = 1


class IndexerRunner:
    """Binds this rank's indexers (raising ``MonoUnsupported`` when the kernel
    cannot serve them; no collective: a host's or ``alone``'s vote decides)."""

    def __init__(
        self,
        inner,
        metadata_builder,
        *,
        hook: IndexerHook | None = None,
        peer_offset: int = 0,
        max_rows: int = kix.MAX_TOKENS,
    ) -> None:
        self.rank = get_tensor_model_parallel_rank()
        self.tp = get_tensor_model_parallel_world_size()
        ix = bind_indexers(inner, self.tp, self.rank)
        if ix is None:
            raise MonoUnsupported("no DSA indexer the kernel serves (logged above)")
        # a replicated build's ILOG
        self.ilog = logits_plane(metadata_builder)
        if ix.replicated and self.ilog is None:
            raise MonoUnsupported("no logits plane for a replicated step's ILOG")
        self.ix = replace(ix, peer_offset=peer_offset)
        self.device = next(iter(ix.modules.values())).wq_b.weight.device
        # the widest step it serves: its buffers are sized for it
        self.rows = max(
            (r for r in range(1, max_rows + 1) if self.ix.serves(r, 1, self.ilog)),
            default=0,
        )
        if not self.rows:
            raise MonoUnsupported("no step width the kernel serves")
        built = max(r for r in range(1, max_rows + 1) if not kix.unfit(self.ix.key(r)))
        if self.rows < built:
            logger.info(
                "mono DSA indexer: the logits plane holds a %d-row step's ILOG; "
                "wider steps (to %d rows) run the production indexer",
                self.rows,
                built,
            )
        widest = self.ix.key(self.rows)
        self.scratch_bytes = kix.scratch_bytes(widest)
        self.plain_bytes = kix.plain_bytes(widest)
        # its regions start ``peer_offset`` into the peer buffer it shares
        self.peer_bytes = peer_offset + kix.peer_bytes(widest)
        # the routing: a compiled stack's is installed before its first trace
        # (``dispatch``), an eager one's at ``attach``
        self.hook = hook
        self._unhook = None

    def serves(self, rows: int, qlen: int) -> bool:
        return (
            0 < rows <= self.rows
            and not rows % qlen
            and self.ix.serves(rows, qlen, self.ilog)
        )

    def launcher(self, rows: int, qlen: int):
        """This step's build, or None: the DSA layers run their own forward."""
        return kix.build(self.ix.key(rows, qlen)) if self.serves(rows, qlen) else None

    def attach(self, scratch, plain, peers, tag0: int) -> None:
        """The step's buffers (the indexers' regions ``peer_offset`` into
        ``peers``) and their first tag. Hooks the indexers if not yet: once
        every rank holds a whole runner, so a refusal before leaves them
        untouched; dropping this (or ``close``) unhooks them."""
        if self.hook is None:
            self.hook = IndexerHook(self.ix.modules)
            self._unhook = weakref.finalize(self, self.hook.remove)
        self.hook.attach(self.ix, scratch, plain, self.ilog, peers, self.rank, tag0)

    # ------------------------------------------- a model's only mono launches
    @classmethod
    def alone(cls, inner, hook: IndexerHook, metadata_builder) -> "IndexerRunner":
        """Every rank's indexers with buffers, mailboxes and width builds of
        their own, or none (``bind_agreed``)."""
        group = get_tp_group().cpu_group
        made = []
        bind_agreed(lambda: made.append(cls(inner, metadata_builder, hook=hook)), group)
        (runner,) = made
        runner._own_buffers(group)
        return runner

    def _own_buffers(self, group) -> None:
        self.scratch = uncached_bytes(-(-self.scratch_bytes // 16) * 16, self.device)
        self.peers, self._finalizer = owned_peer_buffer(
            self,
            self.peer_bytes,
            group,
            self.rank,
            self.tp,
            self.device,
            plain_bytes=self.plain_bytes,
        )
        self.mailboxes = StepMailboxes(self.peers, self.scratch)
        self.widths = WidthBuilds(
            lambda width: self.launcher(*width),
            lambda launcher: [(launcher, kix.ABI)],
            group,
            "DSA indexer mono",
        )
        self.attach(self.scratch, self.peers.plain, self.peers, TAG0)

    def prepare(self, rows: int, qlen: int) -> bool:
        """Whether every rank holds this step's build (``WidthBuilds``)."""
        return self.serves(rows, qlen) and self.widths.prepare((rows, qlen))

    def serving(self, rows: int, qlen: int):
        """The step's forward with the indexers on its build: the mailboxes
        cleared and the peers fenced first, on the step's stream."""
        self.mailboxes.begin_step()
        return self.hook.serving(self.widths[(rows, qlen)])

    def unhook(self) -> None:
        """Give the indexers their own forward back if ``attach`` hooked them."""
        if self._unhook is not None:
            self._unhook()

    def close(self) -> None:
        """``unhook``, and close the peer mappings now (once no rank can launch
        another step)."""
        self.unhook()
        self._finalizer()
