# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fixed native sources and exact lanes retained by one target FULL graph."""

from dataclasses import dataclass

from atom.models.deepseek_v4.mono.binding import input_identity
from atom.models.deepseek_v4.mono.config import BLOCK_TOKENS
from atom.mono.runtime.consensus import MonoUnsupported


@dataclass(frozen=True)
class TargetReplayPublication:
    """Capture-owned integer publisher, never a reusable step authorization.

    Native publishes its fixed GPU staging storage before replay. The graph
    copies those sources to its exact Mono lanes, then generates addresses.
    Session owner/graph/slot generations and the agreed request frontier still
    authorize each invocation; a changed layout takes full publication.
    """

    tokens: object
    positions: object
    metadata: object
    sources: tuple
    source_identity: tuple
    buckets: tuple
    layout: tuple

    @staticmethod
    def _sources(tokens, positions, metadata):
        return (
            tokens,
            positions,
            metadata.state_slot_in,
            metadata.state_slot_out,
            metadata.block_tables,
        )

    @classmethod
    def bind(cls, adapter, tokens, positions, metadata):
        """Bind only after the full startup publication has checked every lane."""
        if (
            not adapter.fused_metadata
            or adapter.ragged
            or adapter.published_rows != tokens.numel()
        ):
            return None
        sources = cls._sources(tokens, positions, metadata)
        return cls(
            tokens,
            positions,
            metadata,
            sources,
            input_identity(sources),
            adapter.published_buckets,
            adapter.graph_layout(),
        )

    def publish(self, adapter, tokens, positions, metadata, frontier):
        """Publish host bookkeeping; device copies belong to the FULL graph.

        A stable continuation has already checked positions, in-place slots,
        table prefixes and lifecycle generations. Source replacement is a
        serving fault, not a rank-local switch into a collective slow path.
        """
        if (
            input_identity(self._sources(tokens, positions, metadata))
            != self.source_identity
        ):
            raise MonoUnsupported("target FULL graph native staging storage changed")
        q = adapter.tokens_per_seq
        contracts = []
        layout = []
        for offset, bucket in self.buckets:
            for start in range(offset, offset + bucket.width, q):
                position, slot, blocks = frontier[start // q]
                layout.append(
                    (start, start + q, slot, offset, bucket.width, bucket.ordinal)
                )
                for i in range(q):
                    end = (position + i + BLOCK_TOKENS) // BLOCK_TOKENS
                    contracts.append((None, position + i, slot, blocks[:end]))
        adapter.published_buckets = self.buckets
        adapter.published_contract = tuple(contracts)
        adapter.published_rows = len(contracts)
        adapter.published_layout = tuple(layout)
        if adapter.graph_layout() != self.layout:
            raise MonoUnsupported("target FULL graph publication layout changed")

    def write_inputs(self):
        """Record the existing exact integer copies against fixed native sources."""
        from atom.models.deepseek_v4.mono.step_meta import write_bucket_inputs

        for offset, bucket in self.buckets:
            write_bucket_inputs(
                bucket,
                self.tokens,
                self.positions,
                self.metadata,
                offset,
                offset // bucket.tokens_per_seq,
            )
