# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Control plane for prefix-cache sidecar state.

The ordinary ``BlockManager`` owns compressed KV blocks.  Some attention
backends need additional state for the same chained-prefix identity, but that
state does not necessarily have the same layout or lifetime as compressed KV:

* SWA KV is page-addressed and windowed.  Its physical pages are independently
  allocated and may be reclaimed while the compressed prefix remains cached.
* DSV4 CSA boundary state is a small, immutable sidecar of each compressed
  128-block.  It is FUSED into the block's SWA chunk (feat/csa-swa-fusion): the
  boundary snapshot lives in a fixed tail byte segment of the SWA chunk and
  rides the SWA block's content-addressing + retention pin, so it needs no
  separate page pool of its own (the associative CsaStatePool it replaced is
  gone).

``StatePool`` is deliberately a control plane, rather than a single flat GPU
allocator.  It provides the common prefix-cache lifecycle and keeps the SWA
component's physical-storage policy isolated (free-list/window/retention),
driving it in lockstep with the compressed pool.
"""

from __future__ import annotations

from dataclasses import dataclass

from atom.model_engine.sequence import Sequence
from atom.model_engine.swa_pool import SlidingWindowPool


@dataclass(frozen=True)
class PrefixStateHit:
    """State sidecars available at one compressed-prefix boundary.

    ``num_cached_blocks`` is the compressed-KV hit after every enabled sidecar
    has applied its own validity rule.  ``swa_hit_start`` identifies the first
    logical block whose SWA page is live. Under the CSA-into-SWA fusion the CSA
    boundary shares the SWA block's fate, so the SWA gate is the CSA gate too.
    """

    num_cached_blocks: int
    swa_hit_start: int


class StatePool:
    """Prefix-cache sidecar-state coordinator.

    The public methods intentionally mirror the current block-manager
    lifecycle.  ``SlidingWindowPool`` is a component, not an inheritance base, so
    it keeps its own free-list / refcount / reclamation policy.

    CSA integration (feat/csa-swa-fusion): the boundary snapshot is fused into
    the SWA chunk, so there is no separate CSA page lifecycle — capture writes
    into the block's SWA chunk (scheduler ships the c4 physical SWA page as the
    destination) and ``bound_hit`` gates the prefix hit purely on SWA presence.
    """

    def __init__(
        self,
        *,
        num_swa_blocks: int,
        swa_window: int,
        block_size: int,
        max_num_batched_tokens: int,
        mtp_k: int,
        full_retain: bool = False,
        retention_interval: int = 0,
        checkpoint_frac: float = 0.5,
        require_csa_boundary_state: bool = False,
    ):
        # SWA has a distinct physical allocation/lifetime and stays a dedicated
        # component.  This is intentionally not a generic ``Block`` pool.
        # full_retain / retention_interval / checkpoint_frac carry the SWA
        # sparse-checkpoint retention policy (and, under the unified-KV arena,
        # the elastic borrow path lives inside this SlidingWindowPool). CSA
        # boundary retention rides this same SWA pin: pinning a checkpoint SWA
        # block keeps its fused CSA state alive across windowing.
        self._swa = SlidingWindowPool(
            num_blocks=num_swa_blocks,
            window=swa_window,
            block_size=block_size,
            max_num_batched_tokens=max_num_batched_tokens,
            mtp_k=mtp_k,
            full_retain=full_retain,
            retention_interval=retention_interval,
            checkpoint_frac=checkpoint_frac,
        )
        self._require_csa_boundary_state = require_csa_boundary_state

    # --------------------- SWA component / compatibility ------------------ #
    @property
    def swa_enabled(self) -> bool:
        return self._swa.enabled

    @property
    def requires_csa_boundary_state(self) -> bool:
        return self._require_csa_boundary_state

    @property
    def swa(self) -> SlidingWindowPool:
        """The SWA component, exposed for component-specific diagnostics/tests."""
        return self._swa

    @property
    def swa_tail_blocks(self) -> int:
        return self._swa.tail_blocks

    def has_free_swa(self, n: int) -> bool:
        return self._swa.has_free(n)

    def swa_admission_blocks(self, seq: Sequence) -> int:
        return self._swa.admission_blocks(seq)

    def bound_hit(
        self,
        seq: Sequence,
        compressed_hit: int,
        block_hashes: list[int],
        compressed_block_ids: list[int],
    ) -> PrefixStateHit:
        """Return the best boundary supported by all materialized sidecars.

        Fused CSA (feat/csa-swa-fusion): the CSA boundary snapshot lives IN the
        terminal block's SWA chunk (content-addressed + retention-pinned), so the
        SWA trailing-window gate IS the CSA gate — a boundary whose SWA window is
        present has its fused CSA state present too. CSA correctness therefore
        rides SWA retention: pin checkpoint SWA blocks (ATOM_SWA_*) to extend how
        far back a hit restores.
        """
        if len(compressed_block_ids) < compressed_hit:
            raise AssertionError("missing physical ids for compressed prefix hit")
        num_cached_blocks = self._swa.bounded_hit(seq, compressed_hit, block_hashes)
        return PrefixStateHit(
            num_cached_blocks=num_cached_blocks,
            swa_hit_start=max(0, num_cached_blocks - self._swa.tail_blocks),
        )

    def claim_swa_cached(self, seq: Sequence, h: int, token_ids: list[int]) -> None:
        self._swa.claim_cached(seq, h, token_ids)

    def append_swa_placeholder(self, seq: Sequence) -> None:
        self._swa.alloc_placeholder(seq)

    def append_new_swa_block(self, seq: Sequence) -> None:
        self._swa.append_new(seq)

    def ensure_swa_for_tokens(
        self, seq: Sequence, num_cached_tokens: int, num_new_tokens: int
    ) -> None:
        self._swa.ensure_for_tokens(seq, num_cached_tokens, num_new_tokens)

    def free_swa_out_of_window(self, seq: Sequence, seq_len: int | None = None) -> None:
        self._swa.free_out_of_window(seq, seq_len)

    def free_swa_after_prefill_chunk(self, seq: Sequence) -> None:
        self._swa.free_after_prefill_chunk(seq)

    def materialize_swa_window(self, seq: Sequence, seq_len: int) -> None:
        self._swa.materialize_window(seq, seq_len)

    def publish_swa_block(
        self, seq: Sequence, logical_block: int, h: int, token_ids: list[int]
    ) -> None:
        self._swa.publish_hash(seq, logical_block, h, token_ids)

    def release(self, seq: Sequence) -> None:
        self._swa.release(seq)
