"""DeepSeek-V4 fixed, full-retain, and unified-arena pool layouts."""

from __future__ import annotations

from typing import Any, Protocol

from atom.kv_cache.dsv4.arena import ArenaGroupSpec
from atom.kv_cache.layout import KvLayoutOptions, KvPoolLayout


class Dsv4LayoutProvider(Protocol):
    def swa_pool_block_bytes(self) -> int: ...

    def swa_pool_num_blocks(self, max_num_seqs: int, max_model_len: int) -> int: ...

    def compute_arena_group_specs(self, available_for_pool: int) -> list[Any]: ...


def compute_dsv4_kv_pool_layout(
    builder: Dsv4LayoutProvider,
    *,
    available_for_pool: int,
    block_bytes: int,
    max_num_seqs: int,
    max_model_len: int,
    swa_window_size: int,
    options: KvLayoutOptions | None = None,
    per_req_cache_bytes: int = 0,
    slots_per_req: int = 1,
    max_per_req_cache_slots: int = 0,
    per_req_cache_equiv_blocks: int = 0,
) -> KvPoolLayout:
    """Partition the DSV4 budget without exposing architecture checks upstream."""
    options = options or KvLayoutOptions.from_environment()
    swa_block_bytes = int(builder.swa_pool_block_bytes())
    compressed_block_bytes = block_bytes - swa_block_bytes
    if swa_block_bytes <= 0 or compressed_block_bytes <= 0:
        raise ValueError(
            "DSV4 layout requires positive SWA and compressed block bytes "
            f"(block={block_bytes}, swa={swa_block_bytes})"
        )

    common = dict(
        manager_kind="dsv4",
        primary_block_bytes=compressed_block_bytes,
        block_bytes=block_bytes,
        available_for_pool=available_for_pool,
        swa_block_bytes=swa_block_bytes,
        swa_window_size=int(swa_window_size or 128),
        per_req_cache_bytes=per_req_cache_bytes,
        slots_per_req=slots_per_req,
        max_per_req_cache_slots=max_per_req_cache_slots,
        per_req_cache_equiv_blocks=per_req_cache_equiv_blocks,
    )

    # The arena owns the whole pool budget.  Select it before fixed/full-retain
    # sizing so those two-pool calculations cannot emit misleading split data.
    if options.unified_arena:
        specs = [
            ArenaGroupSpec.coerce(spec)
            for spec in builder.compute_arena_group_specs(available_for_pool)
        ]
        if specs:
            num_chunks = int(specs[0].num_chunks)
            compressed_capacities = [
                spec.max_compressed_blocks for spec in specs if spec.has_compress
            ]
            num_primary_blocks = (
                min(compressed_capacities) if compressed_capacities else num_chunks
            )
            return KvPoolLayout(
                layout_kind="arena",
                num_primary_blocks=num_primary_blocks,
                num_swa_blocks=num_chunks,
                arena_specs=specs,
                **common,
            )

    if options.full_retain:
        fraction = min(0.9, max(1e-3, options.swa_tail_budget_frac))
        swa_budget = int(available_for_pool * fraction)
        compressed_budget = available_for_pool - swa_budget
        return KvPoolLayout(
            layout_kind="full_retain",
            num_primary_blocks=max(0, compressed_budget // compressed_block_bytes),
            num_swa_blocks=max(0, swa_budget // swa_block_bytes),
            **common,
        )

    num_swa_blocks = int(builder.swa_pool_num_blocks(max_num_seqs, max_model_len))
    swa_reserved = num_swa_blocks * swa_block_bytes
    return KvPoolLayout(
        layout_kind="fixed",
        num_primary_blocks=max(
            0, (available_for_pool - swa_reserved) // compressed_block_bytes
        ),
        num_swa_blocks=num_swa_blocks,
        **common,
    )
