"""Pure CPU tests for architecture-neutral KV pool sizing."""

from types import SimpleNamespace

from atom.kv_cache.dsv4.layout import compute_dsv4_kv_pool_layout
from atom.kv_cache.layout import (
    KvLayoutOptions,
    compute_default_kv_pool_layout,
    compute_total_kv_block_bytes,
)
from atom.model_engine.unified_kv_arena import ArenaGroupSpec


class _Builder:
    def __init__(self, *, block_bytes=100, swa_bytes=60, fixed_swa_blocks=3):
        self._block_bytes = block_bytes
        self._swa_bytes = swa_bytes
        self._fixed_swa_blocks = fixed_swa_blocks

    def compute_block_bytes(self):
        return self._block_bytes

    def swa_pool_block_bytes(self):
        return self._swa_bytes

    def swa_pool_num_blocks(self, max_num_seqs, max_model_len):
        del max_num_seqs, max_model_len
        return self._fixed_swa_blocks

    def compute_arena_group_specs(self, available_for_pool):
        del available_for_pool
        return [
            ArenaGroupSpec(
                name="c4",
                num_chunks=5,
                bytes_per_chunk=128,
                chunk_rows=128,
                owners={"swa": 128, "compress": 32},
            ),
            ArenaGroupSpec(
                name="c128",
                num_chunks=5,
                bytes_per_chunk=128,
                chunk_rows=128,
                owners={"swa": 128, "compress": 1},
            ),
        ]


def _dsv4(builder=None, *, available=1000, options=None, **metadata):
    return compute_dsv4_kv_pool_layout(
        builder or _Builder(),
        available_for_pool=available,
        block_bytes=100,
        max_num_seqs=8,
        max_model_len=1024,
        swa_window_size=128,
        options=options or KvLayoutOptions(),
        **metadata,
    )


def test_dense_default_single_pool():
    layout = compute_default_kv_pool_layout(
        available_for_pool=1000,
        block_bytes=100,
    )
    assert layout.manager_kind == "dense"
    assert layout.layout_kind == "dense"
    assert layout.num_primary_blocks == 10
    assert layout.num_swa_blocks == 0
    assert layout.arena_specs is None


def test_target_and_draft_bytes_are_both_counted():
    target = SimpleNamespace(compute_block_bytes=lambda: 100)
    draft = SimpleNamespace(compute_block_bytes=lambda: 40)
    block_bytes = compute_total_kv_block_bytes([target, draft])
    assert block_bytes == 140
    layout = compute_default_kv_pool_layout(
        available_for_pool=1400,
        block_bytes=block_bytes,
    )
    assert layout.num_primary_blocks == 10


def test_per_request_tensor_deduction_is_not_lost():
    per_req_bytes = 50
    slots = 4
    available_after_state = 1000 - per_req_bytes * slots
    layout = compute_default_kv_pool_layout(
        available_for_pool=available_after_state,
        block_bytes=100,
        per_req_cache_bytes=per_req_bytes,
        max_per_req_cache_slots=slots,
        per_req_cache_equiv_blocks=1,
    )
    assert layout.available_for_pool == 800
    assert layout.num_primary_blocks == 8
    assert layout.max_per_req_cache_slots == 4


def test_dsv4_fixed_split():
    layout = _dsv4()
    assert layout.manager_kind == "dsv4"
    assert layout.layout_kind == "fixed"
    assert layout.primary_block_bytes == 40
    assert layout.num_swa_blocks == 3
    assert layout.num_primary_blocks == (1000 - 3 * 60) // 40


def test_dsv4_full_retain_split():
    layout = _dsv4(options=KvLayoutOptions(full_retain=True, swa_tail_budget_frac=0.2))
    assert layout.layout_kind == "full_retain"
    assert layout.num_swa_blocks == 200 // 60
    assert layout.num_primary_blocks == 800 // 40


def test_dsv4_arena_uses_whole_budget_and_tightest_group():
    layout = _dsv4(
        options=KvLayoutOptions(
            unified_arena=True,
            full_retain=True,
            swa_tail_budget_frac=0.2,
        )
    )
    assert layout.layout_kind == "arena"
    assert layout.num_swa_blocks == 5
    assert layout.num_primary_blocks == 20
    assert [spec.name for spec in layout.arena_specs] == ["c4", "c128"]
