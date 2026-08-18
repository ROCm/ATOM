# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sizing arithmetic for the attention sub-pool specs.

`plan_pools` is the pure half of what used to live inside
`ModelRunner.get_num_blocks()`, which needed a real GPU and so was never
covered. Everything here runs on synthetic specs: no model config is loaded,
so the tests exercise the arithmetic rather than restating a config file.

The central test is `TestParityWithPreSpecArithmetic`, which pins the new
path against the exact expression the runner used before the refactor.
"""

from typing import ClassVar

import pytest

from atom.model_ops.attentions.sub_pool_spec import (
    PAGED_CLASS,
    InsufficientPoolBudget,
    Pool,
    PoolPlan,
    SubPoolSpec,
    merge_specs,
    page_pool,
    plan_pools,
    state_pool,
)

# The sizing layer defines no class names beyond the paged one — a name is
# owned by whatever consumes the count. These tests therefore invent their own
# STATE class names, which also proves the module never special-cases them.
ENTRY_KV = PAGED_CLASS
ENTRY_SWA = "window"
ENTRY_STATE = "recurrent"

GIB = 1 << 30


class TestStatePool:
    def test_declared_extra_entries_are_preserved(self):
        spec = state_pool(ENTRY_STATE, 10, entries_per_req=1, extra_entries=64)
        assert spec.extra_entries == 64

    def test_the_staging_ring_is_sized_here_not_by_the_caller(self, monkeypatch):
        """`state_pool` is the only reader of `OFFLOAD_STATE`.

        A backend that sized the ring itself would be a second env read, and
        the arena and the pool would size themselves from different answers if
        one of them ever drifted -- the arena short exactly the rows the spill
        path addresses.
        """
        monkeypatch.setenv("OFFLOAD_STATE", "1")
        monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "2")
        spec = state_pool(ENTRY_STATE, 10, entries_per_req=3)
        # Groups, not entries: a staging group is `entries_per_req` rows.
        assert spec.staging_entries == 2 * 3
        assert spec.extra_entries == 0

    def test_the_checkpoint_headroom_and_the_staging_ring_add(self, monkeypatch):
        """The regression the merge introduced, and the reason for two fields.

        `STATE_CKPT_EXTRA_ENTRIES` used to arrive through the same parameter
        the staging ring did, and it *overwrites*: setting the headroom deleted
        the ring. The arena then lost the rows `state_entry_views(num_groups +
        slot)` addresses, and GDN's `cache[layer, lo : lo + span]` clamps rather
        than raising -- a short segment, an `entry_bytes` mismatch, and a crash
        inside the packer.
        """
        monkeypatch.setenv("OFFLOAD_STATE", "1")
        monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "2")
        monkeypatch.setenv("STATE_CKPT_EXTRA_ENTRIES", "16")
        spec = state_pool(ENTRY_STATE, 10, entries_per_req=1)
        assert spec.extra_entries == 16
        assert spec.staging_entries == 2

    def test_the_ring_is_absent_by_default(self, monkeypatch):
        monkeypatch.delenv("OFFLOAD_STATE", raising=False)
        spec = state_pool(ENTRY_STATE, 10, entries_per_req=3)
        assert spec.staging_entries == 0


class TestMergeSpecs:
    def test_same_name_sums_entry_bytes(self):
        """Two builders contributing to one pool share its entry index space,
        so their per-entry costs add — this is how an Eagle3 draft KV rides
        the target model's block ids."""
        merged = merge_specs([page_pool(1000), page_pool(250)])
        assert merged[ENTRY_KV].entry_bytes == 1250

    def test_distinct_names_stay_separate(self):
        merged = merge_specs(
            [page_pool(1000), state_pool(ENTRY_STATE, 4096, entries_per_req=1)]
        )
        assert set(merged) == {ENTRY_KV, ENTRY_STATE}
        assert merged[ENTRY_STATE].entry_bytes == 4096

    def test_same_name_in_a_different_pool_is_rejected(self):
        """A name IS an index space; the same name must not mean two things."""
        clash = [
            SubPoolSpec(Pool.PAGE, "x", 10),
            SubPoolSpec(Pool.STATE, "x", 10, entries_per_req=1),
        ]
        with pytest.raises(ValueError, match="pool/multiplicity"):
            merge_specs(clash)

    def test_same_name_with_a_different_multiplicity_is_rejected(self):
        clash = [
            SubPoolSpec(Pool.STATE, "x", 10, entries_per_req=1),
            SubPoolSpec(Pool.STATE, "x", 10, entries_per_req=2),
        ]
        with pytest.raises(ValueError, match="pool/multiplicity"):
            merge_specs(clash)

    def test_two_state_classes_keep_separate_counts(self):
        """SWA blocks and compressor rings share the STATE pool but have
        different index spaces and multiplicities, so they must not collapse
        into one entry size — the two counts are needed downstream."""
        plan = plan_pools(
            [
                page_pool(1000),
                state_pool(ENTRY_SWA, 10, entries_per_req=3, extra_entries=64),
                state_pool(ENTRY_STATE, 500, entries_per_req=1),
            ],
            available_bytes=1_000_000,
            max_num_seqs=8,
        )
        assert plan.entries[ENTRY_SWA] == 8 * 3 + 64
        assert plan.entries[ENTRY_STATE] == 8


class TestPlanPools:
    def test_remainder_only_takes_the_whole_budget(self):
        plan = plan_pools([page_pool(1000)], available_bytes=10_500, max_num_seqs=8)
        assert plan.entries[ENTRY_KV] == 10

    def test_state_floor_is_reserved_before_the_paged_pool(self):
        specs = [page_pool(1000), state_pool(ENTRY_STATE, 500, entries_per_req=2)]
        # 8 seqs x 2 slots x 500B = 8000B reserved, 2000B left -> 2 blocks.
        plan = plan_pools(specs, available_bytes=10_000, max_num_seqs=8)
        assert plan.entries[ENTRY_STATE] == 16
        assert plan.reserved_bytes[ENTRY_STATE] == 8000
        assert plan.entries[ENTRY_KV] == 2

    def test_swa_floor_is_per_request_plus_flat_slack(self):
        specs = [
            page_pool(100),
            state_pool(ENTRY_SWA, 10, entries_per_req=3, extra_entries=64),
        ]
        plan = plan_pools(specs, available_bytes=100_000, max_num_seqs=8)
        assert plan.entries[ENTRY_SWA] == 8 * 3 + 64

    def test_flat_extra_state_entries_take_exactly_their_paged_budget(self):
        specs = [
            page_pool(100),
            state_pool(ENTRY_STATE, 10, entries_per_req=1, extra_entries=3),
        ]
        plan = plan_pools(specs, available_bytes=1_000, max_num_seqs=8)
        assert plan.entries[ENTRY_STATE] == 8 + 3
        assert plan.reserved_bytes[ENTRY_STATE] == (8 + 3) * 10
        assert plan.entries[ENTRY_KV] == 8

    def test_paged_pool_floors_at_zero_rather_than_going_negative(self):
        specs = [page_pool(1_000_000), state_pool(ENTRY_STATE, 100, entries_per_req=1)]
        plan = plan_pools(specs, available_bytes=10_000, max_num_seqs=8)
        assert plan.entries[ENTRY_KV] == 0

    def test_per_request_floor_over_budget_raises_with_the_numbers(self):
        specs = [page_pool(1000), state_pool(ENTRY_STATE, 4096, entries_per_req=1)]
        with pytest.raises(InsufficientPoolBudget) as exc:
            plan_pools(specs, available_bytes=1000, max_num_seqs=64)
        assert exc.value.reserved_bytes == 64 * 4096
        assert exc.value.available_bytes == 1000
        assert exc.value.entries == 64

    def test_stateless_model_never_raises_on_an_empty_budget(self):
        """Without a per-request floor there is nothing that MUST fit, so a
        tiny budget yields zero blocks and lets the caller's assert speak."""
        plan = plan_pools([page_pool(1000)], available_bytes=1, max_num_seqs=8)
        assert plan.entries[ENTRY_KV] == 0

    def test_two_paged_classes_are_rejected(self):
        clash = [page_pool(10), SubPoolSpec(Pool.PAGE, "other", 10)]
        with pytest.raises(ValueError, match="more than one PAGE"):
            plan_pools(clash, available_bytes=1000, max_num_seqs=1)

    def test_reserved_bytes_never_exceed_the_budget(self):
        specs = [
            page_pool(4096),
            state_pool(ENTRY_SWA, 8192, entries_per_req=2, extra_entries=64),
            state_pool(ENTRY_STATE, 65536, entries_per_req=1),
        ]
        plan = plan_pools(specs, available_bytes=8 * GIB, max_num_seqs=128)
        assert plan.total_reserved_bytes <= 8 * GIB


class TestPlanIsTheSingleSourceOfCounts:
    """The runner cross-checks actual allocation against
    `total_reserved_bytes` and publishes `entries` to the engine process, so
    the plan has to stay internally consistent — including after the
    pipeline-parallel reconciliation rewrites the paged count."""

    SPECS: ClassVar = [
        page_pool(4096),
        state_pool(ENTRY_SWA, 8192, entries_per_req=2, extra_entries=64),
        state_pool(ENTRY_STATE, 65536, entries_per_req=1),
    ]

    def _plan(self):
        return plan_pools(self.SPECS, available_bytes=8 * GIB, max_num_seqs=128)

    def test_reserved_bytes_is_exactly_what_gets_allocated(self):
        """The runner cross-checks its allocation against these numbers, so a
        class whose reserved bytes drift from `count * entry_bytes` would make
        that check fire on a pool that is in fact correct."""
        plan = self._plan()
        for name, count in plan.entries.items():
            assert plan.reserved_bytes[name] == count * plan.entry_bytes[name]

    def test_with_paged_entries_rewrites_bytes_too(self):
        plan = self._plan().with_paged_entries(1234)
        assert plan.paged_entries == 1234
        assert plan.reserved_bytes[ENTRY_KV] == 1234 * plan.entry_bytes[ENTRY_KV]

    def test_with_paged_entries_leaves_state_classes_alone(self):
        before = self._plan()
        after = before.with_paged_entries(1)
        for name in (ENTRY_SWA, ENTRY_STATE):
            assert after.entries[name] == before.entries[name]
            assert after.reserved_bytes[name] == before.reserved_bytes[name]

    def test_empty_plan_reads_back_as_no_pool(self):
        """Sizing needs a memory profile, so it runs after model warmup. A
        builder asking for its entry count during warmup must get 0, not an
        AttributeError — that guard is what keeps V4's warmup a no-op."""
        plan = PoolPlan.empty()
        assert plan.entries.get(ENTRY_SWA, 0) == 0
        assert plan.paged_entries == 0
        assert plan.total_reserved_bytes == 0


def _pre_spec_arithmetic(
    *,
    block_bytes: int,
    swa_block_bytes: int,
    per_req_bytes: int,
    slots_per_req: int,
    per_decode: int,
    available: int,
    max_num_seqs: int,
):
    """The sizing expression as it stood before `plan_pools` existed.

    Transcribed from ModelRunner.get_num_blocks(); `block_bytes` is the
    un-stripped figure the old `compute_block_bytes()` returned, i.e. it
    still carries the SWA term.
    """
    tensor_bytes = max_num_seqs * slots_per_req * per_req_bytes if per_req_bytes else 0
    available_for_pool = available - tensor_bytes
    if available_for_pool <= 0:
        raise InsufficientPoolBudget(tensor_bytes, available, max_num_seqs)
    if swa_block_bytes > 0:
        compressed_block_bytes = block_bytes - swa_block_bytes
        num_swa = max_num_seqs * per_decode + 64
        num_kv = max(
            0,
            (available_for_pool - num_swa * swa_block_bytes) // compressed_block_bytes,
        )
    else:
        num_swa = 0
        num_kv = available_for_pool // block_bytes
    return num_kv, num_swa


def _spec_arithmetic(
    *,
    block_bytes: int,
    swa_block_bytes: int,
    per_req_bytes: int,
    slots_per_req: int,
    per_decode: int,
    available: int,
    max_num_seqs: int,
):
    specs = [page_pool(block_bytes - swa_block_bytes)]
    if swa_block_bytes > 0:
        specs.append(
            state_pool(
                ENTRY_SWA,
                swa_block_bytes,
                entries_per_req=per_decode,
                extra_entries=64,
            )
        )
    if per_req_bytes:
        specs.append(
            state_pool(ENTRY_STATE, per_req_bytes, entries_per_req=slots_per_req)
        )
    plan = plan_pools(specs, available, max_num_seqs)
    return plan.entries[ENTRY_KV], plan.entries.get(ENTRY_SWA, 0)


# (block_bytes, swa_block_bytes, per_req_bytes, slots_per_req, per_decode)
_SHAPES = [
    # Stateless MHA / MLA: one paged pool, nothing else.
    (1 << 20, 0, 0, 1, 0),
    # GDN hybrid: paged KV + recurrent state with spec rollback slots.
    (1 << 19, 0, 1 << 22, 4, 0),
    # DeepSeek-V4: compressed KV + paged SWA + compressor ring.
    ((1 << 20) + (7 << 20), 7 << 20, 12_210 << 10, 1, 3),
    # Same, with a fatter window (larger per-request SWA floor).
    ((1 << 20) + (7 << 20), 7 << 20, 12_210 << 10, 1, 9),
]
_BUDGETS = [8 * GIB, 40 * GIB, 137 * GIB]
_SEQ_COUNTS = [1, 64, 256, 512]


class TestParityWithPreSpecArithmetic:
    """On every budget that yields a usable pool, `plan_pools` is bit-identical
    to the arithmetic the runner used before this refactor.

    Budgets too small to serve are the one exception, and only in which error
    surfaces — see `test_unservable_budget_fails_either_way`.
    """

    @pytest.mark.parametrize("shape", _SHAPES)
    @pytest.mark.parametrize("available", _BUDGETS)
    @pytest.mark.parametrize("max_num_seqs", _SEQ_COUNTS)
    def test_identical_counts(self, shape, available, max_num_seqs):
        bb, swa_bb, per_req, slots, per_decode = shape
        kwargs = {
            "block_bytes": bb,
            "swa_block_bytes": swa_bb,
            "per_req_bytes": per_req,
            "slots_per_req": slots,
            "per_decode": per_decode,
            "available": available,
            "max_num_seqs": max_num_seqs,
        }
        try:
            expected = _pre_spec_arithmetic(**kwargs)
        except InsufficientPoolBudget:
            with pytest.raises(InsufficientPoolBudget):
                _spec_arithmetic(**kwargs)
            return
        if expected[0] == 0:
            pytest.skip("unservable budget; covered by the failure-mode test")
        assert _spec_arithmetic(**kwargs) == expected

    @pytest.mark.parametrize("shape", _SHAPES)
    @pytest.mark.parametrize("available", _BUDGETS)
    @pytest.mark.parametrize("max_num_seqs", _SEQ_COUNTS)
    def test_unservable_budget_fails_either_way(self, shape, available, max_num_seqs):
        """When the per-request floors leave no room to page, both forms are
        fatal — they differ only in how.

        The old expression deducted the SWA floor after its budget check, so
        it returned zero paged blocks and let a downstream
        `assert num_kvcache_blocks > 0` complain about block size. Classifying
        SWA as per-request folds its floor into the check, so the failure now
        names the pools that did not fit and how much they needed.
        """
        bb, swa_bb, per_req, slots, per_decode = shape
        kwargs = {
            "block_bytes": bb,
            "swa_block_bytes": swa_bb,
            "per_req_bytes": per_req,
            "slots_per_req": slots,
            "per_decode": per_decode,
            "available": available,
            "max_num_seqs": max_num_seqs,
        }
        try:
            old_kv, _ = _pre_spec_arithmetic(**kwargs)
        except InsufficientPoolBudget:
            old_kv = 0
        if old_kv > 0:
            pytest.skip("budget is servable; covered by the parity test")
        with pytest.raises(InsufficientPoolBudget):
            _spec_arithmetic(**kwargs)


class TestEagle3SharesTheTargetBlockIds:
    def test_draft_bytes_raise_the_per_block_cost_not_the_pool_count(self):
        target_only = plan_pools([page_pool(1000)], 10_000, max_num_seqs=1)
        with_draft = plan_pools(
            [page_pool(1000), page_pool(1000)], 10_000, max_num_seqs=1
        )
        assert set(with_draft.entries) == {ENTRY_KV}
        assert with_draft.entries[ENTRY_KV] == target_only.entries[ENTRY_KV] // 2


def test_staging_entries_are_allocated_but_not_admissible():
    """`staging_entries` buys rows the pool must not lease out. Allocation
    sizes tensors from `entries`; admission divides `admission_entries`."""
    plan = plan_pools(
        [
            page_pool(1000),
            SubPoolSpec(
                Pool.STATE, ENTRY_STATE, 500, entries_per_req=2, staging_entries=3
            ),
        ],
        available_bytes=1_000_000,
        max_num_seqs=8,
    )
    assert plan.entries[ENTRY_STATE] == 8 * 2 + 3
    assert plan.admission_entries[ENTRY_STATE] == 8 * 2
    # The reservation covers every allocated row, staging included.
    assert plan.reserved_bytes[ENTRY_STATE] == (8 * 2 + 3) * 500


def test_checkpoint_headroom_is_allocated_and_admissible():
    """The other cushion, and the opposite answer.

    `extra_entries` is checkpoint capacity: groups the state pool is *meant* to
    hand out, which is the whole of `STATE_CKPT_EXTRA_ENTRIES`. Withholding it
    from admission -- which is what the staging split did before these two
    fields were separated -- leaves `num_groups` at `max_num_seqs` and the
    feature does nothing at all.
    """
    plan = plan_pools(
        [
            page_pool(1000),
            state_pool(ENTRY_STATE, 500, entries_per_req=2, extra_entries=3),
        ],
        available_bytes=1_000_000,
        max_num_seqs=8,
    )
    assert plan.entries[ENTRY_STATE] == 8 * 2 + 3
    assert plan.admission_entries[ENTRY_STATE] == 8 * 2 + 3


def test_admission_entries_defaults_to_entries_without_extras():
    plan = plan_pools(
        [page_pool(1000), state_pool(ENTRY_STATE, 500, entries_per_req=2)],
        available_bytes=1_000_000,
        max_num_seqs=8,
    )
    assert plan.admission_entries[ENTRY_STATE] == plan.entries[ENTRY_STATE] == 16


def test_staging_groups_are_sized_by_multiplicity(monkeypatch):
    """A staging *group* is `entries_per_req` rows. Sizing the ring in bare
    entries would run the last staging group off the end of the tensor."""
    span, k = 3, 2  # GDN: span = 1 + num_spec
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", str(k))
    plan = plan_pools(
        [page_pool(1000), state_pool(ENTRY_STATE, 500, entries_per_req=span)],
        available_bytes=1_000_000,
        max_num_seqs=8,
    )
    num_groups = plan.admission_entries[ENTRY_STATE] // span
    assert num_groups == 8
    # The highest staging group's last row must still be inside the tensor.
    assert (num_groups + k - 1) * span + span <= plan.entries[ENTRY_STATE]


def test_the_ring_sits_past_the_headroom_not_on_top_of_it(monkeypatch):
    """Both cushions at once: the ring must start past every admissible group.

    `state_spills_for_batch` addresses a staging entry as `num_groups + slot`,
    and `num_groups` counts the headroom. A ring sized without it would
    overlap groups the pool leases out -- a spill would read a live request's
    state and store it under an evicted checkpoint's hash.
    """
    span, k, headroom = 2, 2, 5
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", str(k))
    monkeypatch.setenv("STATE_CKPT_EXTRA_ENTRIES", str(headroom * span))
    plan = plan_pools(
        [page_pool(1000), state_pool(ENTRY_STATE, 500, entries_per_req=span)],
        available_bytes=1_000_000,
        max_num_seqs=8,
    )
    num_groups = plan.admission_entries[ENTRY_STATE] // span
    assert num_groups == 8 + headroom
    assert (num_groups + k - 1) * span + span <= plan.entries[ENTRY_STATE]


def test_paged_class_admission_equals_entries():
    """PAGE has no per-request reservation, so the two counts coincide and
    `with_paged_entries` must keep them in step."""
    plan = plan_pools([page_pool(1000)], available_bytes=10_500, max_num_seqs=8)
    assert plan.admission_entries[ENTRY_KV] == plan.entries[ENTRY_KV] == 10
    assert plan.with_paged_entries(4).admission_entries[ENTRY_KV] == 4
