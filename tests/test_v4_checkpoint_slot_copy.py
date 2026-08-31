# SPDX-License-Identifier: MIT

"""Which bytes of a DeepSeek-V4 Active Slot a checkpoint carries, and that a
store/restore round trip moves exactly those and nothing else.

`checkpoint_ranges_for` (tested in `test_state_arena.py`) says which bytes of the
compressor *arena* are live. This file covers the step after it: composing
those with the sliding-window rows that share the slot, turning the result into
byte segments at a slot's real address, and round-tripping them through the
copy planner.

The stubs here SUBCLASS `DeepseekV4AttentionMetadataBuilder` and skip its
construction, which would want a ModelRunner, a model and a GPU. They supply
state, never behaviour, so the arithmetic under test is the shipped arithmetic
and a method that grows a new dependency is picked up rather than silently
stopping to stand for anything.
"""

from __future__ import annotations

import ctypes
from dataclasses import replace
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# Every class below reads unbound methods off the builder, and that module
# does `from aiter import dtypes` at load, which the non-GPU CI runner cannot
# satisfy. Asked of the module actually needed rather than of `aiter`: there
# `aiter` resolves as a namespace package -- the failure reads "cannot import
# name 'dtypes' from 'aiter' (unknown location)", not "no module named" -- so
# a guard on `aiter` can succeed and leave the real import to fail anyway.
#
# `exc_type` is not optional here. The module *is* found, so bare
# `importorskip` treats an ImportError out of it as the caller's mistake: a
# deprecation warning on pytest 9.0 and an error from 9.1, which is what CI
# runs. Naming the type is what the warning itself prescribes, and it keeps
# the skip narrow -- anything that is not an ImportError still fails.
Builder = pytest.importorskip(
    "atom.model_ops.attentions.deepseek_v4_attn",
    reason="the V4 builder's module imports aiter at load",
    exc_type=ImportError,
).DeepseekV4AttentionMetadataBuilder

from atom.model_ops.attentions.paged_state_copy import plan_segmented_copy
from atom.model_ops.attentions.state_arena import (
    StateField,
    checkpoint_ranges_for,
    entry_bytes_for,
    field_extents,
)
from atom.model_ops.attentions.v4_pool_geometry import CSA_RATIO, HCA_RATIO

NEG_INF = float("-inf")
ROW_BYTES = 64
SLOTS = 3

# DeepSeek-V4's field list in miniature: two CSA families, one HCA family that
# a checkpoint owes nothing, and a draft window that is a sliding window and so
# stays whole. Same order as `_state_fields`, which is the order the bytes are
# seen in.
FIELDS = [
    StateField("csa_main_kv", 2, (4, 8), torch.float32),
    StateField("csa_main_score", 2, (4, 8), torch.float32, NEG_INF),
    StateField("hca_main_kv", 2, (16, 8), torch.float32, in_checkpoint=False),
    StateField(
        "hca_main_score", 2, (16, 8), torch.float32, NEG_INF, in_checkpoint=False
    ),
    StateField("state_window", 1, (6, 8), torch.float32),
]
ARENA_BYTES = entry_bytes_for(FIELDS)
ARENA_ROWS = -(-ARENA_BYTES // ROW_BYTES)
ENTRY_ROWS = 5  # the windows that share the slot, in row-space rows
# Rows of that entry a window position actually reaches. Row 2 is interleave
# padding: the real geometry has some, and an image that carried it would pass
# every test below while still being bigger than it has to be.
ENTRY_ROW_RUNS = [(0, 2), (3, 2)]
ENTRY_LIVE_ROWS = sum(count for _, count in ENTRY_ROW_RUNS)
ENTRY_PAD_ROWS = [2]
SLOT_ROWS = ARENA_ROWS + ENTRY_ROWS + 1  # +1 so there is tail padding to drop
SLOT_BYTES = SLOT_ROWS * ROW_BYTES


class _Geo:
    """Only what `_checkpoint_slot_ranges` and `_slot_views` read."""

    arena_rows = ARENA_ROWS
    entry_rows = ENTRY_ROWS
    slot_rows = SLOT_ROWS

    def slot_span(self, physical: int) -> tuple[int, int]:
        return physical * SLOT_ROWS, (physical + 1) * SLOT_ROWS

    def physical_slot(self, group: int) -> int:
        return group

    def entry_row_runs(self) -> list[tuple[int, int]]:
        return list(ENTRY_ROW_RUNS)


class _StubBuilder(Builder):
    """The real builder with the construction skipped, not a look-alike.

    Subclassed rather than given a list of borrowed methods: the list was the
    fragile part -- `_page_unit_regions` grew a call to `_indexer_page_pools`
    and six tests started erroring on a double that had silently stopped
    standing for anything. Inheriting means a new dependency is picked up, and
    only genuinely missing STATE fails, loudly. `__init__` deliberately does
    not chain: the real one wants a ModelRunner, a model and a GPU.
    """

    def __init__(self, plane: torch.Tensor, alignment: int = 256):
        self.pool_geometry = _Geo()
        self._arena_planes = [FIELDS]
        self._checkpoint_range_cache = None
        self._checkpoint_slot_base_cache = None
        # What the ladder actually rounds a checkpoint to, which is where the
        # guard has to look: `kv_cache_block_size * dcp_world_size`.
        self.model_runner = SimpleNamespace(
            config=SimpleNamespace(
                kv_cache_block_size=alignment, decode_context_parallel_size=1
            )
        )
        # The ratios the guard reads come from the model, not from this file's
        # constants -- two dense layers and then the CSA/HCA alternation a V4
        # config declares.
        self.compress_ratios = [0, 0, CSA_RATIO, HCA_RATIO, CSA_RATIO, HCA_RATIO]
        self._plane = plane

    def _plane_row_widths(self):
        return [ROW_BYTES]

    def _slot_views(self):
        geo = self.pool_geometry
        return [
            [self._plane[slice(*geo.slot_span(geo.physical_slot(g)))]]
            for g in range(SLOTS)
        ]


@pytest.fixture
def plane():
    # uint8 so a row is ROW_BYTES elements and byte offsets are element offsets.
    return torch.zeros(SLOTS * SLOT_ROWS, ROW_BYTES, dtype=torch.uint8)


def field_spans() -> dict[str, tuple[int, int]]:
    """Each field's own byte range inside an entry, padding excluded.

    Spelled out here rather than read back from the arena so the round trip is
    checked against the layout it is supposed to have, not against the same
    arithmetic it is exercising.
    """
    spans = {}
    offset = 0
    for field in FIELDS:
        align = max(256, field.align)
        offset = -(-offset // align) * align
        spans[field.name] = (offset, offset + field.bytes_per_entry)
        offset += field.bytes_per_entry
    return spans


def dead_arena_span() -> tuple[int, int]:
    """Byte range spanned by the two HCA fields, which are dead together."""
    spans = field_spans()
    return spans["hca_main_kv"][0], spans["hca_main_score"][1]


def store_and_restore(stub, image_ptrs, image_sizes, live_bytes, src, dst):
    """Scatter group `src` into the image, gather it back into group `dst`.

    Host `memmove` rather than the copy kernel: what is under test is which
    bytes the descriptor names, and running it on the CPU keeps the test off
    a GPU. The same plan serves both directions, as it does in production.
    """
    ranges = Builder._checkpoint_slot_ranges(stub)
    plan = plan_segmented_copy(
        [nbytes for plane in ranges for _, nbytes in plane], image_sizes, live_bytes
    )
    slot_bases = Builder._checkpoint_slot_bases(stub)
    for group, forward in ((src, True), (dst, False)):
        descriptor = np.empty((plan.num_spans, 3), dtype=np.int64)
        plan.write_descriptor(
            descriptor,
            slot_bases[group][None],
            np.asarray(image_ptrs, dtype=np.int64)[None],
            forward=forward,
        )
        for source, destination, nbytes in descriptor:
            ctypes.memmove(int(destination), int(source), int(nbytes))


class TestCheckpointSlotRanges:
    def test_the_ranges_skip_the_dead_fields_and_the_slot_tail(self, plane):
        (ranges,) = Builder._checkpoint_slot_ranges(_StubBuilder(plane))
        dead_start, dead_end = dead_arena_span()

        covered = set()
        for start, nbytes in ranges:
            covered |= set(range(start, start + nbytes))

        assert not covered & set(range(dead_start, dead_end)), "HCA is carried"
        # A window is a sliding window, so every row a position reaches is
        # carried — and the interleave padding between them is not.
        for start, count in ENTRY_ROW_RUNS:
            first = (ARENA_ROWS + start) * ROW_BYTES
            assert set(range(first, first + count * ROW_BYTES)) <= covered
        for row in ENTRY_PAD_ROWS:
            first = (ARENA_ROWS + row) * ROW_BYTES
            assert not covered & set(
                range(first, first + ROW_BYTES)
            ), "interleave padding is carried"
        # Neither the padding the arena is rounded up by nor the slot's own
        # tail alignment belongs to anyone.
        assert not covered & set(
            range((ARENA_ROWS + ENTRY_ROWS) * ROW_BYTES, SLOT_BYTES)
        )

    def test_the_ranges_are_ordered_disjoint_and_inside_the_slot(self, plane):
        (ranges,) = Builder._checkpoint_slot_ranges(_StubBuilder(plane))

        assert ranges == sorted(ranges)
        for (a_start, a_bytes), (b_start, _) in pairwise(ranges):
            assert a_start + a_bytes <= b_start
        for start, nbytes in ranges:
            assert 0 <= start and start + nbytes <= SLOT_BYTES

    def test_the_image_size_is_the_arena_live_bytes_plus_the_windows(self, plane):
        stub = _StubBuilder(plane)

        assert Builder.checkpoint_image_bytes(stub) == (
            sum(n for _, n in checkpoint_ranges_for(FIELDS))
            + ENTRY_LIVE_ROWS * ROW_BYTES
        )
        assert Builder.checkpoint_image_bytes(stub) < SLOT_BYTES

    @pytest.mark.parametrize(
        "block_size, dcp",
        [
            (64, 1),  # under HCA's ratio outright
            (96, 1),  # a multiple of neither
            (32, 2),  # 64 tokens of alignment: dcp does not rescue it
        ],
    )
    def test_an_alignment_a_ratio_does_not_divide_is_refused(
        self, plane, block_size, dcp
    ):
        """HCA owes nothing only while a checkpoint lands on a pool boundary.

        The guard reads the alignment the ladder uses, `kv_cache_block_size *
        decode_context_parallel_size`, against the ratios the model declares.
        """
        stub = _StubBuilder(plane)
        stub.model_runner.config.kv_cache_block_size = block_size
        stub.model_runner.config.decode_context_parallel_size = dcp

        with pytest.raises(ValueError, match="not a multiple of compress"):
            Builder.checkpoint_image_bytes(stub)

    @pytest.mark.parametrize("ratio", [3, 512, 96])
    def test_a_model_ratio_the_shipped_alignment_misses_is_refused(self, ratio, plane):
        """The reachable way to break the premise, at the block size that ships.

        `config.py` forces `kv_cache_block_size` to 256 for every `DeepseekV4*`
        architecture, so a guard written against this file's `CSA_RATIO` /
        `HCA_RATIO` could never fire -- 256 is a multiple of both, and no
        configuration changes that. What a variant *can* change is
        `hf_config.compress_ratios`, and a stride 256 does not divide leaves
        HCA's first pool straddling the boundary, which is the silent-accuracy
        failure the guard exists for.
        """
        stub = _StubBuilder(plane)
        assert stub.model_runner.config.kv_cache_block_size == 256
        stub.compress_ratios = [0, CSA_RATIO, ratio]

        with pytest.raises(ValueError, match="not a multiple of compress"):
            Builder.checkpoint_image_bytes(stub)

    @pytest.mark.parametrize("block_size, dcp", [(0, 1), (256, 0)])
    def test_an_alignment_of_zero_is_refused_as_itself(self, plane, block_size, dcp):
        """Every ratio divides zero, so the ratio check cannot speak for this.

        Folded into that check it would find nothing wrong and raise anyway,
        reporting an alignment that "is not a multiple of compress ratios
        `[]`" -- an accusation with no ratio in it, about a pool geometry that
        simply cannot exist. Its own check, so it can say that instead.
        """
        stub = _StubBuilder(plane)
        stub.model_runner.config.kv_cache_block_size = block_size
        stub.model_runner.config.decode_context_parallel_size = dcp

        with pytest.raises(ValueError, match="not a pool geometry that can exist"):
            Builder.checkpoint_image_bytes(stub)

    def test_dcp_can_make_an_alignment_legal(self, plane):
        """The product is what matters, so a small block can still divide."""
        stub = _StubBuilder(plane)
        stub.model_runner.config.kv_cache_block_size = 64
        stub.model_runner.config.decode_context_parallel_size = 2  # 128 tokens

        assert Builder.checkpoint_image_bytes(stub) > 0


class TestRoundTrip:
    """Store slot 0 into PAGE units, gather it back into slot 1."""

    def test_the_live_bytes_survive_and_the_dead_bytes_are_not_touched(self, plane):
        stub = _StubBuilder(plane)
        live_bytes = Builder.checkpoint_image_bytes(stub)

        # Two distinguishable fills, so a byte that failed to move and a byte
        # that moved when it should not have both show up.
        plane[0 * SLOT_ROWS : 1 * SLOT_ROWS] = 0xA5
        plane[1 * SLOT_ROWS : 2 * SLOT_ROWS] = 0x3C
        before = plane[1 * SLOT_ROWS : 2 * SLOT_ROWS].clone()

        # A checkpoint image: arbitrary units, deliberately not contiguous and
        # not in address order, which is the whole point of PAGE backing.
        units = torch.full((4, live_bytes // 2), 0x11, dtype=torch.uint8)
        chosen = (2, 0, 3)
        store_and_restore(
            stub,
            [units[i].data_ptr() for i in chosen],
            [units.shape[1]] * len(chosen),
            live_bytes,
            src=0,
            dst=1,
        )

        after = plane[1 * SLOT_ROWS : 2 * SLOT_ROWS].reshape(-1)
        source = plane[0 * SLOT_ROWS : 1 * SLOT_ROWS].reshape(-1)
        untouched = before.reshape(-1)

        # Derive what must have moved from the layout, NOT from
        # `_checkpoint_slot_ranges`. Checking the ranges against themselves passes
        # for any self-consistent mistake — including a range shifted the same
        # way on both sides, which is exactly the bug worth catching here.
        dead_start, dead_end = dead_arena_span()
        spans = field_spans()
        entry_runs = [
            (
                (ARENA_ROWS + start) * ROW_BYTES,
                (ARENA_ROWS + start + count) * ROW_BYTES,
            )
            for start, count in ENTRY_ROW_RUNS
        ]
        for start, end in (
            spans["csa_main_kv"],
            spans["csa_main_score"],
            spans["state_window"],  # a draft window, behind the dead fields
            *entry_runs,
        ):
            assert torch.equal(
                after[start:end], source[start:end]
            ), f"bytes [{start}, {end}) did not survive the round trip"

        # The dead fields must still hold slot 1's own fill: carrying them
        # would be waste, writing them from anywhere else would be a bug.
        assert torch.equal(
            after[dead_start:dead_end], untouched[dead_start:dead_end]
        ), "the dead HCA fields were overwritten"
        tail = (ARENA_ROWS + ENTRY_ROWS) * ROW_BYTES
        assert torch.equal(
            after[tail:], untouched[tail:]
        ), "the slot's tail padding was overwritten"
        for row in ENTRY_PAD_ROWS:
            first = (ARENA_ROWS + row) * ROW_BYTES
            assert torch.equal(
                after[first : first + ROW_BYTES],
                untouched[first : first + ROW_BYTES],
            ), "the entry's interleave padding was overwritten"

    def test_a_restore_does_not_read_past_the_image(self, plane):
        """`total_bytes` is the image, so the tail of the last unit is spare."""
        stub = _StubBuilder(plane)
        live_bytes = Builder.checkpoint_image_bytes(stub)
        units = torch.full((live_bytes + 4096,), 0x77, dtype=torch.uint8)
        ranges = Builder._checkpoint_slot_ranges(stub)

        plan = plan_segmented_copy(
            [nbytes for p in ranges for _, nbytes in p], [units.numel()], live_bytes
        )
        descriptor = np.empty((plan.num_spans, 3), dtype=np.int64)
        plan.write_descriptor(
            descriptor,
            Builder._checkpoint_slot_bases(stub)[2][None],
            np.array([[units.data_ptr()]], dtype=np.int64),
            forward=False,
        )

        assert descriptor[:, 2].sum() == live_bytes
        assert (descriptor[:, 0] + descriptor[:, 2]).max() - units.data_ptr() == (
            live_bytes
        )


class TestTheBuilderDeclaresWhatItDrops:
    """`_state_fields` is where the rule actually lives.

    Everything above uses its own field list, so none of it says anything
    about what DeepSeek-V4 itself declares — flipping `in_checkpoint` back on
    in the builder left every one of those tests green. This is the one that
    notices.
    """

    @staticmethod
    def builder_stub():
        class _Stub(Builder):
            _state_dtype = torch.float32
            csa_layers = (2, 4, 6)
            hca_layers = (3, 5)
            csa_main_state_shape = (13, 1024)
            csa_idx_state_shape = (13, 256)
            hca_main_state_shape = (133, 512)
            head_dim = 512
            rope_head_dim = 64
            index_head_dim = 128
            block_size = 256
            win_with_spec = 133
            compress_ratios = (0, 0, 4, 128, 4, 128, 4, -1)
            _kv_fp8 = False
            _indexer_fp4 = False
            _field_window_dtype = torch.bfloat16
            _field_window_layers = (43,)

            def __init__(self):
                pass  # the real one wants a ModelRunner, a model and a GPU

        return _Stub()

    @classmethod
    def build_fields(cls) -> list[StateField]:
        return Builder._state_fields(cls.builder_stub())

    def test_hca_is_the_only_thing_dropped(self):
        dropped = {f.name for f in self.build_fields() if not f.in_checkpoint}

        assert dropped == {"hca_main_kv", "hca_main_score"}, (
            "HCA pools [P, P+128) with no overlap, so a resumer writes every "
            "row it reads; nothing else here is known to be dead at a boundary"
        )

    def test_the_dropped_fields_are_outside_every_range(self):
        fields = self.build_fields()
        carried = set()
        for start, nbytes in checkpoint_ranges_for(fields):
            carried |= set(range(start, start + nbytes))

        for field, start, end in field_extents(fields):
            overlap = carried & set(range(start, end))
            if field.in_checkpoint:
                assert overlap, f"{field.name} is carried but has no range"
            else:
                assert not overlap, f"{field.name} is dropped but has one"

    def test_what_is_dropped_is_versioned_into_the_layout_id(self):
        """Two workers disagreeing on the rule read one image two ways.

        Asks `state_transfer` for the id rather than re-deriving the rule
        beside it: re-deriving asserts the rule twice and the fence not at
        all, so dropping the `nocopy=` segment would leave it green.
        """
        stub = self.builder_stub()

        layout_id = Builder.state_transfer(stub).paged_layout_id

        assert ":nocopy=hca_main_kv,hca_main_score" in layout_id
        # The packing rule shares the id, and both are fenced by the version.
        assert ":entry=packed" in layout_id
        assert layout_id.startswith("dsv4-paged-state-v3:")

    def test_the_layout_id_moves_when_the_rule_does(self):
        """The fence has to notice a field changing sides, not just exist."""
        stub = self.builder_stub()
        before = Builder.state_transfer(stub).paged_layout_id

        stub._state_fields = lambda: [
            replace(f, in_checkpoint=True) for f in Builder._state_fields(stub)
        ]
        after = Builder.state_transfer(stub).paged_layout_id

        assert after != before
        assert ":nocopy=:" in after, "the id moved for some other reason"


class TestPageUnitAddressesAreArithmetic:
    """The addresses `_page_unit_regions` computes are the ones slicing gave.

    Replacing 22 throwaway tensor views per unit with three multiplications is
    only safe if it lands on the same bytes, so this asks both and compares.
    The old expression is written out here rather than kept in production: it
    is the oracle, not a fallback.
    """

    N_CSA = 3
    NUM_BLOCKS = 5
    ENVELOPE_ROWS = 7
    ROW_BYTES = 16
    IDX_ROWS = 4
    IDX_ROW_BYTES = 8

    def build(self):
        plane = torch.zeros(
            self.NUM_BLOCKS * self.ENVELOPE_ROWS, self.ROW_BYTES, dtype=torch.uint8
        )
        idx = torch.zeros(
            self.N_CSA,
            self.NUM_BLOCKS,
            self.IDX_ROWS,
            self.IDX_ROW_BYTES,
            dtype=torch.uint8,
        )

        class _Runner:
            v4_csa_idx_kv = idx
            v4_kv_plane = plane
            v4_kv_plane_rope = None

        class _Geo:
            envelope_rows = self.ENVELOPE_ROWS

        class _Stub(Builder):
            model_runner = _Runner()
            pool_geometry = _Geo()
            csa_layers = tuple(range(self.N_CSA))
            _indexer_fp4 = False
            _page_unit_region_cache = None
            _page_unit_region_owners = ()

            def __init__(self):
                pass  # the real one wants a ModelRunner, a model and a GPU

            def _plane_row_widths(self):
                return [16]

        return _Stub(), plane, idx

    def sliced(self, plane, idx, block_id):
        """What a PAGE unit's regions were before, view by view."""
        start = block_id * self.ENVELOPE_ROWS
        views = [plane[start : start + self.ENVELOPE_ROWS]]
        views += [idx[layer, block_id] for layer in range(self.N_CSA)]
        return [(int(v.data_ptr()), v.numel() * v.element_size()) for v in views]

    def test_every_region_lands_where_a_slice_would_have(self):
        stub, plane, idx = self.build()
        _, sizes = Builder._page_unit_regions(stub)

        for block_id in range(self.NUM_BLOCKS):
            (bases,) = Builder._page_unit_bases(stub, [[block_id]])
            got = list(zip(map(int, bases), map(int, sizes), strict=True))
            assert got == self.sliced(plane, idx, block_id), f"block {block_id}"

    def test_an_image_is_its_units_in_order_addresses_and_sizes_together(self):
        """The plan is cut by one of these and addressed through the other.

        Unit major, region minor in both. Were the two to disagree, a plan cut
        against one order and addressed through the other would put whole
        regions in the wrong unit — a silently wrong checkpoint, not a crash.
        """
        stub, plane, idx = self.build()
        chosen = [3, 0, 4]

        (bases,) = Builder._page_unit_bases(stub, [chosen])
        sizes = Builder._page_unit_stream_sizes(stub, len(chosen))

        assert list(zip(map(int, bases), map(int, sizes), strict=True)) == [
            pair for block in chosen for pair in self.sliced(plane, idx, block)
        ]

    def test_the_regions_are_worked_out_once(self):
        stub, _, _ = self.build()

        first = Builder._page_unit_regions(stub)
        assert Builder._page_unit_regions(stub) is first

    def test_a_non_contiguous_pool_is_refused_not_mis_addressed(self):
        """Affine addressing assumes the layout; say so rather than guess."""
        stub, _, idx = self.build()
        stub.model_runner.v4_csa_idx_kv = idx.transpose(0, 1)

        with pytest.raises(RuntimeError, match="contiguous"):
            Builder._page_unit_regions(stub)


class TestWarmup:
    """That the checkpoint copy path is paid for before a request can pay it."""

    def test_a_backend_without_paged_checkpoints_warms_nothing(self):
        """The hook is on the base builder, so every backend answers it."""
        from atom.model_ops.attentions.backends import AttentionMetadataBuilder

        # No-op by contract: a backend that never copies has nothing to warm,
        # and ModelRunner calls this unconditionally.
        assert AttentionMetadataBuilder.warmup_per_req_cache(object()) is None

    def test_the_runner_warms_the_builder_once_the_pools_are_reachable(self):
        """After the setattr loop, not before: the builder reads them off it.

        Guarded here rather than left to review because nothing else calls
        `warmup_per_req_cache` -- a runner that stopped would restore the
        first-request JIT silently, with every test still green.
        """
        import inspect

        from atom.model_engine import model_runner

        src = inspect.getsource(model_runner.ModelRunner.allocate_kv_cache)
        install = src.index("setattr(self, name, value)")
        warm = src.index("warmup_per_req_cache()")
        assert install < warm, "warmed before the pools were installed"


class TestPageUnitRegionsValidateTheirOwnAddresses:
    """The region cache holds addresses from two allocations, one uncovered.

    `_invalidate_pool_caches` is called from `allocate_per_req_cache`, which
    owns the KV planes; the indexer pools come from `allocate_kv_cache_tensors`
    and it does not. A cache relying on that hook would hold freed base
    pointers after any path that re-runs the first allocation alone, and the
    copy kernel would scatter into whatever the allocator handed that range to
    -- no fault, silent corruption. So it keys on its own addresses.
    """

    def _stub(self, plane, idx):
        stub = _StubBuilder.__new__(_StubBuilder)
        stub.pool_geometry = SimpleNamespace(envelope_rows=2)
        stub._page_unit_region_cache = None
        stub._page_unit_region_owners = ()
        stub._indexer_fp4 = False
        stub.csa_layers = [0]
        stub.model_runner = SimpleNamespace(v4_csa_idx_kv=idx)
        stub._kv_planes = lambda: [plane]
        stub._plane_row_widths = lambda: [ROW_BYTES]
        return stub

    def test_a_moved_indexer_pool_is_noticed_without_the_hook(self):
        first = torch.zeros(4, ROW_BYTES, dtype=torch.uint8)
        idx = torch.zeros(1, 4, 8, dtype=torch.uint8)
        stub = self._stub(first, idx)
        bases, _ = Builder._page_unit_regions(stub)
        assert idx.data_ptr() in set(bases.tolist())

        # The pool is reallocated by the other allocation; nothing calls the
        # invalidator, exactly as today's ordering would allow.
        moved = torch.zeros(1, 4, 8, dtype=torch.uint8)
        assert moved.data_ptr() != idx.data_ptr(), "the test needs a real move"
        stub.model_runner.v4_csa_idx_kv = moved

        again, _ = Builder._page_unit_regions(stub)

        assert moved.data_ptr() in set(again.tolist()), "answered from a freed base"
        assert idx.data_ptr() not in set(again.tolist())

    def test_an_unmoved_pool_is_answered_from_the_cache(self):
        plane = torch.zeros(4, ROW_BYTES, dtype=torch.uint8)
        stub = self._stub(plane, torch.zeros(1, 4, 8, dtype=torch.uint8))

        first = Builder._page_unit_regions(stub)

        assert Builder._page_unit_regions(stub) is first, "rebuilt for nothing"
