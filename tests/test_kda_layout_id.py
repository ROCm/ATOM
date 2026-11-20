# SPDX-License-Identifier: MIT

"""What a K3 state-checkpoint layout id has to encode, and what must move it.

A PAGE image is raw bytes. The scheduler reserves units against one geometry
and a worker reassembles them against its own, so the only thing standing
between a mismatched pair and silently misplaced state is this string. Every
property that changes where a byte lands has to change the id.

The id is asked of `state_transfer()` rather than re-derived beside the test:
re-deriving it would assert that two copies of the same expression agree, which
is what the production one already does with itself.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.state_runtime import StateTransfer
from atom.model_ops.attentions.pool_layout.sub_pool_spec import (
    page_pool,
    plan_pools,
    state_pool,
)

K3 = pytest.importorskip(
    "atom.model_ops.attentions.kimi_mla_gdn_attn",
    reason="the K3 builder's module imports aiter at load",
    exc_type=ImportError,
)
Builder = K3._KimiMLAGDNCommon

SHAPE_K = (3, 4608)
SHAPE_V = (12, 128, 128)
DT_K = torch.bfloat16
DT_V = torch.float32
TP = 8


def builder(
    shape_k=SHAPE_K,
    shape_v=SHAPE_V,
    dt_k=DT_K,
    dt_v=DT_V,
    layers=69,
    num_spec=0,
    pp=1,
    rapidserve=False,
    replayssm=False,
    kpool=1,
    kpool_tail=None,
):
    """`kpool` is the model's (GLM-5.3 pools indexer keys); `kpool_tail` says
    whether this PP stage holds indexer layers, and with them the tail plane."""
    config = SimpleNamespace(
        pipeline_parallel_size=pp,
        enable_rapidserve=rapidserve,
    )
    if kpool_tail is None:
        kpool_tail = kpool > 1
    tail = (256, 3) if kpool_tail else None
    stub = SimpleNamespace(
        model_runner=SimpleNamespace(config=config),
        num_state_layers=lambda: layers,
        num_spec=num_spec,
        replayssm=replayssm,
        _state_shape_for_runner=lambda: (shape_k, shape_v),
        _state_dtypes=lambda: (dt_k, dt_v),
        _kpool_size=lambda: kpool,
        _kpool_tail_plane_shape=lambda: tail,
        _checkpoint_plane_shapes=lambda: [(1, layers)] * 2 + [tail] * bool(tail),
    )
    for name in (
        "_uses_paged_checkpoints",
        "_decode_keep",
        "_decode_image_is_conv_and_ssm",
    ):
        setattr(stub, name, getattr(Builder, name).__get__(stub, type(stub)))
    return stub


@contextmanager
def tp_world(size: int):
    """`state_transfer` reads the TP group, which no test process has.

    Patched on the module under test rather than on `aiter.dist.parallel_state`:
    other test files stub `aiter` into `sys.modules`, so the import path is not
    reliably the one this module bound at load, but the name it bound is.
    """
    original = K3.get_tp_group
    K3.get_tp_group = lambda: SimpleNamespace(world_size=size)
    try:
        yield
    finally:
        K3.get_tp_group = original


def layout_of(**kwargs) -> str:
    with tp_world(kwargs.pop("tp", TP)):
        return Builder.state_transfer(builder(**kwargs)).paged_layout_id


class TestTheIdNamesItsVersionAndItsRule:
    def test_it_is_a_versioned_kda_layout(self):
        assert layout_of().startswith("kda-paged-state-v1:")

    def test_it_states_that_the_whole_slot_is_carried(self):
        """The narrowing rule, so dropping the conv tail later is a `v2` rather
        than a silent reinterpretation of a v1 image."""
        assert ":carry=all" in layout_of()

    def test_it_states_the_plane_order(self):
        """The one thing the shapes cannot say. A reader assembling the image
        interleaved gets every layer but the first wrong."""
        assert ":order=conv-all-layers,ssm-all-layers" in layout_of()


class TestEverythingThatMovesAByteMovesTheId:
    """Each of these changes where a segment starts, how long it is, or how
    many there are. A pair of workers disagreeing on any of them must refuse
    each other's images rather than reassemble them at the wrong offsets."""

    def test_the_conv_shape(self):
        assert layout_of() != layout_of(shape_k=(3, 4096))

    def test_the_ssm_shape(self):
        assert layout_of() != layout_of(shape_v=(12, 128, 64))

    def test_the_conv_dtype(self):
        assert layout_of() != layout_of(dt_k=torch.float16)

    def test_the_ssm_dtype(self):
        """The v side is `ATOM_GDN_SSM_DTYPE`. fp16 and bf16 are the same size,
        so the id is the only thing telling those two apart."""
        assert layout_of() != layout_of(dt_v=torch.bfloat16)
        assert layout_of(dt_v=torch.float16) != layout_of(dt_v=torch.bfloat16)

    def test_the_layer_count(self):
        assert layout_of() != layout_of(layers=68)

    def test_the_tp_size(self):
        assert layout_of() != layout_of(tp=4)

    def test_the_speculative_token_count(self):
        """The conv state is `(conv_kernel - 1 + num_spec, ...)`, so this moves
        the image's size as well as its offsets."""
        assert layout_of() != layout_of(num_spec=2)

    def test_the_same_geometry_gives_the_same_id(self):
        assert layout_of() == layout_of()


class TestPagedCheckpointsAreRefusedWhereSizingWouldRaise:
    """`get_num_blocks` raises on a copying transfer under RapidServe.
    Answering `copy` there would turn "K3 keeps no state cache" into "K3 does
    not start"."""

    def test_pipeline_parallelism_copies(self):
        """The stages agree on one unit geometry, so a checkpoint is a PAGE
        image under PP too rather than a slot competing with admissions."""
        with tp_world(TP):
            transfer = Builder.state_transfer(builder(pp=2))
        assert transfer.copies and not transfer.forks
        assert Builder.paged_checkpoints_span_pipeline

    def test_rapidserve_falls_back_to_fork(self):
        with tp_world(TP):
            transfer = Builder.state_transfer(builder(rapidserve=True))
        assert transfer.forks and not transfer.copies

    def test_the_ordinary_case_copies(self):
        with tp_world(TP):
            transfer = Builder.state_transfer(builder())
        assert transfer.copies and not transfer.forks

    def test_a_copy_is_never_midstep_readable(self):
        """KDA's chunk kernel returns only the final state, so there is no
        interior state to slice a checkpoint out of."""
        with tp_world(TP):
            assert not Builder.state_transfer(builder()).readable_midstep

    @pytest.mark.parametrize(
        "replayssm, num_spec, kpool, keep",
        [
            (True, 2, 1, "snapshot"),
            (True, 0, 1, "snapshot"),
            (False, 2, 1, "snapshot"),
            (False, 0, 1, "settled"),
            (True, 2, 4, "none"),
            (False, 0, 4, "settled"),
        ],
    )
    def test_a_decode_boundary_keeps_what_the_slot_can_give(
        self, replayssm, num_spec, kpool, keep
    ):
        """Under MTP or ReplaySSM the slot is ahead of a decode boundary, and
        the image is rebuilt only where it is conv and SSM alone."""
        with tp_world(TP):
            transfer = Builder.state_transfer(
                builder(replayssm=replayssm, num_spec=num_spec, kpool=kpool)
            )
        assert transfer.decode_keep == keep

    @pytest.mark.parametrize("kpool_tail", [True, False])
    def test_every_pp_stage_of_a_kpool_model_keeps_none(self, kpool_tail):
        """A stage with no indexer layer holds no tail plane, but the head
        files and sizes scratch for every stage, so it answers as they do."""
        with tp_world(TP):
            transfer = Builder.state_transfer(
                builder(replayssm=True, pp=2, kpool=4, kpool_tail=kpool_tail)
            )
        assert transfer.decode_keep == "none"


# ── PP: every stage keeps its images in a store of its own ─────────────────

GIB = 1 << 30
MLA_ROW = 128 * 576
KDA_IMAGE = 6_807_552
KDA_SLOT = 8_380_416
# MLA rows, KDA layers and logged available_for_kv (GiB) per stage of the
# measured PP8 x TP1 run, VLLM_PP_LAYER_PARTITION=13,13,13,13,13,12,12,4; the
# last stage holds the drafter's 5 MLA rows.
PP8_STAGES = list(
    zip(
        (3, 3, 3, 4, 3, 3, 3, 7),
        (10, 10, 10, 9, 10, 9, 9, 2),
        (137.32, 137.37, 157.57, 138.30, 138.54, 152.46, 153.11, 276.36),
    )
)
PP8_ENTRIES = [kda * KDA_IMAGE for _, kda, _ in PP8_STAGES]


class _Contributed(Exception):
    pass


def on_every_stage(monkeypatch, call, stages):
    """`call(rank)` on every PP stage, each gather answered with what every
    stage put in, the way the group's `all_gather_object` would."""
    import torch.distributed as dist

    put = [None] * stages

    def collect(rank):
        def gather(out, obj, group=None):
            put[rank] = obj
            raise _Contributed

        return gather

    for rank in range(stages):
        monkeypatch.setattr(dist, "all_gather_object", collect(rank))
        with pytest.raises(_Contributed):
            call(rank)
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda out, obj, group=None: out.__setitem__(slice(None), put),
    )
    return [call(rank) for rank in range(stages)]


class TestEveryPPStageKeepsItsOwnImages:
    """`ModelRunner` sizes a checkpoint store on every stage, one count for all
    of them and each entry that stage's own image, and the stages then agree
    on one entry an image rather than the largest stage's KV blocks."""

    @pytest.fixture
    def mr(self, monkeypatch):
        mr = pytest.importorskip(
            "atom.model_engine.model_runner",
            reason="the runner's module imports aiter at load",
            exc_type=ImportError,
        )
        import torch.distributed as dist

        monkeypatch.delenv("ATOM_PP_STATE_CHECKPOINT_SLOTS", raising=False)
        monkeypatch.delenv("ATOM_PP_STATE_CHECKPOINT_KV_SHARE", raising=False)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(
            mr, "get_pp_group", lambda: SimpleNamespace(world_size=8, cpu_group=None)
        )
        return mr

    def stage(self, mr, rank, pp=8, prefix_caching=True, stateless_rank=-1):
        rows, kda, gib = PP8_STAGES[rank]
        kda = 0 if rank == stateless_rank else kda
        specs = [page_pool(rows * MLA_ROW)]
        if kda:
            specs.append(
                state_pool(STATE_SLOT_CLASS, kda * KDA_SLOT, entries_per_req=1)
            )
        available = int(gib * GIB)
        stub = SimpleNamespace(
            config=SimpleNamespace(
                pipeline_parallel_size=pp,
                enable_prefix_caching=prefix_caching,
                state_checkpoint_interval_tokens=8192,
            ),
            attn_metadata_builder=SimpleNamespace(
                paged_checkpoints_span_pipeline=True,
                state_transfer=lambda: StateTransfer.copy("kda"),
                checkpoint_image_bytes=lambda: kda * KDA_IMAGE,
                sub_pool_specs=lambda: specs,
            ),
            plan=plan_pools(specs, available, 64),
            available=available,
            transfer=StateTransfer.copy("kda"),
            slot_bytes=kda * KDA_SLOT,
        )
        for name in (
            "_plan_pipeline_checkpoint_store",
            "_local_checkpoint_image_bytes",
            "_local_checkpoint_spec",
            "_agree_pipeline_checkpoint_spec",
        ):
            method = getattr(mr.ModelRunner, name)
            setattr(stub, name, method.__get__(stub, type(stub)))
        return stub

    def stores(self, mr, monkeypatch, **kwargs):
        stubs = [self.stage(mr, rank, **kwargs) for rank in range(8)]
        return stubs, on_every_stage(
            monkeypatch,
            lambda r: stubs[r]._plan_pipeline_checkpoint_store(
                stubs[r].plan, stubs[r].available
            ),
            8,
        )

    def test_every_stage_holds_one_count_at_its_own_image(self, mr, monkeypatch):
        _, stores = self.stores(mr, monkeypatch)
        assert stores == [(1172, entry) for entry in PP8_ENTRIES]

    def test_the_stages_agree_on_one_entry_an_image(self, mr, monkeypatch):
        stubs, stores = self.stores(mr, monkeypatch)
        in_store = on_every_stage(
            monkeypatch,
            lambda r: stubs[r]._agree_pipeline_checkpoint_spec(
                stubs[r]._local_checkpoint_spec(
                    stubs[r].transfer, stubs[r].slot_bytes, stores[r][1]
                )
            ),
            8,
        )
        in_blocks = on_every_stage(
            monkeypatch,
            lambda r: stubs[r]._agree_pipeline_checkpoint_spec(
                stubs[r]._local_checkpoint_spec(stubs[r].transfer, stubs[r].slot_bytes)
            ),
            8,
        )

        assert {spec.units_per_checkpoint for spec in in_store} == {1}
        assert [spec.page_unit_bytes for spec in in_store] == PP8_ENTRIES
        assert {spec.units_per_checkpoint for spec in in_blocks} == {308}
        assert {spec.op_bytes for spec in in_store + in_blocks} == {sum(PP8_ENTRIES)}
        # One joint id per layout, and the two layouts never read each other.
        assert len({spec.layout_id for spec in in_store}) == 1
        assert in_store[0].layout_id != in_blocks[0].layout_id

    def test_without_pp_nothing_is_planned_or_gathered(self, mr, monkeypatch):
        """No store and no gather (no group exists to answer one): the spec
        is the KV-block one it always was."""
        stub = self.stage(mr, 0, pp=1)
        assert stub._plan_pipeline_checkpoint_store(stub.plan, stub.available) is None
        spec = stub._local_checkpoint_spec(stub.transfer, stub.slot_bytes)
        assert (spec.page_unit_bytes, spec.units_per_checkpoint) == (3 * MLA_ROW, 308)

    @pytest.mark.parametrize(
        "kwargs, slots, expected",
        [
            # Prefix caching off: the layout stays, the store holds none.
            ({"prefix_caching": False}, None, [(0, e) for e in PP8_ENTRIES]),
            # A negative count keeps the images in KV blocks.
            ({}, "-1", [None] * 8),
            # Every stage still gathers, so none is left waiting, then all skip
            # the store: `_agree_pipeline_checkpoint_spec` names the stage.
            ({"stateless_rank": 7}, None, [None] * 8),
        ],
    )
    def test_where_no_store_holds_images(
        self, mr, monkeypatch, kwargs, slots, expected
    ):
        if slots is not None:
            monkeypatch.setenv("ATOM_PP_STATE_CHECKPOINT_SLOTS", slots)
        assert self.stores(mr, monkeypatch, **kwargs)[1] == expected
