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
        hf_config=SimpleNamespace(linear_conv_kernel_dim=4),
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

    @pytest.mark.parametrize(
        "num_spec, kpool, keep", [(2, 1, "snapshot"), (0, 1, "settled"), (2, 4, "none")]
    )
    def test_the_midstep_switch_leaves_a_decode_boundary_as_it_was(
        self, monkeypatch, num_spec, kpool, keep
    ):
        """The switch moves where a prefill checkpoint is taken. What a decode
        boundary keeps is still the backend's `_decode_keep`."""
        monkeypatch.setenv("ATOM_K3_KDA_STATE_MIDSTEP", "1")
        monkeypatch.setattr(K3, "flash_kda_chunk", lambda: 32)
        with tp_world(TP):
            transfer = Builder.state_transfer(builder(num_spec=num_spec, kpool=kpool))
        assert transfer.readable_midstep and transfer.midstep_grid == 32
        assert transfer.decode_keep == keep

    def test_without_the_midstep_switch_aiter_is_not_asked_for_the_grid(
        self, monkeypatch
    ):
        """An aiter without ROCm/aiter#7413 has no `FLASH_KDA_CHUNK` to import."""

        def no_grid():
            raise ImportError("FLASH_KDA_CHUNK")

        monkeypatch.setattr(K3, "flash_kda_chunk", no_grid)
        with tp_world(TP):
            assert not Builder.state_transfer(builder()).readable_midstep

    def test_the_midstep_switch_is_refused_under_pipeline_parallelism(
        self, monkeypatch
    ):
        """At startup, in `state_transfer`."""
        monkeypatch.setenv("ATOM_K3_KDA_STATE_MIDSTEP", "1")
        with (
            tp_world(TP),
            pytest.raises(RuntimeError, match="pipeline_parallel_size=2"),
        ):
            Builder.state_transfer(builder(pp=2))
