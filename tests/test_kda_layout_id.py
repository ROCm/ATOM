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
):
    config = SimpleNamespace(
        pipeline_parallel_size=pp,
        enable_rapidserve=rapidserve,
    )
    stub = SimpleNamespace(
        model_runner=SimpleNamespace(config=config, num_gdn_attn_state=layers),
        num_spec=num_spec,
        _state_shape_for_runner=lambda: (shape_k, shape_v),
        _state_dtypes=lambda: (dt_k, dt_v),
    )
    stub._uses_paged_checkpoints = Builder._uses_paged_checkpoints.__get__(
        stub, type(stub)
    )
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
        """The v side is `ATOM_KDA_SSM_DTYPE`. fp16 and bf16 are the same size,
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
    """`get_num_blocks` raises on a copying transfer under PP or RapidServe.
    Answering `copy` there would turn "K3 keeps no state cache" into "K3 does
    not start"."""

    def test_pipeline_parallelism_falls_back_to_fork(self):
        with tp_world(TP):
            transfer = Builder.state_transfer(builder(pp=2))
        assert transfer.forks and not transfer.copies

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
