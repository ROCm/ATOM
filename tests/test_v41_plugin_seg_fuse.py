# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""`model_options.seg_fuse` of the DeepSeek-V4.1 plugin model, items 1 and 7.

Item 1: the FFN seam's fused MXFP8 pair reaches the shared expert's
`gate_up_proj` -- as an argument where ATOM runs the FFN inline, held on the
expert for the length of the call where a plugin owner of `v41.block.ffn`
runs it, whichever way that owner reaches the shared expert. Item 7: the
shared expert's activation is the one-launch SiLU-mul + MXFP8 quant. CPU
only: the modules are stood in for; what is checked is who is handed what,
and that nothing changes with the option off."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter

seg_fuse = None
v4 = None
host = None
recipe = None


@pytest.fixture(scope="module", autouse=True)
def _modules():
    global seg_fuse, v4, host, recipe
    with atom_without_aiter() as load:
        host = load("atom.plugin_host")
        recipe = load("atom.plugin_recipe")
        v4 = load("atom.models.deepseek_v41_plugin.v4")
        seg_fuse = load("atom.models.deepseek_v41_plugin.seg_fuse")
        yield


@pytest.fixture
def items(monkeypatch):
    """Set `model_options.seg_fuse` as the model reads it."""

    def set_items(*numbers):
        plan = SimpleNamespace(model_options=recipe.ModelOptions(seg_fuse=numbers))
        monkeypatch.setattr(host, "_plan", plan)

    monkeypatch.setattr(host, "_plan", None)
    monkeypatch.setattr(seg_fuse, "_logged", set())
    monkeypatch.setattr(seg_fuse, "COUNTS", dict.fromkeys(seg_fuse.COUNTS, 0))
    return set_items


def _linear(native=32, input_size=64):
    return SimpleNamespace(native_a8_group_rows=native, input_size=input_size)


class _Expert:
    """A `PluginExpert` with stand-in projections: `gate_up_proj` records what
    it was handed, `act_down` returns the rows it got."""

    def __new__(cls, native=32):
        expert = v4.PluginExpert.__new__(v4.PluginExpert)
        torch.nn.Module.__init__(expert)
        expert.calls = []

        def gate_up_proj(x, x_scale=None):
            expert.calls.append(("gate_up", x, x_scale))
            return x

        gate_up_proj.native_a8_group_rows = native
        expert.gate_up_proj = gate_up_proj
        expert.act_down = lambda combined, weights: combined
        expert.seg_fuse_input = None
        expert.seg_fuse_act = None
        expert.use_fused_clamp_act_mul = False
        expert.w2 = _linear()
        return expert


def _moe(shared, *, dual=False, comm_fused=False):
    moe = SimpleNamespace(
        shared_experts=shared,
        _use_dual_stream=dual,
        _use_comm_fused_dispatch=comm_fused,
        prefix="layers.3.ffn",
    )
    moe.forward_quant = lambda hidden, fp8, scale: ("inline", hidden, fp8, scale)
    moe.__call__ = lambda hidden: ("module", hidden)
    return moe


def _block(moe, ffn=None):
    return SimpleNamespace(
        ffn=moe, _ffn=moe.__call__ if ffn is None else ffn, layer_name="v41.layers.3"
    )


def _pair(rows=4, width=64):
    hidden = torch.randn(rows, width)
    fp8 = torch.zeros(rows, width, dtype=torch.uint8)
    scale = torch.zeros(rows, width // 32, dtype=torch.uint8)
    return hidden, fp8, scale


# -- the option ---------------------------------------------------------------


def test_off_by_default_nothing_is_bound_or_installed(items):
    assert recipe.ModelOptions().seg_fuse == ()
    assert not seg_fuse.enabled(1) and not seg_fuse.enabled(7)
    shared = _Expert()
    moe = _moe(shared)
    assert seg_fuse.bind_ffn_quant(_block(moe)) is None
    seg_fuse.install_act(moe)
    assert shared.seg_fuse_act is None and shared.seg_fuse_input is None


def test_the_recipe_takes_items_one_and_seven_only():
    options = recipe.ModelOptions.from_mapping({"seg_fuse": [7, 1, 7]})
    assert options.seg_fuse == (1, 7)
    for bad in ([8], [1, 2], ["1"], [True], "1,7", 1):
        with pytest.raises(recipe.RecipeRefused):
            recipe.ModelOptions.from_mapping({"seg_fuse": bad})


@pytest.mark.parametrize(
    "item, other", [(1, "ffn_mhc_fuse_quant"), (7, "shared_act_quant")]
)
def test_an_item_and_the_option_for_the_same_launch_are_refused(item, other):
    with pytest.raises(recipe.RecipeRefused, match="replace the same launch"):
        recipe.ModelOptions.from_mapping({"seg_fuse": [item], other: True})
    recipe.ModelOptions.from_mapping({"seg_fuse": [8 - item], other: True})


# -- item 1 -------------------------------------------------------------------


def test_item_1_inline_hands_the_pair_down_as_an_argument(items):
    items(1)
    moe = _moe(_Expert())
    block = _block(moe)
    block._ffn_quant = seg_fuse.bind_ffn_quant(block)
    assert block._ffn_quant == moe.forward_quant
    assert seg_fuse.ffn_reads_pair(block)
    hidden, fp8, scale = _pair()
    assert block._ffn_quant(hidden, fp8, scale) == ("inline", hidden, fp8, scale)


@pytest.mark.parametrize("flag", ["dual", "comm_fused"])
def test_item_1_inline_off_the_layers_stream_leaves_the_seam_bf16(items, flag):
    items(1)
    moe = _moe(_Expert(), **{flag: True})
    block = _block(moe)
    block._ffn_quant = seg_fuse.bind_ffn_quant(block)
    assert not seg_fuse.ffn_reads_pair(block)


def test_item_1_needs_a_native_mxfp8_shared_expert(items):
    items(1)
    assert seg_fuse.bind_ffn_quant(_block(_moe(None))) is None
    assert seg_fuse.bind_ffn_quant(_block(_moe(_Expert(native=None)))) is None
    block = _block(_moe(_Expert()))
    block._ffn_quant = None
    assert not seg_fuse.ffn_reads_pair(block)


def _owned_block(items, owner):
    """A block whose `v41.block.ffn` a plugin owner runs."""
    items(1)
    shared = _Expert()
    moe = _moe(shared)
    block = _block(moe, ffn=lambda hidden: owner(moe, hidden))
    block._ffn_quant = seg_fuse.bind_ffn_quant(block)
    assert block._ffn_quant is not None and block._ffn_quant != moe.forward_quant
    assert seg_fuse.ffn_reads_pair(block)
    return block, shared


def test_item_1_behind_an_owner_the_expert_reads_the_pair_of_its_own_rows(items):
    """An owner that calls the shared expert itself, on a view of the rows it
    was given: the expert reads the pair, once, and nothing is left behind."""

    def owner(moe, hidden):
        rows = hidden.reshape(-1, hidden.shape[-1])
        return moe.shared_experts(rows)

    block, shared = _owned_block(items, owner)
    hidden, fp8, scale = _pair()
    out = block._ffn_quant(hidden, fp8, scale)
    (call,) = shared.calls
    assert call[0] == "gate_up" and call[1].data_ptr() == fp8.data_ptr()
    assert call[2].data_ptr() == scale.data_ptr()
    assert call[1].shape == (4, 64) and call[2].shape == (4, 2)
    assert out.data_ptr() == fp8.data_ptr()
    assert shared.seg_fuse_input is None
    assert seg_fuse.COUNTS == {"ffn_pair_offered": 1, "ffn_pair_taken": 1}


def test_item_1_the_pair_is_not_used_for_another_tensor(items):
    """Rows the owner copied, or re-gathered, are not the rows the pair was
    written with: the expert quantizes them itself."""

    def owner(moe, hidden):
        return moe.shared_experts(hidden.clone())

    block, shared = _owned_block(items, owner)
    hidden, fp8, scale = _pair()
    block._ffn_quant(hidden, fp8, scale)
    (call,) = shared.calls
    assert call[1].dtype == hidden.dtype and call[2] is None
    assert shared.seg_fuse_input is None
    assert seg_fuse.COUNTS == {"ffn_pair_offered": 1, "ffn_pair_taken": 0}


def test_item_1_the_pair_does_not_outlive_a_call_that_raises(items):
    def owner(moe, hidden):
        raise RuntimeError("declined")

    block, shared = _owned_block(items, owner)
    with pytest.raises(RuntimeError, match="declined"):
        block._ffn_quant(*_pair())
    assert shared.seg_fuse_input is None


def test_item_1_is_given_out_once(items):
    def owner(moe, hidden):
        moe.shared_experts(hidden)
        return moe.shared_experts(hidden)

    block, shared = _owned_block(items, owner)
    hidden, fp8, scale = _pair()
    block._ffn_quant(hidden, fp8, scale)
    first, second = shared.calls
    assert first[1].data_ptr() == fp8.data_ptr() and second[2] is None


@pytest.mark.parametrize(
    "change",
    [
        lambda h: h[None],  # not [rows, width]
        lambda h: h.double(),  # another dtype
        lambda h: h[:2],  # fewer rows
        lambda h: h[1:],  # other rows of the same buffer
    ],
)
def test_seg_fuse_pair_matches_memory_dtype_and_rows(change):
    shared = _Expert()
    hidden, fp8, scale = _pair()
    shared.seg_fuse_input = (hidden, fp8, scale)
    assert shared._seg_fuse_pair(change(hidden)) is None
    assert shared.seg_fuse_input is not None
    taken = shared._seg_fuse_pair(hidden)
    assert taken[0].shape == (4, 64) and taken[1].shape == (4, 2)
    assert shared.seg_fuse_input is None


# -- item 7 -------------------------------------------------------------------


@pytest.fixture
def act_op(monkeypatch):
    """`seg_fuse_ops.v41_silu_mul_mxfp8` stood in for, on a gfx950."""
    import sys
    import types

    ops = types.ModuleType("atom.model_ops.deepseek_v41.seg_fuse_ops")
    ops.v41_silu_mul_mxfp8 = lambda combined, limit: ("act_fp8", limit)
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    monkeypatch.setattr(seg_fuse, "act_quant_supported", lambda width: width == 64)
    return ops.v41_silu_mul_mxfp8


def test_item_7_installs_the_one_launch_activation(items, act_op):
    items(7)
    shared = _Expert()
    seg_fuse.install_act(_moe(shared))
    assert shared.seg_fuse_act is act_op


def test_item_7_is_ignored_where_w2_cannot_read_the_pair(items, act_op):
    items(7)
    for change in (
        lambda e: setattr(e, "use_fused_clamp_act_mul", True),
        lambda e: setattr(e, "w2", _linear(native=None)),
        lambda e: setattr(e, "w2", _linear(input_size=48)),
    ):
        shared = _Expert()
        change(shared)
        seg_fuse.install_act(_moe(shared))
        assert shared.seg_fuse_act is None
    seg_fuse.install_act(_moe(None))


def test_item_7_serves_the_unweighted_activation_only(items, monkeypatch):
    """`PluginExpert._act`: the installed op for the plain activation; a
    weighted one keeps `silu_and_mul` and the multiply."""
    shared = _Expert()
    shared.use_shared_act_quant = False
    shared.swiglu_limit = 10.0
    shared.w2 = SimpleNamespace(
        native_a8_group_rows=32,
        runs_bf16=False,
        quant_type=SimpleNamespace(value=object()),
    )
    seen = []
    shared.seg_fuse_act = lambda combined, limit: (
        seen.append((tuple(combined.shape), limit)) or ("fp8", "scale")
    )
    monkeypatch.setattr(v4._v4, "fused_scale_kwargs", lambda *a: {})

    def silu_and_mul(out, combined, limit):
        out.fill_(1.0)

    monkeypatch.setattr(v4._v4, "aiter_silu_and_mul", silu_and_mul)
    combined = torch.zeros(3, 16)
    assert shared._act(combined, None) == ("fp8", "scale")
    assert seen == [((3, 16), 10.0)]
    out, scale = shared._act(combined, torch.full((3, 1), 2.0))
    assert scale is None and torch.equal(out, torch.full((3, 8), 2.0))
    assert len(seen) == 1
    shared.seg_fuse_act = None
    out, scale = shared._act(combined, None)
    assert scale is None and torch.equal(out, torch.ones(3, 8))
