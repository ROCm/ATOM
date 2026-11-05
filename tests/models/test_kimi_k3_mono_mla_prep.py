# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``ATOM_MONO_MLA_PREP``: an MLA layer's attention seam (and, sub-modes 2 / 3,
its q_a / kv_a, q_b, absorb, rope and KV-cache write) as the tail of the
preceding layer's K2 (``kernels.stages.mla_prep``), host side.

GPU-free: the flags, the bind-time refusals, which K2 builds carry the stage,
that a folded layer takes the stage's outputs while replaying the production
seam's allocations, and sub-mode 3's buffers, arguments and decode entry. Whether the stage is right needs a GPU
(``ATOM_MONO_MLA_PREP_CHECK``).
"""

from types import SimpleNamespace

import pytest
import torch

from atom.models.kimi_k3.mono.config import Dims
from atom.utils import envs

ROWS, HIDDEN = 4, 8


@pytest.fixture
def runner_module():
    pytest.importorskip("flydsl", reason="importing the K3 mono runner needs FlyDSL")
    pytest.importorskip("aiter", reason="the K3 mono runner imports aiter.dist")
    from atom.models.kimi_k3.mono import runner

    return runner


def test_flags_default_off(monkeypatch):
    for name in ("ATOM_MONO_MLA_PREP", "ATOM_MONO_MLA_PREP_LAYERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ATOM_MONO_MLA_PREP_CHECK", raising=False)
    assert envs.ATOM_MONO_MLA_PREP == "0"
    assert envs.ATOM_MONO_MLA_PREP_LAYERS == ""
    assert envs.ATOM_MONO_MLA_PREP_CHECK is False


def _runner(runner_module, prep=None):
    r = object.__new__(runner_module.K3MonoDecodeRunner)
    r.rank = 0
    if prep is not None:
        r.mla_prep = {3: prep}
        r.mla_prep_prev = {2: 3}
    return r


def _prep(blocks=1):
    seam = SimpleNamespace(eps=1e-5, out_eps=2e-5, nw=None, qk=None, out_nw=None)
    return SimpleNamespace(mode=1, blocks=blocks, seam=seam)


def test_unbound_runner_builds_no_prep(runner_module):
    r = _runner(runner_module)
    assert r._prep_key(2, 1) == {}


def test_prep_key_only_on_the_preceding_layer_and_inside_the_width(
    runner_module,
):
    from atom.models.kimi_k3.mono.kernels.stages import mla_prep as k2pr

    r = _runner(runner_module, _prep(blocks=1))
    assert r._prep_key(2, 1) == {
        "mla_prep": 1,
        "prep_nblocks": 1,
        "prep_eps": 1e-5,
        "prep_out_eps": 2e-5,
    }
    assert r._prep_key(2, k2pr.MAX_ROWS)["mla_prep"] == 1
    assert r._prep_key(2, k2pr.MAX_ROWS + 1) == {}
    assert r._prep_key(3, 1) == {}


def test_prep_key_none_where_p0s_sums_do_not_fit_red(runner_module):
    """16 rows on 7+ blocks overflow K2's ``red`` (``build`` asserts, which
    would refuse the whole width): that layer keeps its own seam there."""
    from atom.models.kimi_k3.mono.kernels.stages import mla_prep as k2pr

    r = _runner(runner_module, _prep(blocks=7))
    assert not k2pr.red_fits(16, 8) and k2pr.red_fits(12, 8)
    assert r._prep_key(2, 16) == {}
    assert r._prep_key(2, 12)["prep_nblocks"] == 7


def test_an_unknown_sub_mode_is_refused(runner_module, monkeypatch):
    from atom.mono.runtime.consensus import MonoUnsupported

    r = _runner(runner_module)
    r.plans = []
    monkeypatch.setenv("ATOM_MONO_MLA_PREP", "4")
    with pytest.raises(MonoUnsupported, match="sub-modes are 1"):
        r._bind_mla_prep()
    monkeypatch.setenv("ATOM_MONO_MLA_PREP", "0")
    r._bind_mla_prep()
    assert r.mla_prep == {} and r.mla_prep_prev == {}


def test_a_folded_layer_takes_the_stage_outputs(runner_module):
    """With K2(M-1)'s outputs pending, the layer skips its seam, takes ``x8`` /
    ``xs`` as ``normed`` and ``pref`` as its prefix, and replays the seam's
    allocations (its ``pref`` held for the forward loop)."""
    r = _runner(runner_module, _prep())
    layer = SimpleNamespace(
        self_attention_attn_res=lambda *a: pytest.fail("the seam ran")
    )
    r.inner = SimpleNamespace(layers={3: layer})
    r.mla_prep_check = False
    pend = {
        "pref": torch.zeros(ROWS, HIDDEN),
        "x8": torch.zeros(ROWS, HIDDEN),
        "xs": torch.ones(ROWS, 1),
    }
    r._prep_pending = {3: pend}
    plan = SimpleNamespace(layer_id=3)
    prefix = torch.ones(ROWS, HIDDEN)
    blocks = torch.zeros(ROWS, 4, HIDDEN)
    add = torch.full((ROWS, HIDDEN), 2.0)
    pend["attn"] = None
    got, normed, new_prefix, mirror, attn = r._prepped_seam(
        plan, prefix, blocks, add, None
    )
    assert attn is None
    assert got is layer and new_prefix is pend["pref"]
    assert normed[0] is pend["x8"] and normed[1] is pend["xs"]
    y, y_scale = mirror
    assert y.shape == (ROWS, HIDDEN) and y.dtype == torch.float8_e4m3fn
    assert y_scale.shape == (ROWS, 1)
    assert r._pref_hold_next.shape == prefix.shape
    assert r._prep_pending == {}


def test_an_unfolded_layer_runs_its_seam(runner_module):
    r = _runner(runner_module)
    calls = {}

    def seam(plan, prefix, blocks, add, add2):
        calls["seam"] = True
        return "layer", "normed", "prefix", 1, -1

    r._seam = seam
    plan = SimpleNamespace(layer_id=3)
    out = r._prepped_seam(plan, None, None, None, None)
    assert out == ("layer", "normed", "prefix", None, None) and calls["seam"]


def test_bind_refuses_before_reading_the_seam():
    from atom.models.kimi_k3.mono import weights
    from atom.mono.runtime.consensus import MonoUnsupported

    sa = SimpleNamespace(fuse_input_norm_quant=True)
    off = SimpleNamespace(enabled=False, out_norm=None)
    layer = SimpleNamespace(self_attn=sa, self_attention_attn_res=off)
    with pytest.raises(MonoUnsupported, match="attention residuals off"):
        weights.bind_mla_prep(layer, 3, Dims(8), 1)
    with pytest.raises(MonoUnsupported, match="sub-modes"):
        weights.bind_mla_prep(layer, 3, Dims(8), 4)


def test_prep_key_carries_each_sub_mode_s_constants(runner_module):
    prep = _prep()
    prep.mode = 3
    prep.qa_eps, prep.kva_eps = 1e-6, 2e-6
    prep.rope_neox = False
    prep.cos = torch.zeros(4096, 32)
    r = _runner(runner_module, prep)
    key = r._prep_key(2, 4)
    assert key["mla_prep"] == 3
    assert (key["qa_eps"], key["kva_eps"]) == (1e-6, 2e-6)
    assert (key["rope_neox"], key["rope_maxpos"]) == (False, 4096)


FP8 = torch.float8_e4m3fn


def _prep3(q_heads=16):
    prep = _prep()
    prep.mode, prep.q_heads = 3, q_heads
    for name in ("w_qb", "w_qb_s", "w_k", "w_k_s", "cos", "sin"):
        setattr(prep, name, torch.zeros(4))
    prep.k_scale, prep.q_scale = torch.ones(1), torch.ones(1)
    return prep


def _runner3(runner_module, prep, rows=ROWS, max_rows=8):
    r = _runner(runner_module, prep)
    r.max_rows, r.device = max_rows, "cpu"
    r.inner = SimpleNamespace(config=SimpleNamespace(hidden_size=HIDDEN))
    r._alloc_prep_buffers()
    r._positions = torch.arange(rows, dtype=torch.int64) + 100
    return r


def test_sub_mode_3_buffers(runner_module):
    """q_out at the decode's head stride, zeroed (the padded heads stay 0);
    a dummy run's slots all negative."""
    from atom.models.kimi_k3.mono.kernels.stages import mla_prep as k2pr

    r = _runner3(runner_module, _prep3(q_heads=16))
    b = r._prep_bufs
    assert b["qb"].shape == (8, k2pr.QB) and b["qb"].dtype == torch.bfloat16
    assert b["q_out"].shape == (8, 16, k2pr.ENTRY) and b["q_out"].dtype == FP8
    assert not b["q_out"].float().any()
    assert r._neg_slots.shape == (8,) and (r._neg_slots == -1).all()


def test_sub_mode_3_needs_one_q_out_head_stride(runner_module):
    from atom.mono.runtime.consensus import MonoUnsupported

    r = _runner(runner_module, _prep3(q_heads=16))
    r.mla_prep[7] = _prep3(q_heads=12)
    r.max_rows, r.device = 8, "cpu"
    r.inner = SimpleNamespace(config=SimpleNamespace(hidden_size=HIDDEN))
    with pytest.raises(MonoUnsupported, match="q_out heads"):
        r._alloc_prep_buffers()


def _fc(slots, dtype_q=FP8, cache=None, dummy=False):
    if cache is None:
        # 4 blocks of 4 slots: the stages address slot * 576 B
        cache = torch.zeros(4, 4, 1, 576, dtype=FP8)
    return SimpleNamespace(
        attn_metadata=SimpleNamespace(dtype_q=dtype_q, slot_mapping=slots),
        kv_cache_data={"layer_3": SimpleNamespace(k_cache=cache)},
        context=SimpleNamespace(is_dummy_run=dummy),
    )


def test_sub_mode_3_args(runner_module, monkeypatch):
    from atom.models.kimi_k3.mono.kernels.stages import mla_prep as k2pr

    prep = _prep3()
    r = _runner3(runner_module, prep)
    # a slot mapping longer than the step (the tokens' own are read)
    slots = torch.tensor([5, -1, 9, 2, 7, 7], dtype=torch.int64)
    fc = _fc(slots)
    monkeypatch.setattr(runner_module, "get_forward_context", lambda: fc)
    out = {}
    args = r._prep_args3(3, prep, ROWS, out)
    cache = fc.kv_cache_data["layer_3"].k_cache
    assert args["kvc"] == cache.data_ptr() and args["kvc_sr"] == k2pr.ENTRY
    assert args["slots"] == slots.data_ptr()
    assert args["pos"] == r._positions.data_ptr()
    assert args["qout"] == r._prep_bufs["q_out"].data_ptr()
    assert args["qout_sr"] == 16 * k2pr.ENTRY
    assert args["k_sc"] == prep.k_scale.data_ptr()
    assert out["slots"].tolist() == [5, -1, 9, 2]
    assert out["kvrows"].shape == (16, k2pr.ENTRY)
    assert out["kv3"].shape == (4, 4, k2pr.ENTRY)
    assert out["attn"]["q_out"].shape == (ROWS, 16, k2pr.ENTRY)
    assert out["qb"].shape == (ROWS, k2pr.QB)
    # the ABI's mode-3 arguments, every one
    mode3 = k2pr.ARGS[k2pr.ARGS.index("w_qb") :]
    assert set(args) == set(mode3)

    # a dummy run writes no KV-cache row
    monkeypatch.setattr(
        runner_module, "get_forward_context", lambda: _fc(slots, dummy=True)
    )
    out = {}
    args = r._prep_args3(3, prep, ROWS, out)
    assert args["slots"] == r._neg_slots.data_ptr()
    assert out["slots"].tolist() == [-1] * ROWS


@pytest.mark.parametrize(
    "fc, why",
    [
        (lambda s: _fc(s, dtype_q=torch.bfloat16), "q_out dtype"),
        (
            lambda s: _fc(s, cache=torch.zeros(4, 4, 1, 576, dtype=torch.bfloat16)),
            "KV cache",
        ),
        (
            lambda s: _fc(s, cache=torch.zeros(4, 1, 2, 576, dtype=FP8)[:, :, :1]),
            "KV cache",
        ),
        (lambda s: _fc(s[:2]), "slot mapping"),
        (lambda s: _fc(s.to(torch.int32)), "slot mapping"),
    ],
)
def test_sub_mode_3_refusals(runner_module, monkeypatch, fc, why):
    from atom.mono.runtime.consensus import MonoUnsupported

    prep = _prep3()
    r = _runner3(runner_module, prep)
    slots = torch.arange(ROWS, dtype=torch.int64)
    ctx = fc(slots)
    monkeypatch.setattr(runner_module, "get_forward_context", lambda: ctx)
    with pytest.raises(MonoUnsupported, match=why):
        r._prep_args3(3, prep, ROWS, {})


def test_forward_prepped_from_q_out():
    """Sub-mode 3: the decode starts from the stage's q_out (no q_proj, absorb
    or rope); the latent is returned as is, else g_proj gates it."""
    try:
        from atom.models.kimi_k3.model import KimiFullAttention
    except (ImportError, RuntimeError) as e:  # the GPU stack, absent or no GPU
        pytest.skip(f"atom.models.kimi_k3.model does not import here: {e!r}")

    q_out = torch.zeros(ROWS, 16, 576, dtype=FP8)
    lat = torch.ones(ROWS, 12, 512)
    calls = []

    def prebuilt(q, latent_out=False):
        calls.append((q, latent_out))
        return lat

    def no(*a, **k):
        pytest.fail("the production attention ran")

    sa = SimpleNamespace(
        attn=SimpleNamespace(
            impl=SimpleNamespace(forward_decode_prebuilt=prebuilt, forward_impl=no)
        ),
        g_proj=no,
    )
    got = KimiFullAttention.forward_prepped(
        sa, None, None, None, q_out=q_out, latent_out=True
    )
    assert got is lat and calls == [(q_out, True)]


def test_forward_decode_prebuilt_dummy_run_shapes(monkeypatch):
    try:
        from atom.model_ops import attention_mla as am
    except (ImportError, RuntimeError) as e:  # the GPU stack, absent or no GPU
        pytest.skip(f"attention_mla does not import here: {e!r}")

    ctx = SimpleNamespace(context=SimpleNamespace(is_dummy_run=True))
    cfg = SimpleNamespace(
        hf_config=SimpleNamespace(hidden_size=64), torch_dtype=torch.bfloat16
    )
    monkeypatch.setattr(am, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(am, "get_current_atom_config", lambda: cfg)
    monkeypatch.setattr(am, "_mla_output_width", lambda impl, hidden: 96)
    impl = SimpleNamespace(num_heads=12, kv_lora_rank=512)
    q_out = torch.zeros(ROWS, 16, 576, dtype=FP8)
    run = am.MLAAttention.forward_decode_prebuilt
    assert run(impl, q_out).shape == (ROWS, 96)
    assert run(impl, q_out, latent_out=True).shape == (ROWS, 12, 512)
