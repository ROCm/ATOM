# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``ATOM_DSV41_MONO_BACKEND=aiter``: V4.1 mono decode's layers on AITER's routed
``deepseek_v41_decode_attn_pre`` / ``_layer_post`` v1 ops (``mono/aiter_gate.py``,
``mono/aiter_args.py``, ``mono/aiter_runner.py``).

CPU only, no AITER provider and no GPU:

- the argument mapping: every weight and buffer the runner hands AITER has the
  contract sheet's shape and dtype (meta tensors, TP 2 / 4 / 8, 1 / 6 / 48 rows,
  fold or not, index group or not, bf16 or MXFP8 cache); held as well to AITER's
  own signature (``AITER_SRC``: an AITER checkout routing the ops) and the
  provider's own shape rules (``DSV41_MONO_PROVIDER_SRC``: its checkout) when
  those are given;
- the per-step integers (window rows, window metadata, the selection as pool
  rows) against a by-hand reading of ``indices._indices``;
- the env / gate matrix, and that the default deployment is the one it was;
- the runner's layer loop over a stand-in model: call order, the caller-side
  indexer between the two calls, fold / index / aux arguments, strictness.
"""

import ast
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from aiter_stub import _drop, atom_without_aiter

from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec
from atom.models.deepseek_v41.mono import aiter_args as A
from atom.models.deepseek_v41.mono import aiter_gate as G
from atom.models.deepseek_v41.mono import weights as mono_weights
from atom.models.deepseek_v41.mono.kernels.dims import Dims

META = torch.device("meta")
PRE, POST = A.ATTN_PRE, A.LAYER_POST

# AITER's contract sheet (deepseek_v41_decode.md, sections 4 and 5)
PRE_POSITIONAL = (
    "residual pre_mix hc_fn hc_scale hc_base attn_norm_weight wqkv_a wqkv_a_scale "
    "q_norm_weight kv_norm_weight wq_b wq_b_scale cos_cache sin_cache positions "
    "kv_cache kv_slots pre_out post_out comb_out q_out"
).split()
PRE_KEYWORD = (
    "layer_index sublayer_output post_mix comb_mix residual_out normed_out "
    "q_latent_out q_latent_scale_out index_wq_b index_wq_b_scale index_weights_proj "
    "index_q_out index_q_scale_out index_weights_out aux_out eps hc_eps group stream"
).split()
POST_POSITIONAL = (
    "q kv_cache topk_rows topk_count window_meta attn_sink positions cos_cache "
    "sin_cache wo_a wo_a_scale wo_b wo_b_scale residual pre_mix post_mix comb_mix "
    "hc_fn hc_scale hc_base ffn_norm_weight gate_weight gate_bias w13 w13_scale w2 "
    "w2_scale shared_w13 shared_w13_scale shared_w2 shared_w2_scale residual_out "
    "pre_out post_out comb_out ffn_out"
).split()
POST_KEYWORD = (
    "softmax_scale ring_slots layer_index ffn_normed_out top_k routed_scaling_factor "
    "swiglu_limit eps hc_eps group stream"
).split()

_HF = dict(
    hidden_size=5120,
    hc_mult=4,
    num_attention_heads=64,
    o_groups=8,
    q_lora_rank=1280,
    head_dim=512,
    o_lora_rank=1024,
    n_routed_experts=384,
    moe_intermediate_size=2304,
    index_n_heads=32,
    index_head_dim=128,
)
CONFIG = SimpleNamespace(**_HF)


def _layer_weights(tp, mode, scale_dims=2):
    """A bound layer's parameters on meta, as ``weights.expected`` shapes them,
    with the MXFP4 expert scales aiter's gfx950 ``shuffle_scale`` leaves
    (``[E * N, ceil8(K / 32)]``; ``scale_dims`` 3: the same bytes 3-D)."""
    tensors = {
        name: torch.empty(shape, dtype=dtype, device=META)
        for name, (shape, dtype) in mono_weights.expected(
            CONFIG, mode, Dims(tp)
        ).items()
    }
    inter = -(-(2304 // tp) // 128) * 128
    for name, n, k in (("w13", 2 * inter, 5120), ("w2", 5120, inter)):
        cols = -(-(k // 32) // 8) * 8
        shape = (384 * n, cols) if scale_dims == 2 else (384, n, cols)
        tensors[f"ffn.experts.{name}_weight_scale"] = torch.empty(
            shape, dtype=torch.uint8, device=META
        )
    return mono_weights.LayerWeights(0, mode, tensors)


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize(
    "mode", [AttentionMode.WINDOW, AttentionMode.FULL, AttentionMode.REINDEX]
)
def test_layer_weights_are_the_contracts(tp, mode):
    feeds_index = mode in (AttentionMode.FULL, AttentionMode.REINDEX)
    w = _layer_weights(tp, mode)
    pre, post = A.attn_pre_weights(w, feeds_index), A.layer_post_weights(w)
    want_pre, want_post = A.weight_shapes(tp, feeds_index)
    assert A.shape_errors(pre, want_pre) == []
    assert A.shape_errors(post, want_post) == []
    assert set(pre) == set(want_pre) and set(post) == set(want_post)
    assert set(pre) <= set(PRE_POSITIONAL + PRE_KEYWORD)
    assert set(post) <= set(POST_POSITIONAL + POST_KEYWORD)
    # in place: the block's own parameters, never a copy
    assert pre["wq_b"] is w["attn.wq_b.weight"]
    assert post["shared_w13"] is w["ffn.shared_experts.gate_up_proj.weight"]


def test_tp4_widths_read_off_the_sheet():
    _, post = A.weight_shapes(4, False)
    pre, _ = A.weight_shapes(4, True)
    assert pre["wq_b"][0] == (16 * 512, 1280)
    assert pre["index_wq_b_scale"][0] == (32 * 128 // 32, 1280 // 32)
    assert post["wo_a"][0] == (2 * 1024, 8 * 512)
    assert post["w13"][0] == (384, 2 * 640, 2560)
    assert post["w13_scale"][0] == (384 * 1280, 160)
    assert post["w2_scale"][0] == (384 * 5120, 24)
    assert post["shared_w2"][0] == (5120, 576)


def test_3d_expert_scales_are_the_same_rows():
    w = _layer_weights(4, AttentionMode.WINDOW, scale_dims=3)
    post = A.layer_post_weights(w)
    _, want = A.weight_shapes(4, False)
    assert A.shape_errors(post, want) == []


def test_a_layout_the_contract_does_not_take_is_named():
    w = _layer_weights(4, AttentionMode.WINDOW)
    pre = A.attn_pre_weights(w, False)
    pre["wq_b"] = torch.empty(4096, 1280, dtype=torch.float8_e4m3fn, device=META)
    pre["wq_b_scale"] = torch.empty(1280 // 32, 8192 // 32, device=META).t()
    want, _ = A.weight_shapes(4, False)
    errors = A.shape_errors(pre, want)
    assert any(e.startswith("wq_b is (4096, 1280)") for e in errors)
    assert any(e.startswith("wq_b_scale is") for e in errors)
    del pre["hc_fn"]
    assert "hc_fn missing" in A.shape_errors(pre, want)


# ---------------------------------------------------------------------------
# a call's arguments
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def R():
    """``aiter_runner`` with ATOM's in-tree mono runners stood in (their FlyDSL
    kernels), as ``test_deepseek_v41_mono_hi_routing`` imports ``hi_dispatch``."""
    before = set(sys.modules)
    with pytest.MonkeyPatch.context() as mp:
        for mod, cls in (
            ("runner", "V41MonoDecodeRunner"),
            ("draft_runner", "DraftMonoRunner"),
        ):
            name = f"atom.models.deepseek_v41.mono.{mod}"
            if name in sys.modules:
                continue
            stub = types.ModuleType(name)
            setattr(stub, cls, type(cls, (), {}))
            mp.setitem(sys.modules, name, stub)
        with atom_without_aiter() as load:
            yield SimpleNamespace(
                runner=load("atom.models.deepseek_v41.mono.aiter_runner"),
                dispatch=load("atom.models.deepseek_v41.mono.dispatch"),
            )
        for name in set(sys.modules) - before:
            if name.startswith("atom.models.deepseek_v41.mono"):
                _drop(name)


def _buffers(R, tp, rows, device=META, max_rows=48):
    """A runner's buffers (``_allocate`` itself) narrowed to ``rows``."""
    r = R.runner.V41AiterDecodeRunner.__new__(R.runner.V41AiterDecodeRunner)
    r.tp, r.max_rows = tp, max_rows
    r._allocate(SimpleNamespace(**_HF), device)
    r._use_rows(rows)
    return r


def _rope(device=META):
    table = torch.empty(4096, 32, dtype=torch.bfloat16, device=device)
    return SimpleNamespace(cos_cache=table, sin_cache=table)


def _kv_cache(fp8, device=META):
    if fp8:
        return torch.empty(10000, 528, dtype=torch.uint8, device=device)
    return torch.empty(10000, 512, dtype=torch.bfloat16, device=device)


def _calls(R, tp, rows, *, fold, index, fp8, aux):
    r = _buffers(R, tp, rows)
    w = _layer_weights(tp, AttentionMode.FULL if index else AttentionMode.WINDOW)
    pre_w, post_w = A.attn_pre_weights(w, index), A.layer_post_weights(w)
    residual = torch.empty(1, rows, 4, 5120, dtype=torch.bfloat16, device=META)
    pre_mix = torch.empty(1, rows, 4, dtype=torch.float32, device=META)
    positions = torch.empty(rows, dtype=torch.int64, device=META)
    group, stream = object(), object()
    pre = A.attn_pre_kwargs(
        pre_w,
        layer_index=3,
        residual=residual,
        pre_mix=pre_mix,
        fold=(r.ffn_out, r.gates_ffn[1], r.gates_ffn[2]) if fold else None,
        residual_out=r.res_attn if fold else None,
        rope=_rope(),
        positions=positions,
        kv_cache=_kv_cache(fp8),
        kv_slots=torch.empty(rows, dtype=torch.int32, device=META),
        gates_out=r.gates_attn,
        q_out=r.q,
        index_out=(
            {
                "normed": r.normed, "qr": r.qr, "qr_scale": r.qr_scale,
                "iq": r.iq, "iq_scale": r.iq_scale, "iw": r.iw,
            }
            if index
            else None
        ),
        aux_out=(
            torch.empty(48, 5120, dtype=torch.bfloat16, device=META)[:rows]
            if aux
            else None
        ),
        group=group,
        stream=stream,
    )  # fmt: skip
    post = A.layer_post_kwargs(
        post_w,
        layer_index=3,
        q=r.q,
        kv_cache=_kv_cache(fp8),
        topk_rows=torch.empty(rows, 512, dtype=torch.int32, device=META),
        topk_count=torch.empty(rows, dtype=torch.int32, device=META),
        window_meta=torch.empty(rows, 4, dtype=torch.int32, device=META),
        positions=positions,
        rope=_rope(),
        residual=r.res_attn if fold else residual,
        gates_in=r.gates_attn,
        residual_out=r.res_ffn,
        gates_out=r.gates_ffn,
        ffn_out=r.ffn_out,
        softmax_scale=512**-0.5,
        ring_slots=133,
        group=group,
        stream=stream,
    )
    return pre, post, group, stream


def _sheet_rows(op, tp, s, fp8):
    """The contract sheet's row-tensor shapes and dtypes for an S-row call."""
    hl = 64 // tp
    bf, f32, i32, i64 = torch.bfloat16, torch.float32, torch.int32, torch.int64
    kv = ((10000, 528), torch.uint8) if fp8 else ((10000, 512), bf)
    common = {
        "positions": ((s,), i64),
        "cos_cache": ((4096, 32), bf),
        "sin_cache": ((4096, 32), bf),
        "kv_cache": kv,
        "pre_out": ((s, 4), f32),
        "post_out": ((s, 4), f32),
        "comb_out": ((s, 4, 4), f32),
    }
    if op == PRE:
        return {
            **common,
            "residual": ((s, 4, 5120), bf),
            "pre_mix": ((s, 4), f32),
            "kv_slots": ((s,), i32),
            "q_out": ((s, hl, 512), bf),
            "sublayer_output": ((s, 5120), bf),
            "post_mix": ((s, 4), f32),
            "comb_mix": ((s, 4, 4), f32),
            "residual_out": ((s, 4, 5120), bf),
            "normed_out": ((s, 5120), bf),
            "q_latent_out": ((s, 1280), torch.float8_e4m3fn),
            "q_latent_scale_out": ((s, 40), torch.float8_e8m0fnu),
            "index_q_out": ((s, 32, 128), torch.float8_e4m3fn),
            "index_q_scale_out": ((s, 32), f32),
            "index_weights_out": ((s, 32), f32),
            "aux_out": ((s, 5120), bf),
        }
    return {
        **common,
        "q": ((s, hl, 512), bf),
        "topk_rows": ((s, 512), i32),
        "topk_count": ((s,), i32),
        "window_meta": ((s, 4), i32),
        "residual": ((s, 4, 5120), bf),
        "pre_mix": ((s, 4), f32),
        "post_mix": ((s, 4), f32),
        "comb_mix": ((s, 4, 4), f32),
        "residual_out": ((s, 4, 5120), bf),
        "ffn_out": ((s, 5120), bf),
    }


GROUPS = {
    "fold": ("sublayer_output", "post_mix", "comb_mix", "residual_out"),
    "index": (
        "normed_out q_latent_out q_latent_scale_out index_wq_b index_wq_b_scale "
        "index_weights_proj index_q_out index_q_scale_out index_weights_out"
    ).split(),
}


@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 6, 48])
@pytest.mark.parametrize(
    "fold, index, fp8, aux",
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (True, True, True, True),
        (False, True, True, False),
    ],
)
def test_a_calls_arguments_are_the_contracts(R, tp, rows, fold, index, fp8, aux):
    pre, post, group, stream = _calls(
        R, tp, rows, fold=fold, index=index, fp8=fp8, aux=aux
    )
    # every name AITER's, every positional given, the groups all or none
    assert set(PRE_POSITIONAL) <= set(pre) <= set(PRE_POSITIONAL + PRE_KEYWORD)
    assert set(POST_POSITIONAL) <= set(post) <= set(POST_POSITIONAL + POST_KEYWORD)
    for group_name, on in (("fold", fold), ("index", index)):
        given = [pre.get(name) is not None for name in GROUPS[group_name]]
        assert all(given) if on else not any(given), group_name
    assert (pre["aux_out"] is not None) == aux
    for op, kw in ((PRE, pre), (POST, post)):
        for name, (shape, dtype) in _sheet_rows(op, tp, rows, fp8).items():
            t = kw.get(name)
            if t is None:
                continue
            assert (tuple(t.shape), t.dtype) == (shape, dtype), (op, name)
            assert t.is_contiguous(), (op, name)
        assert kw["group"] is group and kw["stream"] is stream
        assert kw["layer_index"] == 3
    assert post["ring_slots"] == 133 and post["softmax_scale"] == 512**-0.5
    # the defaults the provider bakes in are left to AITER's
    for name in ("eps", "hc_eps", "top_k", "routed_scaling_factor", "swiglu_limit"):
        assert name not in pre and name not in post
    # no output aliases an input or another output
    for kw in (pre, post):
        ptrs = [
            (n, t.untyped_storage())
            for n, t in kw.items()
            if isinstance(t, torch.Tensor)
        ]
        outs = [n for n in kw if n.endswith("_out") and kw[n] is not None]
        for n in outs:
            others = [m for m, _ in ptrs if m != n and kw[m] is not kw[n]]
            assert all(kw[m] is not kw[n] for m in others), n


def _aiter_signature(path: Path, op: str) -> tuple[list, list]:
    """(positional, keyword-only) parameter names of ``op`` in AITER's source."""
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == op)
    return [a.arg for a in fn.args.args], [a.arg for a in fn.args.kwonlyargs]


def test_the_sheet_here_is_aiters_signature():
    """Held to AITER's own source when ``AITER_SRC`` names a checkout routing
    the ops (``zhiding512/dsv41-mono-contract``)."""
    root = os.environ.get("AITER_SRC")
    path = Path(root or "/nonexistent") / "aiter" / "ops" / "deepseek_v41_decode.py"
    if not path.is_file():
        pytest.skip("set AITER_SRC to an AITER checkout that routes the V4.1 ops")
    assert _aiter_signature(path, PRE) == (PRE_POSITIONAL, PRE_KEYWORD)
    assert _aiter_signature(path, POST) == (POST_POSITIONAL, POST_KEYWORD)


@pytest.fixture(scope="module")
def provider_contract():
    root = os.environ.get("DSV41_MONO_PROVIDER_SRC")
    if not root or not (Path(root) / "ops" / "dsv41_mono" / "contract.py").is_file():
        pytest.skip("set DSV41_MONO_PROVIDER_SRC to the dsv41-mono provider checkout")
    sys.path.insert(0, root)
    try:
        from ops.dsv41_mono import contract
    finally:
        sys.path.remove(root)
    yield contract
    for name in [n for n in sys.modules if n == "ops" or n.startswith("ops.")]:
        del sys.modules[name]


@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 48])
@pytest.mark.parametrize(
    "fold, index, fp8", [(False, False, False), (True, True, True)]
)
def test_the_provider_takes_the_calls_shapes(
    R, provider_contract, tp, rows, fold, index, fp8
):
    """The provider's own ``shape_reason`` rules (its ``select()``'s, which read
    shapes, dtypes and scalars only) accept both calls."""
    pre, post, _, _ = _calls(R, tp, rows, fold=fold, index=index, fp8=fp8, aux=fold)
    c = provider_contract
    assert c.attn_pre_shape(c.bind(c.ATTN_PRE, (), pre), tp, META) == rows
    assert c.layer_post_shape(c.bind(c.LAYER_POST, (), post), tp, META) == rows


# ---------------------------------------------------------------------------
# per-step integers
# ---------------------------------------------------------------------------


def test_window_rows_and_metadata_a_layer():
    ring_rel = torch.tensor([7, -1, 300], dtype=torch.int32)
    starts = torch.tensor([0, 1000, 5000])
    slots = A.layer_kv_slots(ring_rel, starts)
    assert slots.dtype == torch.int32
    assert slots.tolist() == [[7, -1, 300], [1007, -1, 1300], [5007, -1, 5300]]
    key_meta = torch.tensor([[3, 128, 99, 1330], [0, 0, 5, 0]], dtype=torch.int32)
    meta = A.layer_window_meta(key_meta, starts)
    assert meta.shape == (3, 2, 4) and meta.dtype == torch.int32
    assert meta[1].tolist() == [[3, 128, 0, 2330], [0, 0, 0, 1000]]
    assert all(meta[i].is_contiguous() for i in range(3))


def test_the_selection_as_pool_rows():
    """``indices._indices``' arithmetic: a column's PAGE through its request's
    block table, valid entries packed first, a pad row none."""
    rpp, page_rows, main_off = 32, 100, 7
    block_tables = torch.tensor([[5, 9, 2], [11, 3, 4]], dtype=torch.int32)
    batch_ids = torch.tensor([1, 0, -1, 0], dtype=torch.int32)
    selected = torch.tensor(
        [[0, 33, -1, -1], [-1, 64, 1, -1], [3, 4, -1, -1], [95, -1, -1, 40]],
        dtype=torch.int32,
    )
    rows, count = A.selection_rows(
        selected, batch_ids, block_tables, rpp, page_rows, main_off
    )
    assert rows.dtype == count.dtype == torch.int32 and rows.shape == (4, 4)

    def by_hand(t):
        b = int(batch_ids[t])
        if b < 0:
            return []
        cols = [c for c in selected[t].tolist() if c >= 0]
        return [
            int(block_tables[b, c // rpp]) * page_rows + main_off + c % rpp
            for c in cols
        ]

    for t in range(4):
        want = by_hand(t)
        assert count[t] == len(want)
        assert rows[t, : len(want)].tolist() == want
        assert rows[t, len(want) :].eq(0).all()
    assert count.tolist() == [2, 2, 0, 2]


def test_the_selection_on_meta_keeps_its_shape():
    rows, count = A.selection_rows(
        torch.empty(6, 512, dtype=torch.int32, device=META),
        torch.empty(8, dtype=torch.int32, device=META),
        torch.empty(4, 64, dtype=torch.int32, device=META),
        32, 4, 0,
    )  # fmt: skip
    assert rows.shape == (6, 512) and count.shape == (6,)
    no_rows, no_count = A.no_selection(6, META)
    assert no_rows.shape == (6, 1) and no_count.shape == (6,)


# ---------------------------------------------------------------------------
# the env / gate matrix
# ---------------------------------------------------------------------------


class _Plugin:
    """``aiter.plugin`` as the gate reads it."""

    def __init__(self, providers=("dsv41-mono-provider",), enabled=(), kill=False):
        self.providers, self.enabled, self.kill = list(providers), tuple(enabled), kill
        self.configured, self.decision = [], None
        self.served = True

    def status(self):
        return {
            "kill_switch": self.kill,
            "enabled": self.enabled,
            "providers": [{"name": n} for n in self.providers],
            "skipped_providers": [],
        }

    def configure(self, *, source, enable):
        self.configured.append((source, tuple(enable)))
        self.enabled = tuple(enable)

    def _decision(self, op):
        if self.kill:
            return None
        if self.served:
            return SimpleNamespace(
                op=SimpleNamespace(name=op), route="provider",
                provider="dsv41-mono-provider", kernel_id=f"k.{op}", reason="selected",
            )  # fmt: skip
        return SimpleNamespace(
            op=SimpleNamespace(name=op), route="native", provider=None,
            kernel_id=None, reason="provider-declined",
        )  # fmt: skip

    def explain(self, op, **kwargs):
        return SimpleNamespace(
            decision=self._decision(op),
            declines={} if self.served else {"dsv41-mono-provider": "declined"},
        )

    def last_decision(self):
        return self.decision

    Op = staticmethod(lambda name, version: SimpleNamespace(name=name, version=version))


def _ops(plugin=None, calls=None):
    calls = [] if calls is None else calls
    plugin = plugin or _Plugin()

    def op(name):
        def run(**kwargs):
            calls.append((name, kwargs))
            plugin.decision = plugin._decision(name)

        return run

    return G.AiterOps(op(PRE), op(POST), plugin)


@pytest.fixture
def env(monkeypatch):
    for name in (
        "ATOM_DSV41_MONO_BACKEND",
        "ATOM_DSV41_MONO_AITER_STRICT",
        "ATOM_DSV41_MONO_AITER_PROVIDER",
        "ATOM_DSV41_MONO_HI",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_the_defaults(env):
    assert G.backend() == "atom" and not G.selected()
    assert G.strict()
    assert G.provider_name() == "dsv41-mono-provider"


@pytest.mark.parametrize(
    "value, backend", [("aiter", "aiter"), (" AITER ", "aiter"), ("atom", "atom")]
)
def test_the_backend_switch(env, value, backend):
    env.setenv("ATOM_DSV41_MONO_BACKEND", value)
    assert G.backend() == backend


def test_an_unknown_backend_is_an_error(env):
    env.setenv("ATOM_DSV41_MONO_BACKEND", "triton")
    with pytest.raises(ValueError, match="one of atom, aiter"):
        G.backend()


@pytest.mark.parametrize("value, strict", [("1", True), ("0", False)])
def test_strictness(env, value, strict):
    env.setenv("ATOM_DSV41_MONO_AITER_STRICT", value)
    assert G.strict() is strict


_CFG = SimpleNamespace(enable_expert_parallel=False)


@pytest.mark.parametrize(
    "case, reason",
    [
        ({}, None),
        ({"dp_ranks": 2}, "DP attention over 2 ranks"),
        ({"config": SimpleNamespace(enable_expert_parallel=True)}, "expert parallel"),
        ({"ops": None}, "aiter cannot route the ops"),
        ({"plugin": _Plugin(kill=True)}, "AITER_PLUGINS=0"),
        ({"plugin": _Plugin(providers=("sp-qkv-provider",))}, "is not installed"),
    ],
)
def test_the_refusal_matrix(env, case, reason):
    ops = case["ops"] if "ops" in case else _ops(case.get("plugin"))
    got = G.refusal(case.get("config", _CFG), case.get("dp_ranks", 0), ops)
    assert (got is None) if reason is None else (reason in got)


def test_another_provider_name(env):
    env.setenv("ATOM_DSV41_MONO_AITER_PROVIDER", "mine")
    assert "'mine' is not installed" in G.refusal(_CFG, 0, _ops())
    assert G.refusal(_CFG, 0, _ops(_Plugin(providers=("mine",)))) is None


def test_load_ops_without_the_ops_is_unavailable(env, monkeypatch):
    real = G.importlib.import_module

    def missing(name):
        if name == "aiter.ops.deepseek_v41_decode":
            raise ImportError("no such module")
        return types.SimpleNamespace() if name == "aiter.plugin" else real(name)

    monkeypatch.setattr(G.importlib, "import_module", missing)
    with pytest.raises(G.Unavailable, match="does not route"):
        G.load_ops()


def test_configure_appends_the_provider_once(env):
    plugin = _Plugin(enabled=("sp-qkv-provider",))
    ops = _ops(plugin)
    assert G.configure(ops) == ("sp-qkv-provider", "dsv41-mono-provider")
    assert G.configure(ops) == ("sp-qkv-provider", "dsv41-mono-provider")
    assert plugin.configured == [(G.SOURCE, ("sp-qkv-provider", "dsv41-mono-provider"))]


def test_explain_and_the_recorded_route(env):
    plugin = _Plugin()
    ops = _ops(plugin)
    assert G.explain(ops, PRE, {}) == (True, f"dsv41-mono-provider k.{PRE}")
    plugin.served = False
    ok, why = G.explain(ops, POST, {})
    assert not ok and "native (provider-declined)" in why and "declined" in why
    plugin.kill = True
    assert G.explain(ops, POST, {}) == (False, "AITER_PLUGINS=0: native")
    # the record of the call that just ran
    plugin.kill, plugin.served = False, True
    assert G.routed(ops, PRE)[0] is False  # nothing ran yet
    ops.attn_pre()
    assert G.routed(ops, PRE)[0] is True
    assert G.routed(ops, POST) == (False, f"the last route recorded is {PRE}'s")
    plugin.served = False
    ops.layer_post()
    assert G.routed(ops, POST) == (False, "native (provider-declined)")


# ---------------------------------------------------------------------------
# install: the default unchanged, the aiter matrix
# ---------------------------------------------------------------------------


@pytest.fixture
def install(R, env, monkeypatch):
    """``install_mono_decode`` on a deployment ATOM's mono serves, with every
    gate it consults recorded."""
    D = R.dispatch
    seen = SimpleNamespace(hi_decide=0, loaded=0, configured=0, created=[])
    gfx1250 = sys.modules.get("atom.models.deepseek_v41.mono.gfx1250")
    if gfx1250 is None:
        import atom.models.deepseek_v41.mono.gfx1250 as gfx1250
    monkeypatch.setattr(gfx1250, "enabled", lambda: False)
    monkeypatch.setattr(D, "config_refusal", lambda cfg: None)
    monkeypatch.setattr(D, "dp_attention_ranks", lambda cfg: 0)

    def decide(*a):
        seen.hi_decide += 1
        return False

    monkeypatch.setattr(D.hi_gate, "decide", decide)
    plugin = _Plugin()

    def load_ops():
        seen.loaded += 1
        return _ops(plugin)

    monkeypatch.setattr(D.aiter_gate, "load_ops", load_ops)
    real_configure = D.aiter_gate.configure

    def configure(ops):
        seen.configured += 1
        return real_configure(ops)

    monkeypatch.setattr(D.aiter_gate, "configure", configure)
    stub = types.ModuleType("atom.models.deepseek_v41.mono.aiter_runner")
    stub.create_runner = lambda model, drafter: seen.created.append(model) or "aiter"
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    D.V41MonoDecodeRunner = lambda *a: "atom"  # the in-tree K1 / K2 runner
    config = SimpleNamespace(
        max_model_len=1 << 20, index_cache_dtype="fp4", enable_expert_parallel=False
    )
    model = torch.nn.Module()

    def run(**environ):
        for k, v in environ.items():
            env.setenv(k, v)
        return D.install_mono_decode(model, config, None)

    return SimpleNamespace(run=run, seen=seen, plugin=plugin, model=model, D=D)


def test_the_default_deployment_is_unchanged(install):
    wrapped = install.run()
    assert isinstance(wrapped, install.D.MonoDecodeModel)
    assert wrapped._mono._create() == "atom"
    # the hi gate decided as before; AITER never imported nor configured
    assert install.seen.hi_decide == 1
    assert install.seen.loaded == install.seen.configured == 0
    assert install.plugin.configured == []


def test_the_default_path_with_hi_off_or_on_is_unchanged(install):
    assert install.run(ATOM_DSV41_MONO_HI="0")._mono._create() == "atom"
    assert install.seen.loaded == 0


def test_the_aiter_backend_installs(install):
    wrapped = install.run(ATOM_DSV41_MONO_BACKEND="aiter")
    assert isinstance(wrapped, install.D.MonoDecodeModel)
    assert wrapped._mono._create() == "aiter"
    assert install.seen.created == [install.model]
    # an explicit backend outranks hi "auto": the hi gate is never asked
    assert install.seen.hi_decide == 0
    assert install.plugin.configured == [(G.SOURCE, ("dsv41-mono-provider",))]


def test_aiter_with_hi_forced_is_a_conflict(install):
    with pytest.raises(RuntimeError, match="both claim"):
        install.run(ATOM_DSV41_MONO_BACKEND="aiter", ATOM_DSV41_MONO_HI="1")


@pytest.mark.parametrize(
    "plugin, reason",
    [
        (_Plugin(providers=()), "is not installed"),
        (_Plugin(kill=True), "AITER_PLUGINS=0"),
    ],
)
def test_a_refused_aiter_backend_is_an_error_when_strict(
    install, monkeypatch, plugin, reason
):
    monkeypatch.setattr(install.D.aiter_gate, "load_ops", lambda: _ops(plugin))
    with pytest.raises(RuntimeError, match=reason):
        install.run(ATOM_DSV41_MONO_BACKEND="aiter")


def test_a_refused_aiter_backend_falls_back_when_not_strict(install, monkeypatch):
    monkeypatch.setattr(
        install.D.aiter_gate, "load_ops", lambda: _ops(_Plugin(providers=()))
    )
    wrapped = install.run(
        ATOM_DSV41_MONO_BACKEND="aiter", ATOM_DSV41_MONO_AITER_STRICT="0"
    )
    assert wrapped._mono._create() == "atom"
    assert install.seen.hi_decide == 1


def test_an_aiter_without_the_ops_is_named(install, monkeypatch):
    def unavailable():
        raise G.Unavailable("aiter does not route the V4.1 decode-layer ops")

    monkeypatch.setattr(install.D.aiter_gate, "load_ops", unavailable)
    with pytest.raises(RuntimeError, match="does not route"):
        install.run(ATOM_DSV41_MONO_BACKEND="aiter")


def test_a_deployment_mono_refuses_is_an_error_for_an_explicit_backend(
    install, monkeypatch
):
    monkeypatch.setattr(install.D, "config_refusal", lambda cfg: "kv cache fp4")
    with pytest.raises(
        RuntimeError, match="BACKEND=aiter, but V4.1 mono decode is off"
    ):
        install.run(ATOM_DSV41_MONO_BACKEND="aiter")
    env_off = install.run(ATOM_DSV41_MONO_AITER_STRICT="0")
    assert env_off is install.model


def test_a_deployment_mono_refuses_is_still_the_original_by_default(
    install, monkeypatch
):
    monkeypatch.setattr(install.D, "config_refusal", lambda cfg: "kv cache fp4")
    assert install.run() is install.model


# ---------------------------------------------------------------------------
# the runner's layer loop
# ---------------------------------------------------------------------------

ROWS, MAX_ROWS, LAYERS = 3, 4, 5


def _topology():
    """window; FULL (selects, produces candidates); REUSE of 1's selection;
    an Engram window layer; REINDEX (selects through 1's candidates)."""
    S = LayerAttentionSpec
    return [
        S(0, 0, AttentionMode.WINDOW),
        S(1, 4, AttentionMode.FULL, kv_owner=1, topk_owner=1, candidate_owner=1),
        S(2, 4, AttentionMode.REUSE, kv_owner=1, topk_owner=1),
        S(3, 0, AttentionMode.WINDOW),
        S(4, 128, AttentionMode.REINDEX, kv_owner=4, topk_owner=4, candidate_owner=1),
    ]


class _Indexer:
    def __init__(self, layer_id, log):
        self.layer_id, self.log = layer_id, log

    def _publish(self, step):
        sel = torch.full((1, ROWS, A.TOPK), -1, dtype=torch.int32)
        sel[0, :, :2] = torch.tensor([[self.layer_id, 1]] * ROWS)
        step.selected[self.layer_id] = sel

    def project(self, normed, qr, qr_scale, cache, step, rope):
        self.log.append(("project", self.layer_id, normed.shape, qr.dtype))
        return ("fp4 query", "weights")

    def score(self, query, weights, cache, step):
        self.log.append(("score", self.layer_id, query))
        self._publish(step)

    def score_quantized(self, query, scaled, cache, step):
        self.log.append(("score_quantized", self.layer_id, query.shape, scaled.shape))
        self._publish(step)


def _model(log):
    topology = _topology()
    blocks = []
    for spec in topology:
        attn = SimpleNamespace(
            indexer=(
                _Indexer(spec.layer_id, log)
                if spec.mode in (AttentionMode.FULL, AttentionMode.REINDEX)
                else None
            ),
            softmax_scale=512**-0.5,
            _compress_batch=lambda normed, cache, step, rope, i=spec.layer_id: (
                log.append(("compress", i, normed.shape))
            ),
        )
        engram = object() if spec.layer_id == 3 else None
        blocks.append(
            SimpleNamespace(
                attn=attn,
                engram=engram,
                engram_forward=lambda residual, a, b, i=spec.layer_id: (
                    log.append(("engram", i)) or residual.clone()
                ),
            )
        )
    return SimpleNamespace(
        topology=topology,
        layers=blocks,
        config=SimpleNamespace(**_HF),
        window_rope=_rope("cpu"),
        global_rope=_rope("cpu"),
        embed=lambda ids: torch.zeros(ids.numel(), 5120, dtype=torch.bfloat16),
        begin_forward=lambda hidden, e: log.append(("begin",)),
        end_forward=lambda hidden, e: log.append(("end",)),
    )


class _Geometry:
    page_bytes, row_bytes = 1024 * 100, 1024

    def window(self, layer, pages):
        return SimpleNamespace(ring_start=1000 * (layer + 1), ring_slots=133)

    def rows_per_page(self, ratio):
        return 128 // ratio

    def main_offset(self, owner):
        return 10 * owner


def _step_and_cache():
    step = SimpleNamespace(
        batch_ids=torch.tensor([0, 1, -1, 0], dtype=torch.int32),
        block_tables=torch.tensor(
            [[2, 3, 7, 8, 9, 1], [5, 6, 4, 4, 4, 4]], dtype=torch.int32
        ),
        positions=torch.tensor([10, 20, 0, 11], dtype=torch.int64),
        selected={},
    )
    cache = SimpleNamespace(
        packed=False,
        geometry=_Geometry(),
        num_pages=64,
        pool=torch.empty(10, 512, dtype=torch.bfloat16),
        rope_positions=lambda step: step.positions,
    )
    return step, cache


@pytest.fixture
def run(R, env, monkeypatch):
    """A runner over ``_model`` (CPU buffers, AITER's ops recorded), and its
    forward, with ``write_step_meta`` and the HC post stood in."""
    M = R.runner
    log, calls = [], []
    model = _model(log)
    plugin = _Plugin()
    r = M.V41AiterDecodeRunner.__new__(M.V41AiterDecodeRunner)
    r.ops, r.group, r.strict = (
        _ops(plugin, calls),
        SimpleNamespace(cpu_group="cpu"),
        True,
    )
    r.model, r.tp, r.rank, r.max_rows = model, 4, 0, MAX_ROWS
    r.aux = SimpleNamespace(
        layer_ids=(3,), buffers=(torch.zeros(MAX_ROWS, 5120, dtype=torch.bfloat16),)
    )
    r.index_fp4 = False
    r.static = [
        (
            A.attn_pre_weights(
                _layer_weights(4, spec.mode), b.attn.indexer is not None
            ),
            A.layer_post_weights(_layer_weights(4, spec.mode)),
        )
        for spec, b in zip(model.topology, model.layers)
    ]
    r._allocate(model.config, torch.device("cpu"))
    r._served, r._ring_starts = set(), None
    step, cache = _step_and_cache()
    md = SimpleNamespace(cache=cache, step=step, engram_embeddings=None)
    monkeypatch.setattr(
        M, "get_forward_context", lambda: SimpleNamespace(attn_metadata=md)
    )

    def write_step_meta(step, geometry, window, ring_rel, key_meta, plans, walk):
        assert plans == {} and walk == (0, ring_rel.shape[0])
        ring_rel.copy_(torch.tensor([5, 6, -1])[: ring_rel.shape[0]])
        key_meta.copy_(torch.tensor([[1, 10, 0, 100], [2, 20, 0, 200], [0, 0, 0, 0]]))

    monkeypatch.setattr(M, "write_step_meta", write_step_meta)
    import atom.model_ops.deepseek_v41.mhc as mhc

    monkeypatch.setattr(
        mhc, "expand_residual", lambda pending, residual, post, comb: residual.clone()
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "stream")
    agreed = []
    monkeypatch.setattr(M, "group_agree", lambda ok, group: agreed.append(ok) or ok)

    def forward():
        ids = torch.zeros(ROWS, dtype=torch.int64)
        return r.forward(ids, None)

    return SimpleNamespace(
        r=r, forward=forward, log=log, calls=calls, plugin=plugin, step=step,
        agreed=agreed, M=M,
    )  # fmt: skip


def test_the_layer_loop_calls_both_ops_with_the_indexer_between(run):
    hidden = run.forward()
    assert hidden.shape == (ROWS, 5120)
    order = [(name, kw["layer_index"]) for name, kw in run.calls]
    assert order == [(op, i) for i in range(LAYERS) for op in (PRE, POST)]
    # the caller-side indexer sits between a selecting layer's two calls
    events = [e[:2] for e in run.log if e[0] in ("compress", "score_quantized")]
    assert events == [
        ("compress", 1), ("score_quantized", 1), ("compress", 4), ("score_quantized", 4)
    ]  # fmt: skip
    assert run.log[0] == ("begin",) and run.log[-1] == ("end",)
    assert ("engram", 3) in run.log


def test_each_calls_arguments(run):
    run.forward()
    pre = {kw["layer_index"]: kw for name, kw in run.calls if name == PRE}
    post = {kw["layer_index"]: kw for name, kw in run.calls if name == POST}
    # layer 0 and the Engram layer settle first: no fold; the rest fold
    for i in range(LAYERS):
        folds = i not in (0, 3)
        assert (pre[i].get("sublayer_output") is not None) == folds, i
        assert (pre[i].get("residual_out") is not None) == folds, i
        indexed = i in (1, 4)
        assert (pre[i].get("index_q_out") is not None) == indexed, i
        assert (pre[i].get("aux_out") is not None) == (i == 3), i
        assert pre[i]["group"] is run.r.group and pre[i]["stream"] == "stream"
        assert post[i]["group"] is run.r.group and post[i]["stream"] == "stream"
        # the window row and keys: ring_rel / key_meta plus the layer's ring start
        start = 1000 * (i + 1)
        assert pre[i]["kv_slots"].tolist() == [5 + start, 6 + start, -1]
        assert post[i]["window_meta"][:, 3].tolist() == [
            100 + start,
            200 + start,
            start,
        ]
        assert post[i]["ring_slots"] == 133
        assert pre[i]["positions"].tolist() == [10, 20, 0]
        assert pre[i]["residual"].shape == (ROWS, 4, 5120)
    # K2's residual is K1's fold output where it folds
    assert post[1]["residual"].data_ptr() == pre[1]["residual_out"].data_ptr()
    # the selection as pool rows: layer 1's, shared by layer 2 (REUSE), layer 4's own
    page_rows = 100  # ratio 4: 32 rows a page, cols 1 and 1 on the first page
    want = [
        [2 * page_rows + 10 + 1, 2 * page_rows + 10 + 1],  # batch 0: cols 1, 1
        [5 * page_rows + 10 + 1, 5 * page_rows + 10 + 1],  # batch 1
        [],  # pad row
    ]
    for i in (1, 2):
        rows, count = post[i]["topk_rows"], post[i]["topk_count"]
        assert count.tolist() == [2, 2, 0]
        assert [rows[t, : len(w)].tolist() for t, w in enumerate(want)] == want
    assert post[1]["topk_rows"] is post[2]["topk_rows"]  # once a step
    assert post[4]["topk_count"].tolist() == [2, 2, 0]
    # ratio 128: a row a page, cols 4 and 1 of batch 0's table (pages 9, 3)
    assert post[4]["topk_rows"][0, :2].tolist() == [9 * 100 + 40, 3 * 100 + 40]
    for i in (0, 3):  # window layers: no selected keys
        assert post[i]["topk_rows"].shape == (ROWS, 1)
        assert post[i]["topk_count"].eq(0).all()


def test_an_fp4_index_plane_projects_from_k1s_normed_row(run):
    run.r.index_fp4 = True
    run.forward()
    projected = [e for e in run.log if e[0] in ("project", "score")]
    assert projected == [
        ("project", 1, (1, ROWS, 5120), torch.float8_e4m3fn),
        ("score", 1, "fp4 query"),
        ("project", 4, (1, ROWS, 5120), torch.float8_e4m3fn),
        ("score", 4, "fp4 query"),
    ]


def test_a_call_routed_off_the_provider_is_an_error_when_strict(run):
    run.plugin.served = False
    with pytest.raises(RuntimeError, match=f"layer 0 {PRE} did not run on dsv41"):
        run.forward()


def test_a_call_routed_off_the_provider_warns_when_not_strict(run):
    run.plugin.served = False
    run.r.strict = False
    run.forward()
    assert len(run.calls) == 2 * LAYERS


def test_prepare_preflights_a_width_once_and_agrees(run, monkeypatch):
    explained = []
    real = run.plugin.explain
    monkeypatch.setattr(
        run.plugin, "explain", lambda op, **kw: explained.append(op) or real(op, **kw)
    )
    assert run.r.prepare(ROWS)
    assert explained == [PRE, POST] * LAYERS and run.agreed == [True]
    assert run.r.prepare(ROWS) and run.agreed == [True]  # memoized
    assert not run.r.prepare(MAX_ROWS + 1)


def test_a_declined_width_is_an_error_when_strict(run):
    run.plugin.served = False
    with pytest.raises(RuntimeError, match="does not serve every call"):
        run.r.prepare(ROWS)
    assert run.agreed == [False]


def test_a_declined_width_runs_aiters_fallback_when_not_strict(run):
    run.plugin.served = False
    run.r.strict = False
    assert run.r.prepare(ROWS)


def test_a_cold_width_under_capture(run, monkeypatch):
    monkeypatch.setattr(run.M, "_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before an eager step"):
        run.r.prepare(ROWS)
    run.r.strict = False
    assert not run.r.prepare(ROWS)
    # a width warmed eagerly is served under capture
    monkeypatch.setattr(run.M, "_capturing", lambda: False)
    assert run.r.prepare(ROWS)
    monkeypatch.setattr(run.M, "_capturing", lambda: True)
    assert run.r.prepare(ROWS)


def test_create_runner_strictness(R, env, monkeypatch):
    M = R.runner
    from atom.mono.runtime.consensus import MonoUnsupported

    def refuse(*a):
        raise MonoUnsupported("layer 0: wq_b is (1, 2)")

    monkeypatch.setattr(M, "V41AiterDecodeRunner", refuse)
    with pytest.raises(RuntimeError, match="wq_b is"):
        M.create_runner(None, None)
    env.setenv("ATOM_DSV41_MONO_AITER_STRICT", "0")
    with pytest.raises(MonoUnsupported):
        M.create_runner(None, None)


def test_the_gate_against_the_real_aiter_plugin(R, env):
    """The installed AITER's own ``aiter.plugin`` (an AITER routing the ops, with
    the provider's entry point visible: venv-fly05): the provider is found and
    enabled, and a call it cannot serve (meta tensors, no GPU) is explained as
    declined with the provider's own reason, without running anything."""
    try:
        ops = G.load_ops()
    except G.Unavailable as why:
        pytest.skip(f"this AITER does not route the V4.1 ops: {why}")
    status = ops.plugin.status()
    if "dsv41-mono-provider" not in [p["name"] for p in status["providers"]]:
        pytest.skip("the dsv41-mono provider's entry point is not visible")
    before = tuple(status["enabled"])
    try:
        assert G.refusal(_CFG, 0, ops) is None
        assert "dsv41-mono-provider" in G.configure(ops)
        pre, post, _, _ = _calls(R, 4, 6, fold=True, index=True, fp8=False, aux=False)
        for op, kw in ((PRE, pre), (POST, post)):
            # meta tensors and no TP group: outside what the provider covers
            ok, why = G.explain(ops, op, {**kw, "group": None, "stream": None})
            assert not ok
            assert (
                "dsv41-mono-provider: declined" in why and "OUT_OF_CONTRACT" in why
            ), why
    finally:
        ops.plugin.configure(source="test restore", enable=list(before))
