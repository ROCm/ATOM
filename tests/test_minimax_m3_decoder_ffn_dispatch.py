# SPDX-License-Identifier: MIT
# Torch-free regression tests for MiniMaxM3DecoderLayer.forward's FFN dispatch.
#
# Defect (found by the Evidence run on the real model with MoE enabled): the SP series collapsed
# main's two FFN calls into ``ffn(hidden_states, x_scale=hidden_states_scale)`` for BOTH layer kinds.
# MiniMaxM3MoE.forward takes no ``x_scale``, so every MoE layer raised TypeError at warmup. On main
# the call with a scale sits inside ``if fuse_post_attention_ar_rmsnorm_quant:`` (never true for MoE
# layers) and the else branch calls ``ffn(hidden_states)``.
#
# The real ``forward`` is extracted from source with ast and run against stand-ins, so no torch, no
# aiter and no weights are needed; the stand-in MoE has the REAL signature (no x_scale), which is
# what makes the old code fail here exactly as it did on the GPU.
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

MODEL = Path(__file__).resolve().parents[1] / "atom/models/minimax_m3/model.py"


class _Act:
    """A tensor stand-in: only ``shape`` is read by the layer; ``kind`` says what it holds."""

    def __init__(self, kind):
        self.kind = kind
        self.shape = (4, 8)


def _extract_forward(calls):
    tree = ast.parse(MODEL.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MiniMaxM3DecoderLayer"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    fn.decorator_list = []
    module = ast.Module(
        [ast.parse("from __future__ import annotations").body[0], fn], []
    )

    def fused_m3_gemma_norm_fp8(hidden_states, weight, eps, residual):
        calls.append("fused_m3_gemma_norm_fp8")
        return _Act("fp8"), "SCALE", residual

    def fused_allreduce_gemma_rms_norm_quant(hidden_states, residual, norm):
        calls.append("fused_allreduce_gemma_rms_norm_quant")
        return _Act("fp8"), "SCALE", residual

    def fused_allreduce_gemma_rms_norm(hidden_states, residual, norm):
        calls.append("fused_allreduce_gemma_rms_norm")
        return _Act("bf16"), residual

    ops = SimpleNamespace(
        minimax_m3_tp_local_residual=lambda residual, n: residual,
        minimax_m3_tp_gather=lambda t, n: t,
        minimax_m3_tp_complete=lambda hidden, attn_out, tokens, n: hidden,
    )
    ns = {
        "torch": SimpleNamespace(ops=SimpleNamespace(aiter=ops)),
        "_linear_consumes_per_token_fp8": lambda linear: linear.fp8,
        "fused_m3_gemma_norm_fp8": fused_m3_gemma_norm_fp8,
        "fused_allreduce_gemma_rms_norm_quant": fused_allreduce_gemma_rms_norm_quant,
        "fused_allreduce_gemma_rms_norm": fused_allreduce_gemma_rms_norm,
    }
    exec(  # noqa: S102 - the real source of the layer, extracted by ast
        compile(ast.fix_missing_locations(module), str(MODEL), "exec"), ns
    )
    return ns["forward"]


class _MoE:
    """The REAL MiniMaxM3MoE call signature: ``forward(self, hidden_states)``, no ``x_scale``."""

    def __init__(self):
        self.seen = []

    def __call__(self, hidden_states):
        self.seen.append(hidden_states)
        return _Act("bf16")


class _Mlp:
    """MiniMaxM3MLP: ``forward(self, x, x_scale=None)``; ``gate_up_proj.fp8`` says whether it
    consumes per-token FP8."""

    def __init__(self, fp8):
        self.gate_up_proj = SimpleNamespace(fp8=fp8)
        self.seen = []

    def __call__(self, x, x_scale=None):
        self.seen.append((x, x_scale))
        return _Act("bf16")


class _Norm:
    """GemmaRMSNorm: ``norm(x)`` or ``norm(x, residual)``; the layer reads weight / variance_epsilon
    for the fused FP8 kernels."""

    weight, variance_epsilon = "w", 1e-6

    def __call__(self, hidden_states, residual=None):
        return _Act("bf16") if residual is None else (_Act("bf16"), residual)


def _layer(*, moe, mlp_fp8=True, qkv_fp8=True, fused_gemma=True, replicated=False):
    attn_calls = []

    class _Attn:
        qkv_proj = SimpleNamespace(fp8=qkv_fp8)

        def __call__(self, positions, hidden_states, hidden_states_scale):
            attn_calls.append(hidden_states_scale)
            return _Act("bf16")

    layer = SimpleNamespace(
        self_attn=_Attn(),
        is_moe_layer=moe,
        input_layernorm=_Norm(),
        post_attention_layernorm=_Norm(),
        _tp_replicated_o_proj=replicated,
        _m3_fused_gemma_fp8=fused_gemma,
    )
    if moe:
        layer.block_sparse_moe = _MoE()
    else:
        layer.mlp = _Mlp(mlp_fp8)
    layer.attn_calls = attn_calls
    return layer


def _run(layer, calls):
    forward = _extract_forward(calls)
    return forward(layer, "positions", _Act("bf16"), _Act("residual"))


@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("mlp_fp8", [True, False])
def test_a_moe_layer_calls_the_moe_without_x_scale_and_gets_bf16(mlp_fp8, replicated):
    """The crash: with every FP8 knob on (the recipe: ptpc_fp8 + fused gemma norm), a MoE layer must
    still call ``block_sparse_moe(hidden_states)`` and hand it BF16, never a scale."""
    layer = _layer(moe=True, mlp_fp8=mlp_fp8, replicated=replicated)
    _run(layer, [])
    (seen,) = (
        layer.block_sparse_moe.seen
    )  # called once, WITHOUT x_scale (the stand-in refuses it)
    assert seen.kind == "bf16"  # the post-attention norm of a MoE layer never quantizes


def test_a_moe_layer_still_feeds_its_attention_the_fp8_input_and_scale():
    """The input norm (before attention) is unaffected: the attention consumes FP8 + scale."""
    layer = _layer(moe=True)
    _run(layer, [])
    assert layer.attn_calls == ["SCALE"]


def test_a_dense_layer_passes_the_fp8_activation_and_its_scale_to_the_mlp():
    layer = _layer(moe=False, mlp_fp8=True)
    _run(layer, [])
    ((x, scale),) = layer.mlp.seen
    assert (x.kind, scale) == ("fp8", "SCALE")  # quantized by the post-attention norm


def test_a_dense_layer_whose_mlp_takes_bf16_gets_bf16_and_no_scale():
    layer = _layer(moe=False, mlp_fp8=False)
    _run(layer, [])
    ((x, scale),) = layer.mlp.seen
    assert (x.kind, scale) == ("bf16", None)


def test_the_layer_never_drops_a_scale_silently_for_a_moe():
    """If a MoE layer ever reached the FFN with a scale (FP8 into a BF16-only consumer) the layer
    must fail loudly rather than drop the scale: the guard is an assertion in the MoE branch.
    """
    tree = ast.parse(MODEL.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MiniMaxM3DecoderLayer"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    branch = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "self.is_moe_layer"
    )
    body = [ast.unparse(s) for s in branch.body]
    assert body[0] == "assert hidden_states_scale is None"
    assert any("self.block_sparse_moe(hidden_states)" in s for s in body)
    assert not any("x_scale" in s for s in body)
