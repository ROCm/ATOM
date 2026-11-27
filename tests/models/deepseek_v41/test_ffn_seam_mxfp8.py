# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The FFN seam writes the MXFP8 pair of its normed rows for the shared expert.

The shared expert's gate_up is a native group32 FP8 GEMM that would quantize
its BF16 input to exactly that pair (``quantize_fp8``), so the FFN seam's
Mega-MHC launch writes it next to the BF16 rows (as the attention seam does
for ``wqkv_a``) and the quantize launch in front of the shared expert goes:
on a dense shared expert (draft layers, rows MegaMoE does not fuse) and on a
target layer whose routed MegaMoE call computes it (``shared_in``).

CPU: where the pair is asked for and where it goes. GPU: the seam's pair is
``quantize_fp8`` of its BF16 norm byte for byte, and the dense shared expert
on real checkpoint weights gives the same bytes from the pair as from the BF16
rows.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

aiter = pytest.importorskip("aiter", reason="the seam and the experts are AITER-backed")

from atom.models.deepseek_v4 import DeepseekV4Args, Expert
from atom.models.deepseek_v4 import MoE as V4MoE
from atom.models.deepseek_v41 import model as v41_model
from atom.models.deepseek_v41.moe import MoE as V41MoE

# ---------------------------------------------------------------- CPU contract


class _Recorder:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result if self.result is not None else args[0]


def _expert(gate_up_rows=32, runs_bf16=False, dtype=torch.float8_e4m3fn):
    expert = Expert.__new__(Expert)
    torch.nn.Module.__init__(expert)
    expert.gate_up_proj = SimpleNamespace(
        native_a8_group_rows=gate_up_rows,
        runs_bf16=runs_bf16,
        weight=torch.empty(0, dtype=dtype),
    )
    return expert


@pytest.mark.parametrize(
    "rows, runs_bf16, dtype, expected",
    [
        (32, False, torch.float8_e4m3fn, True),  # V4.1's native MXFP8 one
        (1, False, torch.float8_e4m3fn, True),  # 1x32 weight scales
        (None, False, torch.float8_e4m3fn, False),  # per-channel / 128x128
        (32, True, torch.float8_e4m3fn, False),  # BF16 fallback: no FP8 input
        (1, False, torch.float4_e2m1fn_x2, False),  # W4A8
    ],
)
def test_which_shared_experts_read_the_pair(rows, runs_bf16, dtype, expected):
    assert _expert(rows, runs_bf16, dtype).reads_mxfp8_input() is expected


def test_the_dense_shared_expert_feeds_the_pair_to_gate_up(monkeypatch):
    """With the pair, gate_up reads it in place of quantizing x; the rest of
    the chain is the one it runs without."""
    import atom.models.deepseek_v4 as v4

    rows, dim, inter = 3, 64, 32
    expert = _expert()
    combined = torch.randn(rows, 2 * inter, dtype=torch.bfloat16)
    expert.gate_up_proj = _Recorder(combined)
    expert.w2 = _Recorder(torch.zeros(rows, dim, dtype=torch.bfloat16))
    expert.w2.native_a8_group_rows = 32
    expert.w2.quant_type = SimpleNamespace(value=None)
    expert.w2.runs_bf16 = False
    expert.swiglu_limit = 10.0
    expert.use_fused_clamp_act_mul = False
    monkeypatch.setattr(v4, "aiter_silu_and_mul", lambda out, c, limit: out.zero_())
    x = torch.randn(rows, dim, dtype=torch.bfloat16)
    x_fp8 = torch.zeros(rows, dim, dtype=torch.float8_e4m3fn)
    x_scale = torch.zeros(rows, dim // 32, dtype=torch.float8_e8m0fnu)

    expert(x, x_fp8=x_fp8, x_scale=x_scale)
    ((args, kwargs),) = expert.gate_up_proj.calls
    assert args[0] is x_fp8 and kwargs == {"x_scale": x_scale}

    expert.gate_up_proj.calls.clear()
    expert(x)
    ((args, kwargs),) = expert.gate_up_proj.calls
    assert args[0] is x and kwargs == {}


def _v4_moe(cls=V4MoE, **values):
    moe = cls.__new__(cls)
    torch.nn.Module.__init__(moe)
    moe.prefix = "layers.3.ffn"
    defaults = {
        "_use_dual_stream": False,
        "_use_comm_fused_dispatch": False,
        "_shared_in_routed": False,
    }
    defaults.update(values)
    for name, value in defaults.items():
        object.__setattr__(moe, name, value)
    return moe


def test_only_a_single_stream_layer_takes_the_pair():
    reads = SimpleNamespace(reads_mxfp8_input=lambda: True)
    assert _v4_moe(shared_experts=reads).takes_mxfp8_input()
    assert not _v4_moe(shared_experts=None).takes_mxfp8_input()
    assert not _v4_moe(
        shared_experts=SimpleNamespace(reads_mxfp8_input=lambda: False)
    ).takes_mxfp8_input()
    assert not _v4_moe(shared_experts=reads, _use_dual_stream=True).takes_mxfp8_input()
    assert not _v4_moe(
        shared_experts=reads, _use_comm_fused_dispatch=True
    ).takes_mxfp8_input()


def _pair(rows=4, dim=64):
    return (
        torch.zeros(rows, dim, dtype=torch.float8_e4m3fn),
        torch.zeros(rows, dim // 32, dtype=torch.float8_e8m0fnu),
    )


def test_a_dense_shared_expert_gets_the_pair():
    x, (x_fp8, x_scale) = torch.randn(4, 64), _pair()
    shared, routed, combined = (torch.randn(4, 64) for _ in range(3))
    moe = _v4_moe(shared_experts=_Recorder(shared))
    moe.experts = SimpleNamespace(combine_accepts_shared=lambda rows: False)
    moe.routed_expert_forward = _Recorder((routed, False))
    moe.combine_outputs = _Recorder(combined)

    assert V4MoE.single_stream_moe_forward(moe, x, x_fp8, x_scale) is combined
    ((args, kwargs),) = moe.shared_experts.calls
    assert args == (x,) and kwargs == {"x_fp8": x_fp8, "x_scale": x_scale}
    ((args, kwargs),) = moe.routed_expert_forward.calls
    assert kwargs["shared_partial"] is shared

    # Without one the shared expert is called as before.
    moe.shared_experts.calls.clear()
    V4MoE.single_stream_moe_forward(moe, x)
    assert moe.shared_experts.calls == [((x,), {})]


def test_a_shared_in_layer_hands_the_pair_to_the_routed_call():
    x, (x_fp8, x_scale) = torch.randn(4, 64), _pair()
    routed, combined = torch.randn(4, 64), torch.randn(4, 64)
    moe = _v4_moe(_shared_in_routed=True, shared_experts=_Recorder())
    moe.routed_with_shared_mxfp8 = _Recorder(routed)
    moe.routed_expert_forward = _Recorder((torch.randn(4, 64), False))
    moe.combine_outputs = _Recorder(combined)

    assert V4MoE.single_stream_moe_forward(moe, x, x_fp8, x_scale) is combined
    assert moe.routed_with_shared_mxfp8.calls == [((x, x_fp8, x_scale), {})]
    assert moe.routed_expert_forward.calls == [] and moe.shared_experts.calls == []
    ((args, _),) = moe.combine_outputs.calls
    assert args == (routed, None)


def test_the_routed_call_carries_the_pair_as_bytes(monkeypatch):
    from atom.model_ops.fused_moe import flydsl_mega_experts as mega

    seen = []
    monkeypatch.setattr(
        mega, "_moe_forward_shared_mxfp8_op", lambda *args: seen.append(args)
    )
    x, logits, (x_fp8, x_scale) = torch.randn(4, 64), torch.randn(4, 8), _pair()
    experts = SimpleNamespace(
        _routing_logits=lambda hidden, router: router, layer_name="layers.3.experts"
    )
    mega.moe_forward_shared_mxfp8(experts, x, logits, x_fp8, x_scale)
    ((hidden, router, fp8, scale, name),) = seen
    assert hidden is x and router is logits and name == "layers.3.experts"
    assert fp8.dtype == scale.dtype == torch.uint8
    assert fp8.data_ptr() == x_fp8.data_ptr()
    assert scale.data_ptr() == x_scale.data_ptr()


def test_the_pair_is_held_on_the_layer_only_while_it_runs(monkeypatch):
    from atom.model_ops.fused_moe import flydsl_mega_experts as mega

    x, (x_fp8, x_scale) = torch.randn(4, 64), _pair()
    held = []

    class Layer:
        def forward_impl(self, hidden, router):
            held.append(self._mega_shared_x_mxfp8)
            return hidden

    layer = Layer()
    context = {"layers.3.experts": layer}
    monkeypatch.setattr(
        "atom.config.get_current_atom_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=context)
        ),
    )
    out = mega._moe_forward_shared_mxfp8(
        x,
        torch.randn(4, 8),
        x_fp8.view(torch.uint8),
        x_scale.view(torch.uint8),
        "layers.3.experts",
    )
    assert out is x
    ((fp8, scale),) = held
    assert fp8.dtype == torch.float8_e4m3fn and scale.dtype == torch.float8_e8m0fnu
    assert layer._mega_shared_x_mxfp8 is None


def test_v41_moe_reshapes_the_pair_with_its_rows():
    hidden = torch.randn(2, 3, 64)
    fp8 = torch.zeros(2, 3, 64, dtype=torch.float8_e4m3fn)
    scale = torch.zeros(2, 3, 2, dtype=torch.float8_e8m0fnu)
    moe = _v4_moe(cls=V41MoE)
    moe.single_stream_moe_forward = _Recorder(torch.randn(6, 64))

    assert V41MoE.forward(moe, hidden, fp8, scale).shape == hidden.shape
    ((args, _),) = moe.single_stream_moe_forward.calls
    assert [tuple(a.shape) for a in args] == [(6, 64), (6, 64), (6, 2)]


class _Block:
    """`Block`'s FFN half with the seam and the MoE recorded."""

    def __init__(self, normed):
        self.normed = normed
        self.ffn = _Recorder(torch.randn(4, 64))

    def prepare_ffn(self, output, residual, pre, post, comb):
        return self.normed, residual, pre, post, comb

    def finish_ffn(self, output, *state):
        return output


def test_the_ffn_gets_the_seams_pair_when_the_seam_writes_one():
    bf16, (fp8, scale) = torch.randn(4, 64), _pair()
    block = _Block((bf16, fp8, scale))
    v41_model.Block.ffn_forward(block, None, None, None, None, None)
    assert block.ffn.calls == [((bf16, fp8, scale), {})]

    block = _Block(bf16)
    v41_model.Block.ffn_forward(block, None, None, None, None, None)
    assert block.ffn.calls == [((bf16,), {})]


@pytest.mark.parametrize(
    "packed, arch_ok, takes, expected",
    [
        (True, True, True, True),
        (False, True, True, False),  # Triton / gfx942 seam: no MXFP8 output
        (True, False, True, False),  # no row-major scales on this arch
        (True, True, False, False),  # the FFN cannot take it
    ],
)
def test_the_ffn_seam_asks_for_the_pair_only_where_it_is_read(
    monkeypatch, packed, arch_ok, takes, expected
):
    block = SimpleNamespace(
        hc_attn_fn=torch.zeros(1),
        hc_ffn_fn=torch.zeros(1),
        attn=SimpleNamespace(
            wqkv_a=SimpleNamespace(native_a8_group_rows=32, runs_bf16=False)
        ),
        ffn=SimpleNamespace(takes_mxfp8_input=lambda: takes),
    )
    monkeypatch.setattr(
        v41_model, "pack_mhc_fn", lambda fn: torch.zeros(1) if packed else None
    )
    monkeypatch.setattr(v41_model, "fuse_quant_supported", lambda: arch_ok)
    v41_model.Block.process_weights_after_loading(block)
    assert block.hc_ffn_fuse_quant is expected


# ---------------------------------------------------------------- GPU bytes

_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
_ROWS = [1, 5, 6, 48, 128, 186, 192, 512, 513, 2048, 4096]
_H, _HC = 5120, 4


def _seam():
    from atom.model_ops.deepseek_v41.mhc_pre_delayed import (
        fuse_quant_supported,
        pack_mhc_fn,
        pre_delayed,
    )

    g = torch.Generator(device="cuda").manual_seed(7)
    fn = torch.randn(_HC * (_HC + 2), _HC * _H, device="cuda", generator=g) * 0.02
    packed = pack_mhc_fn(fn.contiguous())
    if packed is None or not fuse_quant_supported():
        pytest.skip("needs HIP Mega-MHC with row-major MXFP8 scales")
    weights = (
        packed,
        torch.tensor([0.5, 0.5, 0.5], device="cuda"),
        torch.randn(_HC * (_HC + 2), device="cuda", generator=g) * 0.1,
        (1 + 0.1 * torch.randn(_H, device="cuda", generator=g)).to(torch.bfloat16),
    )
    options = {
        "rms_eps": 1e-6,
        "hc_eps": 1e-6,
        "sinkhorn_iters": 20,
        "post_mult": 2.0,
        "norm_eps": 1e-6,
        "w_preshuffle_bf16": True,
    }

    def run(rows, seed, with_post, fuse_quant):
        g = torch.Generator(device="cuda").manual_seed(seed * 7919 + rows)
        spread = torch.exp(torch.randn(rows, 1, 1, device="cuda", generator=g))
        residual = torch.randn(rows, _HC, _H, device="cuda", generator=g) * spread
        pre = torch.rand(rows, _HC, device="cuda", generator=g) + 0.1
        post_kw = {}
        if with_post:
            comb = torch.rand(rows, _HC, _HC, device="cuda", generator=g)
            post_kw = {
                "sublayer_output": (
                    torch.randn(rows, _H, device="cuda", generator=g) * spread[:, 0]
                ).to(torch.bfloat16),
                "post_mix": torch.rand(rows, _HC, device="cuda", generator=g) * 2,
                "combination": comb / comb.sum(-1, keepdim=True),
            }
        return pre_delayed(
            residual.to(torch.bfloat16),
            pre,
            *weights,
            fuse_quant=fuse_quant,
            **options,
            **post_kw,
        )

    return run


def _bytes_equal(a, b):
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@_gpu
@pytest.mark.parametrize("with_post", [True, False])
def test_the_ffn_seams_pair_is_quantize_fp8_of_its_norm(with_post):
    """Every row count, both seam shapes: the pair is byte for byte
    ``quantize_fp8`` of the BF16-only seam's norm, and every BF16 output is the
    BF16-only seam's."""
    from atom.model_ops.blockscale import quantize_fp8

    run = _seam()
    for rows in _ROWS:
        for seed in (1, 2):
            plain = run(rows, seed, with_post, fuse_quant=False)
            fused = run(rows, seed, with_post, fuse_quant=True)
            normed, fp8, scale = fused[1]
            ref_fp8, ref_scale = quantize_fp8(plain[1])
            assert _bytes_equal(fp8, ref_fp8), rows
            assert _bytes_equal(scale, ref_scale), rows
            assert _bytes_equal(normed, plain[1]), rows
            for got, want in zip((fused[0], *fused[2:]), (plain[0], *plain[2:])):
                assert _bytes_equal(got, want), rows


class _NativeLinear:
    """A loaded native group32 FP8 linear's GEMM (`LinearBase.forward`)."""

    runs_bf16 = False
    quant_type = SimpleNamespace(value=None)

    def __init__(self, weight, scale):
        from aiter import QuantType, dtypes

        from atom.model_ops.linear import weight_is_stored_preshuffled
        from atom.model_ops.utils import shuffle_weights

        self.native_a8_group_rows = weight.shape[0] // scale.shape[0]
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = scale
        if weight_is_stored_preshuffled(
            QuantType.per_1x32,
            dtypes.fp8,
            native_group_rows=self.native_a8_group_rows,
        ):
            shuffle_weights(self.weight)

    def __call__(self, x, x_scale=None):
        from atom.model_ops.blockscale import native_quant_linear

        return native_quant_linear(
            x,
            self.weight,
            self.weight_scale,
            x_scale=x_scale,
            weight_group_rows=self.native_a8_group_rows,
            weight_preshuffled=getattr(self.weight, "is_shuffled", False),
        )


def _checkpoint_expert(prefix):
    from safetensors import safe_open

    directory = os.environ.get("ATOM_DSV41_REFERENCE")
    if not directory:
        pytest.skip("Set ATOM_DSV41_REFERENCE for real checkpoint tests")
    root = Path(directory)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    config = json.loads((root / "config.json").read_text())

    def tensor(name):
        key = f"{prefix}.{name}"
        if key not in index:
            pytest.skip(f"{key} not in this checkpoint")
        with safe_open(root / index[key], framework="pt", device="cuda") as handle:
            return handle.get_tensor(key)

    def scale(name):
        s = tensor(f"{name}.scale")
        return s if s.dtype == torch.float8_e8m0fnu else s.view(torch.float8_e8m0fnu)

    expert = Expert.__new__(Expert)
    torch.nn.Module.__init__(expert)
    expert.gate_up_proj = _NativeLinear(
        torch.cat([tensor("w1.weight"), tensor("w3.weight")]).contiguous(),
        torch.cat([scale("w1"), scale("w3")]).contiguous(),
    )
    expert.w2 = _NativeLinear(tensor("w2.weight").contiguous(), scale("w2"))
    # A config without it reads as DeepseekV4Args does.
    expert.swiglu_limit = float(config.get("swiglu_limit", DeepseekV4Args.swiglu_limit))
    expert.use_fused_clamp_act_mul = False
    assert expert.reads_mxfp8_input()
    return expert


@_gpu
@pytest.mark.parametrize(
    "prefix", ["layers.10.ffn.shared_experts", "mtp.0.ffn.shared_experts"]
)
def test_the_dense_shared_expert_on_the_seams_pair_is_bit_identical(prefix):
    """Real shared-expert weights (a target layer, a draft layer): the dense
    chain from the FFN seam's pair gives the bytes it gives from the BF16
    rows it quantizes itself."""
    run, expert = _seam(), _checkpoint_expert(prefix)
    for rows in _ROWS:
        normed, fp8, scale = run(rows, 3, True, fuse_quant=True)[1]
        reference = expert(normed)
        assert torch.isfinite(reference.float()).all(), rows
        assert _bytes_equal(expert(normed, x_fp8=fp8, x_scale=scale), reference), rows
