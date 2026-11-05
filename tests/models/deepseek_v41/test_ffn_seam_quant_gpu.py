# SPDX-License-Identifier: MIT
"""The FFN seam's fused MXFP8 output and its delivery to the shared expert.

The seam (`pre_delayed(fuse_quant=True)`) must give the BF16 rows of the unfused
seam and, next to them, exactly `quantize_fp8` of those rows. `MoE.forward` must
hand that pair to the shared expert's `gate_up_proj` on every dispatch path (a
lost pair only costs a second quant, so nothing else would notice)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)
pytest.importorskip("aiter", reason="the seam and the MoE are AITER-backed")

_HC = 4
_HIDDEN = 5120
_SEAM = {"rms_eps": 1e-6, "hc_eps": 1e-6, "sinkhorn_iters": 20, "post_mult": 2.0}


def _bytes(t):
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize("rows", [1, 48, 97])
def test_fused_seam_rows_are_quantize_fp8_of_its_bf16_rows(rows):
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.deepseek_v41.mhc_pre_delayed import (
        fuse_quant_supported,
        pack_mhc_fn,
        pre_delayed,
    )

    if not fuse_quant_supported():
        pytest.skip("this arch/AITER has no row-major fused MXFP8 seam output")
    gen = torch.Generator(device="cuda").manual_seed(41)

    def randn(*shape, scale=1.0, dtype=torch.float32):
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(dtype)

    fn = randn(_HC * (_HC + 2), _HC * _HIDDEN, scale=0.02)
    packed = pack_mhc_fn(fn)
    if packed is None:
        pytest.skip("HIP Mega-MHC is not available")
    args = (
        randn(rows, _HC, _HIDDEN, dtype=torch.bfloat16),  # residual
        randn(rows, _HC).softmax(-1),  # pre_mix
        packed,
        torch.full((3,), 0.5, device="cuda"),  # hc_scale
        randn(_HC * (_HC + 2), scale=0.1),  # hc_base
        randn(_HIDDEN, scale=0.1, dtype=torch.bfloat16) + 1,  # ffn_norm weight
    )
    seam = dict(
        **_SEAM,
        norm_eps=1e-6,
        w_preshuffle_bf16=True,
        sublayer_output=randn(rows, _HIDDEN, dtype=torch.bfloat16),
        post_mix=randn(rows, _HC).sigmoid() * 2,
        combination=randn(rows, _HC, _HC).softmax(-1),
    )
    plain = pre_delayed(*args, **seam, fuse_quant=False)
    fused = pre_delayed(*args, **seam, fuse_quant=True)
    normed, fp8, scale = fused[1]
    assert torch.equal(_bytes(normed), _bytes(plain[1]))
    for got, want in zip(fused[2:], plain[2:]):
        assert torch.equal(got, want)
    want_fp8, want_scale = quantize_fp8(normed)
    assert torch.equal(_bytes(fp8), _bytes(want_fp8))
    assert torch.equal(_bytes(scale), _bytes(want_scale))


class _GateUp(nn.Module):
    """Records what `Expert.forward` feeds its gate_up."""

    def __init__(self, native):
        super().__init__()
        self.native_a8_group_rows = 32 if native else None
        self.calls = []

    def forward(self, x, x_scale=None):
        self.calls.append((x, x_scale))
        return torch.zeros(x.shape[0], 64, dtype=torch.bfloat16, device=x.device)


def _moe(dim, native=True):
    from atom.models.deepseek_v4 import Expert
    from atom.models.deepseek_v41.moe import MoE

    shared = Expert.__new__(Expert)
    nn.Module.__init__(shared)
    shared.gate_up_proj = _GateUp(native)
    shared.w2 = lambda x: torch.zeros(x.shape[0], dim, dtype=x.dtype, device=x.device)
    shared.swiglu_limit = 0.0
    shared.use_fused_clamp_act_mul = False

    moe = MoE.__new__(MoE)
    nn.Module.__init__(moe)
    moe.dim = dim
    moe.prefix = "v41.layers.0.ffn"
    moe.shared_experts = shared
    moe._use_dual_stream = False
    moe._use_comm_fused_dispatch = False
    moe.alt_stream = torch.cuda.Stream()
    moe.routed_inputs = []

    def routed_expert_forward(x, **kwargs):
        moe.routed_inputs.append(x)
        return torch.zeros_like(x), False

    moe.routed_expert_forward = routed_expert_forward
    moe.combine_outputs = lambda routed, shared, prefix=None: routed + shared
    return moe


def _seam_rows(rows, dim):
    from atom.model_ops.blockscale import quantize_fp8

    hidden = torch.randn(rows, dim, device="cuda").to(torch.bfloat16)
    fp8, scale = quantize_fp8(hidden)
    return hidden, fp8, scale


@pytest.mark.parametrize("path", ["single", "dual", "op_single", "op_dual"])
def test_moe_hands_the_seam_pair_to_the_shared_gate_up(monkeypatch, path):
    import atom.model_ops.module_dispatch_ops as ops

    dim = 128
    moe = _moe(dim)
    hidden, fp8, scale = _seam_rows(7, dim)
    if path.startswith("op_"):
        moe._use_dual_stream = True
        context = {moe.prefix: moe}
        monkeypatch.setattr(
            ops,
            "get_current_atom_config",
            lambda: SimpleNamespace(
                compilation_config=SimpleNamespace(static_forward_context=context)
            ),
        )
        monkeypatch.setattr(
            ops, "_dual_stream_is_active", lambda self, n: path == "op_dual"
        )
        moe((hidden, fp8, scale))
    elif path == "dual":
        moe.dual_stream_moe_forward(hidden, (fp8, scale))
    else:
        moe((hidden, fp8, scale))
    torch.cuda.synchronize()
    ((x, x_scale),) = moe.shared_experts.gate_up_proj.calls
    assert x.dtype == torch.float8_e4m3fn and x.data_ptr() == fp8.data_ptr()
    assert x_scale is not None and x_scale.data_ptr() == scale.data_ptr()
    (routed_in,) = moe.routed_inputs
    assert routed_in.dtype == torch.bfloat16
    assert routed_in.data_ptr() == hidden.data_ptr()


def test_bf16_rows_keep_the_shared_expert_quantizing_itself():
    dim = 128
    moe = _moe(dim)
    hidden, _, _ = _seam_rows(5, dim)
    moe(hidden)
    ((x, x_scale),) = moe.shared_experts.gate_up_proj.calls
    assert x.dtype == torch.bfloat16 and x_scale is None


def test_quantized_input_needs_a_native_mx_gate_up():
    dim = 128
    moe = _moe(dim, native=False)
    hidden, fp8, scale = _seam_rows(3, dim)
    with pytest.raises(AssertionError, match="native MX"):
        moe((hidden, fp8, scale))
