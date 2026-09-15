"""The fused combine+quant A2A kernel must match combine-then-quant exactly
enough, including on the empty-rank rows that carry lse=-inf and o=NaN.

No distributed setup: both kernels are driven directly off a synthetic `recv`
buffer, which is the only thing the all-to-all produces.
"""

import pytest
import torch

triton = pytest.importorskip("triton")

from atom.model_ops.dcp_ops import (  # noqa: E402
    _dcp_a2a_unpack_combine_kernel,
    _dcp_a2a_unpack_combine_quant_kernel,
    _lse_pack_slots,
)

FP8 = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn


def _make_recv(n, b, h, d, dtype, pack, empty_rows=(), seed=0):
    """[N, B, H, D+pack] as the all-to-all delivers it."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    recv = torch.empty((n, b, h, d + pack), dtype=dtype, device="cuda")
    body = torch.randn((n, b, h, d), generator=g).to(dtype).cuda()
    lse = torch.randn((n, b, h), generator=g).cuda().float() * 2.0
    for row in empty_rows:  # a rank owning no KV for this row
        body[:, row] = float("nan")
        lse[:, row] = float("-inf")
    recv[..., :d] = body
    if pack == 1:
        recv[..., d] = lse.to(dtype)
    else:
        bits = lse.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        hi = ((bits >> 16) & 0xFFFF).to(torch.uint16).view(dtype)
        lo = (bits & 0xFFFF).to(torch.uint16).view(dtype)
        recv[..., d] = hi
        recv[..., d + 1] = lo
    return recv


def _run_unfused(recv, b, h, d, n, pack, dtype):
    out = torch.empty((b, h, d), dtype=dtype, device="cuda")
    _dcp_a2a_unpack_combine_kernel[(b, h)](
        recv,
        out,
        out,
        recv.stride(0),
        recv.stride(1),
        recv.stride(2),
        out.stride(0),
        out.stride(1),
        0,
        0,
        n,
        HEAD_DIM=d,
        LSE_PACK=pack,
        N_ROUNDED=triton.next_power_of_2(n),
        WRITE_LSE=False,
    )
    return out


def _run_fused(recv, b, h, d, n, pack):
    out = torch.empty((b, h, d), dtype=FP8, device="cuda")
    scale = torch.empty((b, 1), dtype=torch.float32, device="cuda")
    _dcp_a2a_unpack_combine_quant_kernel[(b,)](
        recv,
        out,
        scale,
        recv.stride(0),
        recv.stride(1),
        recv.stride(2),
        out.stride(0),
        out.stride(1),
        n,
        HEAD_DIM=d,
        H_LOCAL=h,
        LSE_PACK=pack,
        N_ROUNDED=triton.next_power_of_2(n),
        FP8_MAX=float(torch.finfo(FP8).max),
    )
    return out, scale


def _torch_combine_fp32(recv, b, h, d, n, pack):
    """The combine in fp32, with no bf16 intermediate. This is what the fused
    kernel computes; the unfused pair rounds to bf16 in between."""
    body = recv[..., :d].float()
    if pack == 1:
        lse = recv[..., d].float()
    else:
        hi = recv[..., d].view(torch.uint16).to(torch.int64)
        lo = recv[..., d + 1].view(torch.uint16).to(torch.int64)
        bits = ((hi << 16) | lo).to(torch.int32)
        lse = bits.view(torch.float32)
    lse = torch.where(torch.isfinite(lse), lse, torch.full_like(lse, float("-inf")))
    lse_max = lse.amax(dim=0, keepdim=True)
    lse_max = torch.where(torch.isinf(lse_max), torch.zeros_like(lse_max), lse_max)
    glse = (lse - lse_max).exp().sum(dim=0, keepdim=True).log() + lse_max
    factor = (lse - glse).exp()
    factor = torch.where(torch.isfinite(factor), factor, torch.zeros_like(factor))
    body = torch.where(factor[..., None] == 0, torch.zeros_like(body), body)
    return (body * factor[..., None]).sum(dim=0).reshape(b, h * d)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("b,h,d,n", [(51, 4, 256, 4), (8, 4, 256, 4), (1, 2, 128, 2)])
def test_fused_matches_fp32_combine_then_quant(dtype, b, h, d, n):
    pack = _lse_pack_slots(dtype)
    recv = _make_recv(n, b, h, d, dtype, pack)

    row32 = _torch_combine_fp32(recv, b, h, d, n, pack)
    got_q, got_s = _run_fused(recv, b, h, d, n, pack)

    ref_s = row32.abs().amax(dim=1, keepdim=True) / torch.finfo(FP8).max
    ref_s = torch.where(ref_s > 0, ref_s, torch.ones_like(ref_s))
    torch.testing.assert_close(got_s, ref_s, rtol=2e-3, atol=0)

    # fp8 e4m3 is a FLOATING grid, not a uniform one: 3 mantissa bits give a
    # relative half-step of 2**-4. The absolute floor is the subnormal step,
    # scale * 2**-9. Comparing against a single `scale` would be the tolerance
    # for an integer grid and is simply the wrong model.
    deq = got_q.float().reshape(b, h * d) * got_s
    torch.testing.assert_close(
        deq, row32, rtol=2.0**-4, atol=(got_s * 2.0**-9).max().item()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_scale_differs_from_the_bf16_path_only_by_bf16_rounding():
    """DELIBERATE numerics change, pinned here so it cannot drift unnoticed.

    Today: combine -> store bf16 -> per-token quant. Fused: combine -> quant,
    straight out of the fp32 accumulator. The fused scale is therefore taken
    from unrounded values and differs from today's by up to one bf16 step
    (2**-8 = 0.39%). It is the more accurate of the two, but it is NOT
    bit-identical to the path it replaces.
    """
    b, h, d, n, dtype = 51, 4, 256, 4, torch.bfloat16
    pack = _lse_pack_slots(dtype)
    recv = _make_recv(n, b, h, d, dtype, pack)

    bf16_row = _run_unfused(recv, b, h, d, n, pack, dtype).float().reshape(b, h * d)
    bf16_s = bf16_row.abs().amax(dim=1, keepdim=True) / torch.finfo(FP8).max
    _, got_s = _run_fused(recv, b, h, d, n, pack)

    rel = ((got_s - bf16_s).abs() / bf16_s).max().item()
    assert rel < 2.0**-8, f"scale drifted {rel:.5f}, more than one bf16 step"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_empty_rows_do_not_poison_the_scale():
    """A row every rank reports empty must come back all-zero with scale 1.0,
    not NaN -- NaN in the scale would silently destroy the whole o_proj row."""
    b, h, d, n, dtype = 16, 4, 256, 4, torch.bfloat16
    pack = _lse_pack_slots(dtype)
    empty = (0, 7, 15)
    recv = _make_recv(n, b, h, d, dtype, pack, empty_rows=empty)

    got_q, got_s = _run_fused(recv, b, h, d, n, pack)
    assert torch.isfinite(got_s).all(), got_s

    deq = got_q.float().reshape(b, h * d) * got_s
    assert torch.isfinite(deq).all()
    for row in empty:
        assert got_s[row].item() == 1.0
        assert deq[row].abs().max().item() == 0.0


# --------------------------------------------------------------------------
# Gating. Runs without a GPU: the predicate only reads attributes off self, so
# a stand-in exercises the whole matrix. A wrong answer here is a silent
# numerics bug (o_proj told a tensor is quantized when it is not), which is
# exactly the kind of thing that never shows up as a crash.
# --------------------------------------------------------------------------
import types  # noqa: E402

from aiter import QuantType, dtypes  # noqa: E402

from atom.model_ops.attention_mla import MLAAttention  # noqa: E402


def _stub(**over):
    o_proj = types.SimpleNamespace(
        quant_type=QuantType.per_Token, params_dtype=dtypes.fp8, input_scale=None
    )
    for k in ("quant_type", "params_dtype", "input_scale"):
        if k in over:
            setattr(o_proj, k, over.pop(k))
    s = types.SimpleNamespace(pbm_enabled=True, dcp_comm_backend="a2a", o_proj=o_proj)
    for k, v in over.items():
        setattr(s, k, v)
    return s


@pytest.mark.parametrize(
    "over,expect",
    [
        ({}, "fp8"),  # the one good case
        ({"pbm_enabled": False}, None),  # o_proj not fed directly
        ({"dcp_comm_backend": "ag_rs"}, None),  # other backend
        ({"quant_type": QuantType.per_Tensor}, None),  # not per-token
        ({"quant_type": QuantType.per_1x128}, None),  # block scheme
        ({"params_dtype": dtypes.fp4x2}, None),  # not fp8
        ({"input_scale": object()}, None),  # static scale
        ({"o_proj": None}, None),  # no o_proj at all
    ],
)
def test_fused_quant_gating(over, expect):
    got = MLAAttention._dcp_fused_quant_dtype(_stub(**over))
    assert got == (dtypes.fp8 if expect == "fp8" else None)
