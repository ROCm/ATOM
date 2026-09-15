"""The fused combine+quant A2A kernel must match combine-then-quant exactly
enough, including on the empty-rank rows that carry lse=-inf and o=NaN.

No distributed setup: both kernels are driven directly off a synthetic `recv`
buffer, which is the only thing the all-to-all produces.
"""

import types

import pytest
import torch

# Order matters: importing triton without an active GPU driver raises rather
# than failing the import, so the device check has to come first.
if not torch.cuda.is_available():
    pytest.skip(
        "drives Triton kernels on real tensors; needs a real GPU",
        allow_module_level=True,
    )

triton = pytest.importorskip("triton")

from aiter import dtypes as _aiter_dtypes

from atom.model_ops.dcp_ops import (
    _dcp_a2a_unpack_combine_kernel,
    _dcp_a2a_unpack_combine_quant_kernel,
    _lse_pack_slots,
)

# The dtype the production call site passes: _dcp_fused_quant_dtype returns
# aiter's dtypes.fp8, which is e4m3fn on MI355X. Hard-coding e4m3fnuz here
# tested a format this path never sees.
FP8 = _aiter_dtypes.fp8


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
    """A row every rank reports empty must dequantize to zero, not to NaN.

    Scale 0 with a zero reciprocal, which is what aiter stores at these shapes
    and what `kimi_k3/quant.py` mirrors, so the fused and unfused paths stay
    interchangeable for o_proj. aiter lets its clamp run and leaves `-FP8_MAX`
    in the quantized row; zeroing it as well is the one deliberate difference,
    and it is the safer of the two if a consumer ever ignores the scale.
    """
    b, h, d, n, dtype = 16, 4, 256, 4, torch.bfloat16
    pack = _lse_pack_slots(dtype)
    empty = (0, 7, 15)
    recv = _make_recv(n, b, h, d, dtype, pack, empty_rows=empty)

    got_q, got_s = _run_fused(recv, b, h, d, n, pack)
    assert torch.isfinite(got_s).all(), got_s

    deq = got_q.float().reshape(b, h * d) * got_s
    assert torch.isfinite(deq).all()
    for row in empty:
        assert got_s[row].item() == 0.0, "aiter stores a zero scale here"
        assert got_q.float().reshape(b, h * d)[row].abs().max().item() == 0.0
        assert deq[row].abs().max().item() == 0.0


def test_quant_matches_aiter_bit_for_bit():
    """The fused quant must BE aiter's per-token quant, not merely close to it.

    o_proj cannot tell which kernel produced its input, so a 1 ulp difference in
    the scale is not a rounding detail -- it moves elements to an adjacent FP8
    code. Spelling the scale as a divide costs ~0.1% of codes and a different
    scale on over half the rows, and a tolerance test does not see any of it.

    One rank makes the combine exact: global_lse == lse, so factor == 1.0 and
    the fp32 accumulator holds the bf16 input unchanged. That removes the
    intentional bf16-round-trip difference and leaves only the quantizer.
    """
    from aiter import QuantType, get_hip_quant

    n, b, h, d = 1, 64, 4, 256
    pack = _lse_pack_slots(torch.bfloat16)
    recv = _make_recv(n, b, h, d, torch.bfloat16, pack, seed=3)
    recv[..., d:] = 0.0  # lse == 0 on the single rank

    got_q, got_s = _run_fused(recv, b, h, d, n, pack)
    want_q, want_s = get_hip_quant(QuantType.per_Token)(
        recv[0, ..., :d].reshape(b, h * d).contiguous(), quant_dtype=FP8
    )

    assert torch.equal(got_s.reshape(-1), want_s.reshape(-1))
    assert torch.equal(
        got_q.view(torch.uint8).reshape(b, -1), want_q.view(torch.uint8).reshape(b, -1)
    )


def test_lse_and_fused_quant_is_rejected_at_any_group_size():
    """The combination the fused combine cannot serve must be rejected, and the
    single-rank shortcut must not be a way around that.

    The shortcut returns before the body runs, so an assertion placed after it
    would let a single-rank caller through with the LSE silently dropped -- the
    one shape of bug this check exists to prevent.
    """
    from atom.model_ops.dcp_ops import cp_lse_a2a

    o = torch.zeros(2, 4, 8, device="cuda")
    lse = torch.zeros(2, 4, device="cuda")
    for world_size in (1, 4):
        group = types.SimpleNamespace(world_size=world_size, device_group=None)
        with pytest.raises(AssertionError, match="does not emit LSE"):
            cp_lse_a2a(o, lse, group, return_lse=True, quant_dtype=FP8)


def test_non_power_of_two_group():
    """A group size N_ROUNDED has to round up, with a poison slab where it lands.

    Both kernels index n over N_ROUNDED, so a 3-rank group has a fourth lane
    addressing one rank-slab past `recv`. That lane's factor is zero, so an
    unmasked load cannot change a number -- the only symptom is the access
    itself, which is invisible from here and shows up in the field as a fault.
    So this pins what IS observable: that a non-power-of-two group combines
    correctly, and that nothing from beyond `recv` leaks into the result.
    """
    n, b, h, d = 3, 4, 2, 128
    assert triton.next_power_of_2(n) > n, "meaningless for a power-of-two group"
    pack = _lse_pack_slots(torch.bfloat16)

    big = torch.empty((n + 1, b, h, d + pack), dtype=torch.bfloat16, device="cuda")
    big[:n] = _make_recv(n, b, h, d, torch.bfloat16, pack)
    big[n] = 1e4  # where the padding lane points
    recv = big[:n]

    ref = _run_unfused(recv, b, h, d, n, pack, torch.bfloat16)

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
    deq = out.float().reshape(b, -1) * scale
    got = deq.reshape(b, h, d)
    # fp8 has ~2 decimal digits; compare against the bf16 combine it must track.
    torch.testing.assert_close(got, ref.float(), rtol=0.1, atol=0.1)
    assert got.abs().max().item() < 1e3, "a value from beyond `recv` reached the output"
