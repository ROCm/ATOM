# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which dense-projection layouts the K3 mono kernels bind, and which they refuse.

The kernels read a dense weight one of two ways: plain bf16, or the production
per-token FP8 plane in place (E4M3 in AITER's (16, 16) preshuffle, rows padded
to 16, an fp32 scale a padded row). Anything else the weight's records describe
would be indexed as one of those and read wrong, with no error, so ``_dense``
has to refuse it at bind.
The binder compares quant types by name, so these run without AITER; the last
four check the kernels against AITER's own shuffle where it imports: their
indexing, then (on a GPU) the shared GEMV pieces against the dequantized
product and on every finite E4M3 byte.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.models.kimi_k3.mono import weights as W
from atom.models.kimi_k3.mono.config import EXPERTS, HEAD_DIM, HIDDEN, Dims
from atom.mono.runtime.consensus import MonoUnsupported

FP8 = torch.float8_e4m3fn
SHAPE = (200, 256)
PADDED = 208
# AITER's QuantType as the binder reads it
PER_TOKEN = SimpleNamespace(name="per_Token")
PER_1X128 = SimpleNamespace(name="per_1x128")


def fp8_linear(rows=PADDED, k=SHAPE[1], weight=None, **overrides):
    weight = torch.nn.Parameter(
        torch.zeros(rows, k, dtype=FP8) if weight is None else weight,
        requires_grad=False,
    )
    weight.is_shuffled = overrides.pop("is_shuffled", True)
    fields = {
        "weight": weight,
        "params_dtype": weight.dtype,
        "quant_type": PER_TOKEN,
        "native_a8_group_rows": None,
        "bias": None,
        "weight_scale": torch.ones(weight.shape[0], 1, dtype=torch.float32),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_bf16_binds_without_a_scale():
    weight = torch.nn.Parameter(torch.zeros(SHAPE, dtype=torch.bfloat16))
    w, scale = W._dense(SimpleNamespace(weight=weight), "proj", SHAPE)
    assert scale is None
    assert w.data_ptr() == weight.data_ptr()


@pytest.mark.parametrize("padded", [False, True])
def test_per_token_fp8_binds_in_place(padded):
    # ATOM records the logical rows of a plane it padded to the GEMM's tile;
    # one already a multiple of 16 rows is held as is
    record = {"is_output_padded": True, "_output_size_before_padding": SHAPE[0]}
    module = fp8_linear(**(record if padded else {}))
    w, scale = W._dense(module, "proj", SHAPE if padded else (PADDED, SHAPE[1]))
    assert w.data_ptr() == module.weight.data_ptr()
    assert scale.data_ptr() == module.weight_scale.data_ptr()
    assert scale.shape == (PADDED,)


@pytest.mark.parametrize(
    "module, shape, why",
    [
        (fp8_linear(is_shuffled=False), SHAPE, "preshuffle"),
        (fp8_linear(quant_type=PER_1X128), SHAPE, "per-token"),
        (fp8_linear(native_a8_group_rows=128), SHAPE, "native-a8"),
        (fp8_linear(rows=SHAPE[0]), SHAPE, "padded to 16"),
        (fp8_linear(rows=192), SHAPE, "weight shape"),
        (
            fp8_linear(is_output_padded=True, _output_size_before_padding=SHAPE[0] + 1),
            SHAPE,
            "output padding",
        ),
        (fp8_linear(bias=torch.zeros(PADDED)), SHAPE, "bias"),
        (
            fp8_linear(weight=torch.zeros(SHAPE[1], PADDED, dtype=FP8).t()),
            SHAPE,
            "not contiguous",
        ),
        (fp8_linear(weight_scale=None), SHAPE, "scale"),
        (fp8_linear(weight_scale=torch.ones(SHAPE[0])), SHAPE, "scale"),
        (
            fp8_linear(weight_scale=torch.ones(PADDED, dtype=torch.bfloat16)),
            SHAPE,
            "scale",
        ),
        (
            fp8_linear(weight_scale=torch.ones(2 * PADDED)[::2]),
            SHAPE,
            "scale not contiguous",
        ),
    ],
)
def test_unreadable_layouts_are_refused(module, shape, why):
    with pytest.raises(MonoUnsupported, match=why):
        W._dense(module, "proj", shape)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float8_e4m3fnuz])
def test_other_dtypes_are_refused(dtype):
    # E4M3FNUZ is FP8 too (gfx942's), but not the E4M3 the kernels decode
    module = fp8_linear(weight=torch.zeros(PADDED, SHAPE[1], dtype=dtype))
    with pytest.raises(MonoUnsupported, match="bf16 or preshuffled"):
        W._dense(module, "proj", SHAPE)


def _dense_as(module, name, shape):
    """`_dense` stubbed to the layout its module stands for: a scale or None."""
    return None, module.scale


AS_FP8 = SimpleNamespace(scale=torch.ones(1))
AS_BF16 = SimpleNamespace(scale=None)


@pytest.mark.parametrize("fp8", ["in_proj", "f_b_proj"])
def test_a_kda_layer_with_one_fp8_projection_is_refused(monkeypatch, fp8):
    # one K1 build reads in_proj and f_b_proj one way
    monkeypatch.setattr(W, "_dense", _dense_as)
    dims = Dims(8)
    attn = SimpleNamespace(
        _in_proj_fused=True,
        num_local_heads=dims.heads,
        head_dim=HEAD_DIM,
        conv_kernel_size=4,
        _kda_gate_lower_bound=-5.0,
        **{p: AS_FP8 if p == fp8 else AS_BF16 for p in ("in_proj", "f_b_proj")},
    )
    layer = SimpleNamespace(is_linear_attn=True, self_attn=attn)
    with pytest.raises(MonoUnsupported, match="not both bf16 or both FP8"):
        W.bind_kda(layer, 1, dims)


@pytest.mark.parametrize("bf16", ["o_proj", "gate_up_proj", "down_proj"])
def test_a_moe_layer_with_mixed_projections_is_refused(monkeypatch, bf16):
    # one K2 build reads o_proj and both shared-expert projections one way
    monkeypatch.setattr(W, "_dense", _dense_as)
    monkeypatch.setattr(W, "_experts", lambda *a: (None,) * 4)
    monkeypatch.setattr(W, "_mxfp4_to_bf16", lambda *a, **k: None)
    proj = {
        p: AS_BF16 if p == bf16 else AS_FP8
        for p in ("o_proj", "gate_up_proj", "down_proj")
    }
    moe = SimpleNamespace(
        use_latent_moe=True,
        shared_experts=SimpleNamespace(
            gate_up_proj=proj["gate_up_proj"], down_proj=proj["down_proj"]
        ),
        routed_expert_norm=object(),
        gate=SimpleNamespace(weight=torch.zeros(EXPERTS, HIDDEN, dtype=torch.bfloat16)),
        routed_expert_up_proj=None,
    )
    layer = SimpleNamespace(
        block_sparse_moe=moe, self_attn=SimpleNamespace(o_proj=proj["o_proj"])
    )
    with pytest.raises(MonoUnsupported, match="not all bf16 or all FP8"):
        W.bind_moe(layer, 1, Dims(8), 0)


def shuffled(rows, k):
    plain = torch.randint(
        0, 256, (rows, k), dtype=torch.uint8, generator=torch.Generator().manual_seed(0)
    )
    shuffle = pytest.importorskip("aiter.ops.shuffle")
    flat = shuffle.shuffle_weight(plain.view(FP8), layout=(16, 16))
    return plain, flat.view(torch.uint8).reshape(-1)


def test_w8_offset_indexes_the_production_preshuffle():
    # the kernels read the plane ATOM's FP8 linear holds: shuffle_weight at
    # (16, 16). Every 8-element run, then a GEMV chunk's operands as
    # gemv.w8_chunk_ops loads them: lane group g's 8 at kcol + 32 q + 8 g
    gemv = pytest.importorskip("atom.mono.device.gemv", exc_type=ImportError)
    rows, k = 48, 384
    plain, flat = shuffled(rows, k)
    for row in range(rows):
        for col in range(0, k, 8):
            at = gemv.w8_offset(row - row % 16, row % 16, col, k)
            assert torch.equal(flat[at : at + 8], plain[row, col : col + 8])
    for row0 in range(0, rows, 16):
        for lane in range(64):
            for kcol in range(0, k, 128):
                dwords = gemv.w8_lane_dwords(row0, lane, kcol, k)
                for q, at in enumerate(dwords):
                    col = kcol + 32 * q + 8 * (lane // 16)
                    want = plain[row0 + lane % 16, col : col + 8]
                    assert torch.equal(flat[4 * at : 4 * at + 8], want)


def test_fb_offset_indexes_the_production_preshuffle():
    # kda_gates' f_b operands (fb_offset, fb_w8_dword): wave w's rows 16 w ..
    # of head h, lane group g's K columns 32 g + 8 q
    kda = pytest.importorskip(
        "atom.models.kimi_k3.mono.kernels.attention.kda", exc_type=ImportError
    )
    hd = kda.HD
    plain, flat = shuffled(2 * hd, hd)
    for h in range(2):
        for wave in range(hd // 16):
            for lane in range(64):
                base = kda.fb_offset({"lane": lane, "wave": wave, "w8": True}, h)
                for q in range(4):
                    at = 4 * kda.fb_w8_dword(base, q)
                    row = h * hd + 16 * wave + lane % 16
                    col = 32 * (lane // 16) + 8 * q
                    assert torch.equal(flat[at : at + 8], plain[row, col : col + 8])


def _gemv_kernel(rows, k, s, w8):
    """One CTA a 16-row task of an [rows, k] GEMV against s token rows, from
    the ``gemv`` pieces K1 / K2 use: ``loads``, ``rows_to_lds``, ``mfmas``,
    then the reduce (``pair_scales`` / ``row_pair``) -> fp32 ``out[s, rows]``,
    each value bf16-rounded."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr import const_expr, gpu, range_constexpr
    from flydsl.expr.typing import Int64

    from atom.mono.device import gemv
    from atom.mono.device.ops import rsrc
    from atom.mono.plan.execution import THREADS, WAVES

    pairs, xrow = gemv.ROWS // 2, k + gemv.LDS_PAD
    default_stream = fx.Stream(None)

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        xl: fx.Array[fx.Int32, s * xrow // 2, 16]

    name = f"gemv_test_k{k}_s{s}_{'w8' if w8 else 'bf16'}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def gemv_task(w: Int64, scale: Int64, x: Int64, out: Int64):
        tid = fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        row0 = fx.block_idx.x * gemv.ROWS
        lds = fx.SharedAllocator().allocate(Smem).peek()
        red, xl = lds.red.ptr, lds.xl.ptr
        ops = gemv.loads(lane, wave, w, row0, k, w8=w8)
        scales = None
        if const_expr(w8):
            scales = gemv.pair_scales(scale, row0, tid)
        gemv.rows_to_lds(tid, x, k, s, xl, xrow)
        acc = gemv.mfmas(lane, wave, xl, xrow, k, s, ops, w8=w8)
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < pairs * s:
            rp = tid % pairs
            t = tid // pairs
            vs = gemv.row_pair(red, rp, t, scales)
            for i in range_constexpr(2):
                bo.buffer_store(vs[i], rsrc(out), t * rows + row0 + 2 * rp + i)

    @flyc.jit
    def launch(
        w: Int64, scale: Int64, x: Int64, out: Int64, stream: fx.Stream = default_stream
    ):
        gemv_task(w, scale, x, out).launch(
            grid=(rows // gemv.ROWS,), block=(THREADS,), stream=stream
        )

    return launch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("k", [768, 1536])
@pytest.mark.parametrize("s", [1, 5, 16])
@pytest.mark.parametrize("w8", [False, True])
def test_gemv_matches_the_dequantized_product(k, s, w8):
    # small integers: every product and partial sum is exact in fp32, so the
    # kernel's K order cannot change the result and any misread element or
    # wrong row scale shows as a mismatch
    pytest.importorskip("flydsl")
    shuffle = pytest.importorskip("aiter.ops.shuffle")
    rows, dev = 48, torch.device("cuda")
    gen = torch.Generator().manual_seed(k + s)
    plain = torch.randint(-16, 17, (rows, k), generator=gen).float()
    x = torch.randint(-8, 9, (s, k), generator=gen).to(torch.bfloat16)
    if w8:
        w = shuffle.shuffle_weight(plain.to(FP8).to(dev), layout=(16, 16))
        scale = torch.rand(rows, generator=gen) + 0.5
    else:
        w = plain.to(torch.bfloat16).to(dev)
        scale = torch.ones(rows)
    scale = scale.to(dev)
    x = x.to(dev)
    out = torch.zeros(s, rows, device=dev)
    _gemv_kernel(rows, k, s, w8)(
        w.data_ptr(), scale.data_ptr(), x.data_ptr(), out.data_ptr()
    )
    torch.cuda.synchronize()
    sums = x.float().cpu() @ plain.t()
    want = (sums * scale.cpu() if w8 else sums).to(torch.bfloat16).float()
    assert torch.equal(out.cpu(), want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_w8_widens_every_finite_code_exactly():
    # all 254 finite E4M3 bytes (subnormals to 448), each in every row and at
    # every position of a lane's 8-byte run, read back through one-hot token
    # rows: each output is one widened weight, so it must be the byte's value
    pytest.importorskip("flydsl")
    shuffle = pytest.importorskip("aiter.ops.shuffle")
    rows, k, s, dev = 48, 768, 16, torch.device("cuda")
    finite = torch.tensor(
        [b for b in range(256) if b & 0x7F != 0x7F], dtype=torch.uint8
    )
    at = (7 * torch.arange(rows)[:, None] + torch.arange(k)[None, :]) % finite.numel()
    plain = finite[at].view(FP8)
    w = shuffle.shuffle_weight(plain.to(dev), layout=(16, 16))
    scale = torch.ones(rows, device=dev)
    out = torch.zeros(s, rows, device=dev)
    launch = _gemv_kernel(rows, k, s, True)
    for c0 in range(0, k, s):
        x = torch.zeros(s, k, dtype=torch.bfloat16)
        x[torch.arange(s), c0 + torch.arange(s)] = 1
        x = x.to(dev)
        launch(w.data_ptr(), scale.data_ptr(), x.data_ptr(), out.data_ptr())
        torch.cuda.synchronize()
        want = plain.float()[:, c0 : c0 + s].t()
        assert torch.equal(out.cpu(), want), f"columns {c0} .. {c0 + s - 1}"
