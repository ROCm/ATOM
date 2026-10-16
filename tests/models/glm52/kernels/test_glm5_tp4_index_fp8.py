# SPDX-License-Identifier: Apache-2.0
"""Byte-level oracle for ATOM's FP8 indexer key plane.

The GLM-5.2 TP4 MonoKernel's fused indexer writes and reads the same index
cache ATOM's production indexer uses.  Getting the preshuffle, the scale
direction or a segment offset wrong produces a plane that loads without error
and scores worse -- there is no exception to catch -- so the only cheap way to
know is to diff bytes against the writer that owns the format.

Three layers, cheapest first:

1. ``_oracle_plane`` is a transcription of aiter's pure-torch reference
   (``aiter/op_tests/test_indexer_k_quant_and_cache.py``: ``_split_k_scale``,
   ``_write_block_preshuffle``, ``_compute_ref_scale``, ``run_torch``).  It
   needs neither a GPU nor aiter.
2. ``_kernel_plane`` writes the same input through
   ``tp4_layout.index_fp8_data_dword`` / ``index_fp8_scale_dword`` -- the
   address arithmetic ``tp4_kernel.py`` itself calls -- and through
   ``_ue8m0_scale``, the integer-exponent form of the scale the kernel
   computes.  Layer 1 and layer 2 must agree byte for byte.
3. ``test_aiter_writer_matches_oracle`` runs the real production writer
   (``aiter.indexer_k_quant_and_cache``) on a GPU and diffs it against layer 1,
   which is what pins layer 1 to production rather than to itself.
4. ``test_flydsl_writer_matches_oracle_bytes`` runs the kernel's own FP8 write
   expressions as a standalone one-wave-per-token kernel.  That is the only
   layer that can see whether ``cvt_pk_fp8_f32`` emits OCP E4M3 on this arch
   and whether the even-lane dword assembly orders its four bytes correctly.

What this file does NOT prove: the read path.  Dequantization is exercised only
by compiling it; nothing here scores anything.  And layer 4 is a *copy* of the
kernel's twenty-line write block, not a call into it, so the two can drift.
"""

import math

import pytest
import torch

from atom.models.glm52.mono.kernels.tp4_layout import (
    INDEX_ALIGNED_DIM,
    INDEX_DIM,
    INDEX_TILE,
    index_fp8_data_dword,
    index_fp8_scale_dword,
)

# gfx950 / MI355X: OCP e4m3 (torch.float8_e4m3fn), max 448.  gfx942 would be
# e4m3fnuz with max 240 -- see aiter/aiter/utility/dtypes.py:12-13.
FP8_MAX = 448.0
PAGE = 16


def _ue8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    """The fp32 power-of-two scale ATOM's FP8 index writer stores for one row.

    aiter computes ``exp2(ceil(log2(max(amax, 1e-4) / 448)))`` in floating point
    (``cache_kernels.cu:1921-1930``).  That is an exact statement about the
    exponent field: writing ``amax = m * 2**a`` with ``m`` in [1, 2),
    ``amax / 448 = (m / 1.75) * 2**(a - 8)`` and ``m / 1.75`` lies in
    [0.5714, 1.1429], so the smallest enclosing power of two is ``2**(a - 8)``
    when ``m <= 1.75`` and ``2**(a - 7)`` otherwise.  ``m > 1.75`` is exactly
    ``mantissa > 0x600000``.  The kernel computes it this way -- no division by
    448 and no log2, both of which can land on the wrong side of a power of two
    by one ulp and silently halve or double a whole row.
    """

    bits = torch.clamp(amax.to(torch.float32), min=1e-4).view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 8
    exponent = exponent + ((bits & 0x7FFFFF) > 0x600000).to(torch.int32)
    return (exponent.clamp(1, 254) << 23).view(torch.float32)


def _oracle_plane(k: torch.Tensor, slots: list[int], num_blocks: int, page: int = PAGE) -> torch.Tensor:
    """aiter's pure-torch reference writer, transcribed."""

    head_dim = k.shape[-1]
    plane = torch.zeros((num_blocks, page, INDEX_ALIGNED_DIM), dtype=torch.uint8)
    flat = plane.view(num_blocks, page * INDEX_ALIGNED_DIM)
    values = k.to(torch.float32)
    amax = values.abs().max(dim=-1, keepdim=True).values
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / FP8_MAX)))
    quantized = torch.clamp(values / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
    for i, slot in enumerate(slots):
        if slot < 0:
            continue
        block, offset = slot // page, slot % page
        token_tile, token_in_tile = offset // INDEX_TILE, offset % INDEX_TILE
        for col_tile in range(head_dim // INDEX_TILE):
            base = token_tile * INDEX_TILE * head_dim + col_tile * INDEX_TILE * INDEX_TILE + token_in_tile * INDEX_TILE
            flat[block, base : base + INDEX_TILE] = quantized[i, col_tile * INDEX_TILE : (col_tile + 1) * INDEX_TILE]
        scale_offset = page * head_dim + offset * 4
        flat[block, scale_offset : scale_offset + 4] = scale[i].view(torch.uint8)
    return plane


def _kernel_plane(
    k: torch.Tensor,
    slots: list[int],
    num_blocks: int,
    page: int = PAGE,
    *,
    preshuffle: bool = True,
    reciprocal: bool = True,
) -> torch.Tensor:
    """The same write through the kernel's own address arithmetic.

    ``preshuffle=False`` and ``reciprocal=False`` exist so the tests below can
    show that this harness actually rejects the two layouts a reimplementation
    is most likely to produce by accident.
    """

    plane = torch.zeros((num_blocks, page, INDEX_ALIGNED_DIM), dtype=torch.uint8)
    flat = plane.view(-1)
    values = k.to(torch.float32)
    scale = _ue8m0_scale(values.abs().max(dim=-1, keepdim=True).values)
    factor = (1.0 / scale) if reciprocal else scale
    quantized = torch.clamp(values * factor, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
    for i, slot in enumerate(slots):
        if slot < 0:
            continue
        pg, row = slot // page, slot % page
        # Four elements per store: one lane pair of the kernel's write wave.
        for column in range(0, INDEX_DIM, 4):
            if preshuffle:
                dword = index_fp8_data_dword(pg, row, column, page)
            else:
                dword = (pg * page * INDEX_ALIGNED_DIM + row * INDEX_DIM + column) // 4
            flat[dword * 4 : dword * 4 + 4] = quantized[i, column : column + 4]
        dword = index_fp8_scale_dword(pg, row, page)
        flat[dword * 4 : dword * 4 + 4] = scale[i].view(torch.uint8)
    return plane


def _sample(tokens: int, seed: int = 0) -> tuple[torch.Tensor, list[int], int]:
    generator = torch.Generator().manual_seed(seed)
    k = torch.randn((tokens, INDEX_DIM), generator=generator).to(torch.bfloat16)
    slots = list(range(tokens))
    return k, slots, (tokens + PAGE - 1) // PAGE


def test_aligned_dim_is_144():
    assert INDEX_ALIGNED_DIM == 144


def test_scale_matches_the_float_formula():
    """The integer-exponent scale equals exp2(ceil(log2(amax / 448)))."""

    generator = torch.Generator().manual_seed(7)
    amax = torch.cat(
        [
            torch.rand(20000, generator=generator) * 40.0 + 1e-3,
            # Values whose quotient lands exactly on a power of two: amax =
            # 1.75 * 2**m is the boundary the float form can fall off.
            torch.tensor([1.75 * 2.0**m for m in range(-10, 11)]),
            torch.tensor([448.0 * 2.0**m for m in range(-10, 11)]),
        ]
    )
    reference = torch.tensor([2.0 ** math.ceil(math.log2(float(v) / FP8_MAX)) for v in amax], dtype=torch.float32)
    assert torch.equal(_ue8m0_scale(amax), reference)


def test_kernel_layout_matches_oracle_bytes():
    """The kernel's address arithmetic and scale reproduce aiter's plane."""

    k, slots, blocks = _sample(256)
    oracle = _oracle_plane(k, slots, blocks)
    mine = _kernel_plane(k, slots, blocks)
    mismatched = int((oracle != mine).sum())
    assert mismatched == 0, f"{mismatched} of {oracle.numel()} bytes differ"


def test_kernel_layout_matches_oracle_on_shuffled_pages():
    """Physical pages need not be in logical order."""

    k, _, blocks = _sample(256, seed=3)
    generator = torch.Generator().manual_seed(11)
    pages = torch.randperm(blocks, generator=generator).tolist()
    slots = [pages[i // PAGE] * PAGE + i % PAGE for i in range(k.shape[0])]
    assert torch.equal(_oracle_plane(k, slots, blocks), _kernel_plane(k, slots, blocks))


def test_negative_slots_write_nothing():
    k, slots, blocks = _sample(64, seed=5)
    slots = [-1 if i % 3 == 0 else s for i, s in enumerate(slots)]
    mine = _kernel_plane(k, slots, blocks)
    assert torch.equal(_oracle_plane(k, slots, blocks), mine)
    for i, slot in enumerate(slots):
        if slot < 0:
            row = mine.view(-1, PAGE, INDEX_ALIGNED_DIM)[i // PAGE]
            assert int(row.view(-1)[index_fp8_scale_dword(0, i % PAGE, PAGE) * 4 : ][:4].sum()) == 0


def test_padding_is_never_written():
    """The last 12 bytes of each 144-byte row stay zero."""

    k, slots, blocks = _sample(128, seed=9)
    plane = _kernel_plane(k, slots, blocks).view(blocks, PAGE * INDEX_ALIGNED_DIM)
    assert int(plane[:, PAGE * (INDEX_DIM + 4) :].sum()) == 0


def test_harness_rejects_a_row_major_plane():
    """Trap 2 of the format spec: skipping the MFMA 16x16 preshuffle."""

    k, slots, blocks = _sample(64, seed=1)
    assert not torch.equal(_oracle_plane(k, slots, blocks), _kernel_plane(k, slots, blocks, preshuffle=False))


def test_harness_rejects_a_multiplied_scale():
    """Trap 3/6: the writer multiplies by 1/scale, not by the scale."""

    k, slots, blocks = _sample(64, seed=2)
    assert not torch.equal(_oracle_plane(k, slots, blocks), _kernel_plane(k, slots, blocks, reciprocal=False))


def _build_index_writer(tokens: int, page: int = PAGE):
    """A one-wave-per-token FlyDSL kernel holding tp4_kernel's FP8 write verbatim.

    The CPU layers above cannot see whether the kernel's *expressions* lower to
    those bytes: whether ``cvt_pk_fp8_f32`` emits OCP E4M3 on this arch, whether
    the exponent arithmetic survives, whether the even-lane dword assembly puts
    the four bytes in the right order.  Running the whole MonoKernel would
    answer that, but it needs four GPUs and a layer's worth of weights.  This
    runs the same twenty lines on their own.

    The quantize block below is a copy of the ``const_expr(index_fp8)`` branch in
    ``tp4_kernel.py``'s cache stage, and a copy is a thing that can drift -- keep
    them together.
    """

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr import math as fmath
    from flydsl.expr import range_constexpr, rocdl
    from flydsl.expr.typing import Int64, T

    from atom.models.glm52.mono.kernels.tp4_layout import fp8_pe_upper_pair_lane
    from atom.mono.device.ops import bpermute_i32
    from atom.mono.device.ops import rsrc as _rsrc
    from atom.mono.device.ops import xred as _xred

    @flyc.kernel(known_block_size=[64, 1, 1])
    def write_index_rows(k_in: Int64, slot_in: Int64, plane: Int64):
        lane = fx.thread_idx.x
        token = fx.block_idx.x
        r_k, r_plane = _rsrc(k_in), _rsrc(plane)
        slot = fx.Int32(bo.buffer_load(_rsrc(slot_in), token, vec_width=1, dtype=T.i32))
        i0 = lane * 2
        v0 = fx.Float32(fx.BFloat16(bo.buffer_load(r_k, token * INDEX_DIM + i0, vec_width=1, dtype=T.bf16)))
        v1 = fx.Float32(fx.BFloat16(bo.buffer_load(r_k, token * INDEX_DIM + i0 + 1, vec_width=1, dtype=T.bf16)))

        amax = fx.max(fmath.absf(v0), fmath.absf(v1))
        for sh in range_constexpr(6):
            amax = _xred(amax, 32 >> sh, fx.max)
        amax = fx.max(amax, fx.Float32(1.0e-4))
        bits = amax.bitcast(fx.Int32)
        exponent = ((bits >> 23) & fx.Int32(0xFF)) - 8
        exponent = exponent + ((bits & fx.Int32(0x7FFFFF)) > fx.Int32(0x600000)).select(fx.Int32(1), fx.Int32(0))
        exponent = fx.max(fx.min(exponent, fx.Int32(254)), fx.Int32(1))
        qs = (exponent << 23).bitcast(fx.Float32)
        inv = ((fx.Int32(254) - exponent) << 23).bitcast(fx.Float32)
        q0 = fx.min(fx.max(v0 * inv, -FP8_MAX), FP8_MAX)
        q1 = fx.min(fx.max(v1 * inv, -FP8_MAX), FP8_MAX)
        pg, row = slot // page, slot % page
        pair = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & fx.Int32(0xFFFF)
        upper_pair = bpermute_i32(fp8_pe_upper_pair_lane(lane) * 4, pair)
        if lane % 2 == 0:
            bo.buffer_store(pair | (upper_pair << 16), r_plane, index_fp8_data_dword(pg, row, i0, page))
        if lane == 0:
            bo.buffer_store(qs, r_plane, index_fp8_scale_dword(pg, row, page))

    @flyc.jit
    def launch(k_in: Int64, slot_in: Int64, plane: Int64, stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        write_index_rows(k_in, slot_in, plane).launch(grid=(tokens,), block=(64,), stream=stream)

    return launch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_flydsl_writer_matches_oracle_bytes():
    """The kernel's own expressions, on the GPU, against the oracle's bytes."""

    pytest.importorskip("aiter", reason="the writer pulls buffer_ops out of the AITER runtime")
    pytest.importorskip("flydsl")
    tokens = 128
    k, slots, blocks = _sample(tokens, seed=21)
    device = torch.device("cuda")
    plane = torch.zeros((blocks, PAGE, INDEX_ALIGNED_DIM), dtype=torch.uint8, device=device)
    # Keep both alive: the launcher takes raw pointers, so a temporary would be
    # freed and reused before the kernel ran.
    keys = k.to(device).contiguous()
    slot_tensor = torch.tensor(slots, dtype=torch.int32, device=device)
    _build_index_writer(tokens)(keys.data_ptr(), slot_tensor.data_ptr(), plane.data_ptr())
    torch.cuda.synchronize()
    produced = plane.cpu()
    oracle = _oracle_plane(k, slots, blocks)
    mismatched = int((produced != oracle).sum())
    assert mismatched == 0, f"{mismatched} of {produced.numel()} bytes differ from the oracle"


def _bf16_ue8m0_boundary_triplet(m: int) -> tuple[float, float, float]:
    """(below, at, above) bf16-exact amax values bracketing ``1.75 * 2**m``.

    ``1.75 * 2**m`` is the one place ``exp2(ceil(log2(amax / 448)))`` can
    disagree with the exponent-field form: the quotient lands exactly on a
    power of two, and ``ceil`` is a cliff there. bf16 has 7 explicit mantissa
    bits, so its ulp at this exponent is ``2**(m - 7)``, and both the boundary
    value and its neighbors are exact bf16 values -- no float64-to-bf16
    rounding can move them off the point being tested.
    """

    base = 1.75 * (2.0**m)
    ulp = (2.0**-7) * (2.0**m)
    return (base - ulp, base, base + ulp)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_flydsl_writer_matches_aiter_at_ue8m0_boundary():
    """The open question: does the kernel's exponent-field scale agree with
    aiter's actual ``exp2f(ceilf(log2f(amax / 448)))`` float32 computation,
    bit for bit, at ``amax = 1.75 * 2**m`` -- the only point where they could
    disagree -- and at the bf16 values one ulp either side of it.

    aiter computes this with real float32 division and ``log2f`` / ``ceilf``
    / ``exp2f`` (``cache_kernels.cu:1367-1372``, the kernel behind
    ``aiter.indexer_k_quant_and_cache``), not a reciprocal multiply, so the
    1-ulp reciprocal slip this task hypothesized does not apply to the
    division step itself. But ``log2f``/``exp2f`` are still transcendental
    approximations, and nothing elsewhere in this file runs the real op at
    this exact boundary -- random data, as the module docstring for
    ``test_scale_matches_the_float_formula`` notes, essentially never lands
    there. This compares the kernel's own FlyDSL write expressions (not the
    Python oracle) directly against the real op's output bytes, on rows
    engineered to sit on the cliff.
    """

    aiter = pytest.importorskip("aiter")
    pytest.importorskip("flydsl")
    from aiter.ops.cache import indexer_k_quant_and_cache

    assert torch.finfo(aiter.dtypes.fp8).max == FP8_MAX, "this box is not OCP fp8"

    rows: list[float] = []
    for m in range(-10, 11):
        rows.extend(_bf16_ue8m0_boundary_triplet(m))
    tokens = len(rows)
    blocks = (tokens + PAGE - 1) // PAGE
    slots = list(range(tokens))

    k = torch.zeros((tokens, INDEX_DIM), dtype=torch.bfloat16)
    for i, amax in enumerate(rows):
        k[i, 0] = torch.tensor(amax, dtype=torch.bfloat16)

    device = torch.device("cuda")
    keys = k.to(device).contiguous()
    slot_i64 = torch.tensor(slots, dtype=torch.int64, device=device)
    slot_i32 = torch.tensor(slots, dtype=torch.int32, device=device)

    aiter_plane = torch.zeros((blocks, PAGE, INDEX_ALIGNED_DIM), dtype=aiter.dtypes.fp8, device=device)
    indexer_k_quant_and_cache(keys, aiter_plane, slot_i64, INDEX_DIM, "ue8m0", preshuffle=True)
    aiter_bytes = aiter_plane.view(torch.uint8).cpu()

    kernel_plane = torch.zeros((blocks, PAGE, INDEX_ALIGNED_DIM), dtype=torch.uint8, device=device)
    _build_index_writer(tokens)(keys.data_ptr(), slot_i32.data_ptr(), kernel_plane.data_ptr())
    torch.cuda.synchronize()
    kernel_bytes = kernel_plane.cpu()

    mismatched = int((aiter_bytes != kernel_bytes).sum())
    assert mismatched == 0, (
        f"{mismatched} of {aiter_bytes.numel()} bytes differ between aiter's real "
        "writer and the kernel's own expressions at the ue8m0 boundary (amax = "
        "1.75 * 2**m, and the bf16 values one ulp either side of it)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_aiter_writer_matches_oracle():
    """Pin the transcription to the production writer itself."""

    aiter = pytest.importorskip("aiter")
    from aiter.ops.cache import indexer_k_quant_and_cache

    assert torch.finfo(aiter.dtypes.fp8).max == FP8_MAX, "this box is not OCP fp8"
    k, slots, blocks = _sample(256, seed=13)
    device = torch.device("cuda")
    plane = torch.zeros((blocks, PAGE, INDEX_ALIGNED_DIM), dtype=aiter.dtypes.fp8, device=device)
    indexer_k_quant_and_cache(
        k.to(device),
        plane,
        torch.tensor(slots, dtype=torch.int64, device=device),
        INDEX_DIM,
        "ue8m0",
        preshuffle=True,
    )
    produced = plane.view(torch.uint8).cpu()
    oracle = _oracle_plane(k, slots, blocks)
    mismatched = int((produced != oracle).sum())
    assert mismatched == 0, f"{mismatched} of {produced.numel()} bytes differ from aiter"
    assert torch.equal(produced, _kernel_plane(k, slots, blocks))
