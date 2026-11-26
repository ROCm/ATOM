# SPDX-License-Identifier: MIT
"""CSA2 operator/ownership differential tests against pinned upstream methods."""

import pytest
import torch

pytest.importorskip("aiter", reason="the compressor and cached RoPE are AITER kernels")

from atom.model_ops.deepseek_v41.compressor import Compressor
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding


@pytest.mark.parametrize("original_length,base", [(0, 10000), (65536, 160000)])
def test_rope_matches_reference_interleaved_and_inverse(
    reference, original_length, base
):
    torch.manual_seed(723)
    target = RotaryEmbedding(
        64, 1024, base=base, original_length=original_length, factor=16
    )
    frequencies = reference.precompute_freqs_cis(
        64, 1024, original_length, base, 16, 32, 1
    )
    torch.testing.assert_close(target.frequencies, frequencies, rtol=0, atol=0)
    positions = torch.tensor([0, 1, 97, 511, 1023])
    for shape in ((2, 5, 512), (2, 5, 8, 512)):
        source = torch.randn(shape, dtype=torch.bfloat16)
        for inverse in (False, True):
            expected = source.clone()
            reference.apply_rotary_emb(
                expected[..., -64:], frequencies[positions], inverse
            )
            actual = target(source.clone(), positions, inverse=inverse)
            assert torch.equal(actual, expected)


def test_rope_pair_rotates_both_as_two_separate_calls_would():
    """The two-channel entry is a launch count, not a different rotation.

    On CPU this takes the fallback, so what it pins is the contract every
    caller depends on: both tensors rotated, in place, in argument order.
    """
    torch.manual_seed(724)
    target = RotaryEmbedding(64, 1024, base=10000)
    positions = torch.tensor([0, 1, 97, 511, 1023])
    query = torch.randn((2, 5, 8, 512), dtype=torch.bfloat16)
    latent = torch.randn((2, 5, 512), dtype=torch.bfloat16)
    expected = target(query.clone(), positions), target(latent.clone(), positions)

    actual = target.pair(query, latent, positions)

    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert actual[0] is query and actual[1] is latent


class _Step:
    """The fields the window write reads: one request and one padding row."""

    def __init__(self, tokens, slot, device):
        self.running_tokens = tokens
        self.positions = torch.arange(tokens, device=device, dtype=torch.int32)
        self.batch_ids = torch.zeros(tokens, device=device, dtype=torch.int32)
        self.batch_ids[-1] = -1
        self.cu_seqlens_q = torch.tensor([0, tokens - 1], device=device)
        self.slots = torch.tensor([slot], device=device, dtype=torch.int32)


def _window(start, ring):
    """`WindowParams` as `geometry.window` builds it for either layout."""
    params = type("Window", (), {})()
    params.ring_start, params.slot_rows, params.ring_slots = start, ring, ring
    params.ring_stride, params.run_rows = ring, ring
    return params


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("tokens", [5, 13])
@pytest.mark.parametrize("start", [0, 4096])
def test_fused_seam_reproduces_rope_quant_and_window_byte_for_byte(
    planar, tokens, start
):
    """The fused KV seam against the launches it replaces, in both layouts.

    Bytes, not a tolerance: the quantization shares its exponent helper with
    `quantize_fp8`, and the rotation has to land on the same BF16 value the
    removed kernel boundaries used to round to. A tolerance here would pass
    while the cache and the extend rows disagreed by a code.

    The planar arm uses an independent host scatter of quantized bytes. The
    BF16 reference writer walks the ring per request rather than per token.
    Both check the entire allocation, including untouched rows and padding.
    """
    from atom.model_ops.attentions.deepseek_v41.planes import split_source
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.deepseek_v41.rope_window import rope_quant_window
    from atom.model_ops.v4_kernels.state_writes import swa_write

    torch.manual_seed(725)
    device, heads, dim, rope_dim, ring = "cuda", 4, 512, 64, 8
    slot = 3
    rope = RotaryEmbedding(rope_dim, 1024, base=10000).to(device)
    step = _Step(tokens, slot, device)
    # The RoPE entry reads its row count off the first two dims, so both stay
    # in the batched shape the attention hands over and the kernel takes views.
    query = torch.randn(1, tokens, heads, dim, device=device, dtype=torch.bfloat16)
    normed = torch.randn(1, tokens, dim, device=device, dtype=torch.bfloat16)
    window = _window(start, ring)

    rotated = rope(normed.clone(), step.positions)
    values, scales = quantize_fp8(rotated)
    qat = quantize_fp8(rotated, dequantize=True)
    rows = start + (slot + 1) * ring
    base = start + slot * ring
    if planar:
        backing = torch.zeros(
            rows * (dim + dim // 32), device=device, dtype=torch.uint8
        )
        pool = split_source(backing, dim)
        for token in range(max(0, tokens - 1 - ring), tokens - 1):
            row = base + int(step.positions[token]) % ring
            pool[0][row] = values.view(torch.uint8).view(-1, dim)[token]
            pool[1][row] = scales.view(torch.uint8).view(-1, dim // 32)[token]
        actual_backing = torch.zeros_like(backing)
        actual_pool = split_source(actual_backing, dim)
        written = pool[0][base : base + ring]
    else:
        pool = torch.zeros(rows, dim, device=device, dtype=torch.bfloat16)
        swa_write(
            qat.view(tokens, dim),
            step.positions,
            step.cu_seqlens_q,
            step.slots,
            pool,
            window,
            ring,
        )
        actual_pool = torch.zeros_like(pool)
        written = pool[base : base + ring]
    # Without this the ring arms compare two untouched buffers. Rows, not
    # elements: an element is free to be zero, a whole row is not.
    assert int((written != 0).any(-1).sum()) == min(tokens - 1, ring)

    actual_query = query.clone()
    rope_quant_window(
        actual_query.view(tokens, heads, dim),
        normed.view(tokens, dim),
        rope.cos_cache,
        rope.sin_cache,
        step.positions,
        rope_dim=rope_dim,
        ring=(actual_pool, step, window),
    )

    assert torch.equal(actual_query, rope(query.clone(), step.positions))
    if planar:
        assert torch.equal(actual_backing, backing)
    else:
        assert torch.equal(actual_pool, pool)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_fused_seam_emits_the_rows_a_prefill_attends_to():
    """The other half: a prefill stores nothing and hands its rows back.

    Its window write has to stay behind attention, so this mode replaces the
    two `quantize_fp8` calls rather than the window writer, and both of their
    outputs have to come out of the one launch unchanged.
    """
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.deepseek_v41.rope_window import rope_quant_window

    torch.manual_seed(726)
    device, tokens, heads, dim, rope_dim = "cuda", 13, 4, 512, 64
    rope = RotaryEmbedding(rope_dim, 1024, base=10000).to(device)
    positions = torch.arange(tokens, device=device, dtype=torch.int32)
    normed = torch.randn(1, tokens, dim, device=device, dtype=torch.bfloat16)
    query = torch.randn(1, tokens, heads, dim, device=device, dtype=torch.bfloat16)

    rotated = rope(normed.clone(), positions)
    values, scales = quantize_fp8(rotated)
    qat = quantize_fp8(rotated, dequantize=True)

    actual_values = torch.empty_like(values)
    actual_scales = torch.empty_like(scales)
    actual_qat = torch.empty_like(normed)
    rope_quant_window(
        query.view(tokens, heads, dim),
        normed.view(tokens, dim),
        rope.cos_cache,
        rope.sin_cache,
        positions,
        rope_dim=rope_dim,
        qat=actual_qat,
        values=actual_values,
        scales=actual_scales,
    )

    assert torch.equal(actual_qat.view(tokens, dim), qat.view(tokens, dim))
    assert torch.equal(actual_values.view(torch.uint8), values.view(torch.uint8))
    assert torch.equal(actual_scales.view(torch.uint8), scales.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_fused_seam_writes_the_fp8_pools_query_and_rows():
    """The FP8 pool's prefill: the rotated query quantized by the KV row's rule
    (`query_fp8`), and the rows into a `[tokens, 528]` buffer through strided
    views -- the bytes `quantize_fp8` gives, packed values then scales."""
    from atom.model_ops.attentions.deepseek_v41.packed_rows import pack_rows
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.deepseek_v41.rope_window import rope_quant_window

    torch.manual_seed(727)
    device, tokens, heads, dim, rope_dim = "cuda", 13, 4, 512, 64
    rope = RotaryEmbedding(rope_dim, 1024, base=10000).to(device)
    positions = torch.arange(tokens, device=device, dtype=torch.int32)
    normed = torch.randn(1, tokens, dim, device=device, dtype=torch.bfloat16)
    query = torch.randn(1, tokens, heads, dim, device=device, dtype=torch.bfloat16)

    values, scales = quantize_fp8(rope(query.clone(), positions))
    rows = torch.empty(tokens, dim + dim // 32, device=device, dtype=torch.uint8)
    query_values = torch.empty_like(values).view(tokens, heads, dim)
    query_scales = torch.empty_like(scales).view(tokens, heads, dim // 32)
    rope_quant_window(
        query.clone().view(tokens, heads, dim),
        normed.view(tokens, dim),
        rope.cos_cache,
        rope.sin_cache,
        positions,
        rope_dim=rope_dim,
        values=rows[:, :dim].view(values.dtype),
        scales=rows[:, dim:].view(scales.dtype),
        query_fp8=(query_values, query_scales),
    )

    expected_rows = pack_rows(*quantize_fp8(rope(normed.clone(), positions)))
    assert torch.equal(rows, expected_rows.view(tokens, -1))
    assert torch.equal(
        query_values.view(torch.uint8), values.view(torch.uint8).view_as(query_values)
    )
    assert torch.equal(
        query_scales.view(torch.uint8), scales.view(torch.uint8).view_as(query_scales)
    )


def _run_fp8_query(n, query, normed, rope, positions, rope_dim):
    """`rope_quant_window` with `_FP8_HEADS` at `n` (None: the arch's list),
    an MXFP8 query sink and the FP8 KV packed as rows: (KV rows, query
    values, query scales). `query` is not modified."""
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.deepseek_v41 import rope_window

    tokens, heads, dim = query.shape
    fp8, e8m0 = (t.dtype for t in quantize_fp8(normed))
    rows = torch.empty(tokens, dim + dim // 32, device=query.device, dtype=torch.uint8)
    qv = torch.empty(tokens, heads, dim, device=query.device, dtype=fp8)
    qs = torch.empty(tokens, heads, dim // 32, device=query.device, dtype=e8m0)
    saved = rope_window._FP8_HEADS
    rope_window._FP8_HEADS = n
    try:
        rope_window.rope_quant_window(
            query.clone(),
            normed,
            rope.cos_cache,
            rope.sin_cache,
            positions,
            rope_dim=rope_dim,
            values=rows[:, :dim].view(fp8),
            scales=rows[:, dim:].view(e8m0),
            query_fp8=(qv, qs),
        )
    finally:
        rope_window._FP8_HEADS = saved
    return rows, qv, qs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("fp8_heads", [2, 4, 16, 64])
def test_fp8_query_heads_a_program_is_bit_exact(monkeypatch, fp8_heads):
    """Any power-of-two group of MXFP8 query heads a program gives a program a
    head's query and KV bytes; the KV keeps the slot after the query ones."""
    from atom.model_ops.deepseek_v41 import rope_window

    torch.manual_seed(729)
    device, tokens, heads, dim, rope_dim = "cuda", 64, 64, 512, 64
    rope = RotaryEmbedding(rope_dim, 1024, base=10000).to(device)
    positions = torch.randint(0, 1024, (tokens,), device=device, dtype=torch.int32)
    normed = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16)
    query = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16) * 3
    monkeypatch.setattr(rope_window, "_FP8_HEADS_MIN_PROGRAMS_PER_CU", 0)

    def run(n):
        rows, qv, qs = _run_fp8_query(n, query, normed, rope, positions, rope_dim)
        return rows, qv.view(torch.uint8), qs.view(torch.uint8)

    for expected, actual in zip(run(1), run(fp8_heads)):
        assert torch.equal(actual, expected)


def _zero_sign_cleared(values):
    """MXFP8 bytes with an OCP E4M3 -0.0 read as +0.0. FNUZ has no -0.0 (its
    0x80 is NaN), so those bytes are kept as they are."""
    if values.dtype != torch.float8_e4m3fn:
        return values.view(torch.uint8)
    values = values.view(torch.uint8)
    return torch.where(values & 0x7F == 0, 0, values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("rope_dim", [16, 64])
@pytest.mark.parametrize("tokens", [1, 7, 1344, 1536, 2048])
def test_fp8_query_rope_tail_only_is_bit_exact(monkeypatch, rope_dim, tokens):
    """From 16 heads a program only the RoPE tail is rotated when it is whole
    quant groups, the NoPE groups quantized as loaded; 8 keep the full row.
    Same bytes as a program a head but for the sign of a zero, on rows of
    zeros, -0.0 lanes beside either sign, saturating, tiny and ordinary
    values; scales and KV bytes exact."""
    from atom.model_ops.deepseek_v41 import rope_window

    torch.manual_seed(tokens + rope_dim)
    device, heads, dim = "cuda", 64, 512
    rope = RotaryEmbedding(rope_dim, 4096, base=10000).to(device)
    positions = torch.randint(0, 4096, (tokens,), device=device, dtype=torch.int32)
    normed = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16)
    query = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16) * 3
    query[0, : heads // 4] = 0
    # Even and odd lanes, NoPE and RoPE, beside neighbours of either sign.
    query[0, heads // 4 :: 2, ::5] = -0.0
    if tokens > 1:
        query[1, :, ::3] = 3e4
        query[-1, heads // 2 :] = 1e-30
    monkeypatch.setattr(rope_window, "_FP8_HEADS_MIN_PROGRAMS_PER_CU", 0)

    def run(n):
        rows, qv, qs = _run_fp8_query(n, query, normed, rope, positions, rope_dim)
        return rows, _zero_sign_cleared(qv), qs.view(torch.uint8)

    reference = run(1)
    for n in (8, 16, 32):
        for expected, actual in zip(reference, run(n)):
            assert torch.equal(actual, expected)


def test_fp8_query_heads_a_program_follow_the_arch(monkeypatch):
    """gfx1250 tries 16 heads a program then 8, elsewhere 8, unless
    `_FP8_HEADS` says; the first that divides the heads and keeps two
    programs a CU wins. A rank's 8 heads (TP8) keep 8 on gfx1250."""
    pytest.importorskip("aiter", reason="the arch is read through AITER")
    from aiter.jit.utils import chip_info

    from atom.model_ops.deepseek_v41 import rope_window

    monkeypatch.setattr(rope_window, "_cu_count", lambda device: 256)
    # The arch is faked under a cache: clear it whatever the asserts do, so
    # later tests read the real arch.
    try:
        for arch, tokens, heads, expected in (
            ("gfx1250", 1344, 64, 16),
            ("gfx1250", 1344, 8, 8),
            ("gfx1250", 64, 64, 8),
            ("gfx1250", 32, 64, 1),
            ("gfx1250", 1344, 6, 1),
            ("gfx950", 1344, 64, 8),
            ("gfx950", 1344, 8, 8),
            ("gfx942", 1344, 64, 8),
        ):
            monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda arch=arch: arch)
            rope_window._arch_fp8_heads.cache_clear()
            assert rope_window._fp8_heads(tokens, heads, "cuda") == expected
        monkeypatch.setattr(rope_window, "_FP8_HEADS", 4)
        assert rope_window._fp8_heads(1344, 64, "cuda") == 4
        assert rope_window._fp8_heads(1344, 6, "cuda") == 1
    finally:
        rope_window._arch_fp8_heads.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize(
    "heads, tokens, fp8_heads",
    [(8, 1344, 8), (8, 2048, 8), (64, 1344, 16), (64, 2048, 16), (64, 64, 8)],
)
def test_fp8_query_heads_fall_back_along_the_list(
    monkeypatch, heads, tokens, fp8_heads
):
    """With gfx1250's (16, 8), a rank's 8 heads (TP8) still quantize 8 a
    program rather than 1, and so do 64 heads where 16 a program would leave
    too few programs; the bytes match a program a head's."""
    from atom.model_ops.deepseek_v41 import rope_window

    torch.manual_seed(heads + tokens)
    monkeypatch.setattr(rope_window, "_cu_count", lambda device: 256)
    monkeypatch.setattr(rope_window, "_arch_fp8_heads", lambda: (16, 8))
    device, dim, rope_dim = "cuda", 512, 64
    rope = RotaryEmbedding(rope_dim, 1024, base=10000).to(device)
    positions = torch.randint(0, 1024, (tokens,), device=device, dtype=torch.int32)
    normed = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16)
    query = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16)

    def run(n):
        spy = _GridSpy(rope_window._rope_window_kernel)
        monkeypatch.setattr(rope_window, "_rope_window_kernel", spy)
        rows, qv, qs = _run_fp8_query(n, query, normed, rope, positions, rope_dim)
        monkeypatch.setattr(rope_window, "_rope_window_kernel", spy.kernel)
        return (rows, _zero_sign_cleared(qv), qs.view(torch.uint8)), spy.launches

    reference, _ = run(1)
    actual, launches = run(None)
    assert launches == [((tokens * (heads // fp8_heads + 1),), False, fp8_heads)]
    for expected, got in zip(reference, actual):
        assert torch.equal(got, expected)


class _GridSpy:
    """Wraps a Triton kernel to record each launch's grid, `HEAD_BLOCK` and
    `FP8_HEADS`."""

    def __init__(self, kernel):
        self.kernel, self.launches = kernel, []

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            self.launches.append(
                (tuple(grid), kwargs["HEAD_BLOCK"], kwargs["FP8_HEADS"])
            )
            return launch(*args, **kwargs)

        return run


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("heads", [64, 4, 6])
@pytest.mark.parametrize(
    "sink", ["planar_ring", "bf16_ring", "prefill", "query", "query_fp8", "no_query"]
)
def test_fused_seam_head_block_matches_a_program_a_head(monkeypatch, heads, sink):
    """The head-blocked query grid against a program a head, byte for byte.

    One program rotating a token's heads has to give the per-head programs'
    query and leave the KV half -- now slot 1, not slot `heads` -- writing the
    same rows. The launches are recorded, so a gate that never picks the
    head-blocked grid fails here rather than comparing a grid to itself.
    Six heads is no power of two, the MXFP8 query (`query_fp8`) is not the
    in-place rotation, and a query-less KV launch (`no_query`, a prefill whose
    queries rotate separately) has no heads: all three keep a program a head,
    except that the MXFP8 query quantizes 8 heads a program where they divide.
    """
    from atom.model_ops.attentions.deepseek_v41.planes import split_source
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.deepseek_v41 import rope_window

    if sink == "no_query" and heads != 64:
        pytest.skip("no_query has no heads")
    torch.manual_seed(728)
    device, tokens, dim, rope_dim, ring, slot, start = "cuda", 37, 512, 64, 8, 3, 64
    rope = RotaryEmbedding(rope_dim, 1024, base=10000).to(device)
    step = _Step(tokens, slot, device)
    query_heads = 0 if sink == "no_query" else heads
    query = torch.randn(tokens, query_heads, dim, device=device, dtype=torch.bfloat16)
    normed = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16)
    window = _window(start, ring)
    rows = start + (slot + 1) * ring
    kv_half = sink != "query"
    head_block = sink in ("planar_ring", "bf16_ring", "prefill", "query") and (
        (heads & (heads - 1)) == 0
    )
    fp8_heads = 8 if sink == "query_fp8" and heads % 8 == 0 else 1
    expected_grid = (
        tokens * ((1 if head_block else query_heads // fp8_heads) + kv_half),
    )

    def run(flag):
        monkeypatch.setattr(rope_window, "_HEAD_BLOCK", flag)
        monkeypatch.setattr(rope_window, "_FP8_HEADS", 8 if flag else 1)
        monkeypatch.setattr(rope_window, "_FP8_HEADS_MIN_PROGRAMS_PER_CU", 0)
        spy = _GridSpy(rope_window._rope_window_kernel)
        monkeypatch.setattr(rope_window, "_rope_window_kernel", spy)
        out = {"query": query.clone()}
        kwargs = {}
        if sink == "planar_ring":
            out["pool"] = torch.zeros(
                rows * (dim + dim // 32), device=device, dtype=torch.uint8
            )
            kwargs["ring"] = (split_source(out["pool"], dim), step, window)
        elif sink == "bf16_ring":
            out["pool"] = torch.zeros(rows, dim, device=device, dtype=torch.bfloat16)
            kwargs["ring"] = (out["pool"], step, window)
        elif sink in ("prefill", "query_fp8", "no_query"):
            out["qat"] = torch.empty_like(normed)
            # Buffers of the dtypes the seam writes; it overwrites them.
            out["values"], out["scales"] = quantize_fp8(normed)
            kwargs.update(qat=out["qat"], values=out["values"], scales=out["scales"])
        if sink == "query_fp8":
            # Zeroed as bytes: not every FP8 dtype implements a fill.
            def zeros(width, like):
                buffer = torch.zeros(
                    tokens, heads, width, device=device, dtype=torch.uint8
                )
                return buffer.view(like.dtype)

            out["query_values"] = zeros(dim, out["values"])
            out["query_scales"] = zeros(dim // 32, out["scales"])
            kwargs["query_fp8"] = (out["query_values"], out["query_scales"])
        rope_window.rope_quant_window(
            out["query"],
            normed if kv_half else None,
            rope.cos_cache,
            rope.sin_cache,
            step.positions,
            rope_dim=rope_dim,
            **kwargs,
        )
        monkeypatch.setattr(rope_window, "_rope_window_kernel", spy.kernel)
        return out, spy.launches

    (expected, default_launches), (actual, launches) = run(False), run(True)
    assert default_launches == [((tokens * (query_heads + kv_half),), False, 1)]
    assert launches == [(expected_grid, head_block, fp8_heads)]
    if sink in ("packed_ring", "bf16_ring", "prefill", "query"):
        assert not torch.equal(expected["query"], query)
    for name, tensor in expected.items():
        assert torch.equal(actual[name].view(torch.uint8), tensor.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize(
    "cus, tokens, fp8_heads",
    [(256, 1, 1), (256, 63, 1), (256, 64, 8), (80, 19, 1), (80, 20, 8)],
)
def test_fp8_query_groups_heads_from_two_programs_a_cu(
    monkeypatch, cus, tokens, fp8_heads
):
    """64 heads quantize 8 a program from 2 grouped query programs a CU on
    (64 tokens on 256 CUs, 20 on 80) and keep a program a head below."""
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.deepseek_v41 import rope_window

    monkeypatch.setattr(rope_window, "_cu_count", lambda device: cus)
    monkeypatch.setattr(rope_window, "_FP8_HEADS", 8)
    device, heads, dim, rope_dim = "cuda", 64, 512, 64
    rope = RotaryEmbedding(rope_dim, 1024, base=10000).to(device)
    positions = torch.zeros(tokens, device=device, dtype=torch.int32)
    normed = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16)
    query = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16)
    values, scales = quantize_fp8(normed)
    query_fp8 = (
        torch.empty(tokens, heads, dim, device=device, dtype=values.dtype),
        torch.empty(tokens, heads, dim // 32, device=device, dtype=scales.dtype),
    )
    spy = _GridSpy(rope_window._rope_window_kernel)
    monkeypatch.setattr(rope_window, "_rope_window_kernel", spy)
    rope_window.rope_quant_window(
        query,
        normed,
        rope.cos_cache,
        rope.sin_cache,
        positions,
        rope_dim=rope_dim,
        values=values,
        scales=scales,
        query_fp8=query_fp8,
    )
    assert spy.launches == [((tokens * (heads // fp8_heads + 1),), False, fp8_heads)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("ratio", [1, 2])
def test_compressor_all_chunk_boundaries_against_official_decode(
    reference, single_rank, ratio
):
    torch.manual_seed(317)
    args = reference.ModelArgs(
        dim=64, head_dim=64, compress_ratios=(ratio,), max_batch_size=2, max_seq_len=16
    )
    with reference.set_dtype(torch.bfloat16):
        source = reference.Compressor(args, 0)
        target = Compressor(64, 64, ratio, args.norm_eps).cuda()
        upstream = dict(source.named_parameters())
        # Filled one disk tensor at a time through the merged loader, which is
        # how `packed_modules_mapping` fills `wkv_gate` -- so this also pins
        # that the pooling half is shard 0 and the gate is shard 1. Feeding
        # both halves the same numbers would make that unfalsifiable.
        for shard_id, name in enumerate(("wkv",) if ratio == 1 else ("wkv", "wgate")):
            value = (torch.randn(64, 64) * 0.1).bfloat16()
            target.wkv_gate.weight_loader(target.wkv_gate.weight, value, shard_id)
            upstream[f"{name}.weight"].data.copy_(value)
        for norm in (target.norm.weight, upstream["norm.weight"]):
            norm.data.fill_(1)
        hidden = torch.randn(2, 11, 64, dtype=torch.bfloat16)
        expected = []
        for position in range(hidden.shape[1]):
            latent = source(hidden[:, position : position + 1], position)
            if latent is not None:
                expected.append(latent)
        expected = torch.cat(expected, dim=1)
        for chunks in ((11,), (1, 3, 2, 5), (2, 1, 7, 1)):
            position, tail, actual = 0, None, []
            for length in chunks:
                latent, tail = target(
                    hidden[:, position : position + length].cuda(), position, tail
                )
                if latent is not None:
                    actual.append(latent.cpu())
                position += length
            torch.testing.assert_close(
                torch.cat(actual, dim=1), expected, rtol=1 / 128, atol=2**-10
            )
            assert (tail is None) == (ratio == 1)


@pytest.mark.parametrize("ratio,rows", [(1, 32), (2, 64)])
def test_compressor_holds_pool_and_gate_in_one_matrix(single_rank, ratio, rows):
    """One parameter, and the two disk tensors land in their own row ranges.

    A ratio-1 layer ships no `wgate` at all, so its matrix is the pooling half
    alone -- and the merged loader has to refuse a gate shard there rather
    than write past the rows it owns.
    """
    torch.manual_seed(451)
    compressor = Compressor(96, 32, ratio, 1e-6)
    assert [name for name, _ in compressor.named_parameters()] == [
        "wkv_gate.weight",
        "norm.weight",
    ]
    assert compressor.wkv_gate.weight.shape == (rows, 96)
    shards = [torch.full((32, 96), float(i + 1)) for i in range(ratio)]
    for shard_id, value in enumerate(shards):
        compressor.wkv_gate.weight_loader(compressor.wkv_gate.weight, value, shard_id)
    for shard_id, value in enumerate(shards):
        loaded = compressor.wkv_gate.weight[shard_id * 32 : (shard_id + 1) * 32]
        assert torch.equal(loaded.float(), value)
    if ratio == 1:
        with pytest.raises(AssertionError):
            compressor.wkv_gate.weight_loader(compressor.wkv_gate.weight, shards[0], 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_compressor_project_hands_back_strided_halves(single_rank):
    """The halves the two writers read: zero-copy, unit inner stride.

    `fused_compress_attn` and `update_compressor_states` both take a row
    stride but address the head dimension as `col_off + d`, so a trailing
    chunk of one GEMM is exactly what they accept -- and making the halves
    contiguous instead would put the copy back that the fusion removed.
    """
    torch.manual_seed(451)
    compressor = Compressor(96, 32, 2, 1e-6).cuda()
    compressor.wkv_gate.weight.data.normal_(std=0.1)
    values, scores = compressor.project(torch.randn(7, 96, dtype=torch.bfloat16).cuda())
    for half in (values, scores):
        assert half.shape == (7, 32)
        assert half.stride() == (64, 1)
        assert half.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()


@pytest.mark.parametrize("bf16", [False, True])
def test_draft_context_kv_fusion_sees_the_kv_half(
    single_rank, small_config, monkeypatch, bf16
):
    """The fused context-KV GEMM reads a shard of `wqkv_a`, not a `wkv` layer.

    It concatenates one projection per stage and declines when a stage cannot
    supply one. Nothing else asserts that a real attention module supplies it,
    so a rename on the model side turns the fusion off and only costs speed --
    the per-stage fallback it falls back to is numerically identical. A BF16
    shard (no group32 FP8 GEMM) has no FP8 weight to fuse: per stage.
    """
    from types import SimpleNamespace

    from atom.model_ops import linear
    from atom.models.deepseek_v41.config import build_attention_topology
    from atom.models.deepseek_v41.dspark import DeepseekV41DSpark, DraftAttention

    monkeypatch.setattr(linear, "native_group32_runs_bf16", lambda: bf16)
    spec = build_attention_topology(small_config)[1]
    attention = DraftAttention(small_config, spec)
    shard = attention.wkv_shard
    assert shard is not None
    assert shard.output_size == small_config.head_dim
    assert shard.native_a8_group_rows is not None
    rows = attention.wqkv_a.output_sizes[0]
    assert torch.equal(
        shard.weight.view(torch.uint8),
        attention.wqkv_a.weight.view(torch.uint8)[rows:],
    )
    group = attention.wqkv_a.weight_scale_row_group
    assert torch.equal(
        shard.weight_scale.view(torch.uint8),
        attention.wqkv_a.weight_scale.view(torch.uint8)[rows // group :],
    )
    assert shard.runs_bf16 is bf16
    draft = object.__new__(DeepseekV41DSpark)
    torch.nn.Module.__init__(draft)
    draft.mtp = [SimpleNamespace(attn=attention)]
    assert (draft._context_kv_fusion() is None) is bf16


@pytest.mark.parametrize("fp4ptpc", [False, True])
def test_q_leaves_its_norm_in_the_format_its_readers_take(
    single_rank, small_config, fp4ptpc
):
    """`wq_b` and `indexer.wq_b` share one spec and one Q: per-token FP8 for an
    FP4PTPC checkpoint's per-channel GEMMs; for native group32 ones MXFP8, or
    BF16 where those run in BF16."""
    from aiter import QuantType

    from atom.model_ops.layernorm import (
        DualRMSNormBF16,
        DualRMSNormMXFP8,
        DualRMSNormPerToken,
    )
    from atom.model_ops.linear import native_group32_runs_bf16
    from atom.models.deepseek_v41.attention import Attention
    from atom.models.deepseek_v41.config import build_attention_topology

    small_config.fp4ptpc = fp4ptpc
    spec = build_attention_topology(small_config)[1]
    attention = Attention(small_config, spec)
    native = DualRMSNormBF16 if native_group32_runs_bf16() else DualRMSNormMXFP8
    assert type(attention.qk_norm) is (DualRMSNormPerToken if fp4ptpc else native)
    readers = [attention.wq_b]
    if attention.indexer is not None:
        readers.append(attention.indexer.wq_b)
    for reader in readers:
        assert (reader.quant_type == QuantType.per_Token) == fp4ptpc


@pytest.mark.skipif(not torch.cuda.is_available(), reason="inverse RoPE is Triton")
def test_bf16_wo_a_projects_like_a_grouped_einsum(
    single_rank, small_config, monkeypatch
):
    """`_project_out` where `wo_a` runs in BF16 (no group32 FP8 GEMM): inverse
    RoPE, then each group's heads through its `wo_a` rows, written through the
    strided view into `wo_b`'s input; against the CPU rotation and an einsum."""
    from atom.model_ops import linear
    from atom.models.deepseek_v41.attention import Attention
    from atom.models.deepseek_v41.config import build_attention_topology

    monkeypatch.setattr(linear, "native_group32_runs_bf16", lambda: True)
    small_config.o_groups = 2
    attention = Attention(small_config, build_attention_topology(small_config)[1])
    assert attention.wo_a.runs_bf16 and attention.groups == 2
    groups, o_rank = attention.groups, attention.o_rank
    rows, heads, head_dim = 5, attention.heads, attention.head_dim
    torch.manual_seed(31)
    output = torch.randn(rows, heads, head_dim, dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 7, 19, 31])
    weight = torch.randn(groups * o_rank, heads * head_dim // groups).bfloat16()
    rope = RotaryEmbedding(64, 32, base=10000, original_length=0, factor=16)
    rotated = rope(output[None].clone(), positions, inverse=True)[0].float()
    expected = torch.einsum(
        "rgk,gok->rgo",
        rotated.view(rows, groups, -1),
        weight.float().view(groups, o_rank, -1),
    ).reshape(rows, -1)

    attention.wo_a.weight = torch.nn.Parameter(weight.cuda(), requires_grad=False)
    attention.wo_b = torch.nn.Identity()
    actual = attention._project_out(output.cuda(), rope.cuda(), positions.cuda())
    torch.testing.assert_close(
        actual.float().cpu(),
        expected,
        rtol=2**-6,
        atol=2**-6 * expected.abs().max().item(),
    )
