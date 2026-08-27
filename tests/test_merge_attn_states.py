# SPDX-License-Identifier: MIT
"""``merge_attn_states`` -- the LSE merge behind MLA chunked prefill.

``_forward_prefill_cached_chunked`` computes attention twice (causal over the new
tokens, non-causal over each cached chunk) and folds the halves together with
this kernel. It is the only place the two partial softmaxes ever meet, so a
defect here is invisible until output quality drops.

Two things make it worth its own test:

* **The empty-segment path is load-bearing, not an edge case.** Under global-axis
  chunking a short sequence can fall entirely outside a chunk, so its tokens see
  an empty prefix AND an empty suffix -- both ``lse = -inf``. The naive
  ``p_lse - max_lse`` is then ``-inf - (-inf) = NaN`` and the 0/0 scale poisons
  the row. The kernel's ``both_empty`` branch exists for exactly this, and it
  fires on ordinary traffic.
* **The grid maps ``BLOCK_H`` heads per program.** The head axis is masked
  because tensor parallelism need not leave a multiple of ``BLOCK_H``, and a
  wrong mask reads into the next token's rows -- which corrupts data without
  faulting.

The reference is a plain torch implementation in fp32, deliberately NOT a second
Triton kernel: comparing two implementations of the same idea passes happily when
both share a misconception.

Boundary values are the objects under test and are asserted directly, never
filtered out -- ``-inf`` that must stay ``-inf`` is precisely what a
``torch.isfinite`` mask would skip.
"""

from __future__ import annotations

import pytest
import torch

try:
    from atom.model_ops.attentions.triton_merge_attn_states import merge_attn_states

    _MERGE_ERR = None
except ImportError as _e:  # triton/aiter absent on a CPU-only runner
    _MERGE_ERR = str(_e)

needs_gpu = pytest.mark.skipif(
    _MERGE_ERR is not None or not torch.cuda.is_available(),
    reason=f"Triton kernel needs a GPU: {_MERGE_ERR}",
)

DEV = "cuda"
NEG_INF = -float("inf")
FP8 = torch.float8_e4m3fnuz


# ─────────────────────────────────────────────────────────────── reference ──


def _ref_merge(p_out, p_lse, s_out, s_lse, tokens_with_ctx, scale=None):
    """fp32 reference. out is [T, H, D], lse is [H, T]."""
    num_tokens = p_out.shape[0]
    p_lse = p_lse.float().clone()
    s_lse = s_lse.float().clone()
    # FA2 returns +inf where FA3 returns -inf; the kernel normalises to -inf.
    p_lse[p_lse == float("inf")] = NEG_INF
    s_lse[s_lse == float("inf")] = NEG_INF

    out = torch.zeros_like(p_out, dtype=torch.float32)
    lse = torch.zeros_like(p_lse)

    for t in range(num_tokens):
        if t >= tokens_with_ctx:
            out[t] = s_out[t].float()
            lse[:, t] = s_lse[:, t]
            continue
        pl, sl = p_lse[:, t], s_lse[:, t]
        mx = torch.maximum(pl, sl)
        empty = torch.isinf(mx) & (mx < 0)
        safe = torch.where(empty, torch.zeros_like(mx), mx)
        pe, se = torch.exp(pl - safe), torch.exp(sl - safe)
        tot = pe + se
        lse[:, t] = torch.where(
            empty, torch.full_like(mx, NEG_INF), torch.log(tot) + safe
        )
        denom = torch.where(empty, torch.ones_like(tot), tot)
        out[t] = (
            p_out[t].float() * (pe / denom)[:, None]
            + s_out[t].float() * (se / denom)[:, None]
        )

    if scale is not None:
        finfo = torch.finfo(FP8)
        out = (out / scale).clamp(finfo.min, finfo.max)
    return out, lse


def _assert_matches(got_t, ref, p_out, s_out, fp8):
    """Compare against the UNROUNDED fp32 reference, in the output dtype's terms.

    Measuring from the rounded reference instead would fail on exact ties: when
    ``ref`` lands midway between two representable values both neighbours are
    equally correct, and round-to-even (torch) legitimately disagrees with
    round-away (HIP). Observed once per 8.4M elements in the fp8 case.

    Two slack terms on top of what the ideal rounding already costs:
      * 1 ulp of the output dtype -- fp32 op order shifts the pre-rounding value
      * fp32 cancellation -- ``out = p*ps + s*ss`` with ``ps + ss == 1``, so
        where the terms nearly cancel the fp32 REFERENCE is itself uncertain by
        ``~2**-23 * (|p| + |s|)``, independent of ``|out|``
    A criterion missing the first flags large rows that are right to the last
    bit; missing the second, it flags cancellation rows where nothing is wrong.
    """
    got = got_t.float()
    eps = 2**-3 if fp8 else 2**-7  # mantissa bits: fp8 e4m3 -> 3, bf16 -> 7
    ref_q = ref.to(got_t.dtype).float()
    best = (ref_q - ref).abs()  # unavoidable rounding of the ideal value
    slack = ref.abs() * eps + (p_out.abs() + s_out.abs()).float() * 2**-22
    # `err > bound` is False for NaN, so NaN is asserted separately rather than
    # being silently waved through.
    assert not torch.isnan(got).any(), "output contains NaN"
    n_bad = int(((got - ref).abs() > best + slack).sum())
    assert n_bad == 0, f"{n_bad}/{got.numel()} elements beyond 1 ulp + fp32 eps"


def _inputs(num_tokens, num_heads, head_size, seed=0):
    torch.manual_seed(seed)
    mk = lambda: torch.randn(
        num_tokens, num_heads, head_size, device=DEV, dtype=torch.bfloat16
    )
    lse = lambda: torch.randn(num_heads, num_tokens, device=DEV, dtype=torch.float32)
    return mk(), lse(), mk(), lse()


def _run(p_out, p_lse, s_out, s_lse, tokens_with_ctx=None, want_lse=False, scale=None):
    num_tokens, num_heads, head_size = p_out.shape
    tokens_with_ctx = num_tokens if tokens_with_ctx is None else tokens_with_ctx
    out = torch.empty(
        num_tokens,
        num_heads,
        head_size,
        device=DEV,
        dtype=FP8 if scale is not None else torch.bfloat16,
    )
    out_lse = (
        torch.empty(num_heads, num_tokens, device=DEV, dtype=torch.float32)
        if want_lse
        else None
    )
    merge_attn_states(out, p_out, p_lse, s_out, s_lse, out_lse, tokens_with_ctx, scale)
    return out, out_lse


# ──────────────────────────────────────────────────────────────── the merge ──


@needs_gpu
def test_merge_matches_fp32_reference():
    """MLA production shape: 128 heads x 128 head_size."""
    p_out, p_lse, s_out, s_lse = _inputs(512, 128, 128)
    out, _ = _run(p_out, p_lse, s_out, s_lse)
    ref, _ = _ref_merge(p_out, p_lse, s_out, s_lse, 512)
    _assert_matches(out, ref, p_out, s_out, fp8=False)


@needs_gpu
def test_output_lse_matches():
    p_out, p_lse, s_out, s_lse = _inputs(256, 128, 128)
    out, out_lse = _run(p_out, p_lse, s_out, s_lse, want_lse=True)
    ref, ref_lse = _ref_merge(p_out, p_lse, s_out, s_lse, 256)
    _assert_matches(out, ref, p_out, s_out, fp8=False)
    torch.testing.assert_close(out_lse, ref_lse, rtol=0, atol=1e-4)


@needs_gpu
def test_fp8_output_scale():
    p_out, p_lse, s_out, s_lse = _inputs(256, 128, 128)
    scale = torch.tensor([0.5], device=DEV, dtype=torch.float32)
    out, _ = _run(p_out, p_lse, s_out, s_lse, scale=scale)
    ref, _ = _ref_merge(p_out, p_lse, s_out, s_lse, 256, scale)
    _assert_matches(out, ref, p_out, s_out, fp8=True)


@needs_gpu
def test_tokens_without_context_copy_suffix():
    """Tokens at or past ``prefill_tokens_with_context`` take the suffix verbatim.

    Those rows have no prefix at all, so merging would divide by an undefined
    prefix lse; the kernel takes a separate branch for them.
    """
    p_out, p_lse, s_out, s_lse = _inputs(256, 32, 128)
    out, out_lse = _run(p_out, p_lse, s_out, s_lse, tokens_with_ctx=100, want_lse=True)
    ref, ref_lse = _ref_merge(p_out, p_lse, s_out, s_lse, 100)
    _assert_matches(out, ref, p_out, s_out, fp8=False)
    torch.testing.assert_close(out_lse, ref_lse, rtol=0, atol=1e-4)
    # The branch really is a copy, not a merge that happens to agree.
    torch.testing.assert_close(out[100:].float(), s_out[100:].float())


# ───────────────────────────────────────────────────────── boundary values ──


@needs_gpu
@pytest.mark.parametrize("side", ["prefix", "suffix"])
def test_one_empty_segment(side):
    """An empty segment contributes nothing; the other side passes through.

    Its ``lse`` is ``-inf`` and its output rows are zero, which is what aiter
    returns for a zero-length kv segment.
    """
    p_out, p_lse, s_out, s_lse = _inputs(64, 32, 128)
    empty = [0, 7, 63]
    if side == "prefix":
        p_lse[:, empty] = NEG_INF
        p_out[empty] = 0
    else:
        s_lse[:, empty] = NEG_INF
        s_out[empty] = 0

    out, out_lse = _run(p_out, p_lse, s_out, s_lse, want_lse=True)
    ref, ref_lse = _ref_merge(p_out, p_lse, s_out, s_lse, 64)
    _assert_matches(out, ref, p_out, s_out, fp8=False)
    torch.testing.assert_close(out_lse, ref_lse, rtol=0, atol=1e-4)

    kept = s_out if side == "prefix" else p_out
    torch.testing.assert_close(out[empty].float(), kept[empty].float())


@needs_gpu
def test_both_segments_empty_gives_zero_not_nan():
    """The case global-axis chunking creates: a sequence outside the chunk.

    ``max_lse`` is ``-inf`` on both sides. Without the ``both_empty`` guard the
    kernel computes ``-inf - (-inf) = NaN`` and a 0/0 scale, and the NaN then
    spreads through every later merge. Output must be exactly 0 and the merged
    lse must stay ``-inf`` so a downstream merge still sees "nothing here".
    """
    p_out, p_lse, s_out, s_lse = _inputs(64, 32, 128)
    empty = [3, 9, 40]
    for lse, out_t in ((p_lse, p_out), (s_lse, s_out)):
        lse[:, empty] = NEG_INF
        out_t[empty] = 0

    out, out_lse = _run(p_out, p_lse, s_out, s_lse, want_lse=True)

    assert not torch.isnan(out.float()).any()
    assert not torch.isnan(out_lse).any()
    assert (out[empty].float() == 0).all()
    # -inf must SURVIVE as -inf. Comparing it numerically, or masking
    # non-finite values away first, would make this assertion vacuous -- and
    # a finite value here is exactly the defect it guards against.
    assert torch.isinf(out_lse[:, empty]).all()
    assert (out_lse[:, empty] < 0).all()
    # Rows that were not emptied are untouched by the guard.
    ref, ref_lse = _ref_merge(p_out, p_lse, s_out, s_lse, 64)
    _assert_matches(out, ref, p_out, s_out, fp8=False)
    torch.testing.assert_close(out_lse, ref_lse, rtol=0, atol=1e-4)


@needs_gpu
def test_positive_inf_lse_treated_as_empty():
    """FA2 reports an empty segment as ``+inf`` where FA3 reports ``-inf``.

    The kernel normalises ``+inf`` to ``-inf``; without that the exponent
    overflows and the row is lost.
    """
    p_out, p_lse, s_out, s_lse = _inputs(64, 32, 128)
    plus = [2, 11]
    p_lse[:, plus] = float("inf")

    out, out_lse = _run(p_out, p_lse, s_out, s_lse, want_lse=True)
    ref, ref_lse = _ref_merge(p_out, p_lse, s_out, s_lse, 64)

    assert not torch.isnan(out.float()).any()
    _assert_matches(out, ref, p_out, s_out, fp8=False)
    torch.testing.assert_close(out_lse, ref_lse, rtol=0, atol=1e-4)
    # +inf prefix means the suffix wins outright.
    torch.testing.assert_close(out[plus].float(), s_out[plus].float())


# ──────────────────────────────────────────────────────────── grid mapping ──


@needs_gpu
@pytest.mark.parametrize("num_heads", [1, 3, 5, 7, 8, 12, 16, 17, 64, 128])
def test_head_count_not_a_multiple_of_block_h(num_heads):
    """The head axis is blocked, so counts that do not divide it must still work.

    Tensor parallelism divides the head count, and a missing mask in the tail
    program reads into the next token's rows -- silently wrong, never a fault.
    """
    p_out, p_lse, s_out, s_lse = _inputs(48, num_heads, 128)
    out, out_lse = _run(p_out, p_lse, s_out, s_lse, want_lse=True)
    ref, ref_lse = _ref_merge(p_out, p_lse, s_out, s_lse, 48)
    _assert_matches(out, ref, p_out, s_out, fp8=False)
    torch.testing.assert_close(out_lse, ref_lse, rtol=0, atol=1e-4)


@needs_gpu
@pytest.mark.parametrize("head_size", [96, 128, 192])
def test_head_size_padding(head_size):
    """Non-power-of-two head sizes exercise the PADDED_HEAD_SIZE mask."""
    p_out, p_lse, s_out, s_lse = _inputs(48, 16, head_size)
    out, _ = _run(p_out, p_lse, s_out, s_lse)
    ref, _ = _ref_merge(p_out, p_lse, s_out, s_lse, 48)
    _assert_matches(out, ref, p_out, s_out, fp8=False)


@needs_gpu
def test_padded_output_stride():
    """``output`` may carry a wider head stride than the attention outputs.

    The wrapper reads the two strides separately (the backend can pad its own
    buffers); if the kernel used one for both, the writes would land skewed.
    """
    num_tokens, num_heads, head_size = 32, 16, 128
    p_out, p_lse, s_out, s_lse = _inputs(num_tokens, num_heads, head_size)
    padded = torch.zeros(
        num_tokens, num_heads, head_size + 16, device=DEV, dtype=torch.bfloat16
    )
    out = padded[:, :, :head_size]
    assert out.stride(1) != p_out.stride(1)

    merge_attn_states(out, p_out, p_lse, s_out, s_lse, None, num_tokens, None)
    ref, _ = _ref_merge(p_out, p_lse, s_out, s_lse, num_tokens)
    _assert_matches(out, ref, p_out, s_out, fp8=False)
    # Padding lanes stay untouched.
    assert (padded[:, :, head_size:] == 0).all()
