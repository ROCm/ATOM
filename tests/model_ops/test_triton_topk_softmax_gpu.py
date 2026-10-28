# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Triton routing kernel must reproduce AITER's `topk_softmax` exactly.

`rocm_aiter_topk_softmax_impl` swaps one for the other inside captured graphs
and keeps AITER for eager calls, so the same request routes through both
depending on the batch it lands in. Any difference -- an expert id, the order
of tied experts, or the last bit of a weight -- would make outputs depend on
batching, hence bit-exactness rather than a tolerance.
"""

import inspect

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("compares two GPU kernels; needs a real GPU", allow_module_level=True)

aiter = pytest.importorskip("aiter")

from atom.model_ops.topk_softmax_dispatch import triton_topk_softmax_supported
from atom.model_ops.triton_topk_softmax import triton_topk_softmax

EXPERTS, TOPK = 512, 10


def _logits(rows, width, scale, ties):
    x = torch.randn(rows, width, device="cuda") * scale
    if ties:
        # Coarse values: many equal logits, including +0.0 and -0.0.
        x = (x * 4).round() / 4
    return x.to(torch.bfloat16)


def _supported(logits, k, renormalize, shared):
    """`triton_topk_softmax_supported` for fresh outputs in the fused-shared
    layout: [rows, k + shared] buffers, the routed ids through a [:, :k] view."""
    rows = logits.shape[0]
    w = torch.empty(rows, k + shared, device="cuda")
    ids = torch.empty(rows, k + shared, dtype=torch.int32, device="cuda")
    return triton_topk_softmax_supported(w, ids[:, :k], logits, renormalize, shared)


@pytest.mark.parametrize("shared", [0, 1])
@pytest.mark.parametrize("rows", [1, 4, 33, 1024])
@pytest.mark.parametrize("scale,ties", [(2.0, False), (0.05, True), (8.0, False)])
def test_matches_aiter_bit_for_bit(shared, rows, scale, ties):
    torch.manual_seed(rows * 7 + shared)
    logits = _logits(rows, EXPERTS + shared, scale, ties)
    assert _supported(logits, TOPK, True, shared)

    # The fused-shared layout: one [rows, k + shared] buffer per output, the
    # kernels write the routed columns through a [:, :k] view.
    w_ref = torch.full((rows, TOPK + shared), -1.0, device="cuda")
    ids_ref = torch.full(
        (rows, TOPK + shared), EXPERTS, dtype=torch.int32, device="cuda"
    )
    w_new, ids_new = w_ref.clone(), ids_ref.clone()
    scratch = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")

    aiter.topk_softmax(
        w_ref, ids_ref[:, :TOPK], scratch, logits, True, shared,
        "sigmoid" if shared else "",
    )  # fmt: skip
    triton_topk_softmax(w_new, ids_new[:, :TOPK], logits, shared)

    assert torch.equal(ids_new, ids_ref)
    assert torch.equal(w_new, w_ref)


def _aiter_and_triton(logits, k, shared):
    """Both kernels on the fused-shared layout, from the same poisoned
    buffers: ((w_aiter, ids_aiter), (w_triton, ids_triton))."""
    rows = logits.shape[0]
    w_ref = torch.full((rows, k + shared), -1.0, device="cuda")
    ids_ref = torch.full((rows, k + shared), 12345, dtype=torch.int32, device="cuda")
    w_new, ids_new = w_ref.clone(), ids_ref.clone()
    scratch = torch.empty(rows, k, dtype=torch.int32, device="cuda")
    aiter.topk_softmax(
        w_ref, ids_ref[:, :k], scratch, logits, True, shared,
        "sigmoid" if shared else "",
    )  # fmt: skip
    triton_topk_softmax(w_new, ids_new[:, :k], logits, shared)
    return (w_ref, ids_ref), (w_new, ids_new)


# Every shape `triton_topk_softmax_supported` accepts, so a shape where AITER
# takes another code path (as 1024 experts does) cannot be accepted unseen.
SUPPORTED_EXPERTS = (32, 64, 128, 256, 512)
SUPPORTED_TOPK = (1, 8, 10, 32)
SUPPORTED_SHARED = (0, 1, 2, 4, 8)


@pytest.mark.parametrize("shared", SUPPORTED_SHARED)
@pytest.mark.parametrize("k", SUPPORTED_TOPK)
@pytest.mark.parametrize("experts", SUPPORTED_EXPERTS)
def test_every_supported_shape_matches_aiter(experts, k, shared):
    torch.manual_seed(experts + 3 * k + shared)
    width = experts + shared
    logits = torch.cat(
        [
            _logits(24, width, 2.0, False),
            _logits(24, width, 0.05, True),
            _logits(16, width, 8.0, False),
        ]
    )
    assert _supported(logits, k, True, shared)
    (w_ref, ids_ref), (w_new, ids_new) = _aiter_and_triton(logits, k, shared)
    assert torch.equal(ids_new, ids_ref)
    assert torch.equal(_bits(w_new), _bits(w_ref))


@pytest.mark.parametrize("shared", [0, 1])
@pytest.mark.parametrize("k", [10, 32])
@pytest.mark.parametrize("experts", [32, 128, 512])
def test_rows_short_of_finite_logits_match_aiter(experts, k, shared):
    """Row r keeps 1 + r % (k + 1) finite logits (at random places), the rest
    are -inf. Once the finite ones run out AITER picks expert 0 again, with
    weight 0."""
    rows = 64
    torch.manual_seed(experts + k + shared)
    logits = _logits(rows, experts + shared, 2.0, False)
    finite = 1 + torch.arange(rows, device="cuda") % (k + 1)
    rank = torch.rand(rows, experts, device="cuda").argsort(1).argsort(1)
    logits[:, :experts][rank >= finite[:, None]] = float("-inf")
    # As in review: only experts 0..4 finite.
    logits[0, :experts] = _logits(1, experts, 2.0, False)[0]
    logits[0, 5:experts] = float("-inf")
    # No finite logit at all: every weight is NaN in both.
    logits[1, :experts] = float("-inf")

    (w_ref, ids_ref), (w_new, ids_new) = _aiter_and_triton(logits, k, shared)
    assert torch.equal(ids_new, ids_ref)
    finite_rows = torch.arange(rows, device="cuda") != 1
    assert torch.equal(_bits(w_new[finite_rows]), _bits(w_ref[finite_rows]))
    assert bool((ids_new[0, 5:k] == 0).all())
    assert bool((_bits(w_new[0, 5:k]) == 0).all())
    assert bool(w_new[1, :k].isnan().all()) and bool(w_ref[1, :k].isnan().all())
    assert torch.equal(_bits(w_new[1, k:]), _bits(w_ref[1, k:]))


@pytest.mark.parametrize("shared", [0, 1])
@pytest.mark.parametrize("experts", [32, 128, 512])
def test_nan_logits_are_never_picked(experts, shared):
    """NaNs off the first slot of AITER's per-thread vector loads (16 bf16
    below 512 experts, 32 at 512), which its `>` never picks either."""
    rows = 64
    torch.manual_seed(experts + 17 * shared)
    logits = _logits(rows, experts + shared, 2.0, False)
    nan = torch.rand(rows, experts, device="cuda") < 0.3
    nan[:, ::16] = False
    nan[1] = True  # all NaN but the vector-load first slots
    nan[1, ::16] = False
    logits[:, :experts][nan] = float("nan")

    (w_ref, ids_ref), (w_new, ids_new) = _aiter_and_triton(logits, TOPK, shared)
    assert torch.equal(ids_new, ids_ref)
    assert torch.equal(_bits(w_new), _bits(w_ref))
    assert not bool(nan.gather(1, ids_new[:, :TOPK].long()).any())


def test_unsupported_shapes_fall_back():
    logits = torch.zeros(4, 513, dtype=torch.bfloat16, device="cuda")
    assert not _supported(logits, TOPK, False, 1)
    assert not _supported(logits, TOPK, True, 0)  # 513 experts
    assert not _supported(logits[:, ::2], TOPK, True, 0)
    # AITER's fused kernel stops at 512 experts; 1024 takes its
    # softmax-then-top-k path, which rounds the weights differently.
    wide = torch.zeros(4, 1024 + 1, dtype=torch.bfloat16, device="cuda")
    assert not _supported(wide, TOPK, True, 1)
    assert not _supported(wide[:, :1024], TOPK, True, 0)
    assert not _supported(logits[:, :16], 1, True, 0)
    assert not _supported(logits[:, :512], 33, True, 0)
    assert not _supported(wide[:, :515], TOPK, True, 3)


def test_outputs_need_a_unit_last_stride():
    """AITER honours any output stride; the Triton kernel stores along a unit
    one, so it does not take other layouts and its launcher refuses them."""
    rows, shared = 8, 1
    logits = _logits(rows, EXPERTS + shared, 2.0, False)
    w = torch.empty(rows, TOPK + shared, device="cuda")
    ids = torch.empty(rows, TOPK + shared, dtype=torch.int32, device="cuda")
    w_t = torch.empty(TOPK + shared, rows, device="cuda").t()
    ids_t = torch.empty(TOPK + shared, rows, dtype=torch.int32, device="cuda").t()
    assert triton_topk_softmax_supported(w, ids[:, :TOPK], logits, True, shared)
    for out_w, out_ids in ((w_t, ids[:, :TOPK]), (w, ids_t[:, :TOPK])):
        assert not triton_topk_softmax_supported(out_w, out_ids, logits, True, shared)
        with pytest.raises(AssertionError, match="unit last stride"):
            triton_topk_softmax(out_w, out_ids, logits, shared)


@pytest.mark.parametrize("shared", [0, 1])
def test_1024_experts_route_through_aiter(shared, monkeypatch):
    """Past AITER's fused kernel, eager and captured passes both run AITER."""
    from atom.model_ops import topK

    def no_triton(*args, **kwargs):
        raise AssertionError("1024 experts must not take the Triton kernel")

    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "1")
    monkeypatch.setattr(topK, "triton_topk_softmax", no_triton)
    rows, experts, k = 64, 1024, TOPK
    torch.manual_seed(5 + shared)
    logits = _logits(rows, experts + shared, 2.0, False)
    if shared:
        meta_w, meta_ids = _outputs(rows, shared)
        monkeypatch.setattr(topK, "aiter_topK_meta_data", (meta_w, meta_ids))

    def route():
        return topK.rocm_aiter_topk_softmax_impl(
            logits, k, True, shared, experts, "sigmoid" if shared else None
        )

    w_eager, ids_eager = (t.clone() for t in route())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w_graph, ids_graph = route()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(ids_graph[:, :k], ids_eager[:, :k])
    assert torch.equal(_bits(w_graph), _bits(w_eager))


# num_valid_rows (ROCm/aiter#6916): rows at or past the device count are
# padding, routed to id -1 with weight 0; shared columns are written as usual.
AITER_HAS_NUM_VALID_ROWS = (
    "num_valid_rows" in inspect.signature(aiter.topk_softmax).parameters
)
needs_aiter_num_valid_rows = pytest.mark.skipif(
    not AITER_HAS_NUM_VALID_ROWS, reason="AITER predates ROCm/aiter#6916"
)


def _outputs(rows, shared):
    w = torch.full((rows, TOPK + shared), float("nan"), device="cuda")
    ids = torch.full((rows, TOPK + shared), EXPERTS, dtype=torch.int32, device="cuda")
    return w, ids


def _bits(w):
    # Compares -0.0 and +0.0 (and NaNs) as different.
    return w.view(torch.int32)


def _check_pad_rows(w, ids, w_full, ids_full, n):
    """Rows below `n` match the unmasked launch bit for bit; rows at or past
    it carry id -1 and weight +0.0; shared columns match on every row."""
    assert torch.equal(ids[:n], ids_full[:n])
    assert torch.equal(_bits(w[:n]), _bits(w_full[:n]))
    assert bool((ids[n:, :TOPK] == -1).all())
    assert bool((_bits(w[n:, :TOPK]) == 0).all())
    assert torch.equal(ids[:, TOPK:], ids_full[:, TOPK:])
    assert torch.equal(_bits(w[:, TOPK:]), _bits(w_full[:, TOPK:]))


@pytest.mark.parametrize("shared", [0, 1])
@pytest.mark.parametrize("rows", [1, 33, 256])
@pytest.mark.parametrize("count", [None, "zero", "some", "all", "over"])
def test_num_valid_rows_pads_rows(shared, rows, count):
    torch.manual_seed(rows * 5 + shared)
    logits = _logits(rows, EXPERTS + shared, 2.0, False)
    n = {None: rows, "zero": 0, "some": rows // 3, "all": rows, "over": rows + 7}
    n = n[count]
    num_valid_rows = (
        None
        if count is None
        else torch.tensor([n, 12345], dtype=torch.int32, device="cuda")
    )

    w_full, ids_full = _outputs(rows, shared)
    triton_topk_softmax(w_full, ids_full[:, :TOPK], logits, shared)
    w, ids = _outputs(rows, shared)
    triton_topk_softmax(w, ids[:, :TOPK], logits, shared, num_valid_rows)
    _check_pad_rows(w, ids, w_full, ids_full, n)

    if AITER_HAS_NUM_VALID_ROWS:
        w_ref, ids_ref = _outputs(rows, shared)
        scratch = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
        aiter.topk_softmax(
            w_ref, ids_ref[:, :TOPK], scratch, logits, True, shared,
            "sigmoid" if shared else "", num_valid_rows=num_valid_rows,
        )  # fmt: skip
        assert torch.equal(ids, ids_ref)
        assert torch.equal(_bits(w), _bits(w_ref))


@pytest.mark.parametrize("shared", [0, 1])
def test_num_valid_rows_follows_graph_replay(shared):
    rows = 64
    torch.manual_seed(11 + shared)
    logits = _logits(rows, EXPERTS + shared, 2.0, False)
    w_full, ids_full = _outputs(rows, shared)
    triton_topk_softmax(w_full, ids_full[:, :TOPK], logits, shared)

    num_valid_rows = torch.tensor([rows], dtype=torch.int32, device="cuda")
    w, ids = _outputs(rows, shared)
    triton_topk_softmax(w, ids[:, :TOPK], logits, shared, num_valid_rows)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        triton_topk_softmax(w, ids[:, :TOPK], logits, shared, num_valid_rows)

    for n in (rows, 0, 17, rows // 2, 1, rows):
        w.fill_(float("nan"))
        ids[:, :TOPK] = 7
        num_valid_rows.fill_(n)
        graph.replay()
        torch.cuda.synchronize()
        _check_pad_rows(w, ids, w_full, ids_full, n)


@needs_aiter_num_valid_rows
@pytest.mark.parametrize("shared", [0, 1])
def test_impl_passes_num_valid_rows_to_both_paths(shared, monkeypatch):
    """Eager calls take AITER, captured calls take Triton; both must get the
    count and agree bit for bit on every replay."""
    from atom.model_ops import topK

    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "1")
    rows, max_rows = 48, 64
    torch.manual_seed(23 + shared)
    logits = _logits(rows, EXPERTS + shared, 2.0, False)
    if shared:
        meta_w, meta_ids = _outputs(max_rows, shared)
        monkeypatch.setattr(topK, "aiter_topK_meta_data", (meta_w, meta_ids))
    num_valid_rows = torch.tensor([rows], dtype=torch.int32, device="cuda")

    def route():
        w, ids = topK.rocm_aiter_topk_softmax_impl(
            logits, TOPK, True, shared, EXPERTS,
            "sigmoid" if shared else None, num_valid_rows=num_valid_rows,
        )  # fmt: skip
        return w[:rows], ids[:rows]

    route()  # compile the Triton kernel outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w_graph, ids_graph = route()

    for n in (rows, 0, 29):
        num_valid_rows.fill_(n)
        w_eager, ids_eager = (t.clone() for t in route())
        graph.replay()
        torch.cuda.synchronize()
        assert bool((ids_eager[n:, :TOPK] == -1).all())
        assert torch.equal(ids_graph, ids_eager)
        assert torch.equal(_bits(w_graph), _bits(w_eager))


@needs_aiter_num_valid_rows
@pytest.mark.parametrize("shared,offset", [(0, 0), (1, 0), (1, 32)])
def test_select_experts_counts_the_pass_rows(shared, offset, monkeypatch):
    """Through `FusedMoE.select_experts`, which takes the pass's real row count
    from the forward context (`scheduled_tokens_gpu`): the captured Triton
    route follows the count on every replay, bit for bit with the eager AITER
    one, and with a fused shared expert writes the rows `topk_meta_rows` gives
    this (TBO micro-)batch at its offset."""
    from atom.model_ops import topK
    from atom.model_ops.moe import FusedMoE
    from atom.utils import forward_context as fc

    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "1")
    rows, max_rows = 48, 128
    torch.manual_seed(31 + shared + offset)
    logits = _logits(rows, EXPERTS + shared, 2.0, False)
    count = torch.tensor([rows], dtype=torch.int32, device="cuda")
    context = fc.Context(
        positions=torch.zeros(rows, dtype=torch.int64, device="cuda"),
        running_tokens=rows,
        ubatch_token_offset=offset,
        scheduled_tokens_gpu=count,
    )
    monkeypatch.setattr(fc, "_forward_context", fc.ForwardContext(context=context))
    if shared:
        meta_w, meta_ids = _outputs(max_rows, shared)
        monkeypatch.setattr(topK, "aiter_topK_meta_data", (meta_w, meta_ids))

    def route():
        return FusedMoE.select_experts(
            hidden_states=None,
            router_logits=logits,
            top_k=TOPK,
            use_grouped_topk=False,
            renormalize=True,
            num_routing_experts=EXPERTS,
            num_fused_shared_experts=shared,
            fused_shared_experts_scoring_func="sigmoid" if shared else None,
        )

    route()  # compile the Triton kernel outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w_graph, ids_graph = route()
    if shared:
        assert w_graph.data_ptr() == meta_w[offset].data_ptr()
        assert ids_graph.data_ptr() == meta_ids[offset].data_ptr()

    for n in (rows, 0, 29, rows):
        count.fill_(n)
        w_eager, ids_eager = (t.clone() for t in route())
        graph.replay()
        torch.cuda.synchronize()
        assert bool((ids_eager[n:, :TOPK] == -1).all())
        assert bool((_bits(w_eager[n:, :TOPK]) == 0).all())
        assert bool((ids_eager[:n, :TOPK] >= 0).all())
        assert torch.equal(ids_graph, ids_eager)
        assert torch.equal(_bits(w_graph), _bits(w_eager))
    if shared:
        # Rows outside this batch's slice are never written.
        outside = torch.ones(max_rows, dtype=torch.bool, device="cuda")
        outside[offset : offset + rows] = False
        assert bool(meta_w[outside].isnan().all())
        assert bool((meta_ids[outside] == EXPERTS).all())
