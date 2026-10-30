# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Triton routing kernel must reproduce AITER's `topk_softmax` exactly.

`rocm_aiter_topk_softmax_impl` runs it, eager and captured alike, for the
shapes it supports up to `triton_topk_softmax_max_rows()` rows, and AITER
for longer launches, the other shapes or with `ATOM_TRITON_TOPK_SOFTMAX=0`
-- as the layer decided when it was built (`triton_topk_softmax_rows`).
Any difference -- an expert id, the order of tied experts, or the last bit
of a weight -- would route a token differently in a long prefill than in a
short pass, or make that switch change outputs, hence bit-exactness rather
than a tolerance.
"""

import functools
import inspect

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("compares two GPU kernels; needs a real GPU", allow_module_level=True)

aiter = pytest.importorskip("aiter")

from atom.model_ops import topk_softmax_dispatch
from atom.model_ops.topk_softmax_dispatch import (
    triton_topk_softmax_max_rows,
    triton_topk_softmax_rows,
    triton_topk_softmax_supported,
)
from atom.model_ops.triton_topk_softmax import triton_topk_softmax
from atom.utils import envs

# The kernel comparisons run on any GPU; the dispatch tests need this GPU to
# have a measured Triton cutoff.
needs_cutoff = pytest.mark.skipif(
    triton_topk_softmax_max_rows() is None,
    reason="no measured Triton top-k softmax cutoff on this GPU architecture",
)

EXPERTS, TOPK = 512, 10


def _logits(rows, width, scale, ties):
    x = torch.randn(rows, width, device="cuda") * scale
    if ties:
        # Coarse values: many equal logits, including +0.0 and -0.0.
        x = (x * 4).round() / 4
    return x.to(torch.bfloat16)


def _supported(logits, k, renormalize, shared):
    """`triton_topk_softmax_supported` for a routing of these logits, the
    last `shared` of them sigmoid-scored shared experts."""
    experts = logits.shape[1] - shared
    return triton_topk_softmax_supported(
        experts, k, renormalize, shared, "sigmoid" if shared else ""
    )


def _layer_rows(experts, k, shared):
    """What a layer routing `experts` / top-`k` with `shared` sigmoid-scored
    fused shared experts decides when it is built."""
    return triton_topk_softmax_rows(
        experts, k, True, shared, "sigmoid" if shared else None
    )


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
    # AITER's fused kernel stops at 512 experts; 1024 takes its
    # softmax-then-top-k path, which rounds the weights differently.
    wide = torch.zeros(4, 1024 + 1, dtype=torch.bfloat16, device="cuda")
    assert not _supported(wide, TOPK, True, 1)
    assert not _supported(wide[:, :1024], TOPK, True, 0)
    assert not _supported(logits[:, :16], 1, True, 0)
    assert not _supported(logits[:, :512], 33, True, 0)
    assert not _supported(wide[:, :515], TOPK, True, 3)


def test_launcher_needs_a_unit_last_stride():
    """The Triton kernel loads and stores along a unit last stride. The impl's
    outputs always have one (fresh buffers, or column views of the shared
    `aiter_topK_meta_data` rows), as do router logits (a gate's output, or a
    column slice of it); the launcher refuses anything else."""
    rows, shared = 8, 1
    logits = _logits(rows, EXPERTS + shared, 2.0, False)
    w = torch.empty(rows, TOPK + shared, device="cuda")
    ids = torch.empty(rows, TOPK + shared, dtype=torch.int32, device="cuda")
    w_t = torch.empty(TOPK + shared, rows, device="cuda").t()
    ids_t = torch.empty(TOPK + shared, rows, dtype=torch.int32, device="cuda").t()
    wide = _logits(rows, 2 * (EXPERTS + shared), 2.0, False)
    for out_w, out_ids, gating in (
        (w_t, ids[:, :TOPK], logits),
        (w, ids_t[:, :TOPK], logits),
        (w, ids[:, :TOPK], wide[:, ::2]),
    ):
        with pytest.raises(AssertionError, match="unit last stride"):
            triton_topk_softmax(out_w, out_ids, gating, shared)


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
    max_rows = _layer_rows(experts, k, shared)
    assert max_rows is None

    def route():
        return topK.rocm_aiter_topk_softmax_impl(
            logits, k, True, shared, experts, "sigmoid" if shared else None,
            triton_max_rows=max_rows,
        )  # fmt: skip

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


def _record_kernels(monkeypatch, topK):
    """The kernel each later `rocm_aiter_topk_softmax_impl` call launches."""
    used = []
    triton_fn, aiter_fn = topK.triton_topk_softmax, aiter.topk_softmax

    @functools.wraps(triton_fn)
    def via_triton(*args, **kwargs):
        used.append("triton")
        return triton_fn(*args, **kwargs)

    @functools.wraps(aiter_fn)
    def via_aiter(*args, **kwargs):
        used.append("aiter")
        return aiter_fn(*args, **kwargs)

    monkeypatch.setattr(topK, "triton_topk_softmax", via_triton)
    monkeypatch.setattr(aiter, "topk_softmax", via_aiter)
    return used


# With the switch on, eager and captured passes both run Triton; off, both
# run AITER. Either way both get the count.
SWITCH = pytest.mark.parametrize("switch,kernel", [("1", "triton"), ("0", "aiter")])


@needs_cutoff
@needs_aiter_num_valid_rows
@SWITCH
@pytest.mark.parametrize("shared", [0, 1])
def test_impl_passes_num_valid_rows_to_both_paths(shared, switch, kernel, monkeypatch):
    """Eager and captured calls run the same kernel, both get the count and
    agree bit for bit on every replay."""
    from atom.model_ops import topK

    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", switch)
    used = _record_kernels(monkeypatch, topK)
    rows, max_rows = 48, 64
    torch.manual_seed(23 + shared)
    logits = _logits(rows, EXPERTS + shared, 2.0, False)
    if shared:
        meta_w, meta_ids = _outputs(max_rows, shared)
        monkeypatch.setattr(topK, "aiter_topK_meta_data", (meta_w, meta_ids))
    num_valid_rows = torch.tensor([rows], dtype=torch.int32, device="cuda")
    max_rows = _layer_rows(EXPERTS, TOPK, shared)

    def route():
        w, ids = topK.rocm_aiter_topk_softmax_impl(
            logits, TOPK, True, shared, EXPERTS,
            "sigmoid" if shared else None, num_valid_rows=num_valid_rows,
            triton_max_rows=max_rows,
        )  # fmt: skip
        return w[:rows], ids[:rows]

    route()  # eager: compiles the Triton kernel, when on, outside the capture
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
    assert used == [kernel] * (2 + 3)


@needs_cutoff
@needs_aiter_num_valid_rows
@SWITCH
@pytest.mark.parametrize("shared,offset", [(0, 0), (1, 0), (1, 32)])
def test_select_experts_counts_the_pass_rows(
    shared, offset, switch, kernel, monkeypatch
):
    """Through `FusedMoE.select_experts`, which takes the pass's real row count
    from the forward context (`scheduled_tokens_gpu`): the captured route
    follows the count on every replay, bit for bit with the eager one (same
    kernel), and with a fused shared expert writes the rows `topk_meta_rows`
    gives this (TBO micro-)batch at its offset."""
    from atom.model_ops import topK
    from atom.model_ops.moe import FusedMoE
    from atom.utils import forward_context as fc

    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", switch)
    used = _record_kernels(monkeypatch, topK)
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
    triton_max_rows = _layer_rows(EXPERTS, TOPK, shared)

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
            triton_max_rows=triton_max_rows,
        )

    route()  # eager: compiles the Triton kernel, when on, outside the capture
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
    assert used == [kernel] * (2 + 4)
    if shared:
        # Rows outside this batch's slice are never written.
        outside = torch.ones(max_rows, dtype=torch.bool, device="cuda")
        outside[offset : offset + rows] = False
        assert bool(meta_w[outside].isnan().all())
        assert bool((meta_ids[outside] == EXPERTS).all())


@needs_cutoff
@pytest.mark.parametrize("count", [None, "some"])
@pytest.mark.parametrize("past_cutoff", [False, True])
@pytest.mark.parametrize("experts,k,shared", [(512, 10, 1), (256, 8, 1), (128, 8, 0)])
def test_impl_matches_aiter_on_both_sides_of_the_row_cutoff(
    experts, k, shared, past_cutoff, count, monkeypatch
):
    """At `triton_topk_softmax_max_rows()` rows the impl runs the Triton
    kernel, one row more AITER's; on both sides, eager and replayed, with or
    without a row count, it matches AITER's kernel bit for bit, and so does
    the other kernel run directly on the same launch."""
    from atom.model_ops import topK

    if count and not AITER_HAS_NUM_VALID_ROWS:
        pytest.skip("AITER predates ROCm/aiter#6916")
    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "1")
    cutoff = triton_topk_softmax_max_rows()
    assert cutoff > 0, "no launch would take the Triton kernel"
    rows = cutoff + past_cutoff
    torch.manual_seed(rows + experts + shared)
    width = experts + shared
    logits = torch.cat(
        [
            _logits(rows // 2, width, 2.0, False),
            _logits(rows - rows // 2, width, 0.05, True),
        ]
    )
    num_valid_rows = (
        None
        if count is None
        else torch.tensor([rows * 2 // 3], dtype=torch.int32, device="cuda")
    )
    scoring_func = "sigmoid" if shared else ""

    def outputs():
        # The shared columns' ids are prefilled (`init_aiter_topK_meta_data`).
        w = torch.full((rows, k + shared), float("nan"), device="cuda")
        ids = torch.full((rows, k + shared), experts, dtype=torch.int32, device="cuda")
        return w, ids

    w_ref, ids_ref = outputs()
    scratch = torch.empty(rows, k, dtype=torch.int32, device="cuda")
    kwargs = {} if num_valid_rows is None else {"num_valid_rows": num_valid_rows}
    aiter.topk_softmax(
        w_ref, ids_ref[:, :k], scratch, logits, True, shared, scoring_func, **kwargs
    )
    w_tri, ids_tri = outputs()
    triton_topk_softmax(w_tri, ids_tri[:, :k], logits, shared, num_valid_rows)

    if shared:
        monkeypatch.setattr(topK, "aiter_topK_meta_data", outputs())
    used = _record_kernels(monkeypatch, topK)

    max_rows = _layer_rows(experts, k, shared)
    assert max_rows == cutoff

    def route():
        return topK.rocm_aiter_topk_softmax_impl(
            logits, k, True, shared, experts, scoring_func or None,
            num_valid_rows=num_valid_rows, triton_max_rows=max_rows,
        )  # fmt: skip

    w_eager, ids_eager = (t.clone() for t in route())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w_graph, ids_graph = route()
    graph.replay()
    torch.cuda.synchronize()

    assert used == ["aiter" if past_cutoff else "triton"] * 2
    for w, ids in ((w_eager, ids_eager), (w_graph, ids_graph), (w_tri, ids_tri)):
        assert torch.equal(ids, ids_ref)
        assert torch.equal(_bits(w), _bits(w_ref))
    if num_valid_rows is not None:
        assert bool((ids_ref[rows * 2 // 3 :, :k] == -1).all())


@needs_cutoff
@SWITCH
def test_launches_only_compare_the_row_count(switch, kernel, monkeypatch):
    """Launches, direct or through `FusedMoE.select_experts`, read neither the
    switch nor the CU count and run no shape check: the layer decided once,
    when it was built, and flipping the switch afterwards does not move it."""
    from atom.model_ops import moe, topK
    from atom.model_ops.moe import FusedMoE

    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", switch)
    shared, rows = 1, 64
    max_rows = _layer_rows(EXPERTS, TOPK, shared)
    assert (max_rows is not None) == (kernel == "triton")

    calls = []
    get_env = envs.__getattr__

    def counting_getattr(name):
        if name == "ATOM_TRITON_TOPK_SOFTMAX":
            calls.append(name)
        return get_env(name)

    monkeypatch.setattr(envs, "__getattr__", counting_getattr)
    for module, name in (
        (topk_softmax_dispatch, "_cu_num"),
        (topk_softmax_dispatch, "_gfx"),
        (topk_softmax_dispatch, "triton_topk_softmax_max_rows"),
        (topk_softmax_dispatch, "triton_topk_softmax_supported"),
        (topk_softmax_dispatch, "triton_topk_softmax_rows"),
        (moe, "triton_topk_softmax_rows"),
    ):
        fn = getattr(module, name)

        def counted(*args, _fn=fn, _name=name, **kwargs):
            calls.append(_name)
            return _fn(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)
    monkeypatch.setenv("ATOM_TRITON_TOPK_SOFTMAX", "0" if switch == "1" else "1")
    used = _record_kernels(monkeypatch, topK)
    monkeypatch.setattr(topK, "aiter_topK_meta_data", _outputs(rows, shared))
    torch.manual_seed(41)
    logits = _logits(rows, EXPERTS + shared, 2.0, False)

    for _ in range(3):
        topK.rocm_aiter_topk_softmax_impl(
            logits, TOPK, True, shared, EXPERTS, "sigmoid", triton_max_rows=max_rows
        )
        FusedMoE.select_experts(
            hidden_states=None,
            router_logits=logits,
            top_k=TOPK,
            use_grouped_topk=False,
            renormalize=True,
            num_routing_experts=EXPERTS,
            num_fused_shared_experts=shared,
            fused_shared_experts_scoring_func="sigmoid",
            mask_pad_rows=False,
            triton_max_rows=max_rows,
        )
    torch.cuda.synchronize()
    assert used == [kernel] * 6
    assert calls == []
