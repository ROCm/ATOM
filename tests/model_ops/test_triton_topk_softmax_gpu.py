# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""`triton_topk_softmax` must match AITER's `topk_softmax` bit for bit: a
launch takes one kernel or the other by its row count, so any difference
would route a token differently in a long prefill than in a short pass."""

import functools
import inspect
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("compares two GPU kernels; needs a real GPU", allow_module_level=True)

aiter = pytest.importorskip("aiter")

from conftest import atom_config_double

from atom import config
from atom.model_engine.model_runner import RapidServeModelRunner
from atom.model_ops import moe, topK
from atom.model_ops.triton_topk_softmax import triton_topk_softmax
from atom.utils import forward_context as fc

EXPERTS, TOPK = 512, 10
MI355X_CUS = 256
AITER_HAS_NUM_VALID_ROWS = (
    "num_valid_rows" in inspect.signature(aiter.topk_softmax).parameters
)


@pytest.fixture
def gpu(monkeypatch):
    """Set the architecture, CU count and config the cutoff reads."""

    def set_gpu(arch="gfx950", cus=MI355X_CUS, cu_masked=False, cfg=None):
        if cfg is None:
            cfg = SimpleNamespace(
                enable_rapidserve=cu_masked, disagg_constrained=cu_masked
            )
        monkeypatch.setattr(topK, "get_gfx_runtime", lambda: arch)
        monkeypatch.setattr(topK, "get_cu_num", lambda: cus)
        monkeypatch.setattr(topK, "get_current_atom_config", lambda: cfg)
        topK.triton_topk_softmax_rows.cache_clear()

    set_gpu()
    yield set_gpu
    topK.triton_topk_softmax_rows.cache_clear()


def _logits(rows, width, scale=2.0, ties=False, dtype=torch.bfloat16):
    x = torch.randn(rows, width, device="cuda") * scale
    if ties:
        # Coarse values: many equal logits, including +0.0 and -0.0.
        x = (x * 4).round() / 4
    return x.to(dtype)


def _bits(w):
    # Compares -0.0 and +0.0 (and NaNs) as different.
    return w.view(torch.int32)


def _outputs(rows, k, shared):
    # The fused-shared layout: the shared columns' ids are prefilled.
    w = torch.full((rows, k + shared), float("nan"), device="cuda")
    ids = torch.full((rows, k + shared), 12345, dtype=torch.int32, device="cuda")
    return w, ids


def _aiter(logits, k, shared, num_valid_rows=None, renormalize=True):
    w, ids = _outputs(logits.shape[0], k, shared)
    kwargs = {} if num_valid_rows is None else {"num_valid_rows": num_valid_rows}
    scratch = torch.empty(logits.shape[0], k, dtype=torch.int32, device="cuda")
    aiter.topk_softmax(
        w, ids[:, :k], scratch, logits, renormalize, shared,
        "sigmoid" if shared else "", **kwargs,
    )  # fmt: skip
    return w, ids


def _triton(logits, k, shared, num_valid_rows=None):
    w, ids = _outputs(logits.shape[0], k, shared)
    triton_topk_softmax(w, ids[:, :k], logits, shared, num_valid_rows)
    return w, ids


def _rows(experts, k, shared):
    return topK.triton_topk_softmax_rows(
        experts, k, True, shared, "sigmoid" if shared else None
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shared", [0, 1, 2, 4, 8])
@pytest.mark.parametrize("k", [4, 8, 10, 32])
@pytest.mark.parametrize("experts", [32, 64, 128, 256, 512])
def test_every_supported_shape_matches_aiter(experts, k, shared, dtype, gpu):
    assert _rows(experts, k, shared) is not None
    torch.manual_seed(experts + 3 * k + shared)
    width = experts + shared
    logits = torch.cat(
        [
            _logits(24, width, 2.0, dtype=dtype),
            _logits(24, width, 0.05, ties=True, dtype=dtype),
            _logits(16, width, 8.0, dtype=dtype),
        ]
    )
    w_ref, ids_ref = _aiter(logits, k, shared)
    w, ids = _triton(logits, k, shared)
    assert torch.equal(ids, ids_ref)
    assert torch.equal(_bits(w), _bits(w_ref))


@pytest.mark.parametrize(
    "arch,cus,fractions,max_rows",
    [
        ("gfx950", 256, [0.5], 2560),
        ("gfx950", 256, [0.75], 1280),
        ("gfx950", 80, [0.3, 0.5], 480),
        ("gfx942", 304, [0.5], None),
    ],
)
def test_cu_masked_cutoff_counts_the_streams_cus(
    arch, cus, fractions, max_rows, gpu, monkeypatch
):
    gpu(arch, cus, cu_masked=True)
    monkeypatch.setattr(RapidServeModelRunner, "_CU_POOL_FRACTIONS", fractions)
    assert _rows(512, 10, 1) == max_rows


@pytest.mark.parametrize("fractions", [[0.5], [0.75], [0.3, 0.5]])
def test_cu_masked_cutoff_matches_the_streams_masks(fractions, gpu, monkeypatch):
    """The fewest CUs any of RapidServe's CU-masked streams gets."""
    total = torch.cuda.get_device_properties(0).multi_processor_count
    gpu("gfx950", total, cu_masked=True)
    monkeypatch.setattr(RapidServeModelRunner, "_CU_POOL_FRACTIONS", fractions)
    cus = min(
        sum(w.bit_count() for w in RapidServeModelRunner._cu_mask_for_fraction(f, up))
        for f in fractions
        for up in (False, True)
    )
    assert _rows(512, 10, 1) == 20 * cus


@pytest.mark.parametrize(
    "rapidserve,constrained",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_masked_cus_count_only_on_cu_masked_streams(
    rapidserve, constrained, gpu, monkeypatch
):
    cfg = atom_config_double(
        enable_rapidserve=rapidserve,
        disagg_constrained=constrained,
        torch_dtype=torch.bfloat16,
    )
    monkeypatch.setattr(config, "_current_atom_config", cfg)
    gpu(cfg=config.get_current_atom_config())
    monkeypatch.setattr(RapidServeModelRunner, "_CU_POOL_FRACTIONS", [0.5])
    assert _rows(128, 8, 0) == (2560 if rapidserve and constrained else 5120)


def _first_load_slots(experts):
    # AITER's per-thread vector loads start every 16 bf16, every 32 at 512.
    first = torch.zeros(experts, dtype=torch.bool, device="cuda")
    first[:: 32 if experts == 512 else 16] = True
    return first


def _edge_logits(pattern, rows, experts, shared, k):
    """Rows with -inf and NaN logits, no NaN where AITER's vector loads start
    (there AITER may pick it)."""
    logits = _logits(rows, experts + shared)
    routed = logits[:, :experts]
    rand = torch.rand(rows, experts, device="cuda")
    first = _first_load_slots(experts)
    if pattern == "neg_inf":
        # Row r keeps 1 + r % (k + 1) finite logits; row 1 none.
        finite = 1 + torch.arange(rows, device="cuda") % (k + 1)
        routed[rand.argsort(1).argsort(1) >= finite[:, None]] = float("-inf")
        routed[1] = float("-inf")
    elif pattern == "nan":
        nan = (rand < 0.3) & ~first
        nan[1] = ~first
        routed[nan] = float("nan")
    elif pattern == "nan_and_neg_inf":
        # No finite logit; row 0 is NaN but for the -inf first slots.
        nan = (rand < 0.5) & ~first
        nan[0] = ~first
        routed[nan] = float("nan")
        routed[~nan] = float("-inf")
    else:  # "nan_short": fewer finite logits than k, among NaNs
        finite = first & (rand < 0.5)
        finite[0] = False
        routed[:, ~first] = float("nan")
        routed[first & ~finite] = float("-inf")
    return logits


@pytest.mark.parametrize("pattern", ["neg_inf", "nan", "nan_and_neg_inf", "nan_short"])
@pytest.mark.parametrize("k", [10, 32])
@pytest.mark.parametrize("shared", [0, 1])
@pytest.mark.parametrize("experts", [32, 128, 512])
def test_rows_short_of_finite_logits_match_aiter(experts, shared, k, pattern):
    """Once a row's finite logits run out, both kernels pick expert 0 again
    with weight 0; a row with none gets NaN routed weights. A NaN is never
    picked."""
    torch.manual_seed(experts + k + shared)
    logits = _edge_logits(pattern, 64, experts, shared, k)
    routed = logits[:, :experts].float()
    w_ref, ids_ref = _aiter(logits, k, shared)
    w, ids = _triton(logits, k, shared)

    assert torch.equal(ids, ids_ref)
    count = routed.isfinite().sum(1)
    some = count > 0
    assert torch.equal(_bits(w[some]), _bits(w_ref[some]))
    assert bool(w[~some, :k].isnan().all()) and bool(w_ref[~some, :k].isnan().all())
    assert torch.equal(_bits(w[:, k:]), _bits(w_ref[:, k:]))
    assert not bool(routed.isnan().gather(1, ids[:, :k].long()).any())
    for r in range(64):
        n = min(int(count[r]), k)
        assert bool((ids[r, n:k] == 0).all())
        if n:
            assert bool((_bits(w[r, n:k]) == 0).all())


@pytest.mark.parametrize("shared", [0, 1])
@pytest.mark.parametrize("experts", [32, 128, 512])
def test_a_nan_expert_0_is_skipped_for_the_lowest_cleared_or_neg_inf(experts, shared):
    # AITER's first vector load may pick a NaN expert 0; Triton never does.
    k = 10
    logits = torch.full((3, experts + shared), float("nan"), device="cuda")
    logits[0, 7] = 1.0
    logits[1, 5] = float("-inf")
    logits[1, 9] = float("-inf")
    logits[:, experts:] = 0.5
    w, ids = _triton(logits.to(torch.bfloat16), k, shared)
    assert bool((ids[0, :k] == 7).all())
    assert w[0, 0].item() == 1.0 and bool((_bits(w[0, 1:k]) == 0).all())
    assert bool((ids[1, :k] == 5).all()) and bool(w[1, :k].isnan().all())
    assert bool((ids[2, :k] == 0).all()) and bool(w[2, :k].isnan().all())


def test_launcher_needs_a_unit_last_stride():
    rows, shared = 8, 1
    logits = _logits(rows, EXPERTS + shared)
    w, ids = _outputs(rows, TOPK, shared)
    w_t = torch.empty(TOPK + shared, rows, device="cuda").t()
    ids_t = torch.empty(TOPK + shared, rows, dtype=torch.int32, device="cuda").t()
    wide = _logits(rows, 2 * (EXPERTS + shared))
    for out_w, out_ids, gating in (
        (w_t, ids[:, :TOPK], logits),
        (w, ids_t[:, :TOPK], logits),
        (w, ids[:, :TOPK], wide[:, ::2]),
    ):
        with pytest.raises(AssertionError, match="unit last stride"):
            triton_topk_softmax(out_w, out_ids, gating, shared)


def test_no_rows_launch_nothing():
    w, ids = _triton(_logits(0, EXPERTS + 1), TOPK, 1)
    assert w.shape == ids.shape == (0, TOPK + 1)


def _record_kernels(monkeypatch):
    """The kernel each later `rocm_aiter_topk_softmax_impl` call launches."""
    used = []

    def recorded(name, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            used.append(name)
            return fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        topK, "triton_topk_softmax", recorded("triton", topK.triton_topk_softmax)
    )
    monkeypatch.setattr(aiter, "topk_softmax", recorded("aiter", aiter.topk_softmax))
    return used


@pytest.mark.parametrize(
    "width,shared,scoring,renormalize,kernel",
    [
        (EXPERTS, 0, None, True, "triton"),
        (EXPERTS + 1, 1, "sigmoid", True, "triton"),
        (EXPERTS, 1, None, True, "triton"),  # unscored shared: prefilled
        (EXPERTS, 0, None, False, "aiter"),
        # MoRI all2all fusion leaves a scored shared logit to the dispatch.
        (EXPERTS + 1, 0, "sigmoid", True, "aiter"),
    ],
)
def test_impl_picks_the_kernel_from_the_launch(
    width, shared, scoring, renormalize, kernel, gpu, monkeypatch
):
    rows = 16
    logits = _logits(rows, width)
    monkeypatch.setattr(topK, "aiter_topK_meta_data", _outputs(rows, TOPK, shared))
    used = _record_kernels(monkeypatch)
    topK.rocm_aiter_topk_softmax_impl(
        logits, TOPK, renormalize, shared, EXPERTS, scoring
    )  # fmt: skip
    assert used == [kernel]


@pytest.mark.parametrize("shared", [0, 1])
@pytest.mark.parametrize("rows", [1, 33, 256])
@pytest.mark.parametrize("count", ["zero", "some", "all", "over"])
def test_num_valid_rows_pads_rows(shared, rows, count):
    """Rows at or past the count (ROCm/aiter#6916) get id -1 and weight 0;
    shared columns are written on every row."""
    torch.manual_seed(rows * 5 + shared)
    logits = _logits(rows, EXPERTS + shared)
    n = {"zero": 0, "some": rows // 3, "all": rows, "over": rows + 7}[count]
    num_valid_rows = torch.tensor([n, 12345], dtype=torch.int32, device="cuda")
    w_full, ids_full = _triton(logits, TOPK, shared)
    w, ids = _triton(logits, TOPK, shared, num_valid_rows)

    assert torch.equal(ids[:n], ids_full[:n])
    assert torch.equal(_bits(w[:n]), _bits(w_full[:n]))
    assert bool((ids[n:, :TOPK] == -1).all())
    assert bool((_bits(w[n:, :TOPK]) == 0).all())
    assert torch.equal(_bits(w[:, TOPK:]), _bits(w_full[:, TOPK:]))
    if AITER_HAS_NUM_VALID_ROWS:
        w_ref, ids_ref = _aiter(logits, TOPK, shared, num_valid_rows)
        assert torch.equal(ids, ids_ref)
        assert torch.equal(_bits(w), _bits(w_ref))


@pytest.mark.parametrize("shared,offset", [(0, 0), (1, 0), (1, 32)])
def test_select_experts_counts_the_pass_rows(shared, offset, gpu, monkeypatch):
    """Through `FusedMoE.select_experts`, which reads the pass's row count from
    the forward context: the captured route follows it on every replay, bit
    for bit with the eager one, and with a fused shared expert writes this
    (TBO micro-)batch's `topk_meta_rows` only."""
    used = _record_kernels(monkeypatch)
    rows, max_rows = 48, 128
    torch.manual_seed(31 + shared + offset)
    logits = _logits(rows, EXPERTS + shared)
    count = torch.tensor([rows], dtype=torch.int32, device="cuda")
    context = fc.Context(
        positions=torch.zeros(rows, dtype=torch.int64, device="cuda"),
        running_tokens=rows,
        ubatch_token_offset=offset,
        scheduled_tokens_gpu=count,
    )
    monkeypatch.setattr(fc, "_forward_context", fc.ForwardContext(context=context))
    meta_w, meta_ids = _outputs(max_rows, TOPK, shared)
    monkeypatch.setattr(topK, "aiter_topK_meta_data", (meta_w, meta_ids))

    def route():
        return moe.FusedMoE.select_experts(
            hidden_states=None,
            router_logits=logits,
            top_k=TOPK,
            use_grouped_topk=False,
            renormalize=True,
            num_routing_experts=EXPERTS,
            num_fused_shared_experts=shared,
            fused_shared_experts_scoring_func="sigmoid" if shared else None,
        )

    route()  # compiles the Triton kernel outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w_graph, ids_graph = route()
    if shared:
        assert w_graph.data_ptr() == meta_w[offset].data_ptr()

    for n in (rows, 0, 29, rows):
        count.fill_(n)
        w_eager, ids_eager = (t.clone() for t in route())
        graph.replay()
        torch.cuda.synchronize()
        assert bool((ids_eager[n:, :TOPK] == -1).all())
        assert bool((ids_eager[:n, :TOPK] >= 0).all())
        assert torch.equal(ids_graph, ids_eager)
        assert torch.equal(_bits(w_graph), _bits(w_eager))
    assert used == ["triton"] * (2 + 4)
    if shared:
        outside = torch.ones(max_rows, dtype=torch.bool, device="cuda")
        outside[offset : offset + rows] = False
        assert bool(meta_w[outside].isnan().all())


@pytest.mark.parametrize("count", [False, True])
@pytest.mark.parametrize("past_cutoff", [False, True])
@pytest.mark.parametrize("experts,k,shared", [(512, 10, 1), (256, 8, 1), (128, 8, 0)])
def test_impl_matches_aiter_on_both_sides_of_the_row_cutoff(
    experts, k, shared, past_cutoff, count, gpu, monkeypatch
):
    """At the cutoff the impl runs Triton, one row past it AITER; eager and
    replayed, both match AITER's kernel bit for bit."""
    if count and not AITER_HAS_NUM_VALID_ROWS:
        pytest.skip("AITER predates ROCm/aiter#6916")
    max_rows = _rows(experts, k, shared)
    rows = max_rows + past_cutoff
    torch.manual_seed(rows + experts + shared)
    width = experts + shared
    logits = torch.cat(
        [
            _logits(rows // 2, width),
            _logits(rows - rows // 2, width, 0.05, ties=True),
        ]
    )
    num_valid_rows = (
        torch.tensor([rows * 2 // 3], dtype=torch.int32, device="cuda")
        if count
        else None
    )
    w_ref, ids_ref = _aiter(logits, k, shared, num_valid_rows)
    if shared:
        monkeypatch.setattr(topK, "aiter_topK_meta_data", _outputs(rows, k, shared))
    used = _record_kernels(monkeypatch)

    def route():
        return topK.rocm_aiter_topk_softmax_impl(
            logits, k, True, shared, experts, "sigmoid" if shared else None,
            num_valid_rows=num_valid_rows,
        )  # fmt: skip

    w_eager, ids_eager = (t.clone() for t in route())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w_graph, ids_graph = route()
    graph.replay()
    torch.cuda.synchronize()

    assert used == ["aiter" if past_cutoff else "triton"] * 2
    for w, ids in ((w_eager, ids_eager), (w_graph, ids_graph)):
        assert torch.equal(ids, ids_ref)
        assert torch.equal(_bits(w), _bits(w_ref))
