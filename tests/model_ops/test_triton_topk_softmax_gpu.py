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

from atom.model_ops.triton_topk_softmax import (
    triton_topk_softmax,
    triton_topk_softmax_supported,
)

EXPERTS, TOPK = 512, 10


def _logits(rows, width, scale, ties):
    x = torch.randn(rows, width, device="cuda") * scale
    if ties:
        # Coarse values: many equal logits, including +0.0 and -0.0.
        x = (x * 4).round() / 4
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("shared", [0, 1])
@pytest.mark.parametrize("rows", [1, 4, 33, 1024])
@pytest.mark.parametrize("scale,ties", [(2.0, False), (0.05, True), (8.0, False)])
def test_matches_aiter_bit_for_bit(shared, rows, scale, ties):
    torch.manual_seed(rows * 7 + shared)
    logits = _logits(rows, EXPERTS + shared, scale, ties)
    assert triton_topk_softmax_supported(logits, TOPK, True, shared)

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


def test_unsupported_shapes_fall_back():
    logits = torch.zeros(4, 513, dtype=torch.bfloat16, device="cuda")
    assert not triton_topk_softmax_supported(logits, TOPK, False, 1)
    assert not triton_topk_softmax_supported(logits, TOPK, True, 0)  # 513 experts
    assert not triton_topk_softmax_supported(logits[:, ::2], TOPK, True, 0)


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
