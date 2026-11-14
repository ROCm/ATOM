# SPDX-License-Identifier: MIT
"""Live admission/draft counts reuse kernels without changing their results."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("runtime-count regression requires a GPU", allow_module_level=True)

from atom.model_ops.attentions.deepseek_v41 import planes
from atom.model_ops.deepseek_v41 import dspark
from tests.attentions.deepseek_v41.test_draft_fp8 import (
    assert_csr_equal,
    csr_oracle,
    mxfp8,
)


def variants(kernel):
    return len(kernel.device_caches[torch.cuda.current_device()][0])


@pytest.mark.parametrize("operation", ["reset", "gather", "gather_planes"])
def test_window_counts_reuse_one_variant(operation):
    layers, capacity, ring, dim = 3, 32, 13, 512
    sources = [
        torch.empty(
            capacity * ring * (dim + dim // 32), device="cuda", dtype=torch.uint8
        )
        for _ in range(layers)
    ]
    views = [planes.split_source(source, dim) for source in sources]
    for layer, (data, scale) in enumerate(views):
        data.copy_(torch.arange(data.numel(), device="cuda").view_as(data) % 64 + layer)
        scale.fill_(127)
    originals = [tuple(p.clone() for p in pair) for pair in views]
    ptrs = torch.tensor([[p.data_ptr() for p in pair] for pair in views], device="cuda")
    layer_ids = torch.arange(layers, device="cuda")
    order = torch.tensor([2, 0, 1], device="cuda")
    slot_ids = torch.arange(capacity - 1, -1, -1, device="cuda")
    kernel = {
        "reset": planes._reset_windows,
        "gather": planes._gather_windows,
        "gather_planes": planes._gather_window_planes,
    }[operation]
    compiled = None
    for count in (1, 2, 3, 7, 16, 17, 23):
        slots = slot_ids[:count]
        if operation == "reset":
            for pair, original in zip(views, originals):
                for dst, src in zip(pair, original):
                    dst.copy_(src)
            planes.reset_windows(sources, ptrs, slots, ring, dim)
            for pair, original in zip(views, originals):
                for actual, old in zip(pair, original):
                    expected = old.clone().view(capacity, ring, -1)
                    expected[slots] = 0
                    assert torch.equal(actual.view_as(expected), expected)
        else:
            expected = tuple(
                torch.stack(
                    [
                        views[layer][part].view(capacity, ring, -1)[slots]
                        for layer in (2, 0, 1)
                    ]
                )
                for part in range(2)
            )
            if operation == "gather_planes":
                actual = planes.gather_window_planes(ptrs, order, slots, ring, dim)
                assert all(torch.equal(a, b) for a, b in zip(actual, expected))
            else:
                actual = planes.gather_windows(
                    sources, ptrs, layer_ids, order, slots, ring, dim
                )
                torch.testing.assert_close(
                    actual, mxfp8(expected).bfloat16(), rtol=0, atol=0
                )
        torch.cuda.synchronize()
        if compiled is None:
            compiled = variants(kernel)
        assert (
            variants(kernel) == compiled
        ), f"{operation} compiled again at count={count}"


def test_draft_csr_counts_reuse_one_variant():
    # All these counts share the same bounded scan tile (BLOCK_BATCH=16).
    # Only the live request count changes, including its alignment.
    compiled = None
    for count in (17, 19, 21, 23, 32):
        anchors = torch.arange(count, device="cuda") * 3 - 1
        _, indices, _ = dspark.draft_step_indices(anchors, 13, 8, 5)
        actual = dspark.draft_attention_csr(indices, 13)
        assert_csr_equal(actual, csr_oracle(indices, 13))
        if compiled is None:
            compiled = variants(dspark._draft_csr)
        assert (
            variants(dspark._draft_csr) == compiled
        ), f"CSR compiled again at count={count}"
