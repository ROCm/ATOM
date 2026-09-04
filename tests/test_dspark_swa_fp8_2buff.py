#!/usr/bin/env python3
"""Offline gate for DSpark native 2buff fp8 SWA: kernel vs reference + roundtrip.

Validates the read side of the fp8 draft window (DSPARK_SWA_FP8_PLAN.md step 3):
`dspark_paged_window_gather_2buff` must (a) bit-match its torch reference, and
(b) round-trip the write side (`swa_write_2buff_prepacked`) — write a token at
`pos == anchor`, gather a window ending at `anchor`, and recover the dequantized
value at the last slot with unfilled slots zeroed. No model / engine needed."""

import pytest
import torch
from import_guard import skip_if_dependency_missing

# Broad on purpose: under bare non-GPU pytest this import chain fails in more
# ways than ImportError, and every one of them means the same thing here.
try:
    from aiter import dtypes

    import atom.model_ops.v4_kernels  # noqa: F401  (heavy import chain)
except ImportError as _e:
    skip_if_dependency_missing(_e, "requires full atom import env")

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
    CSA_RATIO,
    DENSE_RATIO,
    HCA_RATIO,
    UnifiedPoolGeometry,
)
from atom.model_ops.v4_kernels.state_writes import (
    dspark_paged_window_gather_2buff,
    dspark_paged_window_gather_2buff_reference,
    swa_write_2buff_prepacked,
)
from atom.model_ops.v4_kernels.v4_quant import (
    V4_DIM_QK,
    V4_DIM_QK_PACKED,
    V4_DIM_ROPE,
    dequantize_v4_2buff_to_bf16,
    quantize_bf16_to_v4_2buff_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="2buff gather is a GPU (Triton) kernel"
)

dev = "cuda"
RATIOS = [DENSE_RATIO, CSA_RATIO, HCA_RATIO, CSA_RATIO, HCA_RATIO]


def _geometry(ring_slots, num_slots):
    """A pool whose window rows are the CSA class's — the widest layer stride,
    so the interleaving the gather has to follow is actually non-trivial."""
    return UnifiedPoolGeometry(
        RATIOS,
        num_blocks=3,
        num_slots=num_slots,
        ring_slots=ring_slots,
        block_size=256,
    )


def _pools(geometry, block_tables_seed):
    torch.manual_seed(block_tables_seed)
    src = torch.randn(geometry.plane_rows, V4_DIM_QK, dtype=torch.bfloat16, device=dev)
    nope, rope = quantize_bf16_to_v4_2buff_triton(src)
    return nope, rope


def test_gather_2buff_matches_reference():
    # `W <= cache_size` is a precondition of the gather (a wider draft window
    # would read rows the ring already recycled); keep the test inside it.
    bs, ring_slots, num_slots, W = 3, 11, 6, 10
    geometry = _geometry(ring_slots, num_slots)
    window = geometry.window_params(CSA_RATIO)
    nope, rope = _pools(geometry, 0)
    anchor = torch.tensor([20, 3, 40], dtype=torch.int32, device=dev)
    # Non-identity slots: indexing by batch id instead would still pass on arange.
    slots = torch.tensor([3, 0, 5], dtype=torch.int32, device=dev)

    out = dspark_paged_window_gather_2buff(nope, rope, slots, anchor, W, window)
    ref = dspark_paged_window_gather_2buff_reference(
        nope, rope, slots, anchor, W, window
    )
    torch.cuda.synchronize()
    assert out.shape == (bs, W, V4_DIM_QK) and out.dtype == torch.bfloat16
    assert torch.equal(out, ref), (out.float() - ref.float()).abs().max()


def test_write_then_gather_roundtrip():
    bs, ring_slots, num_slots, W = 2, 13, 8, 12
    geometry = _geometry(ring_slots, num_slots)
    params = geometry.window_params(CSA_RATIO)
    num_pages = geometry.plane_rows
    anchor = torch.tensor([15, 30], dtype=torch.int32, device=dev)
    cu = torch.tensor([0, 1, 2], dtype=torch.int32, device=dev)  # 1 tok/req
    slots = torch.tensor([6, 1], dtype=torch.int32, device=dev)

    nope = torch.zeros(num_pages, V4_DIM_QK_PACKED, dtype=dtypes.fp8, device=dev)
    rope = torch.zeros(num_pages, V4_DIM_ROPE, dtype=torch.bfloat16, device=dev)
    main_kv = torch.randn(bs, V4_DIM_QK, dtype=torch.bfloat16, device=dev)
    k_packed, k_rope = quantize_bf16_to_v4_2buff_triton(main_kv.contiguous())
    swa_write_2buff_prepacked(
        k_packed, k_rope, anchor.clone(), cu, slots, nope, rope, params, 1
    )
    window = dspark_paged_window_gather_2buff(nope, rope, slots, anchor, W, params)
    torch.cuda.synchronize()

    want = dequantize_v4_2buff_to_bf16(k_packed, k_rope).to(torch.bfloat16)
    assert torch.equal(window[:, W - 1, :], want)  # anchor slot == written token
    assert window[:, : W - 1, :].abs().max() == 0.0  # unfilled slots zeroed


if __name__ == "__main__":
    test_gather_2buff_matches_reference()
    test_write_then_gather_roundtrip()
    print("PASS")
