#!/usr/bin/env python3
"""Offline gate for DSpark native 2buff fp8 SWA: kernel vs reference + roundtrip.

Validates the read side of the fp8 draft window (DSPARK_SWA_FP8_PLAN.md step 3):
`dspark_paged_window_gather_2buff` must (a) bit-match its torch reference, and
(b) round-trip the write side (`swa_write_2buff_prepacked`) — write a token at
`pos == anchor`, gather a window ending at `anchor`, and recover the dequantized
value at the last slot with unfilled slots zeroed. No model / engine needed."""

import pytest
import torch

try:
    import atom.model_ops.v4_kernels  # noqa: F401  (heavy import chain)
    from aiter import dtypes
except Exception as _e:  # pragma: no cover - bare-pytest import env
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)

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


def _pools(num_pages, block_tables_seed):
    torch.manual_seed(block_tables_seed)
    src = torch.randn(num_pages, V4_DIM_QK, dtype=torch.bfloat16, device=dev)
    nope, rope = quantize_bf16_to_v4_2buff_triton(src)
    return nope, rope


def test_gather_2buff_matches_reference():
    bs, block_size, max_blocks, num_pages, W = 3, 8, 6, 128, 10
    nope, rope = _pools(num_pages, 0)
    anchor = torch.tensor([20, 3, 40], dtype=torch.int32, device=dev)
    bt = torch.zeros(bs, max_blocks, dtype=torch.int32, device=dev)
    bt[0] = torch.tensor([1, 2, 3, 4, 5, 0], dtype=torch.int32)
    bt[1] = torch.tensor([6, 7, 8, 9, 0, 0], dtype=torch.int32)
    bt[2] = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)

    out = dspark_paged_window_gather_2buff(nope, rope, bt, anchor, W, block_size)
    ref = dspark_paged_window_gather_2buff_reference(
        nope, rope, bt, anchor, W, block_size
    )
    torch.cuda.synchronize()
    assert out.shape == (bs, W, V4_DIM_QK) and out.dtype == torch.bfloat16
    assert torch.equal(out, ref), (out.float() - ref.float()).abs().max()


def test_write_then_gather_roundtrip():
    bs, block_size, max_blocks, num_pages, W = 2, 8, 8, 256, 12
    anchor = torch.tensor([15, 30], dtype=torch.int32, device=dev)
    cu = torch.tensor([0, 1, 2], dtype=torch.int32, device=dev)  # 1 tok/req
    bt = torch.zeros(bs, max_blocks, dtype=torch.int32, device=dev)
    bt[0] = torch.arange(1, 1 + max_blocks, dtype=torch.int32)
    bt[1] = torch.arange(20, 20 + max_blocks, dtype=torch.int32)

    nope = torch.zeros(num_pages, V4_DIM_QK_PACKED, dtype=dtypes.fp8, device=dev)
    rope = torch.zeros(num_pages, V4_DIM_ROPE, dtype=torch.bfloat16, device=dev)
    main_kv = torch.randn(bs, V4_DIM_QK, dtype=torch.bfloat16, device=dev)
    k_packed, k_rope = quantize_bf16_to_v4_2buff_triton(main_kv.contiguous())
    swa_write_2buff_prepacked(
        k_packed, k_rope, anchor.clone(), cu, bt, nope, rope, block_size, 1
    )
    window = dspark_paged_window_gather_2buff(nope, rope, bt, anchor, W, block_size)
    torch.cuda.synchronize()

    want = dequantize_v4_2buff_to_bf16(k_packed, k_rope).to(torch.bfloat16)
    assert torch.equal(window[:, W - 1, :], want)  # anchor slot == written token
    assert window[:, : W - 1, :].abs().max() == 0.0  # unfilled slots zeroed


if __name__ == "__main__":
    test_gather_2buff_matches_reference()
    test_write_then_gather_roundtrip()
    print("PASS")
