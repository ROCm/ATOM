# SPDX-License-Identifier: MIT
"""MiniMax-M3 shapes for ``aiter.mxfp4_moe_scatter_reduce``.

Run 37867633847 died in
``moe_scatter_reduce::scatter_reduce_kernel_impl<6144, 5, 8, true>`` during
32768-token PD warmup (MEMORY_APERTURE_VIOLATION). The kernel only skipped
``sorted_pos < 0``; a corrupt ``reverse_sorted`` entry past ``flat_out`` rows
plus BM>=128 nontemporal loads faulted the GPU.

This file locks the M3 TP4 contract (H=6144, TOPK=5) and the NT path (MB=128):

  * legal reverse maps match a torch reference;
  * negative and ``>= max_sorted`` indices are skipped without a launch error.

Needs a patched aiter (kernel takes ``max_sorted`` and bounds-checks). Apply
``.github/patches/aiter_moe_scatter_reduce_bounds.patch`` (or install
``ROCm/aiter@mengqing/moe_scatter_reduce_bounds``). CI:
``.github/workflows/verify-aiter-scatter-bounds.yaml``.

    AITER_LOG_LEVEL=WARNING AMD_SERIALIZE_KERNEL=3 \\
      python -m pytest tests/test_minimax_m3_moe_scatter_reduce.py -v -s
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("ROCm GPU required", allow_module_level=True)

pytest.importorskip("aiter", reason="scatter_reduce lives in aiter")
import aiter
from aiter import dtypes

if not hasattr(aiter, "mxfp4_moe_scatter_reduce"):
    pytest.skip("aiter build has no mxfp4_moe_scatter_reduce", allow_module_level=True)

H = 6144
TOPK = 5
NE = 129  # M3 experts; dispatch key ignores NE


def _reference(flat_out, reverse_sorted, sorted_weights, M, topk, max_sorted):
    D = flat_out.shape[1]
    out = torch.zeros(M, D, dtype=torch.float32, device=flat_out.device)
    rs = reverse_sorted.view(M, topk)
    for t in range(M):
        acc = torch.zeros(D, dtype=torch.float32, device=flat_out.device)
        for k in range(topk):
            sp = int(rs[t, k].item())
            if sp < 0 or sp >= max_sorted:
                continue
            acc = acc + flat_out[sp].float() * float(sorted_weights[sp].item())
        out[t] = acc
    return out


def _raise_pending_launch_error():
    try:
        torch.ones(1, device="cuda").add_(1)
    except RuntimeError as exc:
        pytest.fail(f"a kernel launch failed silently: {exc}")


def _scatter(flat, rs, weights, out, mb):
    aiter.mxfp4_moe_scatter_reduce(
        flat_out=flat,
        reverse_sorted=rs,
        sorted_weights=weights,
        out=out,
        NE=NE,
        TOPK=TOPK,
        D_HIDDEN=H,
        MB=mb,
    )
    _raise_pending_launch_error()
    torch.cuda.synchronize()


@pytest.mark.parametrize("mb", (16, 128), ids=("no_nt", "nt_bm128"))
@pytest.mark.parametrize("M", (1, 8, 64))
def test_m3_scatter_reduce_matches_reference(M, mb):
    torch.manual_seed(M + mb)
    max_sorted = ((M * TOPK + mb - 1) // mb) * mb + mb
    flat = torch.randn(max_sorted, H, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(max_sorted, dtype=torch.float32, device="cuda")
    rs = torch.arange(M * TOPK, dtype=torch.int32, device="cuda")
    out = torch.full((M, H), float("nan"), dtype=torch.bfloat16, device="cuda")
    _scatter(flat, rs, weights, out, mb)
    ref = _reference(flat, rs, weights, M, TOPK, max_sorted)
    err = (out.float() - ref).abs().max().item()
    assert torch.isfinite(out).all()
    assert err < 2e-2, f"max abs err {err}"


@pytest.mark.parametrize("mb", (16, 128), ids=("no_nt", "nt_bm128"))
def test_m3_scatter_reduce_oob_indices_do_not_fault(mb):
    """The C40 failure mode: reverse_sorted past flat_out + NT loads."""
    torch.manual_seed(mb)
    M, max_sorted = 4, 32
    flat = torch.randn(max_sorted, H, dtype=torch.bfloat16, device="cuda")
    weights = torch.ones(max_sorted, dtype=torch.float32, device="cuda")
    rs = torch.tensor(
        [
            0,
            -1,
            max_sorted,
            max_sorted + 99,
            1,
            -7,
            1 << 20,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        ],
        dtype=torch.int32,
        device="cuda",
    )
    assert rs.numel() == M * TOPK
    out = torch.full((M, H), float("nan"), dtype=torch.bfloat16, device="cuda")
    _scatter(flat, rs, weights, out, mb)
    ref = _reference(flat, rs, weights, M, TOPK, max_sorted)
    err = (out.float() - ref).abs().max().item()
    assert torch.isfinite(out).all()
    assert err < 2e-2, f"max abs err {err}"


def test_m3_scatter_reduce_nt_path_smoke():
    """BM=128 turns on NT hints — the path the 32K warmup used."""
    torch.manual_seed(0)
    M, mb = 256, 128
    max_sorted = ((M * TOPK + mb - 1) // mb) * mb
    flat = torch.randn(max_sorted, H, dtype=torch.bfloat16, device="cuda") * 0.01
    weights = torch.rand(max_sorted, dtype=torch.float32, device="cuda")
    rs = torch.arange(M * TOPK, dtype=torch.int32, device="cuda") % max_sorted
    out = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
    _scatter(flat, rs, weights, out, mb)
    assert torch.isfinite(out).all()
