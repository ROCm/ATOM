# SPDX-License-Identifier: MIT
"""The MiniMax-M3.1 draft's one-GEMM context K/V (`MiniMaxM3DSpark._context_kv_weight`):
the K/V rows of several preshuffled MXFP8 qkv weights, row-sliced on 16-row
tiles and concatenated, give those columns of each full GEMM. The GEMM kernel is
chosen per output width, so the fp32 accumulation order can differ between the
two calls: a rare element (measured <= 0.02%) lands one bf16 step away."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _mx8_weight(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(
        120, 130, (n, k // 32), device="cuda", dtype=torch.uint8, generator=g
    )
    return w, s.view(torch.float8_e8m0fnu)


@pytest.mark.parametrize("rows", [8, 56, 448])
def test_concat_kv_rows_gemm(rows):
    from aiter.ops.shuffle import shuffle_weight

    from atom.model_ops.blockscale import native_quant_linear, quantize_fp8

    N, K, Q, KV = 2304, 6144, 2048, 128
    x = (torch.randn(rows, K, device="cuda") * 2).to(torch.bfloat16)
    xq, xs = quantize_fp8(x)
    full, parts_w, parts_s = [], [], []
    for layer in range(5):
        w, s = _mx8_weight(N, K, layer)
        w_sh = shuffle_weight(w, layout=(16, 16))
        full.append(
            native_quant_linear(
                xq, w_sh, s, x_scale=xs, weight_group_rows=1, weight_preshuffled=True
            )
        )
        parts_w.append(w_sh.narrow(0, Q, 2 * KV))
        parts_s.append(s.narrow(0, Q, 2 * KV))
    kv = native_quant_linear(
        xq, torch.cat(parts_w), torch.cat(parts_s), x_scale=xs, weight_group_rows=1,
        weight_preshuffled=True,
    )  # fmt: skip
    bad = 0
    for layer in range(5):
        got, ref = kv[:, layer * 2 * KV : (layer + 1) * 2 * KV], full[layer][:, Q:]
        # one bf16 step is at most 2**-7 of the value
        torch.testing.assert_close(got, ref, atol=2e-3, rtol=2**-6)
        bad += int((got != ref).sum())
    assert bad <= 1e-3 * kv.numel()
