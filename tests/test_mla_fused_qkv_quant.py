# SPDX-License-Identifier: MIT
import pytest
import torch
from aiter.ops.quant import per_tensor_quant_hip
from aiter.test_common import checkAllclose

from atom.model_ops.triton_fused_qkv_quant import fused_qkv_per_tensor_quant

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="requires a ROCm GPU"
)


@pytest.mark.parametrize("tokens", [0, 1, 3, 5, 88, 512, 1001, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_strided_qkv_matches_hip(tokens, dtype):
    torch.manual_seed(tokens)
    q = torch.randn((tokens, 12, 256), device="cuda", dtype=dtype)[..., :192]
    k = torch.randn((tokens, 12, 192), device="cuda", dtype=dtype)
    v = torch.randn((tokens, 12, 256), device="cuda", dtype=dtype)[..., 128:]
    out = fused_qkv_per_tensor_quant(q, k, v)
    for i, x in enumerate((q, k, v)):
        assert out[i].shape == x.shape
        assert out[i].is_contiguous()
        assert out[3 + i].shape == (1,)
        if tokens:
            # Whole 16-element vectors, enough rows for the reference HIP op.
            ref, scale = per_tensor_quant_hip(
                x.contiguous().view(-1, 4096 if x.numel() % 4096 == 0 else 128),
                quant_dtype=torch.float8_e4m3fn,
            )
            assert (
                checkAllclose(
                    out[i].float(),
                    ref.view(x.shape).float(),
                    rtol=0,
                    atol=0,
                    tol_err_ratio=0,
                    msg="quantized output",
                )
                == 0
            )
            assert (
                checkAllclose(
                    out[3 + i],
                    scale,
                    rtol=0,
                    atol=0,
                    tol_err_ratio=0,
                    msg="descale",
                )
                == 0
            )
        else:
            assert (
                checkAllclose(
                    out[3 + i],
                    torch.full_like(out[3 + i], 1e-6),
                    rtol=0,
                    atol=0,
                    tol_err_ratio=0,
                    msg="empty descale",
                )
                == 0
            )
    for i in (1, 2):
        assert (
            checkAllclose(
                out[5 + i],
                out[3 + i].clamp_min(1e-6) * 2,
                rtol=0,
                atol=0,
                tol_err_ratio=0,
                msg="gather descale",
            )
            == 0
        )


@pytest.mark.parametrize("tokens", [1, 7, 512])
def test_no_padding_reads_and_zero_input(tokens):
    storage = torch.full((tokens, 4, 79), 64, device="cuda", dtype=torch.bfloat16)
    q = storage[..., 3:66:2]
    q.fill_(1)
    k = torch.zeros((tokens + 1, 4, 13), device="cuda", dtype=q.dtype)
    v = k.transpose(0, 1)
    out = fused_qkv_per_tensor_quant(q, k, v)
    assert (
        checkAllclose(
            out[0].float() * out[3],
            q.float(),
            rtol=1e-6,
            atol=0,
            tol_err_ratio=0,
            msg="strided input",
        )
        == 0
    )
    for i in (1, 2):
        assert torch.count_nonzero(out[i].float()).item() == 0
        assert out[3 + i].item() == pytest.approx(1e-6)
        assert out[5 + i].item() == pytest.approx(2e-6)


def test_graph_replay_updates_scales_and_outputs():
    q = torch.ones((512, 12, 192), device="cuda", dtype=torch.bfloat16)
    v = q[..., :128]
    for _ in range(3):
        fused_qkv_per_tensor_quant(q, q, v)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fused_qkv_per_tensor_quant(q, q, v)
    for value in (3, 0, -7):
        q.fill_(value)
        graph.replay()
        for i, x in enumerate((q, q, v)):
            scale = out[3 + i]
            assert (
                checkAllclose(
                    out[i].float() * scale,
                    x.float(),
                    rtol=1e-6,
                    atol=0,
                    tol_err_ratio=0,
                    msg="graph replay",
                )
                == 0
            )
