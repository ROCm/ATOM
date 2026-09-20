# SPDX-License-Identifier: MIT
import pytest
import torch

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires a ROCm GPU", allow_module_level=True)

from aiter.ops.quant import per_tensor_quant_hip
from aiter.test_common import checkAllclose

from atom.model_ops.triton_fused_qkv_quant import fused_qkv_per_tensor_quant


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


@pytest.mark.parametrize("tokens", [0, 1, 3, 5, 1001, 8192])
@pytest.mark.parametrize("rope_layout", ["shared", "expanded", "per_head"])
def test_split_k_matches_materialized_concat(tokens, rope_layout):
    torch.manual_seed(tokens)
    q = torch.randn((tokens, 12, 256), device="cuda", dtype=torch.bfloat16)[..., :192]
    kv = torch.randn((tokens, 12, 256), device="cuda", dtype=q.dtype)
    k, v = kv.split([128, 128], dim=-1)
    heads = 12 if rope_layout == "per_head" else 1
    # Nonzero offset and stride(-1)=2 also exercise the RoPE address calculation.
    rope = torch.randn((tokens, heads, 131), device="cuda", dtype=q.dtype)[..., 2:130:2]
    if rope_layout == "expanded":
        rope = rope.expand(-1, 12, -1)
    full_k = torch.cat((k, rope.expand(-1, 12, -1)), dim=-1)
    reference = fused_qkv_per_tensor_quant(q, full_k, v)
    actual = fused_qkv_per_tensor_quant(q, k, v, k_rope=rope)
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0)
    assert actual[1].shape == full_k.shape and actual[1].is_contiguous()


@pytest.mark.parametrize("tokens", [1, 257])
def test_split_k_tail_guards_and_amax(tokens):
    q = torch.zeros((tokens, 3, 20), device="cuda", dtype=torch.bfloat16)
    v = torch.zeros((tokens, 3, 7), device="cuda", dtype=q.dtype)
    k = torch.full((tokens, 3, 29), 128, device="cuda", dtype=q.dtype)[..., 1:27:2]
    rope = torch.full((tokens, 1, 17), 128, device="cuda", dtype=q.dtype)[..., 2:16:2]
    for kval, rval in ((9, -2), (2, -19), (0, 0)):
        k.fill_(kval)
        rope.fill_(rval)
        result = fused_qkv_per_tensor_quant(q, k, v, k_rope=rope)
        full_k = torch.cat((k, rope.expand(-1, 3, -1)), -1)
        scale = max(abs(kval), abs(rval)) / 448 or 1e-6
        assert result[4].item() == pytest.approx(scale)
        torch.testing.assert_close(
            result[1].float(),
            (full_k.float() / scale).to(torch.float8_e4m3fn).float(),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("tokens", [1, 512])
def test_split_k_graph_replay_updates_rope_scale(tokens):
    q = torch.zeros((tokens, 12, 192), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((tokens, 12, 128), device="cuda", dtype=q.dtype)
    v = torch.zeros_like(k)
    rope = torch.ones((tokens, 1, 64), device="cuda", dtype=q.dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused_qkv_per_tensor_quant(q, k, v, k_rope=rope)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = fused_qkv_per_tensor_quant(q, k, v, k_rope=rope)
        for value in (3, 0, -17):
            rope.fill_(value)
            graph.replay()
            stream.synchronize()
            assert result[4].item() == pytest.approx(
                abs(value) / 448 if value else 1e-6
            )
            assert result[6].item() == pytest.approx(max(abs(value) / 448, 1e-6) * 2)
            full_k = torch.cat((k, rope.expand(-1, 12, -1)), -1)
            torch.testing.assert_close(
                result[1].float() * result[4], full_k.float(), rtol=1e-6, atol=0
            )
    torch.cuda.current_stream().wait_stream(stream)
