# SPDX-License-Identifier: MIT
import pytest
import torch

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires a ROCm GPU", allow_module_level=True)

from atom.model_ops.triton_fused_qkv_quant import fused_kv_per_tensor_quant
from atom.model_ops.utils import quant_fp8_per_tensor


@pytest.mark.parametrize("tokens", [0, 1, 17, 1001, 1024, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matches_dynamic_hip(tokens, dtype):
    torch.manual_seed(tokens)
    k = torch.randn((tokens, 12, 192), device="cuda", dtype=dtype)
    v = torch.randn((tokens, 12, 128), device="cuda", dtype=dtype) * 3
    k8, v8, ks, vs = fused_kv_per_tensor_quant(k, v)
    for x, actual, scale in ((k, k8, ks), (v, v8, vs)):
        assert actual.shape == x.shape and actual.is_contiguous()
        assert actual.dtype == torch.float8_e4m3fn
        assert scale.dtype == torch.float32 and scale.shape == (1,)
        if tokens:
            ref, ref_scale = quant_fp8_per_tensor(x)
            torch.testing.assert_close(actual.float(), ref.float(), rtol=0, atol=0)
            torch.testing.assert_close(scale, ref_scale, rtol=0, atol=0)
            torch.testing.assert_close(
                scale, x.float().abs().max().reshape(1) / 448, rtol=1e-6, atol=0
            )
        else:
            assert scale.item() == pytest.approx(1e-6)


@pytest.mark.parametrize("tokens", [1, 513, 4097])
def test_tail_guards_and_independent_amax(tokens):
    # Guard values would dominate amax if any masked load read past the input.
    tensors = []
    for dim, value in ((13, 2.0), (7, -9.0)):
        n = tokens * 3 * dim
        storage = torch.full((n + 256,), 128, device="cuda", dtype=torch.bfloat16)
        x = storage[:n].view(tokens, 3, dim)
        x.zero_()
        x.view(-1)[-1] = value
        tensors.append(x)
    k, v = tensors
    k8, v8, ks, vs = fused_kv_per_tensor_quant(k, v)
    for x, actual, scale, amax in ((k, k8, ks, 2), (v, v8, vs, 9)):
        assert scale.item() == pytest.approx(amax / 448)
        torch.testing.assert_close(actual.float() * scale, x.float(), rtol=1e-6, atol=0)


@pytest.mark.parametrize("tokens", [1, 1024])
def test_graph_replay_recomputes_scale_on_current_stream(tokens):
    k = torch.ones((tokens, 12, 192), device="cuda", dtype=torch.bfloat16)
    v = torch.ones((tokens, 12, 128), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused_kv_per_tensor_quant(k, v)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            k8, v8, ks, vs = fused_kv_per_tensor_quant(k, v)
        for kval, vval in ((3, -7), (0, 0), (1, 19)):
            k.fill_(kval)
            v.fill_(vval)
            graph.replay()
            stream.synchronize()
            for x, actual, scale, val in ((k, k8, ks, kval), (v, v8, vs, vval)):
                assert scale.item() == pytest.approx(abs(val) / 448 if val else 1e-6)
                torch.testing.assert_close(
                    actual.float() * scale, x.float(), rtol=1e-6, atol=0
                )
    torch.cuda.current_stream().wait_stream(stream)


def test_rejects_strided_input():
    k = torch.empty((5, 12, 256), device="cuda", dtype=torch.bfloat16)[..., :192]
    v = torch.empty((5, 12, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contiguous"):
        fused_kv_per_tensor_quant(k, v)


def test_rejects_mismatched_heads():
    k = torch.empty((5, 12, 192), device="cuda", dtype=torch.bfloat16)
    v = torch.empty((5, 16, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="matching tokens/heads"):
        fused_kv_per_tensor_quant(k, v)
