# SPDX-License-Identifier: MIT
"""The vLLM attention override preserves native caller-owned output slices."""

from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip("aiter", reason="the native V4 model reaches AITER ops")
pytest.importorskip("vllm", reason="the plugin imports the vLLM metadata bridge")

from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import QKNormRopeOut
from atom.models import deepseek_v4 as native
from atom.plugin.vllm.models import deepseek_v4 as plugin
from atom.utils.forward_context import AttnState


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    return request.param


@pytest.fixture
def layer():
    # No weights or KV pool are needed to exercise the model's dispatch and
    # buffer contract. Only the final attention computation is replaced below.
    layer = plugin.DeepseekV4AttentionVllm.__new__(plugin.DeepseekV4AttentionVllm)
    torch.nn.Module.__init__(layer)
    layer.n_local_heads = 2
    layer.head_dim = 2
    return layer


@pytest.mark.parametrize("path", ["padded_prefill", "prefill", "decode", "no_metadata"])
@pytest.mark.parametrize("supplied_output", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
def test_sparse_output_clipping_and_padding(
    monkeypatch, device, layer, path, supplied_output, transposed
):
    num_in = 6
    num_real = 3 if path == "padded_prefill" else num_in
    metadata = NS(
        state=AttnState.DECODE if path == "decode" else AttnState.PREFILL_NATIVE,
        # A decode graph must keep its padded rows despite a shorter bid tensor.
        batch_id_per_q_token=torch.zeros(3 if path == "decode" else num_real),
    )
    fc = NS(
        context=NS(is_dummy_run=False),
        attn_metadata=None if path == "no_metadata" else metadata,
    )
    monkeypatch.setattr(plugin, "get_forward_context", lambda: fc)
    query = torch.arange(24, device=device, dtype=torch.bfloat16).reshape(6, 2, 2)
    qkn = QKNormRopeOut(q_sa=query, kv=query[:, 0])
    positions = torch.arange(num_in, device=device)
    tensors = [torch.ones(num_in, 2, device=device) for _ in range(6)]
    parent = torch.full((num_in + 2, 2, 2), -9.0, device=device)
    if transposed:
        parent = parent.transpose(1, 2)
    output = parent[1:-1] if supplied_output else None
    calls = []

    def sparse(self, clipped_qkn, clipped_positions, *args, out=None):
        calls.append(clipped_positions.numel())
        torch.testing.assert_close(clipped_qkn.q_sa, query[:num_real])
        torch.testing.assert_close(clipped_qkn.kv, qkn.kv[:num_real])
        torch.testing.assert_close(clipped_positions, positions[:num_real])
        assert all(t.shape[0] == num_real for t in args)
        if supplied_output:
            assert out.data_ptr() == output.data_ptr()
            assert out.shape == (num_real, 2, 2)
        else:
            assert out is None
            out = torch.empty(num_real, 2, 2, device=device)
        out.copy_(query[:num_real] + 10)
        return out.reshape(num_real, -1)

    monkeypatch.setattr(native.DeepseekV4Attention, "_sparse_attention", sparse)
    kwargs = {"out": output} if supplied_output else {}
    result = layer._sparse_attention(qkn, positions, *tensors, **kwargs)
    expected = torch.zeros(num_in, 2, 2, device=device)
    expected[:num_real] = query[:num_real] + 10
    torch.testing.assert_close(result, expected.view(num_in, -1))
    assert calls == [num_real]
    if supplied_output:
        assert (result.data_ptr() == output.data_ptr()) == (not transposed)
        torch.testing.assert_close(output, expected)
    else:
        assert torch.all(parent == -9)
    assert torch.all(parent[[0, -1]] == -9)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_dummy_forward_zeros_the_supplied_slice(
    monkeypatch, device, layer, transposed, bypass
):
    fc = NS(context=NS(is_dummy_run=not bypass), attn_metadata=None)
    monkeypatch.setenv("ATOM_V4_BYPASS_ATTN", "1" if bypass else "0")
    monkeypatch.setattr(plugin, "get_forward_context", lambda: fc)
    monkeypatch.setattr(native, "get_forward_context", lambda: fc)
    parent = torch.full((5, 2, 2), -9.0, device=device)
    if transposed:
        parent = parent.transpose(1, 2)
    output = parent[1:4]
    qkn = QKNormRopeOut(q_sa=torch.ones_like(output))
    result = layer._sparse_attention(qkn, torch.arange(3, device=device), out=output)
    assert result.shape == (3, 4)
    assert (result.data_ptr() == output.data_ptr()) == (not transposed)
    assert torch.all(result == 0)
    assert torch.all(output == 0) and torch.all(parent[[0, -1]] == -9)
