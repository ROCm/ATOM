"""The recurrent GDN kernel must read packed Q/K/V views without repacking."""

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter")

from atom.model_ops.fla_ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)


@pytest.mark.parametrize(
    "num_seqs,window,query_heads,value_heads,key_dim,value_dim",
    [(1, 3, 2, 2, 16, 32), (16, 3, 8, 24, 128, 128)],
)
def test_packed_qkv_matches_contiguous(
    num_seqs, window, query_heads, value_heads, key_dim, value_dim
):
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")

    torch.manual_seed(19)
    tokens = num_seqs * window
    q_width = query_heads * key_dim
    v_width = value_heads * value_dim
    packed = torch.randn(
        1, tokens, 2 * q_width + v_width, device="cuda", dtype=torch.bfloat16
    )
    q, k, v = torch.split(packed, [q_width, q_width, v_width], dim=-1)
    q = q.view(1, tokens, query_heads, key_dim)
    k = k.view(1, tokens, query_heads, key_dim)
    v = v.view(1, tokens, value_heads, value_dim)
    assert not q.is_contiguous() and not k.is_contiguous() and not v.is_contiguous()

    gate = torch.randn(1, tokens, value_heads, device="cuda", dtype=torch.bfloat16)
    beta = torch.rand(1, tokens, value_heads, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(
        tokens + 1,
        value_heads,
        key_dim,
        value_dim,
        device="cuda",
        dtype=torch.float32,
    )
    indices = torch.arange(1, tokens + 1, device="cuda", dtype=torch.int32).view(
        num_seqs, window
    )
    cu_seqlens = torch.arange(0, tokens + 1, window, device="cuda", dtype=torch.int32)

    packed_out, packed_state = fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=state.clone(),
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
    )
    reference_out, reference_state = fused_recurrent_gated_delta_rule(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        gate,
        beta,
        initial_state=state.clone(),
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
    )
    torch.testing.assert_close(packed_out, reference_out, rtol=0, atol=0)
    torch.testing.assert_close(packed_state, reference_state, rtol=0, atol=0)

    output_buffer = torch.empty_like(reference_out)
    direct_out, direct_state = fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=state.clone(),
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
        out=output_buffer,
    )
    assert direct_out.data_ptr() == output_buffer.data_ptr()
    torch.testing.assert_close(direct_out, reference_out, rtol=0, atol=0)
    torch.testing.assert_close(direct_state, reference_state, rtol=0, atol=0)


def test_packed_qkv_matches_contiguous_without_varlen():
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")

    torch.manual_seed(23)
    batch, tokens, heads, dim = 2, 3, 2, 32
    width = heads * dim
    packed = torch.randn(batch, tokens, 3 * width, device="cuda", dtype=torch.bfloat16)
    q, k, v = (
        tensor.view(batch, tokens, heads, dim)
        for tensor in torch.split(packed, width, dim=-1)
    )
    gate = torch.randn(batch, tokens, heads, device="cuda", dtype=torch.bfloat16)
    beta = torch.rand(batch, tokens, heads, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(batch, heads, dim, dim, device="cuda", dtype=torch.float32)
    indices = (
        torch.arange(batch, device="cuda", dtype=torch.int32)
        .view(batch, 1)
        .expand(batch, tokens)
        .contiguous()
    )

    packed_out, packed_state = fused_recurrent_gated_delta_rule(
        q, k, v, gate, beta, initial_state=state.clone(), ssm_state_indices=indices
    )
    reference_out, reference_state = fused_recurrent_gated_delta_rule(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        gate,
        beta,
        initial_state=state.clone(),
        ssm_state_indices=indices,
    )
    torch.testing.assert_close(packed_out, reference_out, rtol=0, atol=0)
    torch.testing.assert_close(packed_state, reference_state, rtol=0, atol=0)
