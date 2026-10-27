# SPDX-License-Identifier: MIT
import os

import pytest
import torch

from . import oracle_kernels as oracle
from .reference import load_reference


def test_fp4_nibble_order_and_signed_values():
    packed = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8
    )
    expected = torch.tensor(
        [[0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]]
    )
    torch.testing.assert_close(oracle.unpack_fp4(packed), expected, rtol=0, atol=0)
    assert torch.signbit(oracle.unpack_fp4(packed)[0, 8])


def test_fp4_round_to_even_and_nonzero_zero_scale():
    values = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6] * 4])
    packed, scale = oracle.fp4_act_quant(values)
    assert scale.float().item() == 1
    torch.testing.assert_close(
        oracle.unpack_fp4(packed)[0, :8],
        torch.tensor([0.0, 1, 1, 2, 2, 4, 4, 6]),
        rtol=0,
        atol=0,
    )
    _, scale = oracle.fp4_act_quant(
        torch.zeros(1, 16), 16, scale_dtype=torch.float8_e4m3fn
    )
    assert scale.float().item() == 2**-9


def test_fp8_activation_scale_rounds_up():
    values = torch.full((1, 32), 449.0)
    quant, scale = oracle.act_quant(values, 32, "ue8m0", torch.float8_e8m0fnu)
    assert scale.float().item() == 2
    assert quant.float().amax() <= 448


def test_sparse_attention_counts_sink_once_and_retains_duplicate_entries():
    query = torch.zeros(1, 1, 2, 32, dtype=torch.bfloat16)
    values = torch.full((1, 1, 32), 6.0, dtype=torch.bfloat16)
    indices = torch.tensor([[[0, 0, -1]]], dtype=torch.int32)
    actual = oracle.sparse_attn(query, values, torch.zeros(2), indices, 32**-0.5)
    torch.testing.assert_close(actual, torch.full_like(query, 4), rtol=0, atol=0)
    assert not oracle.sparse_attn(
        query, values, torch.zeros(2), indices.fill_(-1), 32**-0.5
    ).any()


def test_sinkhorn_positive_mixes_and_stochastic_residual():
    generator = torch.Generator().manual_seed(41)
    mixes = torch.randn(2, 3, 24, generator=generator)
    pre, post, comb = oracle.hc_split_sinkhorn(mixes, torch.ones(3), torch.zeros(24))
    assert (pre > 0).all() and (post > 0).all()
    torch.testing.assert_close(comb.sum(-1), torch.ones(2, 3, 4), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(comb.sum(-2), torch.ones(2, 3, 4), rtol=1e-5, atol=1e-5)


def test_pinned_official_model_prefill_decode():
    model_dir = os.environ.get("ATOM_DSV41_REFERENCE")
    if not model_dir:
        pytest.skip("Set ATOM_DSV41_REFERENCE to the pinned HF snapshot")
    with load_reference(model_dir) as reference, reference.set_dtype(torch.bfloat16):
        args = reference.ModelArgs(
            max_batch_size=1,
            max_seq_len=16,
            temperature=0,
            dim=64,
            vocab_size=128,
            moe_inter_dim=64,
            n_layers=5,
            n_heads=4,
            q_lora_rank=32,
            head_dim=64,
            rope_head_dim=32,
            o_groups=4,
            o_lora_rank=32,
            n_routed_experts=4,
            n_activated_experts=2,
            window_size=4,
            compress_ratios=(0, 2, 2, 1, 1, 0),
            kv_source_layers=(1, 3),
            index_source_layers=(1, 3, 4),
            index_n_heads=4,
            index_head_dim=32,
            index_topk=4,
            candidate_source_layer=3,
            candidate_topk_blocks=2,
            candidate_block_size=2,
            swiglu_limit=10,
        )
        model = reference.Transformer(args)
        torch.manual_seed(41)
        with torch.no_grad():
            for name, value in model.named_parameters():
                if value.dtype == torch.float4_e2m1fn_x2:
                    value.view(torch.uint8).copy_(
                        torch.randint(0, 256, value.shape, dtype=torch.uint8)
                    )
                elif value.dtype == torch.float8_e8m0fnu:
                    value.copy_(torch.full(value.shape, 2**-5))
                elif name.endswith("norm.weight"):
                    value.fill_(1)
                elif "hc_" in name or name.endswith(("bias", "attn_sink")):
                    value.zero_()
                else:
                    value.copy_(
                        (torch.randn(value.shape).float() * 0.2).to(value.dtype)
                    )
        tokens = torch.tensor([[3, 5, 7, 11]])
        model(tokens[:, :3], 0)
        _, decoded, _ = model(tokens[:, 3:], 3)
        _, prefilled, _ = model(tokens, 0)
        assert torch.isfinite(decoded).all()
        torch.testing.assert_close(decoded, prefilled, rtol=1e-2, atol=1e-2)
