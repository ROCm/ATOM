# SPDX-License-Identifier: MIT
"""CPU tests of M3 TP output exchange, residual ownership and fake schemas.

The attention/GEMM/transport boundaries are mocked. These tests check layout
and orchestration, not GPU kernel correctness or quantization accuracy.
"""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("aiter")

from atom.config import use_custom_atom_config
from atom.distributed import sp_kernels, ulysses_sp
from atom.model_ops.minimax_m3 import tp_o_proj as ops
from atom.models import minimax_m3 as model


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_qkv_stays_sharded_and_only_o_proj_loads_full_weights(
    monkeypatch, sparse, enabled
):
    from conftest import atom_config_double

    from atom.model_ops import linear

    cfg = SimpleNamespace(
        hidden_size=32,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        sparse_attention_config={
            "sparse_block_size": 128,
            "sparse_num_index_heads": 4,
            "sparse_index_dim": 8,
            "sparse_topk_blocks": 4,
            "sparse_attention_freq": [1],
        },
    )
    settings = atom_config_double(m3_tp_replicated_o_proj=enabled)
    monkeypatch.setattr(model, "Attention", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(model, "get_rope", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(model, "_minimax_m3_cos_sin_cache", lambda *args: None)
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: 4)
    cls = model.MiniMaxM3SparseAttention if sparse else model.MiniMaxM3Attention
    checkpoint = torch.arange(32 * 64).reshape(32, 64).to(torch.bfloat16)
    q_checkpoint = torch.arange(64 * 32).reshape(64, 32).to(torch.bfloat16)
    for rank in range(4):
        group = SimpleNamespace(world_size=4, rank_in_group=rank)
        monkeypatch.setattr(linear, "get_tp_group", lambda group=group: group)
        with use_custom_atom_config(settings), torch.no_grad():
            attention = cls(cfg, 0)
            attention.o_proj.weight_loader(attention.o_proj.weight, checkpoint)
            attention.qkv_proj.weight_loader(
                attention.qkv_proj.weight, q_checkpoint, "q"
            )
        assert attention.q_size == 16 and attention.kv_size == 8
        assert attention.qkv_proj.tp_dim == 0 and attention.qkv_proj.tp_size == 4
        torch.testing.assert_close(
            attention.qkv_proj.weight[:16],
            q_checkpoint[rank * 16 : (rank + 1) * 16].to(
                attention.qkv_proj.weight.dtype
            ),
        )
        expected = checkpoint if enabled else checkpoint[:, rank * 16 : (rank + 1) * 16]
        torch.testing.assert_close(
            attention.o_proj.weight, expected.to(attention.o_proj.weight.dtype)
        )
        assert isinstance(
            attention.o_proj,
            linear.ReplicatedLinear if enabled else linear.RowParallelLinear,
        )


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU-only test attempted GPU initialization")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


@pytest.mark.parametrize("tokens", [1, 3, 4, 5, 8193])
def test_attention_exchanges_only_output_and_preserves_real_metadata(
    monkeypatch, tokens
):
    width, world = 8, 4
    heads = torch.arange(tokens * width * world, dtype=torch.float32).reshape(
        tokens, world, width
    )
    padded = ops._pad_tokens(heads, world)
    local = padded.shape[0] // world
    positions = torch.arange(tokens)
    qkv = torch.zeros(tokens, width + 2)
    q, k, v = qkv[:, :width], qkv[:, -2:-1], qkv[:, -1:]
    monkeypatch.setattr(ulysses_sp, "_SP_WORLD_SIZE", 1)
    monkeypatch.setattr(ulysses_sp, "_prefer_all_gather", lambda *args: False)
    for rank in range(world):
        group = SimpleNamespace(world_size=world, rank_in_group=rank)
        monkeypatch.setattr(ops, "get_tp_group", lambda group=group: group)

        def attention(rank=rank, **kwargs):
            assert kwargs["query"] is q and kwargs["key"] is k and kwargs["value"] is v
            assert kwargs["qkv"] is qkv and kwargs["position"] is positions
            return heads[:, rank].contiguous()

        def exchange(output, source, received_group, *, group=group, rank=rank):
            assert received_group is group
            # A QKV input exchange would have different columns/data.
            torch.testing.assert_close(source.reshape(-1, width), padded[:, rank])
            start = rank * local
            output.copy_(padded[start : start + local].permute(1, 0, 2))

        monkeypatch.setattr(sp_kernels, "all_to_all_into", exchange)
        config = SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={
                    "attn": SimpleNamespace(impl=SimpleNamespace(forward=attention))
                }
            )
        )
        with use_custom_atom_config(config):
            actual, scale = ops.minimax_m3_tp_attention(
                q, k, v, positions, "attn", qkv, False, world
            )
        expected = padded[rank * local : (rank + 1) * local].reshape(local, -1)
        torch.testing.assert_close(actual, expected)
        assert scale.shape == (local, 1)
        fake, _ = ops._attention_fake(q, k, v, positions, "attn", qkv, False, world)
        assert fake.shape == actual.shape


@pytest.mark.parametrize("tokens", [0, 1, 2, 3, 4, 7, 32, 129])
def test_attention_is_added_once_before_replicated_residual(monkeypatch, tokens):
    world, width = 4, 8
    generator = torch.Generator().manual_seed(12)
    residual = torch.randn(tokens, width, generator=generator)
    attention = torch.randn(tokens, width, generator=generator)
    padded_attention = ops._pad_tokens(attention, world)
    partials = torch.randn(world, padded_attention.shape[0], width, generator=generator)
    expected = residual + attention + partials.sum(dim=0)[:tokens]
    contributions = []
    for rank in range(world):

        def reduce(x, **kwargs):
            assert kwargs == {"ca_fp8_quant": False}
            contributions.append(x.clone())
            return x.clone()

        group = SimpleNamespace(world_size=world, rank_in_group=rank, all_reduce=reduce)
        monkeypatch.setattr(ops, "get_tp_group", lambda group=group: group)
        owned = ops.minimax_m3_tp_local_residual(attention, world)
        result = ops.minimax_m3_tp_complete(partials[rank], owned, tokens, world)
        assert result.shape == (tokens, width)
    if tokens:
        # The accumulated replicated residual is added after the collective.
        torch.testing.assert_close(
            residual + torch.stack(contributions).sum(0), expected
        )
    else:
        assert contributions == []


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.float32])
def test_gather_preserves_bits_and_global_rows(monkeypatch, dtype):
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8).to(dtype)
    group = SimpleNamespace(world_size=4)
    monkeypatch.setattr(ops, "get_tp_group", lambda: group)

    def gather(value, *, group):
        assert group.world_size == 4
        if dtype.itemsize == 1:
            assert value.dtype == torch.bfloat16
        return value.repeat(4, 1)

    monkeypatch.setattr(ops, "_all_gather_tokens", gather)
    result = ops.minimax_m3_tp_gather(x, 4)
    assert result.dtype == dtype
    assert torch.equal(result.view(torch.uint8), x.repeat(4, 1).view(torch.uint8))
    assert ops._gather_fake(x, 4).shape == result.shape


@pytest.mark.parametrize("is_moe", [True, False])
@pytest.mark.parametrize("has_residual", [True, False])
def test_decoder_keeps_accumulated_residual_outside_allreduce(
    monkeypatch, is_moe, has_residual
):
    group = SimpleNamespace(world_size=4, rank_in_group=1)
    monkeypatch.setattr(ops, "get_tp_group", lambda: group)
    hidden = torch.ones(7, 8)
    completed = torch.full_like(hidden, 17)
    gather = Mock(side_effect=lambda x, world: x.repeat(world, 1))
    complete = Mock(return_value=completed)
    monkeypatch.setattr(torch.ops.aiter, "minimax_m3_tp_gather", gather)
    monkeypatch.setattr(torch.ops.aiter, "minimax_m3_tp_complete", complete)
    monkeypatch.setattr(
        torch.ops.aiter,
        "minimax_m3_tp_local_residual",
        ops.minimax_m3_tp_local_residual,
    )
    for name in (
        "fused_allreduce_gemma_rms_norm",
        "fused_allreduce_gemma_rms_norm_quant",
    ):
        monkeypatch.setattr(
            model, name, Mock(side_effect=AssertionError("extra all-reduce"))
        )
    moe = Mock(return_value=torch.ones(8, 8))
    mlp = Mock(side_effect=lambda x, x_scale: x)
    attention = Mock(return_value=torch.full((2, 8), 3.0))
    layer = SimpleNamespace(
        _tp_replicated_o_proj=True,
        _m3_fused_gemma_fp8=False,
        is_moe_layer=is_moe,
        self_attn=attention,
        input_layernorm=lambda x, r=None: x * 2 if r is None else ((x + r) * 2, x + r),
        post_attention_layernorm=lambda x, r: ((x + r) / 2, x + r),
        block_sparse_moe=moe,
        mlp=mlp,
    )
    layer._forward_tp_replicated_o_proj = MethodType(
        model.MiniMaxM3DecoderLayer._forward_tp_replicated_o_proj, layer
    )
    incoming = torch.full_like(hidden, 2) if has_residual else None
    output, residual = model.MiniMaxM3DecoderLayer.forward(
        layer, torch.arange(7), hidden, incoming
    )
    assert output is completed
    torch.testing.assert_close(
        residual, hidden if incoming is None else hidden + incoming
    )
    assert attention.call_args.kwargs["hidden_states"].shape == (7, 8)
    args = complete.call_args.args
    assert args[0].shape == (8, 8) and args[2:] == (7, 4)
    # Only the attention increment is reduced; the accumulated residual is not.
    torch.testing.assert_close(args[1], torch.full((2, 8), 3.0))
    if is_moe:
        moe.assert_called_once()
        gather.assert_not_called()
    else:
        gather.assert_called_once()
        mlp.assert_called_once()


def test_final_norm_adds_residual_without_another_reduction(monkeypatch):
    group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(model, "get_pp_group", lambda: group)
    monkeypatch.setattr(
        model,
        "fused_allreduce_gemma_rms_norm",
        Mock(side_effect=AssertionError("extra all-reduce")),
    )
    hidden = torch.ones(5, 8)
    layers = [
        lambda p, h, r: (h + 1, torch.full_like(h, 10)),
        lambda p, h, r: (h + 2, r),
    ]
    instance = SimpleNamespace(
        _tp_replicated_o_proj=True,
        start_layer=0,
        end_layer=2,
        layers=layers,
        aux_hidden_state_layers=(),
        norm=lambda x, r: ((x + r) * 3, x + r),
    )
    result = model.MiniMaxM3Model.forward(
        instance, None, torch.arange(5), inputs_embeds=hidden
    )
    torch.testing.assert_close(result, torch.full_like(hidden, 42))


def test_moe_has_separate_global_output_fake_schema():
    x = torch.empty((3, 8), device="meta", dtype=torch.bfloat16)
    logits = torch.empty((3, 128), device="meta")
    result = torch.ops.aiter.minimax_m3_tp_moe(x, logits, "unused", 4)
    assert result.shape == (12, 8) and result.dtype == torch.bfloat16
    completed = torch.ops.aiter.minimax_m3_tp_complete(result, x, 9, 4)
    assert completed.shape == (9, 8)


def test_symbolic_custom_ops_keep_original_layer_token_count():
    from torch.fx.experimental.proxy_tensor import make_fx

    def transitions(q, residual):
        attention, _ = torch.ops.aiter.minimax_m3_tp_attention(
            q, q, q, q[:, 0], "unused", q, True, 4
        )
        local_residual = torch.ops.aiter.minimax_m3_tp_local_residual(residual, 4)
        partial = torch.ops.aiter.minimax_m3_tp_moe(
            local_residual, attention, "unused", 4
        )
        return torch.ops.aiter.minimax_m3_tp_complete(
            partial, local_residual, q.shape[0], 4
        )

    graph = make_fx(transitions, tracing_mode="symbolic")(
        torch.empty(9, 2048), torch.empty(9, 6144)
    )
    # Execute the same graph on Meta tensors at decode and prefill sizes. A
    # specialized ceil(T/4) or shape-preserving MoE fake would fail this.
    for tokens in (1, 3, 16, 8193, 32768):
        output = graph(
            torch.empty(tokens, 2048, device="meta"),
            torch.empty(tokens, 6144, device="meta"),
        )
        assert output.shape == (tokens, 6144)


def test_moe_capacity_includes_dummy_rows_for_odd_token_budget():
    from atom.model_ops.fused_moe.config import moe_kernel_token_capacity

    config = SimpleNamespace(
        m3_tp_replicated_o_proj=True,
        tensor_parallel_size=4,
        max_num_batched_tokens=32769,
        sequence_parallel_size=1,
        enable_dp_attention=False,
    )
    assert moe_kernel_token_capacity(config, dp_size=1, use_all2all=False) == 32772
