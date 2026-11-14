# SPDX-License-Identifier: MIT
"""Kimi-K3 latent MoE at decode on simulated TP8 ranks: up to 64 rows, after
the one all-reduce, each rank runs `latent_tail` on its own columns and an
all-gather along the hidden dim hands back the routed + shared sum, which the
next layer's attention residual takes with no second addend. Above 64 rows
the block keeps its tail."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter", reason="atom.models.kimi_k3.model imports aiter")

from atom.model_ops import kimi_k3 as kimi_k3_ops
from atom.model_ops import module_dispatch_ops
from atom.model_ops.kimi_k3 import attention_residual, latent_tail
from atom.models import qwen4_exp
from atom.models.kimi_k3 import model as kimi
from tests.models.test_kimi_k3_attn_res_rows import _torch_attn_res
from tests.models.test_kimi_k3_moe_row_tail import (  # noqa: F401
    WORLD,
    _folded,
    ranks,
    split_op_on_cpu,
)

# The smallest dims latent_tail tiles at TP8.
HIDDEN, LATENT, EXPERTS = WORLD * 32, 7 * 256, 4
COLS = HIDDEN // WORLD


class _Stub(nn.Module):
    """Stands in for a sub-module class; the test sets `fn` after `__init__`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class _Attention(_Stub):
    """A KDA mixer whose o_proj partial sum is this rank's share of its input."""

    fuse_input_norm_quant = False
    input_quant_prefix = "self_attn.in_proj"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.o_proj = SimpleNamespace(reduce_results=True)
        self.fn = lambda x: x / WORLD


def _rms_norm(x, weight, eps):
    x = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).bfloat16()


def _attn_res(prefix, bank, score_weight, eps, add, norm_weight, out_eps, add2, q):
    """`apply_attn_res` in torch; like the kernel, it skips a folded zero."""
    if add2 is not None and attention_residual._is_folded_zero(add2):
        add2 = None
    return _torch_attn_res(
        prefix, bank, score_weight, eps, add, norm_weight, out_eps, add2, q
    )


@pytest.fixture(autouse=True)
def tail_calls(monkeypatch, ranks, split_op_on_cpu):  # noqa: F811
    """Stub sub-modules at TP8 with dual-stream off
    (ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0) and an enabled custom all-gather;
    `latent_tail` in torch, logging each call's rank and operands."""
    for name in ("ReplicatedLinear", "FusedMoE", "KimiMLP", "RMSNorm"):
        monkeypatch.setattr(kimi, name, _Stub)
    monkeypatch.setattr(kimi, "MergedBF16Linears", _Stub)
    monkeypatch.setattr(kimi, "KimiKDAAttention", _Attention)
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_world_size", lambda: WORLD)
    config = SimpleNamespace(
        enable_tbo=False,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(kimi, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(module_dispatch_ops, "get_current_atom_config", lambda: config)
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    monkeypatch.setenv("ATOM_USE_CUSTOM_ALL_GATHER", "1")
    group = SimpleNamespace(
        device_communicator=SimpleNamespace(ca_comm=SimpleNamespace(disabled=False))
    )
    monkeypatch.setattr(kimi, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        torch.ops.aiter,
        "kimi_k3_post_attention_forward",
        module_dispatch_ops.kimi_k3_post_attention_forward,
    )
    monkeypatch.setattr(kimi_k3_ops, "apply_attn_res", _attn_res)

    def all_gather(x, use_custom=False, dim=0):
        assert use_custom and dim == -1
        return torch.cat(qwen4_exp.get_tp_group().all_gather(x).chunk(WORLD), -1)

    monkeypatch.setattr(
        kimi, "tensor_model_parallel_all_gather", all_gather, raising=False
    )
    calls = []

    def column_tail(latent, norm_weight, eps, up_proj_rows, shared):
        calls.append((kimi.get_tensor_model_parallel_rank(), up_proj_rows, shared))
        x = _rms_norm(latent, norm_weight, eps).float()
        return (x @ up_proj_rows.float().T + shared.float()).bfloat16()

    monkeypatch.setattr(latent_tail, "latent_tail", column_tail)
    return calls


def _weights():
    gen = torch.Generator().manual_seed(5)

    def rand(fan_in, *shape):
        return (torch.randn(*shape, generator=gen) / fan_in**0.5).bfloat16()

    return {
        "gate": rand(HIDDEN, HIDDEN, EXPERTS),
        "down": rand(HIDDEN, HIDDEN, LATENT),
        "shared": rand(HIDDEN, HIDDEN, HIDDEN),
        "up": rand(LATENT, HIDDEN, LATENT),
        "norm": (1 + 0.1 * torch.randn(LATENT, generator=gen)).bfloat16(),
    }


def _blocks(w):
    """A merged-front latent block per rank, whose stub experts and shared
    down_proj write this rank's share of the partial sums."""
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_token=2,
        routed_expert_hidden_size=LATENT,
        moe_intermediate_size=HIDDEN,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_shared_experts=1,
        latent_moe_use_norm=True,
        rms_norm_eps=1e-6,
    )
    blocks = []
    for rank in range(WORLD):
        b = kimi.KimiSparseMoeBlock(
            config, prefix=f"rank{rank}.mlp", alt_stream=object()
        )
        share = (rank + 1) / (WORLD * (WORLD + 1) / 2)
        b.front_proj.merged_weight = True
        b.front_proj.fn = lambda x: (x, x @ w["gate"], x @ w["down"])
        b.experts.fn = lambda x, logits, out, s=share: out.copy_(
            s * torch.tanh(x * logits.sum(-1, keepdim=True))
        )
        b.shared_experts.act_and_down_proj = lambda x, out, s=share: out.copy_(
            s * torch.tanh(x @ w["shared"])
        )
        b.shared_experts.down_proj = SimpleNamespace(supports_out=lambda: True)
        b.routed_expert_norm.weight, b.routed_expert_norm.eps = w["norm"], 1e-6
        b.routed_expert_norm.fn = lambda x: _rms_norm(x, w["norm"], 1e-6)
        b.routed_expert_up_proj.weight = w["up"]
        b.routed_expert_up_proj.fn = lambda x: x @ w["up"].T
        blocks.append(b)
    return blocks


def _is_sum(output, routed, shared):
    """Whether `output` is `routed + shared` up to two BF16 roundings: the
    current tail rounds `routed`, the column tail the sum."""
    exact = routed.float() + shared.float()
    bound = 2**-7 * (routed.float().abs() + shared.float().abs())
    return bool(((output.float() - exact).abs() <= bound).all())


@pytest.mark.parametrize("tokens", [1, 64, 65])
def test_block_gathers_the_column_tail(ranks, tail_calls, tokens):  # noqa: F811
    w = _weights()
    blocks = _blocks(w)
    x = torch.randn(tokens, HIDDEN, generator=torch.Generator().manual_seed(tokens))
    x = x.bfloat16()

    def on_rank(rank):
        block = blocks[rank]
        block.column_tail_rows, rows = 0, getattr(block, "column_tail_rows", 0)
        base = block.split_moe_forward(x)
        block.column_tail_rows = rows
        return base, block.split_moe_forward(x), block(x)

    results = ranks(on_rank)
    if tokens > latent_tail.MAX_TOKENS:
        assert not tail_calls
        for base, split, op in results:
            assert all(map(torch.equal, base + op, split + split))
        return
    # One call per rank from `split_moe_forward`, one through the split op.
    assert sorted(rank for rank, *_ in tail_calls) == sorted(2 * [*range(WORLD)])
    for rank, up_proj_rows, shared_cols in tail_calls:
        cols = slice(rank * COLS, (rank + 1) * COLS)
        assert torch.equal(up_proj_rows, w["up"][cols])
        assert torch.equal(shared_cols, results[rank][0][1][:, cols])
    for (routed, shared), (output, none), (first, second) in results:
        assert none is None and torch.equal(first, output) and _folded(second)
        assert _is_sum(output, routed, shared)


@pytest.mark.parametrize(
    "custom_gather, ca_comm",
    [
        ("0", SimpleNamespace(disabled=False)),
        ("1", None),
        ("1", SimpleNamespace(disabled=True)),
    ],
    ids=["variable-off", "no-ca-comm", "ca-comm-disabled"],
)
def test_block_keeps_the_replicated_tail_without_custom_gather(
    monkeypatch, ranks, tail_calls, custom_gather, ca_comm  # noqa: F811
):
    """With `ATOM_USE_CUSTOM_ALL_GATHER=0` or no enabled `ca_comm`, where
    aiter's custom all-gather raises, every rank keeps the replicated tail."""
    monkeypatch.setenv("ATOM_USE_CUSTOM_ALL_GATHER", custom_gather)
    group = SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=ca_comm))
    monkeypatch.setattr(kimi, "get_tp_group", lambda: group)
    blocks = _blocks(_weights())
    assert all(block.column_tail_rows == 0 for block in blocks)
    x = torch.randn(8, HIDDEN, generator=torch.Generator().manual_seed(8)).bfloat16()
    results = ranks(lambda rank: blocks[rank].split_moe_forward(x))
    assert not tail_calls and all(shared is not None for _, shared in results)


def _layers(blocks, layer_num=5):
    """A KDA MoE decoder layer per rank around `blocks`, mid attention-
    residual block (size 4) so that it banks nothing."""
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        rms_norm_eps=1e-6,
        num_experts=EXPERTS,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        kimi_kda_layers=[layer_num],
        attn_res_block_size=4,
    )
    gen = torch.Generator().manual_seed(layer_num)
    score_weights = torch.randn(2, HIDDEN, generator=gen)
    norm_weights = (1 + 0.1 * torch.randn(2, HIDDEN, generator=gen)).bfloat16()
    layers = []
    for rank, block in enumerate(blocks):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(kimi, "KimiSparseMoeBlock", lambda *a, b=block, **k: b)
            layer = kimi.KimiDecoderLayer(
                SimpleNamespace(hf_config=config, quant_config=None),
                prefix=f"rank{rank}.layers.{layer_num}",
                layer_num=layer_num,
            )
        layer.self_attention_attn_res.score_weight = score_weights[0]
        layer.mlp_attn_res.score_weight = score_weights[1]
        layer.input_layernorm.weight = norm_weights[0]
        layer.post_attention_layernorm.weight = norm_weights[1]
        layers.append(layer)
    return layers


@pytest.mark.parametrize("tokens", [8, 65])
def test_next_layer_takes_the_gathered_sum(ranks, tokens):  # noqa: F811
    """A layer's pending adds fed to the next through `KimiDecoderLayer.forward`:
    up to 64 rows, its second is a folded zero, and the next layer's output
    is the one it gives with `pending_add2=None`."""
    blocks = _blocks(_weights())
    layers = _layers(blocks)
    gen = torch.Generator().manual_seed(tokens)
    x, add, add2 = torch.randn(3, tokens, HIDDEN, generator=gen).bfloat16()
    bank = torch.randn(tokens, 2, HIDDEN, generator=gen).bfloat16()

    def two_layers(rank):
        prefix, hidden, shared, bank_out = layers[rank](None, x, bank, add, add2)
        folded = attention_residual._is_folded_zero(shared)
        if folded:
            shared.fill_(float("nan"))  # the next attention residual skips it
        nxt = layers[rank](None, prefix, bank_out, hidden, shared)
        alone = layers[rank](None, prefix, bank_out, hidden) if folded else None
        return (hidden, shared), nxt, alone

    got = ranks(two_layers)
    for block in blocks:
        block.column_tail_rows = 0
    base = ranks(two_layers)
    for ((hidden, shared), nxt, alone), ((routed, base_shared), want, _) in zip(
        got, base
    ):
        if tokens > latent_tail.MAX_TOKENS:
            assert alone is None and torch.equal(hidden, routed)
            assert torch.equal(shared, base_shared)
            assert all(map(torch.equal, nxt, want))
            continue
        assert alone is not None and _is_sum(hidden, routed, base_shared)
        # Its own second addend is a folded zero too, never written.
        assert attention_residual._is_folded_zero(nxt[2])
        assert all(torch.equal(nxt[i], alone[i]) for i in (0, 1, 3))
