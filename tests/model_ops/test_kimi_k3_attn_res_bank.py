# SPDX-License-Identifier: MIT
"""Kimi-K3's block-residual bank: allocated once per forward, each closing
layer copies its prefix into a slot. Every layer's output matches the loop that
grew the candidates by `torch.cat`, bit for bit."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter", reason="atom.models.kimi_k3.model imports aiter")

from atom.model_ops import kimi_k3 as kimi_k3_ops
from atom.model_ops.attention_residual import AttnRes
from atom.models.kimi_k3 import model as kimi
from tests.models.test_kimi_k3_attn_res_rows import _torch_attn_res

LAYERS, BLOCK, HIDDEN, TOKENS = 25, 12, 64, 5


class _Norm(nn.Module):
    """RMSNorm in torch."""

    def __init__(self, dim, eps, **kwargs):
        super().__init__()
        self.weight = torch.ones(dim, dtype=torch.bfloat16)
        self.variance_epsilon = eps

    def forward(self, x):
        ms = x.float().pow(2).mean(-1, keepdim=True)
        y = x.float() * torch.rsqrt(ms + self.variance_epsilon)
        return (y * self.weight.float()).to(x.dtype)


class _Attention(nn.Module):
    """A KDA mixer: tanh of a projection."""

    fuse_input_norm_quant, input_quant_prefix = False, "self_attn.in_proj"

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x @ self.w)


class _Moe(nn.Module):
    """An MoE handing back its routed and shared outputs apart."""

    row_shardable = False

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x @ self.w, torch.tanh(x)


def _layers(monkeypatch):
    """`KimiDecoderLayer.__init__` around the stand-ins: KDA + MoE layers."""
    for name, stub in (
        ("RMSNorm", _Norm),
        ("ReplicatedLinear", lambda *args, **kwargs: nn.Identity()),
        ("KimiKDAAttention", _Attention),
        ("KimiSparseMoeBlock", _Moe),
    ):
        monkeypatch.setattr(kimi, name, stub)
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        rms_norm_eps=1e-6,
        num_experts=4,
        first_k_dense_replace=0,
        kimi_kda_layers=range(LAYERS),
        attn_res_block_size=BLOCK,
        num_hidden_layers=LAYERS,
    )
    layers = []
    for i in range(LAYERS):
        layer = kimi.KimiDecoderLayer(
            SimpleNamespace(hf_config=config, quant_config=None),
            prefix=f"layers.{i}",
            layer_num=i,
        )
        gen = torch.Generator().manual_seed(i)
        w = (torch.randn(2, HIDDEN, HIDDEN, generator=gen) / 8).bfloat16()
        layer.self_attn.w, layer.block_sparse_moe.w = w
        for site in (layer.self_attention_attn_res, layer.mlp_attn_res):
            site.out_norm.weight = (
                1 + 0.1 * torch.randn(HIDDEN, generator=gen)
            ).bfloat16()
            site.score_weight = torch.randn(HIDDEN, generator=gen)
        layers.append(layer)
    return layers, config


def _plain(site):
    """`site` mixing whatever candidates it is handed, as before the bank."""
    twin = AttnRes(site.proj, site.norm, site.out_norm)
    twin.score_weight = site.score_weight
    return twin


def _cat_reference(layers, x):
    """Each layer's `(prefix, routed, shared, candidates)` from the loop that
    appended a closing layer's prefix to the candidates with `torch.cat`."""
    bank = x.new_zeros(TOKENS, 0, HIDDEN)
    prefix, add, add2, outs = x, None, None, []
    for i, layer in enumerate(layers):
        h, prefix = _plain(layer.self_attention_attn_res)(prefix, bank, add, add2)
        if i % BLOCK == 0:
            bank, prefix = torch.cat([bank, prefix.unsqueeze(1)], dim=1), None
        h, prefix = _plain(layer.mlp_attn_res)(prefix, bank, layer.self_attn(h))
        add, add2 = layer.block_sparse_moe(h)
        outs.append((prefix, add, add2, bank))
    return outs


@torch.inference_mode()
def test_bank_layers_match_the_cat_loop(monkeypatch):
    monkeypatch.setattr(kimi_k3_ops, "apply_attn_res", _torch_attn_res)
    layers, config = _layers(monkeypatch)
    gen = torch.Generator().manual_seed(LAYERS)
    x = torch.randn(TOKENS, HIDDEN, generator=gen).bfloat16()
    bank = kimi.KimiLinearModel.block_bank(SimpleNamespace(config=config), x)
    assert bank.shape == (TOKENS, -(-LAYERS // BLOCK), HIDDEN)
    bank.fill_(float("nan"))  # a slot read before its block closes shows

    prefix, add, add2 = x, None, None
    for layer, want in zip(layers, _cat_reference(layers, x), strict=True):
        prefix, add, add2, out_bank = layer(None, prefix, bank, add, add2)
        assert out_bank is bank
        assert all(map(torch.equal, (prefix, add, add2), want[:3]))
        assert torch.equal(bank[:, : want[3].shape[1]], want[3])

    # the model's output site mixes the whole bank, every block closed
    final = _plain(layers[-1].mlp_attn_res)
    got = final(prefix, bank, add, add2)
    assert all(map(torch.equal, got, final(want[0], want[3], want[1], want[2])))
