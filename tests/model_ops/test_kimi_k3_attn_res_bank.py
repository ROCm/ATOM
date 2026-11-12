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
    """An RMSNorm in torch, for the site with no candidates yet."""

    def __init__(self, gen):
        super().__init__()
        self.weight = (1 + 0.1 * torch.randn(HIDDEN, generator=gen)).bfloat16()
        self.variance_epsilon = 1e-6

    def forward(self, x):
        ms = x.float().pow(2).mean(-1, keepdim=True)
        y = x.float() * torch.rsqrt(ms + self.variance_epsilon)
        return (y * self.weight.float()).to(x.dtype)


def _site(gen, block_size=None, layer_idx=0):
    norm = _Norm(gen)
    site = AttnRes(SimpleNamespace(), norm, norm, True, block_size, layer_idx)
    site.score_weight = torch.randn(HIDDEN, generator=gen)
    return site


def _plain(site):
    """`site` mixing whatever candidates it is handed, as before the bank."""
    twin = AttnRes(site.proj, site.norm, site.out_norm, True)
    twin.score_weight = site.score_weight
    return twin


class _Layer(kimi.KimiDecoderLayer):
    """`KimiDecoderLayer.forward` over stand-ins: a tanh mixer, and an FFN that
    returns a shared output on odd layers, as an MoE layer does."""

    is_linear_attn, post_attention_op = True, False

    def __init__(self, layer_idx):
        nn.Module.__init__(self)
        gen = torch.Generator().manual_seed(layer_idx)
        self.self_attention_attn_res = _site(gen, BLOCK, layer_idx)
        self.mlp_attn_res = _site(gen)
        self.w = (torch.randn(2, HIDDEN, HIDDEN, generator=gen) / 8).bfloat16()
        self.moe = layer_idx % 2 == 1

    def self_attn(self, x):
        return torch.tanh(x @ self.w[0])

    def _ffn(self, x):
        return x @ self.w[1], torch.tanh(x) if self.moe else None


def _cat_reference(layers, x):
    """Each layer's output from the loop that appended a closing layer's
    prefix to the candidates with `torch.cat`."""
    bank = x.new_zeros(TOKENS, 0, HIDDEN)
    prefix, add, add2, outs = x, None, None, []
    for i, layer in enumerate(layers):
        h, prefix = _plain(layer.self_attention_attn_res)(prefix, bank, add, add2)
        if i % BLOCK == 0:
            bank, prefix = torch.cat([bank, prefix.unsqueeze(1)], dim=1), None
        prefix, add, add2 = layer._post_attention(layer.self_attn(h), prefix, bank)
        outs.append((prefix, add, add2, bank))
    return outs


def _same(a, b):
    return a is b if a is None or b is None else torch.equal(a, b)


@torch.inference_mode()
def test_bank_layers_match_the_cat_loop(monkeypatch):
    monkeypatch.setattr(kimi_k3_ops, "apply_attn_res", _torch_attn_res)
    layers = [_Layer(i) for i in range(LAYERS)]
    gen = torch.Generator().manual_seed(LAYERS)
    x = torch.randn(TOKENS, HIDDEN, generator=gen).bfloat16()
    config = SimpleNamespace(num_hidden_layers=LAYERS, attn_res_block_size=BLOCK)
    bank = kimi.KimiLinearModel.block_bank(SimpleNamespace(config=config), x)
    assert bank.shape == (TOKENS, -(-LAYERS // BLOCK), HIDDEN)
    bank.fill_(float("nan"))  # a slot read before its block closes shows

    prefix, add, add2 = x, None, None
    for layer, want in zip(layers, _cat_reference(layers, x), strict=True):
        prefix, add, add2, out_bank = layer(None, prefix, bank, add, add2)
        assert out_bank is bank
        assert all(map(_same, (prefix, add, add2), want[:3]))
        assert torch.equal(bank[:, : want[3].shape[1]], want[3])

    # the model's output site mixes the whole bank, every block closed
    final = _site(gen)
    got = final(prefix, bank, add, add2)
    assert all(map(_same, got, final(want[0], want[3], want[1], want[2])))
