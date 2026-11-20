# SPDX-License-Identifier: MIT
"""Kimi-K3's block-residual bank: allocated once per forward, each closing
layer copies its prefix into a slot. Every layer's output matches the loop that
grew the candidates by `torch.cat`, bit for bit."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter
from torch import nn

from atom.config import CompilationLevel


@pytest.fixture(scope="module")
def modules():
    # Initialize Torch's backend before the CPU import stand-ins expose Triton.
    from torch._dynamo.backends.registry import lookup_backend

    lookup_backend("eager")
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            ops=load("atom.model_ops.kimi_k3"),
            AttnRes=load("atom.model_ops.attention_residual").AttnRes,
            kimi=load("atom.models.kimi_k3.model"),
        )


def _torch_attn_res(
    prefix,
    block_residual,
    score_weight,
    eps,
    add_hidden=None,
    out_norm_weight=None,
    out_eps=1e-6,
    add_hidden2=None,
    quant_dtype=None,
):
    assert quant_dtype is None
    ps = prefix.float()
    for add in (add_hidden, add_hidden2):
        if add is not None:
            ps = ps + add.float()
    candidates = torch.cat([block_residual.float(), ps[:, None]], 1)
    scores = (candidates * score_weight).sum(-1) * torch.rsqrt(
        candidates.pow(2).mean(-1) + eps
    )
    mixed = (scores.softmax(-1)[..., None] * candidates).sum(1)
    mixed = mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + out_eps)
    mixed = mixed * out_norm_weight.float()
    return mixed.to(prefix.dtype), prefix if add_hidden is None else ps.to(prefix.dtype)


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


def _layers(monkeypatch, kimi):
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
    twin = type(site)(site.proj, site.norm, site.out_norm)
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
def test_bank_layers_match_the_cat_loop(monkeypatch, modules):
    monkeypatch.setattr(modules.ops, "apply_attn_res", _torch_attn_res)
    kimi = modules.kimi
    layers, config = _layers(monkeypatch, kimi)
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


def _model_without_layers(modules, block_size=BLOCK):
    model = object.__new__(modules.kimi.KimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        attn_res_block_size=block_size, num_hidden_layers=LAYERS, hidden_size=HIDDEN
    )
    return model


@pytest.mark.parametrize("block_size,slots", [(BLOCK, 3), (None, 0)])
def test_pp_placeholders_have_a_full_zeroed_bank(
    modules, monkeypatch, block_size, slots
):
    kimi = modules.kimi
    monkeypatch.setattr(kimi, "_normalize_kimi_config", lambda config: None)
    monkeypatch.setattr(kimi, "pp_holds_embedding", lambda config: False)
    monkeypatch.setattr(
        kimi, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    monkeypatch.setattr(
        kimi, "make_layers", lambda *args, **kwargs: (0, 0, nn.ModuleList())
    )
    config = SimpleNamespace(
        vocab_size=128,
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        attn_res_block_size=block_size,
        num_shared_experts=0,
    )
    model = kimi.KimiLinearModel(
        SimpleNamespace(
            hf_config=config,
            quant_config=None,
            compilation_config=SimpleNamespace(level=CompilationLevel.NO_COMPILATION),
        )
    )
    values = model.make_empty_intermediate_tensors(
        TOKENS, torch.bfloat16, torch.device("cpu")
    )
    assert values["hidden_states"].shape == (TOKENS, HIDDEN)
    assert values["block_residual"].shape == (TOKENS, slots, HIDDEN)
    for name in ("hidden_states", "block_residual"):
        value = values[name]
        assert value.dtype == torch.bfloat16 and value.device.type == "cpu"
        assert value.storage_offset() == 0
        assert torch.count_nonzero(value).item() == 0


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("slot", [0, 1, 2])
def test_close_writes_only_the_slot_in_the_bank_view(modules, compiled, offset, slot):
    site = modules.AttnRes(
        nn.Identity(), nn.Identity(), block_size=BLOCK, layer_idx=slot * BLOCK
    )

    def close(prefix, bank):
        return site.maybe_close_block(prefix, bank)

    if compiled:
        close = torch.compile(close, backend="eager", fullgraph=True, dynamic=True)
    for tokens in (2, TOKENS):
        storage = torch.arange((tokens + 2) * 3 * HIDDEN, dtype=torch.float32)
        storage = storage.reshape(tokens + 2, 3, HIDDEN).bfloat16()
        bank = storage[offset : offset + tokens]
        prefix = -(torch.arange(tokens * HIDDEN).reshape(tokens, HIDDEN) + 1)
        prefix = prefix.bfloat16()
        expected = storage.clone()
        expected[offset : offset + tokens, slot] = prefix

        out_bank, remainder = close(prefix, bank)

        assert out_bank is bank
        assert remainder is None
        # Whole backing storage catches writes to neighboring rows or slots.
        torch.testing.assert_close(storage, expected, rtol=0, atol=0)


def test_new_bank_slot_copy_stays_in_one_dynamo_graph(modules):
    model = _model_without_layers(modules)
    site = modules.AttnRes(nn.Identity(), nn.Identity(), block_size=BLOCK, layer_idx=0)

    def close_first(prefix):
        bank = model.block_bank(prefix)
        bank.fill_(float("nan"))
        bank, remainder = site.maybe_close_block(prefix, bank)
        assert remainder is None
        return bank[:, 0]

    compiled = torch.compile(close_first, backend="eager", fullgraph=True, dynamic=True)
    for tokens in (2, 5):
        prefix = torch.randn(tokens, HIDDEN).bfloat16()
        torch.testing.assert_close(compiled(prefix), prefix, rtol=0, atol=0)
