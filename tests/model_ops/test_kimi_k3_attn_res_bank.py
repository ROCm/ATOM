# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Kimi-K3's block-residual bank: allocated once per forward, each closing
layer copies its prefix into a slot. Every layer's output matches the loop that
grew the candidates by `torch.cat`, bit for bit."""

import ast
from contextlib import nullcontext
from pathlib import Path
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
    num_blocks=None,
):
    assert quant_dtype is None
    if num_blocks is not None:
        block_residual = block_residual[:, :num_blocks]
    ps = prefix.float()
    for add in (add_hidden, add_hidden2):
        if add is not None:
            ps = ps + add.float()
    candidates = torch.cat([block_residual.float(), ps[:, None]], 1)
    scores = (candidates * score_weight).sum(-1) * torch.rsqrt(
        candidates.pow(2).mean(-1) + eps
    )
    mixed = (scores.softmax(-1)[..., None] * candidates).sum(1)
    if out_norm_weight is not None:
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
        vocab_size=128,
        num_shared_experts=0,
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


def _cat_reference(layers, x, stage_ends=()):
    """Each layer's `(prefix, routed, shared, candidates)` from the loop that
    appended a closing layer's prefix to the candidates with `torch.cat`."""
    bank = x.new_zeros(x.shape[0], 0, HIDDEN)
    prefix, add, add2, outs = x, None, None, []
    for i, layer in enumerate(layers):
        h, prefix = _plain(layer.self_attention_attn_res)(prefix, bank, add, add2)
        if i % BLOCK == 0:
            bank, prefix = torch.cat([bank, prefix.unsqueeze(1)], dim=1), None
        h, prefix = _plain(layer.mlp_attn_res)(prefix, bank, layer.self_attn(h))
        add, add2 = layer.block_sparse_moe(h)
        outs.append((prefix, add, add2, bank))
        if i + 1 in stage_ends:
            prefix, add, add2 = prefix + add + add2, None, None
    return outs


@torch.inference_mode()
def test_bank_layers_match_the_cat_loop(monkeypatch, modules):
    monkeypatch.setattr(modules.ops, "apply_attn_res", _torch_attn_res)
    kimi = modules.kimi
    layers, config = _layers(monkeypatch, kimi)
    gen = torch.Generator().manual_seed(LAYERS)
    x = torch.randn(TOKENS, HIDDEN, generator=gen).bfloat16()
    bank = kimi.KimiLinearModel.block_bank(
        SimpleNamespace(config=config, start_layer=0), x
    )
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
    model.start_layer = 0
    return model


@pytest.mark.parametrize(
    "start,block_size,slots",
    [(0, BLOCK, 0), (12, BLOCK, 1), (13, BLOCK, 2), (13, None, 0)],
)
def test_pp_placeholders_hold_only_the_incoming_prefix(
    modules, monkeypatch, start, block_size, slots
):
    kimi = modules.kimi
    monkeypatch.setattr(kimi, "_normalize_kimi_config", lambda config: None)
    monkeypatch.setattr(kimi, "pp_holds_embedding", lambda config: False)
    monkeypatch.setattr(
        kimi, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False, world_size=2)
    )
    monkeypatch.setattr(
        kimi, "make_layers", lambda *args, **kwargs: (start, start, nn.ModuleList())
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
            enable_tbo=False,
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


def _model_stage(monkeypatch, modules, layers, config, start, end, model_class=None):
    kimi = modules.kimi
    group = SimpleNamespace(
        is_first_rank=start == 0,
        is_last_rank=end == LAYERS,
        world_size=1 if start == 0 and end == LAYERS else 2,
    )
    monkeypatch.setattr(kimi, "get_pp_group", lambda: group)
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(kimi, "pp_holds_embedding", lambda config: group.is_first_rank)
    monkeypatch.setattr(kimi, "_normalize_kimi_config", lambda config: None)
    monkeypatch.setattr(
        kimi, "make_layers", lambda *a, **k: (start, end, nn.ModuleList(layers))
    )
    monkeypatch.setattr(
        kimi,
        "VocabParallelEmbedding",
        lambda v, h: nn.Embedding(v, h, dtype=torch.bfloat16),
    )
    model = (model_class or kimi.KimiLinearModel)(
        SimpleNamespace(
            hf_config=config,
            quant_config=None,
            enable_tbo=False,
            compilation_config=SimpleNamespace(level=CompilationLevel.NO_COMPILATION),
        )
    )
    assert isinstance(model._column_gather_setup, kimi._ColumnGatherSetup)
    assert list(model.children())[-1] is model._column_gather_setup
    if group.is_last_rank:
        model.output_attn_res.score_weight = torch.ones(HIDDEN)
    return model


def _vllm_forward_class(modules):
    # Run the plugin's actual forwarding code without importing a vLLM engine.
    path = Path(modules.kimi.__file__).parents[2] / "plugin/vllm/models/kimi_k3.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {"_k3_residual_stream", "KimiLinearModelVllm"}
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    assert len(nodes) == len(names)
    namespace = {
        "torch": torch,
        "kimi_k3_base": modules.kimi,
        "KimiDecoderLayer": modules.kimi.KimiDecoderLayer,
        "IntermediateTensors": modules.kimi.IntermediateTensors,
        "get_pp_group": lambda: modules.kimi.get_pp_group(),
    }
    exec(  # noqa: S102
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["KimiLinearModelVllm"]


@pytest.mark.parametrize("ends", [(LAYERS,), (12, LAYERS), (7, 19, LAYERS)])
@pytest.mark.parametrize("inputs_embeds", [False, True])
@pytest.mark.parametrize("plugin_aux", [False, True])
@torch.inference_mode()
def test_model_forward_allocates_and_transports_written_prefix(
    monkeypatch, modules, ends, inputs_embeds, plugin_aux
):
    monkeypatch.setattr(modules.ops, "apply_attn_res", _torch_attn_res)
    layers, config = _layers(monkeypatch, modules.kimi)
    x = torch.randn(
        TOKENS, HIDDEN, generator=torch.Generator().manual_seed(7)
    ).bfloat16()
    reference = _cat_reference(layers, x, ends[:-1])
    ids = torch.arange(TOKENS)
    model_class = _vllm_forward_class(modules) if plugin_aux else None
    aux_ids = (0, 12, 24, LAYERS)
    incoming = None
    for start, end in zip((0, *ends[:-1]), ends):
        model = _model_stage(
            monkeypatch, modules, layers, config, start, end, model_class
        )
        if plugin_aux:
            model.set_aux_hidden_state_layers(aux_ids)
        if start == 0:
            model.embed_tokens.weight[:TOKENS].copy_(x)
        prior = None if incoming is None else incoming["block_residual"].clone()
        allocations = []
        allocate = model.block_bank

        def tracked_bank(*args, allocate=allocate, allocations=allocations):
            bank = allocate(*args)
            allocations.append(bank)
            return bank

        monkeypatch.setattr(model, "block_bank", tracked_bank)
        result = model(ids, None, incoming, x if inputs_embeds and start == 0 else None)
        assert len(allocations) == 1
        assert allocations[0].shape == (TOKENS, 3, HIDDEN)
        if prior is not None:
            assert torch.equal(incoming["block_residual"], prior)
        prefix, add, add2, bank = reference[end - 1]
        if end < LAYERS:
            assert torch.equal(result["hidden_states"], prefix + add + add2)
            assert torch.equal(result["block_residual"], bank)
            assert result["block_residual"].is_contiguous()
            incoming = result
        else:
            if plugin_aux:
                result, aux = result
                expected_aux = [
                    (
                        x
                        if i == 0
                        else modules.kimi.KimiDecoderLayer.aux_hidden_state(
                            reference[i - 1]
                        )
                    )
                    for i in aux_ids
                    if start <= i <= end
                ]
                assert len(aux) == len(expected_aux)
                assert all(torch.equal(a, b) for a, b in zip(aux, expected_aux))
            expected, _ = _plain(model.output_attn_res)(prefix, bank, add, add2)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_attention_sites_pass_the_full_bank_and_precomputed_bound(monkeypatch, modules):
    bank = torch.randn(TOKENS, 3, HIDDEN)
    prefix = torch.randn(TOKENS, HIDDEN)
    seen = []

    def mix(*args):
        seen.append((args[1], args[-1]))
        return args[0], args[0]

    monkeypatch.setattr(modules.ops, "apply_attn_res", mix)
    for layer in (1, 12, 13, 25):
        site = modules.AttnRes(
            nn.Identity(), nn.Identity(), block_size=BLOCK, layer_idx=layer
        )
        site.score_weight = torch.ones(HIDDEN)
        site(prefix, bank)
    assert [count for _, count in seen] == [1, 1, 2, 3]
    assert all(value is bank for value, _ in seen)


@pytest.mark.parametrize("blocks", [None, 1])
def test_receive_requires_the_preceding_prefix(modules, blocks):
    model = _model_without_layers(modules)
    model.start_layer = 13
    received = None if blocks is None else torch.zeros(TOKENS, blocks, HIDDEN)
    with pytest.raises(ValueError, match="received block prefix"):
        model.block_bank(torch.zeros(TOKENS, HIDDEN), received)


def _mono_methods(modules):
    # Exercise the real orchestration methods without importing FlyDSL kernels.
    from atom.models.kimi_k3.mono.config import layer_block

    path = Path(modules.kimi.__file__).parent / "mono" / "runner.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "K3MonoDecodeRunner"
    )
    names = {"forward", "_layer", "_fallback", "_seam", "_write_aux"}
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(cls.body) == len(names)
    namespace = {
        "torch": torch,
        "nullcontext": nullcontext,
        "MonoUnsupported": RuntimeError,
        "width_key": lambda *args: None,
        "kda_metadata": lambda ctx: None,
        "get_forward_context": lambda: None,
        "layer_block": layer_block,
    }
    # This is the checked-in production code, with only GPU imports omitted.
    exec(  # noqa: S102
        compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["K3MonoDecodeRunner"]()


@torch.inference_mode()
def test_mono_fallback_reuses_forward_with_one_aux_tap(monkeypatch, modules):
    monkeypatch.setattr(modules.ops, "apply_attn_res", _torch_attn_res)
    layers, config = _layers(monkeypatch, modules.kimi)
    model = _model_stage(monkeypatch, modules, layers, config, 0, LAYERS)
    ids = torch.arange(4)
    x = model.get_input_embeddings(ids)
    reference = _cat_reference(layers, x)
    runner = _mono_methods(modules)
    runner.inner, runner.widths, runner.ix = model, {None: None}, None
    runner.plans = [
        SimpleNamespace(layer_id=i, mono=False, ffn_only=False) for i in range(LAYERS)
    ]
    runner.mailboxes = SimpleNamespace(begin_step=lambda: None)
    runner._raise_on_given_up_waits = lambda rows: None
    taps, hooks, allocations = [], [], []
    runner.aux = SimpleNamespace(
        layer_ids=tuple(range(LAYERS)),
        extract=lambda output, layer: layer.aux_hidden_state(output),
        write=lambda i, value: taps.append((i, value)),
    )
    allocate = model.block_bank

    def bank(prefix):
        result = allocate(prefix)
        allocations.append(result)
        return result

    monkeypatch.setattr(model, "block_bank", bank)
    handles = [
        layer.register_forward_hook(lambda *args: hooks.append(args))
        for layer in layers
    ]
    try:
        result = runner.forward(ids, None, 1)
    finally:
        for handle in handles:
            handle.remove()
    assert len(allocations) == 1 and allocations[0].shape == (4, 3, HIDDEN)
    assert hooks == []
    assert [i for i, _ in taps] == list(range(LAYERS))
    for (_, value), output in zip(taps, reference):
        assert torch.equal(
            value, modules.kimi.KimiDecoderLayer.aux_hidden_state(output)
        )
    prefix, add, add2, bank = reference[-1]
    expected, _ = _plain(model.output_attn_res)(prefix, bank, add, add2)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layers,blocks", [(1, 1), (12, 1), (13, 2), (61, 6), (93, 8)])
def test_shared_bank_allocator_covers_the_model_layers(modules, layers, blocks):
    model = _model_without_layers(modules)
    model.config.num_hidden_layers = layers
    assert model.block_bank(torch.zeros(TOKENS, HIDDEN)).shape == (
        TOKENS,
        blocks,
        HIDDEN,
    )


@pytest.mark.parametrize("layer_id", [6, 11, 12])
@torch.inference_mode()
def test_aggregate_reads_the_successor_window(monkeypatch, modules, layer_id):
    monkeypatch.setattr(modules.ops, "apply_attn_res", _torch_attn_res)
    layers, _ = _layers(monkeypatch, modules.kimi)
    x = torch.randn(TOKENS, HIDDEN).bfloat16()
    prefix, add, add2, compact = _cat_reference(layers, x)[layer_id]
    bank = x.new_full((TOKENS, 3, HIDDEN), float("nan"))
    bank[:, : compact.shape[1]].copy_(compact)
    layer, successor = layers[layer_id : layer_id + 2]
    layer.check_aggregate_source(successor)
    result = layer.aggregate_hidden_state((prefix, add, add2, bank), successor)
    res = successor.self_attention_attn_res
    expected, _ = _torch_attn_res(
        prefix, compact, res.score_weight, res.eps, add, None, res.out_eps, add2
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
