# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Kimi's rank views, dispatch, and public AITER projection/add contract."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            model=load("atom.models.kimi_k3.model"),
            tail=load("atom.model_ops.kimi_k3.attention_residual"),
            plugin=load("atom.plugin"),
            simulated_tp=load("atom.distributed.simulated_tp"),
        )


@pytest.fixture
def block(modules, monkeypatch):
    model = modules.model
    ctx = SimpleNamespace(is_prefill=False)
    config = SimpleNamespace(enable_tbo=False)
    pp = SimpleNamespace(world_size=1)
    comm = SimpleNamespace(disabled=False)
    group = SimpleNamespace(
        device_group=object(),
        rank_in_group=0,
        world_size=8,
        device_communicator=SimpleNamespace(ca_comm=comm),
    )
    monkeypatch.setattr(
        model, "get_forward_context", lambda: SimpleNamespace(context=ctx)
    )
    monkeypatch.setattr(model, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(model, "get_pp_group", lambda: pp)
    monkeypatch.setattr(
        model,
        "get_tp_group",
        lambda: group,
    )
    monkeypatch.setattr(modules.plugin, "is_plugin_mode", lambda: False)
    monkeypatch.setenv("ATOM_USE_CUSTOM_ALL_GATHER", "1")
    block = SimpleNamespace(
        tp_size=8,
        hidden_dim=7168,
        moe_hidden_size=3584,
        front_proj=SimpleNamespace(merged_weight=object()),
        shared_experts=SimpleNamespace(
            down_proj=SimpleNamespace(supports_out=lambda: True)
        ),
        fuse_routed_norm_quant=False,
        fuse_routed_triton_mxfp4_quant=False,
        routed_expert_up_proj=SimpleNamespace(
            weight=torch.empty((7168, 3584), dtype=torch.bfloat16)
        ),
        routed_expert_norm=SimpleNamespace(
            weight=torch.ones((3584,), dtype=torch.bfloat16), eps=1e-6
        ),
        _apply_routed_norm_and_up_proj=lambda x: x.new_full((x.shape[0], 7168), 3),
        context=ctx,
        config=config,
        pp=pp,
        comm=comm,
        tp_group=group,
    )
    for name in (
        "_column_tail_producer_supported",
        "_column_tail_eligible",
        "_can_column_tail",
        "_finish_latent_tail",
    ):
        setattr(block, name, MethodType(getattr(model.KimiSparseMoeBlock, name), block))
    return block


@pytest.mark.parametrize(
    "tokens,eligible", [(1, True), (17, True), (64, True), (65, False)]
)
def test_tail_capability_bounds(block, tokens, eligible):
    latent = torch.zeros((tokens, 3584), dtype=torch.bfloat16)
    shared = torch.zeros((tokens, 7168), dtype=torch.bfloat16)
    assert block._can_column_tail(latent, shared) is eligible


@pytest.mark.parametrize(
    "reason", ["prefill", "pp", "tbo", "plugin", "comm", "quant", "simulated_tp"]
)
def test_tail_keeps_safe_fallbacks(block, modules, monkeypatch, reason):
    if reason == "prefill":
        block.context.is_prefill = True
    elif reason == "pp":
        block.pp.world_size = 2
    elif reason == "tbo":
        block.config.enable_tbo = True
    elif reason == "plugin":
        monkeypatch.setattr(modules.plugin, "is_plugin_mode", lambda: True)
    elif reason == "comm":
        block.comm.disabled = True
    elif reason == "simulated_tp":
        block.tp_group.world_size = 4
        modules.simulated_tp._patch_group(block.tp_group, logical=8, physical=4)
    else:
        block.fuse_routed_norm_quant = True
    latent = torch.zeros((4, 3584), dtype=torch.bfloat16)
    shared = torch.ones((4, 7168), dtype=torch.bfloat16)
    assert not block._can_column_tail(latent, shared)
    routed, pending = block._finish_latent_tail(latent, shared)
    assert pending is shared and torch.all(routed == 3)


@pytest.mark.parametrize("rank", range(8))
def test_model_tail_uses_rank_views_and_gathers_once(block, modules, monkeypatch, rank):
    latent = torch.zeros((3, 3584), dtype=torch.bfloat16)
    shared = torch.ones((3, 7168), dtype=torch.bfloat16)
    seen = []

    def project(x, norm, eps, weight, add):
        seen.append((weight, add))
        assert x is latent and norm is block.routed_expert_norm.weight and eps == 1e-6
        return torch.full((3, 896), rank, dtype=torch.bfloat16)

    def gather(local, *, use_custom, dim):
        assert use_custom and dim == -1 and local.shape == (3, 896)
        return torch.cat([local] * 8, dim=-1)

    monkeypatch.setattr(modules.model, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(modules.model, "tensor_model_parallel_all_gather", gather)
    monkeypatch.setattr(modules.tail, "column_tail", project)
    out, pending = block._finish_latent_tail(latent, shared)
    assert pending is None and out.shape == shared.shape
    assert len(seen) == 1
    weight, add = seen[0]
    assert weight.shape == (896, 3584) and add.shape == (3, 896)
    assert (
        weight.untyped_storage().data_ptr()
        == block.routed_expert_up_proj.weight.untyped_storage().data_ptr()
    )
    assert weight.storage_offset() == rank * 896 * 3584
    assert add.untyped_storage().data_ptr() == shared.untyped_storage().data_ptr()
    assert add.storage_offset() == rank * 896 and add.stride() == (7168, 1)


def test_summed_tail_has_a_real_fresh_zero_at_the_split_op_boundary(modules):
    value = torch.full((3, 7168), 7, dtype=torch.bfloat16)
    first, zero = modules.model.KimiSparseMoeBlock._split_op_pair((value, None))
    assert first is value and torch.count_nonzero(zero) == 0
    assert zero.data_ptr() != value.data_ptr()
    zero.fill_(11)
    _, next_zero = modules.model.KimiSparseMoeBlock._split_op_pair((value, None))
    assert torch.count_nonzero(next_zero) == 0
    assert torch.all(value == 7)


@pytest.fixture
def gpu_tail(modules):
    if not torch.cuda.is_available():
        pytest.skip("column-tail numerical qualification requires a GPU")
    return modules.tail


def _gpu_operands(tokens, *, rank=3, padded=False):
    width, hidden, columns = 3584, 7168, 896
    latent = torch.randn(
        (tokens, width + (16 if padded else 0)), dtype=torch.bfloat16, device="cuda"
    )[:, :width]
    weight = torch.randn(
        (hidden, width + (32 if padded else 0)), dtype=torch.bfloat16, device="cuda"
    )[:, :width]
    weight.mul_(0.02)
    norm = torch.ones((width,), dtype=torch.bfloat16, device="cuda")
    shared = torch.randn((tokens, hidden), dtype=torch.bfloat16, device="cuda") * 0.5
    cols = slice(rank * columns, (rank + 1) * columns)
    return latent, norm, weight[cols], shared[:, cols]


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("padded", [False, True])
def test_tail_qualifies_normalized_bf16_input(gpu_tail, seed, padded):
    from atom.model_ops.layernorm import rmsnorm2d_fwd_

    torch.manual_seed(seed)
    width, hidden = 3584, 7168
    latent = torch.randn(
        (64, width + (16 if padded else 0)), dtype=torch.bfloat16, device="cuda"
    )[:, :width]
    weight = torch.randn(
        (hidden, width + (32 if padded else 0)), dtype=torch.bfloat16, device="cuda"
    )[:, :width]
    weight.mul_(0.02)
    norm = 1 + 0.1 * torch.randn((width,), dtype=torch.bfloat16, device="cuda")
    shared = 0.5 * torch.randn((64, hidden), dtype=torch.bfloat16, device="cuda")
    before = [x.clone() for x in (latent, norm, weight, shared)]
    for tokens in range(1, 65):
        normalized = rmsnorm2d_fwd_(latent[:tokens], norm, 1e-6, width)
        projection = (normalized.double() @ weight.double().T).float()
        want = projection + shared[:tokens].float()
        shares = []
        for rank in range(8):
            cols = slice(rank * 896, (rank + 1) * 896)
            out = gpu_tail.column_tail(
                latent[:tokens], norm, 1e-6, weight[cols], shared[:tokens, cols]
            )
            assert out.shape == (tokens, 896) and out.dtype == torch.bfloat16
            assert out.is_contiguous()
            shares.append(out)
        actual = torch.cat(shares, dim=1).float()
        scale = torch.stack(
            (projection.abs(), shared[:tokens].float().abs(), want.abs())
        ).amax(0)
        ulp = torch.exp2(torch.floor(torch.log2(scale.clamp_min(1e-30))) - 7)
        assert torch.max((actual - want).abs() / ulp) <= 1, (seed, padded, tokens)
    for original, copy in zip((latent, norm, weight, shared), before):
        assert torch.equal(original, copy)


@pytest.mark.parametrize("sign", [-1, 1])
def test_tail_adds_before_bf16_rounding(gpu_tail, sign):
    latent, norm, weight, shared = _gpu_operands(6)
    latent.fill_(1)
    weight.zero_()
    weight[:, 0] = sign
    weight[:, 1] = sign / 256
    shared.fill_(sign / 256)
    out = gpu_tail.column_tail(latent, norm, 1e-6, weight, shared)
    expected = torch.full_like(out, sign * (1 + 1 / 128))
    assert torch.equal(out, expected)
    wrong = torch.full_like(out, sign * (1 + 1 / 256)) + shared
    assert not torch.equal(wrong, expected)


@pytest.mark.parametrize("tokens", [1, 17, 64])
def test_tail_graph_replays_changed_inputs(gpu_tail, tokens):
    latent, norm, weight, shared = _gpu_operands(tokens, padded=True)
    gpu_tail.column_tail(latent, norm, 1e-6, weight, shared)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gpu_tail.column_tail(latent, norm, 1e-6, weight, shared)
    for i in range(4):
        latent.fill_(i + 1)
        shared.fill_((i - 2) / 4)
        expected = gpu_tail.column_tail(latent, norm, 1e-6, weight, shared)
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, expected)


def test_tail_fullgraph_uses_the_same_normalization(gpu_tail):
    latent, norm, weight, shared = _gpu_operands(17)
    compiled = torch.compile(gpu_tail.column_tail, fullgraph=True)
    expected = gpu_tail.column_tail(latent, norm, 1e-6, weight, shared)
    assert torch.equal(compiled(latent, norm, 1e-6, weight, shared), expected)
