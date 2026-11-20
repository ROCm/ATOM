# SPDX-License-Identifier: MIT
"""Real MoE construction/loading with small CPU tensors and observed consumers."""

from contextlib import nullcontext
from enum import Enum
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from aiter_stub import atom_without_aiter
from torch import nn


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            kimi=load("atom.models.kimi_k3.model"),
            ops=load("atom.model_ops.kimi_k3"),
            linear=load("atom.model_ops.linear"),
            loader=load("atom.model_loader.loader"),
            dispatch=load("atom.model_ops.module_dispatch_ops"),
            moe=load("atom.model_ops.moe"),
        )


class _Kernel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fn = lambda x: x

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


@pytest.fixture
def make_block(modules, monkeypatch):
    kimi = modules.kimi
    for name in ("FusedMoE", "RMSNorm"):
        monkeypatch.setattr(kimi, name, _Kernel)
    monkeypatch.setenv("ATOM_LOADER_NUM_THREADS", "1")
    monkeypatch.setenv("ATOM_LOADER_STRICT_COVERAGE", "1")
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    config = SimpleNamespace(
        enable_tbo=False,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(kimi, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(modules.dispatch, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(
        torch.ops.aiter,
        "maybe_dual_stream_split_forward",
        modules.dispatch.maybe_dual_stream_split_forward,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(kimi, "tensor_model_parallel_all_reduce", lambda x: x * 2)
    seen, gemms = {}, []

    def mm(x, weight, bias=None, otype=None, out=None, **kwargs):
        gemms.append(weight)
        value = F.linear(x.double(), weight.double(), bias).to(otype or x.dtype)
        return value if out is None else out.copy_(value)

    def situ(x, beta, linear_beta, quant=False):
        seen["gate_up"] = x
        gate, up = x.chunk(2, dim=-1)
        return gate - up, None

    for module in (kimi, modules.linear):
        if hasattr(module, "tgemm"):
            monkeypatch.setattr(module, "tgemm", SimpleNamespace(mm=mm))
    monkeypatch.setattr(modules.ops, "situ_and_mul_maybe_quant", situ)

    def make(tp=1, shared=2, latent=8, experts=6):
        group = SimpleNamespace(rank_in_group=0, world_size=tp, rank=0)
        monkeypatch.setattr(modules.linear, "get_tp_group", lambda: group)
        monkeypatch.setattr(kimi, "get_tensor_model_parallel_world_size", lambda: tp)
        hf = SimpleNamespace(
            hidden_size=16,
            num_hidden_layers=1,
            num_experts=experts,
            num_experts_per_token=2,
            routed_expert_hidden_size=latent,
            moe_intermediate_size=8,
            num_shared_experts=shared,
            moe_renormalize=True,
            moe_router_activation_func="sigmoid",
            latent_moe_use_norm=True,
            rms_norm_eps=1e-6,
            hidden_act="situ",
        )
        moe = kimi.KimiSparseMoeBlock(hf, prefix="layers.1.block_sparse_moe")
        generator = torch.Generator().manual_seed(0)
        checkpoint = {}
        for name, param in moe.named_parameters():
            param.data.copy_(torch.randint(-2, 3, param.shape, generator=generator))
            weight = param.detach().clone()
            owner = moe.get_submodule(name.rsplit(".", 1)[0])
            if getattr(owner, "tp_dim", None) == 0:
                weight = torch.cat(
                    [part.repeat(tp, 1) for part in weight.chunk(2, dim=0)]
                )
            elif getattr(owner, "tp_dim", None) == 1:
                weight = weight.repeat(1, tp)
            checkpoint[name] = weight

        def experts(x, logits, out=None):
            seen["routed_input"], seen["router_logits"] = x, logits
            value = x + logits.sum(-1, keepdim=True)
            return value if out is None else out.copy_(value)

        moe.experts.fn = experts
        monkeypatch.setattr(
            modules.loader,
            "safetensors_weights_iterator",
            lambda *args, **kwargs: iter(checkpoint.items()),
        )

        def load(plugin=False):
            if plugin:
                return modules.loader.load_model_in_plugin_mode(
                    moe,
                    SimpleNamespace(
                        hf_config=hf,
                        load_dummy=None,
                        plugin_config=SimpleNamespace(
                            is_plugin_mode=True,
                            is_vllm=True,
                            model_config=SimpleNamespace(model="cpu-checkpoint"),
                        ),
                    ),
                )
            return modules.loader.load_model(moe, "cpu-checkpoint", hf)

        return SimpleNamespace(
            moe=moe, load=load, checkpoint=checkpoint, seen=seen, gemms=gemms
        )

    return make


def _run(block, tokens=5, method="split_moe_forward", dtype=torch.bfloat16):
    block.seen.clear()
    block.gemms.clear()
    x = torch.arange(tokens * 16).reshape(tokens, 16).remainder(5).sub(2).to(dtype)
    out = getattr(block.moe, method)(x)
    return out, dict(block.seen), list(block.gemms)


@pytest.mark.parametrize("tp,plugin", [(1, False), (8, False), (1, True)])
@pytest.mark.parametrize("tokens", [1, 16, 128])
def test_loaded_front_feeds_real_consumers(make_block, tp, plugin, tokens):
    block = make_block(tp=tp)
    moe = block.moe
    want, want_seen, before = _run(block, tokens)
    params = dict(moe.named_parameters())
    names = set(moe.state_dict())
    assert block.load(plugin) == names
    merged = moe.front_proj.merged_weight
    assert merged is not None
    assert dict(moe.named_parameters()).keys() == params.keys()
    assert all(moe.get_parameter(name) is param for name, param in params.items())
    assert set(moe.state_dict()) == names
    start = 0
    for linear in moe.front_proj.linears():
        weight = linear.weight
        assert weight.untyped_storage().data_ptr() == merged.data_ptr()
        assert weight.storage_offset() == start * 16
        start += weight.shape[0]
    assert merged.shape == (start, 16)
    actual, seen, after = _run(block, tokens, method="forward")
    assert len(after) == len(before) - 2
    assert sum(w is merged for w in after) == 1
    for key in want_seen:
        torch.testing.assert_close(seen[key], want_seen[key], rtol=0, atol=0)
    assert (
        seen["routed_input"].untyped_storage().data_ptr()
        == seen["gate_up"].untyped_storage().data_ptr()
    )
    for got, expected in zip(actual, want):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    moe.front_proj.process_weights_after_loading()
    assert moe.front_proj.merged_weight is merged
    block.checkpoint["gate.weight"].fill_(1)
    block.load(plugin)
    assert moe.front_proj.merged_weight is merged
    _, reloaded, _ = _run(block, tokens)
    torch.testing.assert_close(
        reloaded["router_logits"],
        F.linear(
            torch.arange(tokens * 16).reshape(tokens, 16).remainder(5).sub(2).float(),
            torch.ones(6, 16),
        ).bfloat16(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("plugin", [False, True])
def test_child_padding_hook_runs_before_front_merge(make_block, plugin):
    block = make_block()
    want, _, before = _run(block)
    block.moe.gate.output_row_alignment = 4
    block.load(plugin)
    assert block.moe.gate.is_output_padded
    assert block.moe.front_proj.merged_weight is None
    actual, seen, after = _run(block)
    assert len(after) == len(before)
    assert seen["router_logits"].shape == (5, 6)
    for got, expected in zip(actual, want):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "change",
    ["quantized", "native", "bias", "dtype", "strided", "shape", "reduce", "subclass"],
)
def test_unsupported_linear_keeps_its_dispatch(make_block, modules, change):
    block = make_block()
    linear = block.moe.gate
    if change == "quantized":
        linear.quant_type = modules.kimi.QuantType.per_Token
    elif change == "native":
        linear.native_a8_group_rows = 32
    elif change == "bias":
        linear.bias = nn.Parameter(torch.zeros(6))
    elif change == "dtype":
        linear.weight.data = linear.weight.float()
    elif change == "strided":
        linear.weight.data = linear.weight.T.contiguous().T
    elif change == "shape":
        linear.output_size += 1
    elif change == "reduce":
        linear.reduce_results = True
    else:

        class CustomLinear(type(linear)):
            pass

        linear.__class__ = CustomLinear
    block.moe.front_proj.process_weights_after_loading()
    assert block.moe.front_proj.merged_weight is None
    # The unsupported backend remains responsible for this projection.
    linear.forward = lambda x: x.new_full((x.shape[0], 6), 7)
    _, seen, gemms = _run(block)
    assert torch.equal(seen["router_logits"], torch.full((5, 6), 7).bfloat16())
    assert len(gemms) == 4


def test_plugin_enum_and_non_bf16_input(make_block, modules):
    block = make_block()
    block.load(plugin=True)
    block.moe.front_proj.merged_weight = None
    plugin_quant = Enum("PluginQuant", {"No": modules.kimi.QuantType.No.value})
    for linear in block.moe.front_proj.linears():
        linear.quant_type = plugin_quant.No
    block.moe.front_proj.process_weights_after_loading()
    merged = block.moe.front_proj.merged_weight
    assert merged is not None
    _, _, gemms = _run(block, dtype=torch.float16)
    assert len(gemms) == 5
    assert all(w is not merged for w in gemms)


@pytest.mark.parametrize("shared,latent", [(0, 8), (2, None)])
def test_other_moe_configurations_keep_original_path(make_block, shared, latent):
    block = make_block(shared=shared, latent=latent)
    assert block.moe.front_proj is None
    want, _, _ = _run(block)
    block.load()
    actual, _, _ = _run(block)
    for got, expected in zip(actual, want):
        if expected is None:
            assert got is None
        else:
            torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_dual_stream_consumer_keeps_reading_merged_storage(make_block, monkeypatch):
    block = make_block(tp=8)
    stream = SimpleNamespace(wait_stream=lambda other: None)
    block.moe.alt_stream = stream
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda other: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, other: None)
    x = torch.ones(5, 16).bfloat16()

    def run():
        block.gemms.clear()
        return block.moe._dual_stream_split(x)

    want = run()
    block.load()
    actual = run()
    torch.testing.assert_close(actual, want, rtol=0, atol=0)
    assert len(block.gemms) == 5
    assert all(w is not block.moe.front_proj.merged_weight for w in block.gemms)
