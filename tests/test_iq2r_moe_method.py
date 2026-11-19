# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
from import_guard import skip_if_dependency_missing

# The IQ2R config parser tests live in tests/test_quant_config.py, which runs
# without AITER.
try:
    # aiter/triton absent under bare non-GPU pytest
    import atom.model_ops.moe as moe_mod
except ImportError as exc:
    skip_if_dependency_missing(exc, "requires full atom import env")


@pytest.fixture
def aiter_iq2r():
    """AITER's GLM-5.3 IQ2R-v2 format code; older AITER builds lack it."""
    pytest.importorskip("aiter.ops.iq2r_v2")
    return pytest.importorskip("aiter.iq2r_glm53")


def test_iq2r_weight_loader_requires_exact_shapes():
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.tp_size = 1
    parameter = torch.nn.Parameter(
        torch.empty((2, 8), dtype=torch.uint8), requires_grad=False
    )
    parameter.iq2r_name = "w13_weight"
    loaded = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    method.load_weight(parameter, loaded)
    assert torch.equal(parameter, loaded)

    with pytest.raises(ValueError, match="shape mismatch"):
        method.load_weight(parameter, loaded[:, :-1])


def test_iq2r_weight_loader_checks_the_tile_n():
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.tp_size = 4
    method.layout = "glm53-packed-v2"
    parameter = torch.nn.Parameter(
        torch.zeros(1, dtype=torch.int32), requires_grad=False
    )
    parameter.iq2r_name = "w2_iq2r_tile_n"
    method.load_weight(parameter, torch.tensor([128], dtype=torch.int32))
    assert parameter.item() == 128

    for loaded in ([64], [128, 128]):
        with pytest.raises(ValueError, match="kernels need 128"):
            method.load_weight(parameter, torch.tensor(loaded, dtype=torch.int32))


@pytest.mark.parametrize(
    "layout,entries", [("glm53-packed-v2", 512), ("glm53-packed-v3", 1024)]
)
def test_iq2r_accepts_the_v2_and_v3_layouts(layout, entries):
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    moe_mod.Iq2rMoEMethod.__init__(method, None, SimpleNamespace(), layout=layout)
    assert method.codebook_entries == entries


@pytest.mark.parametrize("layout", [None, "bogus", "glm53-packed-v1"])
def test_iq2r_rejects_other_layouts(layout):
    moe = SimpleNamespace()
    with pytest.raises(ValueError, match="unsupported IQ2R layout"):
        moe_mod.Iq2rMoEMethod.__init__(
            object.__new__(moe_mod.Iq2rMoEMethod), None, moe, layout=layout
        )


def _glm53_packed_method(
    tp_size: int, tp_rank: int, intermediate: int, entries: int = 512
):
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.codebook_entries = entries
    method.layout = {512: "glm53-packed-v2", 1024: "glm53-packed-v3"}[entries]
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            tp_size=tp_size, tp_rank=tp_rank, ep_size=1, dp_size=1
        ),
        max_num_tokens=8,
    )
    method.classes = 4
    method.hidden_size = 6144
    method.intermediate_size = intermediate
    method.tp_size = tp_size
    method.tp_rank = tp_rank
    return method


def _glm53_packed_layer(method, device="meta", num_experts=256):
    layer = torch.nn.Module()
    layer.has_bias = False
    # The shared expert stays FP8, outside the MoE.
    layer.num_fused_shared_experts = 0
    with torch.device(device):
        method.create_weights(
            layer,
            num_experts=num_experts,
            hidden_size=6144,
            intermediate_size_per_partition=method.intermediate_size,
            params_dtype=torch.bfloat16,
        )
    return layer


@pytest.mark.parametrize("entries", [512, 1024])
def test_iq2r_glm53_packed_create_weights_at_tp4(aiter_iq2r, entries):
    from aiter.iq2r_glm53 import iq2r_glm53_aux_bytes, iq2r_glm53_gate_bytes
    from aiter.ops.iq2r_format import IQ2RMetadata

    method = _glm53_packed_method(
        tp_size=4, tp_rank=1, intermediate=512, entries=entries
    )
    layer = _glm53_packed_layer(method)

    down = IQ2RMetadata(logical_n=6144, logical_k=512, codebook_entries=entries)
    assert tuple(layer.w13_weight.shape) == (256, iq2r_glm53_gate_bytes(1024, entries))
    assert tuple(layer.w13_weight_scale.shape) == (
        256,
        iq2r_glm53_aux_bytes(1024, 6144, 4, gate=True, codebook_entries=entries),
    )
    assert tuple(layer.w2_weight.shape) == (256, down.data_bytes)
    assert tuple(layer.w2_weight_scale.shape) == (
        256,
        iq2r_glm53_aux_bytes(6144, 512, 4, gate=False, codebook_entries=entries),
    )
    for name in ("w13_iq2r_tile_n", "w2_iq2r_tile_n"):
        parameter = getattr(layer, name)
        assert (tuple(parameter.shape), parameter.dtype) == ((1,), torch.int32)
    assert (method.tp_size, method.tp_rank, method.classes) == (4, 1, 4)


@pytest.mark.parametrize("entries", [512, 1024])
@pytest.mark.parametrize("load_dummy", [None, "empty", "zero", "xavier"])
def test_iq2r_glm53_packed_post_load_checks_codebook_range(
    aiter_iq2r, monkeypatch, load_dummy, entries
):
    method = _glm53_packed_method(
        tp_size=8, tp_rank=0, intermediate=256, entries=entries
    )
    layer = _glm53_packed_layer(method, device="cpu", num_experts=2)
    for aux in (layer.w13_weight_scale, layer.w2_weight_scale):
        aux.data[:, : entries * 8] = 0x38  # E4M3 1.0, inside the kernels' range
    monkeypatch.setattr(
        moe_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(load_dummy=load_dummy),
    )
    monkeypatch.setattr(moe_mod, "_IQ2R_GLM53_WORKSPACES", {})
    method.process_weights_after_loading(layer)

    # subnormal: copies form wrongly (the last entry of copy 0 is checked too)
    layer.w2_weight_scale.data[0, 5 if load_dummy else entries * 8 - 1] = 0x05
    if load_dummy:  # dummy weights carry no codebooks
        method.process_weights_after_loading(layer)
    else:
        with pytest.raises(ValueError, match="codebook bytes"):
            method.process_weights_after_loading(layer)


def test_glm53_weights_mapping_targets_every_compiled_tensor(aiter_iq2r):
    compiler = pytest.importorskip("aiter.quantization.iq2r.glm5.compile")
    from atom.model_loader.weight_names import CheckpointNameRewriter
    from atom.models.deepseek_v2 import GlmMoeDsaForCausalLM

    layer = _glm53_packed_layer(_glm53_packed_method(4, 0, 512))
    rewriter = CheckpointNameRewriter(
        weights_mapping=GlmMoeDsaForCausalLM.weights_mapping
    )
    mapped = sorted(
        rewriter.rewrite(key)
        for projection in ("gate_up", "down")
        for key in compiler.iq2r_compiled_tensor_keys(3, projection).values()
    )
    assert mapped == sorted(
        f"model.layers.3.mlp.experts.{name}" for name, _ in layer.named_parameters()
    )


@pytest.mark.parametrize(
    "tp_size,ep_size,dp_size,fused_shared",
    [(2, 1, 1, 0), (1, 1, 1, 0), (4, 2, 1, 0), (4, 1, 2, 0), (8, 1, 1, 1)],
)
def test_iq2r_glm53_packed_rejects_unsupported_parallelism(
    tp_size, ep_size, dp_size, fused_shared
):
    method = _glm53_packed_method(tp_size, 0, 2048 // tp_size)
    method.moe.moe_parallel_config.ep_size = ep_size
    method.moe.moe_parallel_config.dp_size = dp_size
    layer = torch.nn.Module()
    layer.has_bias = False
    layer.num_fused_shared_experts = fused_shared
    with (
        pytest.raises(NotImplementedError, match="glm53-packed-v2"),
        torch.device("meta"),
    ):
        method.create_weights(
            layer,
            num_experts=256 + fused_shared,
            hidden_size=6144,
            intermediate_size_per_partition=2048 // tp_size,
            params_dtype=torch.bfloat16,
        )


@pytest.mark.parametrize("entries", [512, 1024])
@pytest.mark.parametrize("tp_size,tp_rank", [(4, 3), (8, 5)])
def test_iq2r_glm53_packed_loader_slices_the_rank_shard(
    aiter_iq2r, tp_size, tp_rank, entries
):
    from aiter.iq2r_glm53 import (
        iq2r_glm53_aux_bytes,
        iq2r_glm53_gate_bytes,
        iq2r_glm53_slice_down_aux,
        iq2r_glm53_slice_gate,
        iq2r_glm53_slice_gate_aux,
    )
    from aiter.ops.iq2r_format import IQ2RMetadata, iq2r_slice_input_data

    shard = 2048 // tp_size
    method = _glm53_packed_method(tp_size, tp_rank, shard, entries)
    down = IQ2RMetadata(logical_n=6144, logical_k=2048, codebook_entries=entries)
    generator = torch.Generator().manual_seed(tp_size)

    def full(columns):
        return torch.randint(0, 256, (1, columns), generator=generator).to(torch.uint8)

    def aux_bytes(n, k, gate):
        return iq2r_glm53_aux_bytes(n, k, 4, gate=gate, codebook_entries=entries)

    weights = {
        "w13_weight": full(iq2r_glm53_gate_bytes(4096, entries)),
        "w13_weight_scale": full(aux_bytes(4096, 6144, True)),
        "w2_weight": full(down.data_bytes),
        "w2_weight_scale": full(aux_bytes(6144, 2048, False)),
    }
    start = 2 * shard * tp_rank
    expected = {
        "w13_weight": iq2r_glm53_slice_gate(
            weights["w13_weight"], start, 2 * shard, entries
        ),
        "w13_weight_scale": iq2r_glm53_slice_gate_aux(
            weights["w13_weight_scale"], start, 2 * shard, 4096, 4, entries
        ),
        "w2_weight": iq2r_slice_input_data(
            weights["w2_weight"], down, shard * tp_rank, shard
        ),
        "w2_weight_scale": iq2r_glm53_slice_down_aux(
            weights["w2_weight_scale"], shard * tp_rank, shard, 2048, 4, entries
        ),
    }
    # The rank's shard has the shape create_weights allocates.
    assert expected["w13_weight_scale"].shape[1] == aux_bytes(2 * shard, 6144, True)
    assert expected["w2_weight_scale"].shape[1] == aux_bytes(6144, shard, False)
    assert expected["w13_weight"].shape[1] == iq2r_glm53_gate_bytes(2 * shard, entries)
    for name, value in expected.items():
        parameter = torch.nn.Parameter(torch.empty_like(value), requires_grad=False)
        parameter.iq2r_name = name
        method.load_weight(parameter, weights[name])
        assert torch.equal(parameter, value), name


def test_iq2r_glm53_packed_apply_runs_the_glm53_moe(aiter_iq2r, monkeypatch):
    method = _glm53_packed_method(tp_size=4, tp_rank=0, intermediate=512)
    layer = SimpleNamespace(
        num_fused_shared_experts=0,
        w13_weight=torch.empty((256, 3), dtype=torch.uint8),
        w13_weight_scale=torch.empty((256, 5), dtype=torch.uint8),
        w2_weight=torch.empty((256, 2), dtype=torch.uint8),
        w2_weight_scale=torch.empty((256, 7), dtype=torch.uint8),
    )
    hidden = torch.randn(2, 6144, dtype=torch.bfloat16)
    # FusedMoE routes before apply: top-8 over the 256 routed experts. The
    # kernels take int32 ids and FP32 weights.
    topk_weights = torch.full((2, 8), 0.1, dtype=torch.bfloat16)
    topk_ids = torch.arange(16, dtype=torch.int64).reshape(2, 8)
    calls = []
    monkeypatch.setattr(
        method, "_glm53_workspace", lambda device: ("workspace", device)
    )
    shared_calls = []

    def moe_out(*args, shared=None):
        calls.append(args)
        shared_calls.append(shared)

    monkeypatch.setattr(aiter_iq2r, "iq2r_glm53_moe_out", moe_out)
    arguments = {
        "layer": layer,
        "x": hidden,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "global_num_experts": 256,
        "activation": moe_mod.ActivationType.Silu,
    }

    output = method.apply(**arguments)

    assert len(calls) == 1
    assert torch.equal(calls[0][0], hidden)
    assert calls[0][1] is layer.w13_weight
    assert calls[0][5].dtype == torch.float32
    assert torch.equal(calls[0][5], topk_weights.float())
    assert calls[0][6].dtype == torch.int32
    assert torch.equal(calls[0][6], topk_ids.int())
    assert calls[0][5].shape == calls[0][6].shape == (2, 8)
    assert calls[0][7] is output
    assert calls[0][8] == ("workspace", hidden.device)
    assert shared_calls == [None]  # no shared expert attached
    with pytest.raises(ValueError, match="does not support"):
        method.apply(**{**arguments, "activation": moe_mod.ActivationType.Swiglu})
    with pytest.raises(ValueError, match="BF16 rows"):
        method.apply(**{**arguments, "x_scale": torch.ones(2, 1)})
    with pytest.raises(ValueError, match="router weights"):
        method.apply(**{**arguments, "apply_router_weight_on_input": True})


def test_iq2r_glm53_apply_fuses_the_attached_shared_expert(aiter_iq2r, monkeypatch):
    """An attached shared expert goes to the IQ2R launches for small batches only."""
    method = _glm53_packed_method(tp_size=4, tp_rank=0, intermediate=512)
    layer = SimpleNamespace(
        num_fused_shared_experts=0,
        w13_weight=torch.empty((256, 3), dtype=torch.uint8),
        w13_weight_scale=torch.empty((256, 5), dtype=torch.uint8),
        w2_weight=torch.empty((256, 2), dtype=torch.uint8),
        w2_weight_scale=torch.empty((256, 7), dtype=torch.uint8),
    )
    generator = torch.Generator().manual_seed(0)
    gate, up = (torch.randn((512, 6144), generator=generator) * 0.02 for _ in range(2))
    down = torch.randn((6144, 512), generator=generator) * 0.02
    method.attach_shared_expert(layer, gate, up, down)
    packed = layer.iq2r_shared
    assert packed.gate.numel() == 2 * 512 * 6144 and packed.down.numel() == 512 * 6144
    assert packed.gate_scales.numel() == 2 * 512 * 192
    assert packed.down_scales.numel() == 512 * 6144 // 32

    seen = []
    monkeypatch.setattr(method, "_glm53_workspace", lambda device: None)
    monkeypatch.setattr(
        aiter_iq2r, "iq2r_glm53_moe_out", lambda *args, shared=None: seen.append(shared)
    )
    limit = method.fused_shared_tokens()
    assert limit == aiter_iq2r.FUSED_SHARED_TOKENS
    for tokens in (1, limit, limit + 1):
        method.apply(
            layer=layer,
            x=torch.zeros((tokens, 6144), dtype=torch.bfloat16),
            topk_weights=torch.zeros((tokens, 8)),
            topk_ids=torch.zeros((tokens, 8), dtype=torch.int32),
            global_num_experts=256,
            activation=moe_mod.ActivationType.Silu,
        )
    assert seen == [packed, packed, None]


def test_dequant_fp8_block_keeps_fp32():
    """The shared expert is dequantized in FP32 before its MXFP8 packing; the
    indexer's BF16 helper rounds the same values."""
    from atom.models.deepseek_v2 import _dequant_fp8_block, _dequant_fp8_block_to_bf16

    weight = torch.randn((256, 384)).to(torch.float8_e4m3fn)
    block = torch.rand((2, 3)) + 0.5
    expected = weight.float() * block.repeat_interleave(128, 0).repeat_interleave(
        128, 1
    )
    dequantized = _dequant_fp8_block(weight, block)
    assert dequantized.dtype == torch.float32
    assert torch.equal(dequantized, expected)
    assert torch.equal(_dequant_fp8_block_to_bf16(weight, block), expected.bfloat16())


def _shared_fp8_linear(rows, columns, generator):
    weight = (torch.randn((rows, columns), generator=generator) * 0.1).to(
        torch.float8_e4m3fn
    )
    scale = torch.rand((rows // 128, columns // 128), generator=generator) + 0.5
    return SimpleNamespace(weight=weight, weight_scale=scale)


def _iq2r_deepseek_moe(monkeypatch, *, use_dual_stream, load_dummy=False):
    """A DeepseekV2MoE holding only what the shared-expert fusion reads."""
    deepseek_v2 = pytest.importorskip("atom.models.deepseek_v2")
    monkeypatch.setattr(
        deepseek_v2,
        "get_current_atom_config",
        lambda: SimpleNamespace(load_dummy=load_dummy),
    )
    attached = []
    method = SimpleNamespace(
        attach_shared_expert=lambda layer, *weights: attached.append(weights),
        fused_shared_tokens=lambda: 256,
    )
    generator = torch.Generator().manual_seed(0)
    module = object.__new__(deepseek_v2.DeepseekV2MoE)
    torch.nn.Module.__init__(module)
    module.experts = SimpleNamespace(quant_method=method)
    module.shared_experts = SimpleNamespace(
        gate_up_proj=_shared_fp8_linear(1024, 6144, generator),
        down_proj=_shared_fp8_linear(6144, 512, generator),
    )
    module._use_dual_stream = use_dual_stream
    module._iq2r_shared_fusable = True
    module._use_moe_dispatch_op = True
    module.fused_moe_forward_rows = 0
    return module, attached


@pytest.mark.parametrize("use_dual_stream", [False, True])
def test_iq2r_shared_fusion_does_not_need_dual_stream(monkeypatch, use_dual_stream):
    """The shared expert is attached, in FP32, with or without dual-stream."""
    monkeypatch.delenv("ATOM_IQ2R_FUSE_SHARED", raising=False)
    module, attached = _iq2r_deepseek_moe(monkeypatch, use_dual_stream=use_dual_stream)
    module.process_weights_after_loading()

    assert module.fused_moe_forward_rows == 256
    ((gate, up, down),) = attached
    assert gate.dtype == up.dtype == down.dtype == torch.float32
    assert gate.shape == up.shape == (512, 6144) and down.shape == (6144, 512)
    gate_up = module.shared_experts.gate_up_proj
    full = gate_up.weight.float() * gate_up.weight_scale.repeat_interleave(
        128, 0
    ).repeat_interleave(128, 1)
    assert torch.equal(gate, full[:512]) and torch.equal(up, full[512:])


@pytest.mark.parametrize(
    "env,load_dummy,fusable",
    [("0", False, True), ("1", True, True), ("1", False, False)],
)
def test_iq2r_shared_fusion_off(monkeypatch, env, load_dummy, fusable):
    """ATOM_IQ2R_FUSE_SHARED=0, dummy weights or non-IQ2R experts keep the
    separate shared MLP."""
    monkeypatch.setenv("ATOM_IQ2R_FUSE_SHARED", env)
    module, attached = _iq2r_deepseek_moe(
        monkeypatch, use_dual_stream=False, load_dummy=load_dummy
    )
    module._iq2r_shared_fusable = fusable
    module.process_weights_after_loading()
    assert module.fused_moe_forward_rows == 0
    assert attached == []


@pytest.mark.parametrize("use_dual_stream", [False, True])
def test_moe_dispatch_takes_the_fused_path_up_to_its_rows(monkeypatch, use_dual_stream):
    """`maybe_dual_stream_forward` runs `fused_moe_forward` for 1 to
    `fused_moe_forward_rows` rows, and otherwise the dual- or single-stream
    path, as without the hook."""
    dispatch = pytest.importorskip("atom.model_ops.module_dispatch_ops")
    monkeypatch.setattr(dispatch, "plugin_graph_is_piecewise", lambda: False)
    monkeypatch.setattr(
        pytest.importorskip("atom.utils.tbo.ubatching"), "tbo_active", lambda: False
    )
    monkeypatch.setattr(
        dispatch, "get_forward_context", lambda: SimpleNamespace(context=None)
    )
    paths = []

    def path(name):
        return lambda hidden: paths.append(name) or hidden

    module = SimpleNamespace(
        _use_dual_stream=use_dual_stream,
        _dual_stream_token_threshold=1024,
        _dual_stream_decode_token_threshold=1024,
        fused_moe_forward_rows=256,
        fused_moe_forward=path("fused"),
        dual_stream_moe_forward=path("dual"),
        single_stream_moe_forward=path("single"),
    )
    without_hook = SimpleNamespace(
        **{
            k: v
            for k, v in vars(module).items()
            if k not in ("fused_moe_forward_rows", "fused_moe_forward")
        }
    )
    static_forward_context = {"fused": module, "plain": without_hook}
    monkeypatch.setattr(
        dispatch,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context=static_forward_context
            )
        ),
    )
    other = "dual" if use_dual_stream else "single"
    for rows in (1, 256, 257, 1024, 1025):
        for name in ("fused", "plain"):
            paths.clear()
            hidden = torch.zeros((rows, 4))
            assert dispatch.maybe_dual_stream_forward(hidden, name) is hidden
            fused = name == "fused" and rows <= 256
            expected = "fused" if fused else other if rows <= 1024 else "single"
            assert paths == [expected], (rows, name)


def test_deepseek_moe_forward_reaches_the_fused_path_without_dual_stream(
    monkeypatch,
):
    """With dual-stream off, IQ2R experts still dispatch through the custom op,
    which takes the fused path for small batches."""
    if not torch.cuda.is_available():
        pytest.skip("maybe_dual_stream_forward is registered for CUDA only")
    module, _ = _iq2r_deepseek_moe(monkeypatch, use_dual_stream=False)
    module.process_weights_after_loading()
    dispatch = pytest.importorskip("atom.model_ops.module_dispatch_ops")
    module.prefix = "model.layers.3.mlp"
    module.experts.hidden_size = 4
    monkeypatch.setattr(
        dispatch,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={module.prefix: module}
            )
        ),
    )
    paths = []
    module.fused_moe_forward = lambda hidden: paths.append("fused") or hidden + 1
    module.single_stream_moe_forward = (
        lambda hidden: paths.append("single") or hidden + 2
    )
    for rows, path in ((256, "fused"), (257, "single")):
        paths.clear()
        out = module(torch.zeros((rows, 4), device="cuda"))
        assert paths == [path]
        assert torch.equal(out, torch.full_like(out, 1.0 if path == "fused" else 2.0))
