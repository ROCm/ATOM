# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ast
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.model_ops.monokernel.config import (
    GLM5_GRAPH_BATCHES,
    ConvStateLayout,
    KimiDecodeGeometry,
    conv_state_offset,
    conv_state_shape,
    glm5_kernel_samples,
    glm5_tp_config,
)
from atom.model_ops.monokernel.dispatch import select_backend
from atom.model_ops.monokernel.glm import layout as glm_layout


def _kimi_mono_module():
    import importlib

    from tests.aiter_stub import stubbed_aiter

    with stubbed_aiter():
        return importlib.import_module("atom.models.kimi_k3_mono")


def _glm_mono_module():
    import importlib

    from tests.aiter_stub import stubbed_aiter

    with stubbed_aiter():
        return importlib.import_module("atom.models.glm52_mono")


def _preshuffle_linear_weight(weight):
    import torch

    raw = weight.view(torch.uint8)
    *lead, rows, packed_cols = raw.shape
    lane_k = 16 // raw.element_size()
    tiled = raw.reshape(
        *lead,
        rows // 16,
        16,
        packed_cols // 32,
        32 // lane_k,
        lane_k,
    )
    nlead = len(lead)
    order = list(range(nlead)) + [nlead, nlead + 2, nlead + 3, nlead + 1, nlead + 4]
    return tiled.permute(*order).contiguous().reshape_as(raw).view(weight.dtype)


def _preshuffled_mxfp4_linear(source):
    import torch

    from atom.model_ops.monokernel.formats import dequantize_mxfp4, quantize_mxfp4

    packed, scale = quantize_mxfp4(source)
    rows = source.shape[0]
    fp4_dtype = torch.float4_e2m1fn_x2
    shuffled_weight = _preshuffle_linear_weight(packed).view(fp4_dtype)
    shuffled_weight.is_shuffled = True
    groups = scale.shape[1]
    shuffled_scale = (
        scale.reshape(rows // 32, 2, 16, groups // 8, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .reshape(rows, groups)
    )
    linear = SimpleNamespace(
        input_size=source.shape[1],
        output_size=source.shape[0],
        quant_type=SimpleNamespace(name="per_1x32"),
        params_dtype=fp4_dtype,
        weight=shuffled_weight,
        weight_scale=shuffled_scale,
    )
    expected = dequantize_mxfp4(packed, scale).to(torch.bfloat16)
    return linear, expected


def _preshuffled_per_token_fp8_linear(source):
    import torch

    fp8_dtype = torch.float8_e4m3fn
    scale = source.abs().amax(dim=1, keepdim=True).float() / 448.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    native = (source.float() / scale).clamp(-448, 448).to(fp8_dtype)
    shuffled = _preshuffle_linear_weight(native)
    shuffled.is_shuffled = True
    linear = SimpleNamespace(
        input_size=source.shape[1],
        output_size=source.shape[0],
        quant_type=SimpleNamespace(name="per_Token"),
        params_dtype=fp8_dtype,
        weight=shuffled,
        weight_scale=scale,
        is_output_padded=False,
    )
    return linear, (native.float() * scale).to(torch.bfloat16)


def _kimi_decode_context(samples, actual_tokens=None, state_indices=None):
    actual_tokens = samples if actual_tokens is None else actual_tokens
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=actual_tokens,
        num_spec_decodes=0,
        num_actual_tokens=actual_tokens,
        replayssm=False,
        non_spec_state_indices_tensor=state_indices,
    )
    return SimpleNamespace(
        context=SimpleNamespace(is_prefill=False),
        ubatch_slices=None,
        attn_metadata=SimpleNamespace(kda_metadata=metadata),
        kv_cache_data={},
    )


def test_public_package_import_is_lazy():
    code = """
import sys
import atom.model_ops.monokernel
assert 'atom.model_ops.monokernel.glm.kernel' not in sys.modules
assert 'atom.model_ops.monokernel.k3.kernel' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_peer_buffer_allocation_failure_is_collective(monkeypatch):
    import torch

    from atom.model_ops.monokernel import runtime

    gathers = []
    monkeypatch.setattr(runtime.torch.cuda, "current_device", lambda: 0)

    def fail_allocation(*_args, **_kwargs):
        raise RuntimeError("rank 0 allocation failed")

    def all_gather_object(output, local, *, group):
        gathers.append((local, group))
        output[:] = [local, (None, (b"peer-handle", 0))]

    monkeypatch.setattr(runtime.torch, "zeros", fail_allocation)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda **_kwargs: pytest.fail("failed readiness must skip the final barrier"),
    )

    group = object()
    with pytest.raises(RuntimeError, match="allocation/export failed.*rank 0 allocation failed"):
        runtime.SymmetricPeerBuffer(256, rank=0, npes=2, group=group)

    assert len(gathers) == 1
    assert gathers[0][1] is group


def test_peer_buffer_remote_open_failure_is_collective(monkeypatch):
    import torch

    from atom.model_ops.monokernel import runtime

    class Storage:
        device = torch.device("cuda", 0)

        @staticmethod
        def data_ptr():
            return 0x1000

    gathers = []
    closed = []
    open_calls = []
    monkeypatch.setattr(runtime.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(runtime.torch, "zeros", lambda *_args, **_kwargs: Storage())
    monkeypatch.setattr(runtime, "get_allocation_base", lambda _address: 0x1000)
    monkeypatch.setattr(runtime, "get_ipc_handle", lambda _base: b"local-handle")

    def open_ipc_handle(handle):
        open_calls.append(handle)
        if len(open_calls) == 2:
            raise RuntimeError("rank 0 open failed")
        return 0x2000

    monkeypatch.setattr(runtime, "open_ipc_handle", open_ipc_handle)
    monkeypatch.setattr(runtime, "close_ipc_handle", closed.append)

    def all_gather_object(output, local, *, group):
        gathers.append((local, group))
        if len(gathers) == 1:
            output[:] = [
                local,
                (None, (b"peer-1-handle", 16)),
                (None, (b"peer-2-handle", 32)),
            ]
        else:
            output[:] = [local, None, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda **_kwargs: pytest.fail("failed open must skip the final barrier"),
    )
    monkeypatch.setattr(
        runtime.torch,
        "tensor",
        lambda *_args, **_kwargs: pytest.fail("failed open must skip address publication"),
    )

    group = object()
    with pytest.raises(RuntimeError, match="open failed.*rank 0 open failed"):
        runtime.SymmetricPeerBuffer(256, rank=0, npes=3, group=group)

    assert len(gathers) == 2
    assert all(call_group is group for _, call_group in gathers)
    assert closed == [0x2000]


def test_tp_validation_consensus_propagates_peer_failure(monkeypatch):
    import torch

    from atom.model_ops.monokernel.dispatch import (
        MonoUnsupported,
        tp_uniform_local_validation,
    )

    expected_group = object()

    def all_gather_object(output, local, *, group):
        assert local is None
        assert group is expected_group
        output[:] = [None, "ValueError: rank-local layout"]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    with pytest.raises(MonoUnsupported, match="rank 1: ValueError: rank-local layout"):
        tp_uniform_local_validation(
            None,
            group=expected_group,
            world_size=2,
            context="weight mapping failed",
        )


def test_bundled_aiter_compatibility_imports():
    torch = pytest.importorskip("torch")
    pytest.importorskip("flydsl")
    if not torch.cuda.is_available():
        pytest.skip("AITER import requires a visible ROCm device")
    pytest.importorskip("aiter")

    import atom.model_ops.monokernel.k3.staged  # noqa: F401


def test_scaled_mfma_uses_flydsl_v0341_operand_abi():
    root = Path(__file__).parents[1] / "atom" / "model_ops" / "monokernel"
    paths = (
        root / "mxfp8_linear.py",
        root / "k3" / "kernel.py",
        root / "k3" / "router_projection.py",
    )
    scaled_calls = []
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "gemm":
                continue
            assert not isinstance(node.args[2], (ast.List, ast.Tuple))
            assert not isinstance(node.args[3], (ast.List, ast.Tuple))
            keywords = {keyword.arg for keyword in node.keywords}
            if {"scale_a", "scale_b"} <= keywords:
                scaled_calls.append(node)
    assert scaled_calls


def test_glm_int64_metadata_uses_low_int32_words_for_all_rows():
    import torch

    values = torch.tensor([7, 129, -1, 2**31 - 1], dtype=torch.int64)
    words = values.view(torch.int32)

    assert values.dtype is torch.int64
    assert torch.equal(words[::2], values.to(torch.int32))
    assert words[1].item() == 0
    assert words[2].item() == 129


def test_glm_kernel_indexes_int64_metadata_as_word_offsets():
    kernel = (
        Path(__file__).parents[1]
        / "atom"
        / "model_ops"
        / "monokernel"
        / "glm"
        / "kernel.py"
    )
    tree = ast.parse(kernel.read_text())
    helpers = {
        name: next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        for name in ("row_position", "row_slot", "row_index_bounds")
    }
    for name, helper in helpers.items():
        forbidden = [
            node
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "_uniform")
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "readfirstlane"
                )
            )
        ]
        assert not forbidden, f"{name} must preserve lane-varying sample indices"

    for helper_name, pointer_name in (
        ("row_position", "positions"),
        ("row_slot", "slot_mapping"),
    ):
        helper = helpers[helper_name]
        load = next(
            node
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "buffer_load"
        )
        resource = load.args[0]
        assert isinstance(resource, ast.Call)
        assert isinstance(resource.args[0], ast.Name)
        assert resource.args[0].id == pointer_name
        offset = load.args[1]
        assert isinstance(offset, ast.BinOp) and isinstance(offset.op, ast.Mult)
        assert isinstance(offset.left, ast.Name) and offset.left.id == "s"
        assert isinstance(offset.right, ast.Constant) and offset.right.value == 2
        dtype = next(keyword.value for keyword in load.keywords if keyword.arg == "dtype")
        assert isinstance(dtype, ast.Attribute)
        assert isinstance(dtype.value, ast.Name) and dtype.value.id == "T"
        assert dtype.attr == "i32"


def test_native_decode_flag(monkeypatch):
    from atom.utils import envs

    name = "ATOM_NATIVE_DECODE_MONOKERNEL"
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) == "off"
    for value in ("off", "auto", "mono", "staged"):
        monkeypatch.setenv(name, value.upper())
        assert getattr(envs, name) == value
    monkeypatch.setenv(name, "on")
    with pytest.raises(ValueError, match="off, auto, mono, staged"):
        getattr(envs, name)


@pytest.mark.parametrize(
    "override",
    [
        {"mode": "off"},
        {"samples": 1},
        {"tp_size": 4},
        {"kv_cache_dtype": "auto"},
        {"native": False},
        {"decode": False},
        {"mtp": True},
        {"dpa": True},
        {"dcp": True},
        {"plugin": True},
        {"kv_cache_dtype": "fp8"},
        {"external_indexer": False},
        {"cache_layout": "split"},
    ],
)
def test_unsupported_glm_forward_falls_back(override):
    args = dict(
        model="glm52",
        mode="auto",
        samples=4,
        tp_size=8,
        kv_cache_dtype="bf16",
    )
    args.update(override)
    assert select_backend(**args) is None


def test_kimi_conv_state_layout_contracts():
    channels = 10
    channel = 7
    assert [
        conv_state_offset(ConvStateLayout.CHANNEL_MAJOR, channel, time, channels)
        for time in range(3)
    ] == [21, 22, 23]
    assert [
        conv_state_offset(ConvStateLayout.TIME_MAJOR, channel, time, channels)
        for time in range(3)
    ] == [7, 17, 27]
    assert conv_state_shape(ConvStateLayout.CHANNEL_MAJOR, 5, channels) == (5, 10, 3)
    assert conv_state_shape(ConvStateLayout.TIME_MAJOR, 5, channels) == (5, 3, 10)


def test_kimi_adapter_constructs_time_major_ops(monkeypatch):
    import torch

    module = _kimi_mono_module()
    captured = []

    class FakeOp:
        def __init__(self, *_args, **kwargs):
            captured.append(kwargs)

    op_module = types.ModuleType("atom.model_ops.monokernel.k3.op")
    op_module.KimiK3MonoKernel = FakeOp
    monkeypatch.setitem(sys.modules, op_module.__name__, op_module)
    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(
        module,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group=object(), device_group=object()),
    )
    geometry = KimiDecodeGeometry(1, 8, 10, "fp16", False)
    monkeypatch.setattr(module, "KimiK3PreparedWeights", object)
    module._KimiLayerOp(
        SimpleNamespace(layer_idx=1),
        weights=object(),
        prepared_weights=object(),
        geometry=geometry,
        backend="mono",
        kind="kda",
        launch_width=8,
    )

    assert captured[0]["conv_state_layout"] is ConvStateLayout.TIME_MAJOR
    assert captured[0]["conv_state_rows"] == 10
    assert captured[0]["state_dtype"] is torch.float16
    assert captured[0]["mtp"] is True


def test_kimi_announces_backend_once_on_rank_zero(monkeypatch):
    module = _kimi_mono_module()
    rank = [0]
    messages = []

    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank[0])
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(
        module,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group=object(), device_group=object()),
    )
    monkeypatch.setattr(module, "_layer_weights", lambda *_args: object())
    monkeypatch.setattr(
        module,
        "prepare_kimi_k3_weights",
        lambda *_args, **_kwargs: SimpleNamespace(
            validate_source=lambda *_: None
        ),
    )
    monkeypatch.setattr(module, "_KimiLayerOp", lambda *_args: object())
    monkeypatch.setattr(module, "tp_uniform_local_validation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.logger, "info", lambda *args: messages.append(args))
    monkeypatch.setattr(module.torch.cuda, "is_current_stream_capturing", lambda: False)

    def runner():
        value = object.__new__(module.KimiMonoDecode)
        value._mode = "auto"
        value._atom_config = SimpleNamespace(kv_cache_dtype="bf16")
        value._ops = {}
        value._weights = {}
        value._prepared = {}
        value._outputs = {}
        value._chains = {}
        value._refused = set()
        value._announced = set()
        return value

    layers = [
        SimpleNamespace(layer_idx=layer_idx, is_linear_attn=True, block_sparse_moe=object())
        for layer_idx in (1, 2)
    ]
    rank_zero = runner()
    geometry = KimiDecodeGeometry(4, 1, 10, "fp16", False)
    rank_zero._op(layers[0], geometry, "kda", 1)
    rank_zero._op(layers[0], geometry, "kda", 1)
    rank_zero._op(layers[1], geometry, "kda", 1)
    rank[0] = 1
    runner()._op(layers[0], geometry, "kda", 1)

    assert messages == [
        ("Kimi-K3 MonoKernel on: path=%s backend=%s q=%d", "kda", "mono", 1)
    ]


def test_glm_announces_samples_once_on_rank_zero(monkeypatch):
    module = _glm_mono_module()
    rank = [0]
    messages = []
    layers = [SimpleNamespace(layer_idx=layer_idx) for layer_idx in (1, 2)]

    class FakeOwned:
        op = object()
        def close(self):
            pass

    op_module = types.ModuleType("atom.model_ops.monokernel.glm.op")
    op_module.prepare_glm5_weights = lambda *_args: {}
    monkeypatch.setitem(sys.modules, op_module.__name__, op_module)
    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank[0])
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(module, "get_tp_group", lambda: SimpleNamespace(cpu_group=object()))
    monkeypatch.setattr(module, "_bf16_vector", lambda tensor, *_args: tensor)
    monkeypatch.setattr(module, "_layer_weights", lambda *_args: object())
    monkeypatch.setattr(module, "_GlmLayerOp", lambda *_args: FakeOwned())
    monkeypatch.setattr(module, "tp_uniform_local_validation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.logger, "info", lambda *args: messages.append(args))

    def runner():
        value = object.__new__(module.Glm52MonoDecode)
        value._lm = SimpleNamespace(model=SimpleNamespace(norm=SimpleNamespace(weight=object())))
        value._atom_config = SimpleNamespace(
            hf_config=SimpleNamespace(index_topk=2048), tensor_parallel_size=8,
            kv_cache_dtype="bf16", speculative_config=None,
        )
        value._mode = "auto"
        value._required = False
        value._shard = glm5_tp_config(8)
        value._ops, value._weights, value._prepared, value._runtimes = {}, {}, {}, {}
        value._refused, value._announced = set(), set()
        value._mono_layers = lambda: layers
        return value

    rank_zero = runner()
    assert rank_zero._prepare(4, 1)
    assert rank_zero._prepare(4, 1)
    assert rank_zero._prepare(8, 1)
    rank[0] = 1
    assert runner()._prepare(4, 1)
    assert messages == [
        ("GLM-5.2 MonoKernel on: S=%d chunk=%d", 4, 4),
        ("GLM-5.2 MonoKernel on: S=%d chunk=%d", 8, 8),
    ]
    rank_zero.close()
    assert rank_zero._announced == set()


def test_model_specific_backend_selection():
    common = dict(samples=8, tp_size=8, kv_cache_dtype="fp8")
    assert select_backend("glm52", "auto", **common) is None
    assert select_backend("glm52", "staged", **common) is None
    glm_bf16 = dict(samples=8, tp_size=8, kv_cache_dtype="bf16")
    assert select_backend("glm52", "auto", **glm_bf16) == "mono"
    assert select_backend("glm52", "mono", **glm_bf16) == "mono"
    assert select_backend("glm52", "staged", **glm_bf16) is None
    assert select_backend("kimi_k3", "auto", **common) == "mono"
    assert select_backend("kimi_k3", "mono", **common) == "mono"
    assert select_backend("kimi_k3", "auto", **common, is_kda=False) is None
    assert select_backend("kimi_k3", "auto", **common, has_moe=False) is None


def test_glm_c8_geometry_and_full_graph_ladder():
    tp8, tp4 = glm5_tp_config(8), glm5_tp_config(4)
    assert (tp8.local_heads, tp8.inter) == (8, 256)
    assert (tp4.local_heads, tp4.inter) == (16, 512)
    assert tp8.local_heads * 8 == tp4.local_heads * 4
    assert tp8.inter * 8 == tp4.inter * 4
    assert tp4.n_experts + tp4.num_shared_experts == 257
    for batch in GLM5_GRAPH_BATCHES:
        samples = batch * 6
        assert select_backend(
            "glm52", "auto", samples=samples, tp_size=4, kv_cache_dtype="fp8",
            mtp=True, query_length=6,
        ) == "mono"
        chunk = glm5_kernel_samples(samples, 6)
        assert chunk in (6, 12) and chunk % 6 == 0 and samples % chunk == 0
        samples = batch * 5
        assert select_backend(
            "glm52", "auto", samples=samples, tp_size=4, kv_cache_dtype="fp8",
            mtp=True, query_length=5,
        ) == "mono"
        chunk = glm5_kernel_samples(samples, 5)
        assert chunk in (5, 10) and chunk % 5 == 0 and samples % chunk == 0


def test_glm_q6_sparse_rows_are_intra_request_causal():
    slots = [100 + i for i in range(12)]
    flat, indptr, expected = [], [0], []
    for request in range(2):
        request_slots = slots[request * 6 : (request + 1) * 6]
        for token in range(6):
            row = [10 + request] + request_slots[: token + 1]
            expected.append(row)
            flat.extend(row)
            indptr.append(len(flat))
    for sample, row in enumerate(expected):
        actual = glm_layout.sparse_cache_rows(flat, sample=sample, topk=2048, sparse_kv_indptr=indptr)
        assert actual == row
        request, token = divmod(sample, 6)
        assert set(actual).isdisjoint(slots[request * 6 + token + 1 : (request + 1) * 6])


def test_glm_tp4_symmetric_and_split_schedule_contracts():
    cfg, samples = glm5_tp_config(4), 12
    scratch, symmetric = glm_layout.layout(samples, cfg.local_heads, 4, 2048, inter=cfg.inter)
    part = 4 * samples * cfg.hidden * 8
    assert scratch["mid"] >= 0
    assert symmetric["_part_stride"] == part and symmetric["_bytes"] == 4 * part
    stages = dict(glm_layout.stage_tasks(
        samples, cfg.local_heads, 2048, expert_mxfp4=True, inter=cfg.inter
    ))
    assert stages["split"] == samples * 2 * (2048 // 32)
    assert stages["ug"] >= samples * 9 * (cfg.inter // glm_layout.UG_TILE)


def test_glm_required_c8_decode_does_not_fallback():
    module = _glm_mono_module()
    runner = object.__new__(module.Glm52MonoDecode)
    runner._required = True
    with pytest.raises(module.MonoUnsupported, match="required MonoKernel decode failed"):
        runner._unsupported("missing FP8 cache")


def test_glm_chunk12_covers_routing_expert_tiles_and_down_lds():
    assert glm_layout.sample_wave_batches(12) == 2
    assert glm_layout.ug_task_rounds(256) == 1
    assert glm_layout.ug_task_rounds(512) == 2
    routed = [
        wave + batch * glm_layout.WAVES
        for batch in range(glm_layout.sample_wave_batches(12))
        for wave in range(glm_layout.WAVES)
        if wave + batch * glm_layout.WAVES < 12
    ]
    expert_tiles = [
        cta + task_round * glm_layout.BLOCKS
        for task_round in range(glm_layout.ug_task_rounds(512))
        for cta in range(glm_layout.BLOCKS)
    ]
    assert routed == list(range(12))
    assert expert_tiles == list(range(512))
    assert [
        glm_layout.split_acc_head(group, lane_group, element)
        for group in range(2)
        for lane_group in range(2)
        for element in range(4)
    ] == list(range(16))
    assert [glm_layout.split_score_column(wave, 0) for wave in range(8)] == list(range(8))
    assert [glm_layout.fp8_kv_upper_pair_lane(lane) for lane in range(0, 16, 4)] == [2, 6, 10, 14]
    assert [glm_layout.fp8_pe_upper_pair_lane(lane) for lane in range(0, 16, 2)] == list(range(1, 16, 2))
    assert glm_layout.down_x_words(12, 512, True) == 27_648


def test_glm_layout_covers_only_requested_decode_batches():
    for samples in (4, 8):
        scratch, symmetric = glm_layout.layout(samples, 8, 8, 2048)
        assert scratch["_bytes"] > 0
        assert symmetric["_bytes"] > 0


def test_glm_flat_and_paged_sparse_indices_are_physical_rows():
    indices = [90, 91, 92, 93, 40, 41, 42, 43]
    assert list(
        glm_layout.sparse_cache_rows(
            indices,
            sample=0,
            topk=4,
            cur_pos=1,
        )
    ) == [0, 1]
    assert list(
        glm_layout.sparse_cache_rows(
            indices,
            sample=1,
            topk=4,
            cur_pos=8,
        )
    ) == [40, 41, 42, 43]

    physical = [17, 3, 99, 8, 70]
    indptr = [0, 2, 5]
    assert glm_layout.sparse_cache_rows(
        physical,
        sample=0,
        topk=4,
        sparse_kv_indptr=indptr,
    ) == [17, 3]
    assert glm_layout.sparse_cache_rows(
        physical,
        sample=1,
        topk=4,
        sparse_kv_indptr=indptr,
    ) == [99, 8, 70]


def test_glm_padded_rows_use_safe_physical_row_without_cache_store():
    physical = [17]
    indptr = [0, 1, 1, 1, 1]

    active = glm_layout.paged_row_contract(physical, indptr, 0)
    assert active == {
        "active": True,
        "context": 1,
        "index_base": 0,
        "safe_row": 17,
        "write_cache": True,
    }
    for sample in range(1, 4):
        padded = glm_layout.paged_row_contract(physical, indptr, sample)
        assert padded == {
            "active": False,
            "context": 0,
            "index_base": 0,
            "safe_row": 0,
            "write_cache": False,
        }
        assert glm_layout.sparse_cache_rows(
            physical,
            sample=sample,
            topk=4,
            sparse_kv_indptr=indptr,
        ) == []


def test_glm_full_and_shared_layers_use_one_sparse_buffer():
    import torch

    module = _glm_mono_module()
    shared = torch.arange(16, dtype=torch.int32)
    full_indexer = SimpleNamespace(sparse_kv_indices_buffer=shared)
    full = SimpleNamespace(
        self_attn=SimpleNamespace(
            mla_attn=SimpleNamespace(
                impl=SimpleNamespace(sparse_kv_indices_buffer=shared)
            ),
            indexer=full_indexer,
        )
    )
    reused = SimpleNamespace(
        self_attn=SimpleNamespace(
            mla_attn=SimpleNamespace(
                impl=SimpleNamespace(sparse_kv_indices_buffer=shared)
            ),
            indexer=None,
        )
    )

    assert module._shared_sparse_buffer([full, reused]).data_ptr() == shared.data_ptr()
    reused.self_attn.mla_attn.impl.sparse_kv_indices_buffer = shared.clone()
    with pytest.raises(module.MonoUnsupported, match="do not share"):
        module._shared_sparse_buffer([full, reused])


def test_glm_bf16_attention_layout_mapping():
    import torch

    module = _glm_mono_module()
    from atom.model_ops.monokernel.weights import linear_bf16

    weight = torch.arange(2 * 5 * 3, dtype=torch.float32).to(torch.bfloat16).view(10, 3)
    linear = SimpleNamespace(
        input_size=3,
        output_size=10,
        weight=weight,
        quant_type=SimpleNamespace(name="No"),
        params_dtype=torch.bfloat16,
    )
    assert (
        linear_bf16(
            linear,
            name="kv_b_proj",
            logical_rows=10,
            logical_cols=3,
        ).data_ptr()
        == weight.data_ptr()
    )

    w_uk, w_uv = module._split_kv_b(
        weight,
        heads=2,
        nope_dim=2,
        value_dim=3,
        kv_lora=3,
    )
    by_head = weight.view(2, 5, 3)
    assert torch.equal(w_uk.view(2, 3, 2), by_head[:, :2].transpose(1, 2))
    assert torch.equal(w_uv.view(2, 3, 3), by_head[:, 2:])

    linear.weight = weight.float()
    with pytest.raises(
        module.MonoUnsupported,
        match=r"kv_b_proj BF16 weight has dtype torch\.float32",
    ):
        linear_bf16(
            linear,
            name="kv_b_proj",
            logical_rows=10,
            logical_cols=3,
        )


@pytest.mark.skipif(
    not hasattr(__import__("torch"), "float8_e4m3fn"),
    reason="E4M3 FP8 dtype is unavailable",
)
def test_glm_recipe_per_token_fp8_attention_mapping():
    import torch

    module = _glm_mono_module()
    from atom.model_ops.monokernel.weights import linear_bf16

    source = torch.randn(16, 64, dtype=torch.bfloat16)
    linear, expected = _preshuffled_per_token_fp8_linear(source)
    restored = linear_bf16(
        linear,
        name="kv_b_proj",
        logical_rows=16,
        logical_cols=64,
    )

    assert restored.is_contiguous()
    assert torch.equal(restored, expected)
    w_uk, w_uv = module._split_kv_b(
        restored,
        heads=2,
        nope_dim=2,
        value_dim=6,
        kv_lora=64,
    )
    by_head = expected.view(2, 8, 64)
    assert torch.equal(w_uk.view(2, 64, 2), by_head[:, :2].transpose(1, 2))
    assert torch.equal(w_uv.view(2, 6, 64), by_head[:, 2:])


def test_glm_default_off_does_not_inspect_runtime_config():
    runner = _glm_mono_module().Glm52MonoDecode(None, object(), "off")
    assert runner._enabled is False
    assert runner._ops == {}


def test_glm_default_page_size_accepted_and_segmented_refused(monkeypatch):
    module = _glm_mono_module()
    cfg = module.GLM5_CONFIG
    hf_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        hidden_size=cfg.hidden,
        num_attention_heads=cfg.local_heads * 8,
        q_lora_rank=cfg.q_lora,
        kv_lora_rank=cfg.kv_lora,
        qk_rope_head_dim=cfg.pe_dim,
        qk_nope_head_dim=cfg.nope_dim,
        v_head_dim=cfg.v_dim,
        n_routed_experts=cfg.n_experts,
        num_experts_per_tok=cfg.top_k,
        n_shared_experts=cfg.num_shared_experts,
        index_topk=2048,
        routed_scaling_factor=cfg.route_scale,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        rms_norm_eps=module.EPS,
        moe_intermediate_size=cfg.inter * 8,
    )
    atom_config = SimpleNamespace(
        hf_config=hf_config,
        tensor_parallel_size=8,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_dp_attention=False,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=1,
        enable_expert_parallel=False,
        enable_tbo=False,
        enable_tbo_decode=False,
        speculative_config=None,
        kv_cache_dtype="bf16",
    )
    layer = SimpleNamespace(
        mlp=SimpleNamespace(experts=object()),
        self_attn=SimpleNamespace(indexer=object(), skip_topk=False),
    )
    causal_lm = SimpleNamespace(
        model=SimpleNamespace(layers=[layer], start_layer=0, end_layer=1)
    )
    monkeypatch.setattr(module, "is_plugin_mode", lambda: False)

    monkeypatch.setattr(
        module,
        "envs",
        SimpleNamespace(
            ATOM_USE_TRITON_MLA_SHUFFLE_KV=False,
            ATOM_MLA_PAGE_SIZE=1,
        ),
    )
    assert module.Glm52MonoDecode(causal_lm, atom_config, "auto")._enabled

    module.envs.ATOM_MLA_PAGE_SIZE = 2
    assert not module.Glm52MonoDecode(causal_lm, atom_config, "auto")._enabled


def test_glm_graph_warmup_dispatch_counts_padded_rows(monkeypatch):
    import torch

    module = _glm_mono_module()
    samples = 4
    runner = object.__new__(module.Glm52MonoDecode)
    runner._enabled = True
    runner._mode = "auto"
    runner._required = False
    runner._shard = glm5_tp_config(8)
    runner._atom_config = SimpleNamespace(
        tensor_parallel_size=8,
        kv_cache_dtype="bf16",
        speculative_config=None,
        enable_dp_attention=False,
        decode_context_parallel_size=1,
    )
    runner._lm = SimpleNamespace(
        model=SimpleNamespace(aux_hidden_state_layers=[], layers=[])
    )
    runner._mono_layers = lambda: []
    runner._prepare = lambda _samples, _query_length: True
    metadata = SimpleNamespace(
        max_seqlen_q=1,
        slot_mapping=torch.tensor([7, -1, -1, -1], dtype=torch.int64),
        sparse_kv_indptr=torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
    )
    context = SimpleNamespace(is_prefill=False, scheduled_bs=1, running_tokens=samples)
    monkeypatch.setattr(
        module,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=context,
            attn_metadata=metadata,
            ubatch_slices=None,
            kv_cache_data={},
        ),
    )
    monkeypatch.setattr(module, "is_plugin_mode", lambda: False)
    monkeypatch.setattr(
        module,
        "_shared_sparse_buffer",
        lambda _layers: torch.arange(samples, dtype=torch.int32),
    )

    assert runner.supports(
        torch.arange(samples),
        torch.arange(samples, dtype=torch.int64),
        None,
        None,
    )


def test_glm_c8_full_graph_padding_dispatches_q6(monkeypatch):
    import torch

    module = _glm_mono_module()
    samples, active = 16 * 6, 8 * 6
    runner = object.__new__(module.Glm52MonoDecode)
    runner._enabled, runner._mode, runner._required = True, "auto", True
    runner._shard = glm5_tp_config(4)
    runner._atom_config = SimpleNamespace(
        tensor_parallel_size=4, kv_cache_dtype="fp8",
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=5),
        enable_dp_attention=False, decode_context_parallel_size=1,
    )
    runner._lm = SimpleNamespace(model=SimpleNamespace(aux_hidden_state_layers=[], layers=[]))
    runner._mono_layers = lambda: []
    seen = []
    runner._prepare = lambda rows, query_length: seen.append((rows, query_length)) or True
    metadata = SimpleNamespace(
        max_seqlen_q=6,
        slot_mapping=torch.cat((
            torch.arange(active, dtype=torch.int64),
            torch.full((samples - active,), -1, dtype=torch.int64),
        )),
        sparse_kv_indptr=torch.cat((
            torch.arange(active + 1, dtype=torch.int32),
            torch.full((samples - active,), active, dtype=torch.int32),
        )),
    )
    context = SimpleNamespace(is_prefill=False, scheduled_bs=8, running_tokens=samples)
    monkeypatch.setattr(module, "get_forward_context", lambda: SimpleNamespace(
        context=context, attn_metadata=metadata, ubatch_slices=None, kv_cache_data={}
    ))
    monkeypatch.setattr(module, "is_plugin_mode", lambda: False)
    monkeypatch.setattr(module, "_shared_sparse_buffer", lambda _layers: torch.arange(samples, dtype=torch.int32))
    assert runner.supports(
        torch.arange(samples), torch.arange(samples, dtype=torch.int64), None, None
    )
    assert seen == [(samples, 6)]


def test_glm_fp8_fused_576_cache_has_explicit_device_io():
    kernel = (Path(__file__).parents[1] / "atom" / "model_ops" / "monokernel" / "glm" / "kernel.py").read_text()
    assert 'cache_fp8 = kv_cache_dtype == "fp8"' in kernel
    assert "slot * (QK_DIM // 4)" in kernel
    assert "_fp8_to_bf16x8(raw[0], raw[1])" in kernel
    assert "_fp8_to_bf16x8(pe_raw[0], pe_raw[1])" in kernel


def test_shared_linear_weight_unshuffle_round_trip():
    import torch

    from atom.model_ops.monokernel.weights import _unshuffle_linear_weight

    native = torch.arange(16 * 64, dtype=torch.int32).to(torch.uint8).view(16, 64)
    shuffled = _preshuffle_linear_weight(native)
    shuffled.is_shuffled = True
    restored = _unshuffle_linear_weight(shuffled)
    assert torch.equal(restored, native)

    from atom.model_ops.monokernel.packing import pack_a16w4_weight

    assert torch.equal(pack_a16w4_weight(restored), shuffled.reshape(-1))


def test_shared_linear_scale_unshuffle_round_trip():
    import torch

    from atom.model_ops.monokernel.weights import _unshuffle_linear_scale

    native = torch.arange(2 * 256 * 16, dtype=torch.int32).to(torch.uint8).view(512, 16)
    shuffled = (
        native.view(16, 2, 16, 2, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .view_as(native)
    )
    restored = _unshuffle_linear_scale(shuffled, experts=2, rows=256, groups=16)
    assert torch.equal(restored, native.view(2, 256, 16))

    from atom.model_ops.monokernel.packing import pack_a16w4_scale

    assert torch.equal(pack_a16w4_scale(restored), shuffled.reshape(-1))


def test_kimi_runner_close_is_idempotent():
    KimiMonoDecode = _kimi_mono_module().KimiMonoDecode

    class Owned:
        def __init__(self):
            self.calls = 0

        def close(self):
            self.calls += 1

    runner = object.__new__(KimiMonoDecode)
    owned = Owned()
    runner._ops = {(1, 1, "staged", "kda", False): owned}
    runner._weights = {}
    runner._prepared = {(1, "staged", False): object()}
    runner._outputs = {(1, 1, "kda"): object()}
    runner._chains = {(1, 1): (object(), object())}
    runner._refused = set()
    runner._announced = {("kda", "staged", 4)}
    runner._prebuilt = True
    runner.close()
    runner.close()
    assert owned.calls == 1
    assert runner._prepared == {}
    assert runner._outputs == {}
    assert runner._chains == {}
    assert runner._announced == set()


def test_kimi_prepared_weights_are_shared_across_sample_runners(monkeypatch):
    import torch

    module = _kimi_mono_module()
    static_names = (
        "w_kda_in_packed", "w_kda_o_packed", "w_router",
        "w_ug", "s_ug", "w_dn", "s_dn",
    )
    prepared = SimpleNamespace(
        **{name: torch.zeros(1, dtype=torch.uint8) for name in static_names},
        **{
            name: SimpleNamespace(
                weight=torch.zeros(1, dtype=torch.uint8),
                scale=torch.zeros(1, dtype=torch.uint8),
            )
            for name in ("latent_down", "shared_up", "shared_down", "latent_up")
        },
    )
    validations = []
    prepared.validate_source = lambda weights, backend=None: validations.append((weights, backend))
    weights = module.LayerWeights(1, {"source": torch.zeros(1)})
    preparations = []

    class Owned:
        def __init__(
            self,
            _layer,
            layer_weights,
            prepared_weights,
            geometry,
            _backend,
            _kind,
            launch_width,
        ):
            self.weights = layer_weights
            self.prepared = prepared_weights
            self.static = [getattr(prepared_weights, name) for name in static_names]
            self.static += [
                tensor
                for name in ("latent_down", "shared_up", "shared_down", "latent_up")
                for tensor in (
                    getattr(prepared_weights, name).weight,
                    getattr(prepared_weights, name).scale,
                )
            ]
            self.activation = torch.empty(launch_width, 2)
            self.scratch = torch.empty(launch_width, 3)

        def close(self):
            pass

    runner = object.__new__(module.KimiMonoDecode)
    runner._mode = "auto"
    runner._atom_config = SimpleNamespace(kv_cache_dtype="bf16")
    runner._ops = {}
    runner._weights = {}
    runner._prepared = {}
    runner._refused = set()
    runner._announced = set()
    layer = SimpleNamespace(layer_idx=3, is_linear_attn=True, block_sparse_moe=object())
    monkeypatch.setattr(module, "_layer_weights", lambda *_args: weights)
    monkeypatch.setattr(
        module,
        "prepare_kimi_k3_weights",
        lambda weights, backend, *, mtp: (
            preparations.append((weights, backend, mtp)) or prepared
        ),
    )
    monkeypatch.setattr(module, "_KimiLayerOp", Owned)
    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(module, "get_tp_group", lambda: SimpleNamespace(cpu_group=object(), device_group=object()))
    monkeypatch.setattr(module, "tp_uniform_local_validation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.torch.cuda, "is_current_stream_capturing", lambda: False)

    op4 = runner._op(
        layer, KimiDecodeGeometry(4, 1, 3, "fp16", False), "kda", 8
    )
    op8 = runner._op(
        layer, KimiDecodeGeometry(8, 1, 3, "fp16", False), "kda", 8
    )

    assert preparations == [(weights, "mono", False)]
    assert validations == []
    assert op4 is op8
    assert op4.weights is weights
    assert op4.prepared is prepared
    runner._outputs = {}
    runner._chains = {}
    runner._prebuilt = False
    runner.close()
    assert runner._prepared == {}


def test_kimi_prebuilds_kda_and_mla_tail_once():
    module = _kimi_mono_module()
    layers = [
        SimpleNamespace(layer_idx=0, is_linear_attn=True),
        SimpleNamespace(
            layer_idx=1,
            is_linear_attn=True,
            block_sparse_moe=object(),
        ),
        SimpleNamespace(
            layer_idx=2,
            is_linear_attn=False,
            block_sparse_moe=object(),
        ),
    ]
    runner = object.__new__(module.KimiMonoDecode)
    runner._enabled = True
    runner._prebuilt = False
    runner._geometry_template = KimiDecodeGeometry(1, 8, 10, "fp16", False)
    runner._lm = SimpleNamespace(
        model=SimpleNamespace(layers=layers, start_layer=0, end_layer=3)
    )
    seen = []
    runner._op = lambda layer, geometry, kind, width: seen.append(
        (layer.layer_idx, geometry.groups, geometry.q, kind, width)
    )

    runner.prepare()
    runner.prepare()

    assert seen == [(1, 1, 8, "kda", 8), (2, 1, 8, "tail", 8)]
    assert runner._prebuilt


def test_kimi_q1_prebuilds_shared_chunk_widths():
    module = _kimi_mono_module()
    layers = [
        SimpleNamespace(
            layer_idx=1,
            is_linear_attn=True,
            block_sparse_moe=object(),
        ),
        SimpleNamespace(
            layer_idx=2,
            is_linear_attn=False,
            block_sparse_moe=object(),
        ),
    ]
    runner = object.__new__(module.KimiMonoDecode)
    runner._enabled = True
    runner._prebuilt = False
    runner._geometry_template = KimiDecodeGeometry(1, 1, 3, "fp16", False)
    runner._lm = SimpleNamespace(
        model=SimpleNamespace(layers=layers, start_layer=0, end_layer=2)
    )
    seen = []
    runner._op = lambda layer, _geometry, kind, width: seen.append(
        (layer.layer_idx, kind, width)
    )

    runner.prepare()

    assert seen == [
        (layer, kind, width)
        for layer, kind in ((1, "kda"), (2, "tail"))
        for width in (8, 4, 2, 1)
    ]


def test_kimi_default_off_does_not_inspect_runtime_config():
    KimiMonoDecode = _kimi_mono_module().KimiMonoDecode
    runner = KimiMonoDecode(None, object(), "off")
    assert runner._enabled is False
    assert runner._ops == {}
    assert runner._prepared == {}
    assert runner._outputs == {}
    assert runner._geometry_template is None
    assert runner._prebuilt is False


@pytest.mark.parametrize("samples", (4, 8))
def test_kimi_inputs_embeds_are_eligible(monkeypatch, samples):
    import torch

    module = _kimi_mono_module()
    runner = object.__new__(module.KimiMonoDecode)
    runner._enabled = True
    runner._mode = "auto"
    runner._atom_config = SimpleNamespace(tensor_parallel_size=8, kv_cache_dtype="bf16")
    runner._geometry_template = KimiDecodeGeometry(1, 1, 3, "fp16", False)
    monkeypatch.setattr(
        module,
        "get_forward_context",
        lambda: _kimi_decode_context(
            samples, state_indices=torch.arange(samples, dtype=torch.int32)
        ),
    )
    input_ids = torch.arange(samples)
    positions = torch.arange(samples)
    inputs_embeds = torch.zeros(samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16)

    assert runner.supports(input_ids, positions, None, inputs_embeds)
    assert not runner.supports(input_ids, positions, object(), inputs_embeds)
    assert not runner.supports(input_ids, positions, None, inputs_embeds.float())
    assert not runner.supports(input_ids, positions, None, inputs_embeds[:, :-1])
    assert not runner.supports(input_ids, positions, None, inputs_embeds.T.contiguous().T)


@pytest.mark.parametrize(("samples", "actual_tokens"), ((4, 3), (8, 5)))
def test_kimi_graph_padding_dispatches(monkeypatch, samples, actual_tokens):
    import torch

    module = _kimi_mono_module()
    runner = object.__new__(module.KimiMonoDecode)
    runner._enabled = True
    runner._mode = "auto"
    runner._atom_config = SimpleNamespace(tensor_parallel_size=8, kv_cache_dtype="bf16")
    runner._geometry_template = KimiDecodeGeometry(1, 1, 3, "fp16", False)
    state_indices = torch.cat(
        (
            torch.arange(actual_tokens, dtype=torch.int32),
            torch.full((samples - actual_tokens,), -1, dtype=torch.int32),
        )
    )
    monkeypatch.setattr(
        module,
        "get_forward_context",
        lambda: _kimi_decode_context(samples, actual_tokens, state_indices),
    )

    assert runner.supports(
        torch.arange(samples),
        torch.arange(samples),
        None,
        torch.zeros(samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("samples", (4, 8))
def test_kimi_padded_forward_propagates_negative_slots_without_state_mutation(
    monkeypatch, samples
):
    import torch

    module = _kimi_mono_module()
    actual_tokens = samples - 1
    state_indices = torch.cat(
        (
            torch.arange(actual_tokens, dtype=torch.int32),
            torch.tensor([-1], dtype=torch.int32),
        )
    )
    conv_state = torch.randn(4, 3, 3, dtype=torch.bfloat16)
    recurrent_state = torch.randn(4, 3, dtype=torch.float16)
    before = (conv_state.clone(), recurrent_state.clone())
    seen = []

    class FakeOp:
        block_write_idx = 0

        @staticmethod
        def forward(hidden, _blocks, indices, conv, recurrent, *, x_out, **_kwargs):
            seen.append(indices)
            assert conv.data_ptr() == conv_state.data_ptr()
            assert recurrent.data_ptr() == recurrent_state.data_ptr()
            x_out.copy_(hidden)
            return x_out

    layer = SimpleNamespace(
        layer_idx=1,
        is_linear_attn=True,
        block_sparse_moe=object(),
        _forward_hooks={},
        _forward_hooks_with_kwargs={},
    )
    model = SimpleNamespace(
        get_input_embeddings=lambda _ids: pytest.fail("inputs_embeds must be used"),
        layers=[layer],
        start_layer=0,
        end_layer=1,
        output_attn_res=lambda hidden, *_args: (hidden, None),
    )
    runner = object.__new__(module.KimiMonoDecode)
    runner._lm = SimpleNamespace(model=model)
    runner._geometry_template = KimiDecodeGeometry(1, 1, 3, "fp16", False)
    runner._op = lambda *_args: SimpleNamespace(op=FakeOp())
    runner._output = lambda *_args: torch.empty_like(inputs_embeds)
    context = _kimi_decode_context(samples, actual_tokens, state_indices)
    context.kv_cache_data = {
        "layer_1": SimpleNamespace(k_cache=conv_state, v_cache=recurrent_state)
    }
    monkeypatch.setattr(module, "get_forward_context", lambda: context)
    inputs_embeds = torch.zeros(samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16)

    runner.forward(torch.arange(samples), torch.arange(samples), inputs_embeds)

    assert len(seen) == 1
    assert torch.equal(seen[0], state_indices)
    assert seen[0][-1].item() < 0
    assert torch.equal(conv_state, before[0])
    assert torch.equal(recurrent_state, before[1])


def test_kimi_negative_slot_device_guards_cover_staged_and_mono_paths():
    root = Path(__file__).parents[1] / "atom" / "model_ops" / "monokernel" / "k3"
    recurrence = (root / "kda_recurrence.py").read_text()
    mono = (root / "kernel.py").read_text()

    assert "if slot >= 0:\n            decode()\n        else:\n            zero_output()" in recurrence
    assert "if (input_slot >= 0) & (output_slot >= 0):" in mono
    assert mono.count("valid_state = (input_slot >= 0) & (output_slot >= 0)") >= 3


def test_kimi_forward_uses_inputs_embeds(monkeypatch):
    import torch

    module = _kimi_mono_module()
    samples = 4
    inputs_embeds = torch.zeros(samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16)

    def unexpected_embedding(_input_ids):
        raise AssertionError("inputs_embeds must bypass token embedding")

    model = SimpleNamespace(
        get_input_embeddings=unexpected_embedding,
        layers=[],
        start_layer=0,
        end_layer=0,
        output_attn_res=lambda hidden, *_args: (hidden, None),
    )
    runner = object.__new__(module.KimiMonoDecode)
    runner._lm = SimpleNamespace(model=model)
    runner._geometry_template = KimiDecodeGeometry(1, 1, 3, "fp16", False)
    monkeypatch.setattr(
        module,
        "get_forward_context",
        lambda: _kimi_decode_context(
            samples, state_indices=torch.arange(samples, dtype=torch.int32)
        ),
    )

    output = runner.forward(torch.arange(samples), torch.arange(samples), inputs_embeds)

    assert output is inputs_embeds


def test_kimi_kda_a_log_is_fp32(monkeypatch):
    import torch

    module = _kimi_mono_module()
    tensor = torch.zeros(1, dtype=torch.bfloat16)
    a_log = torch.zeros(12, dtype=torch.bfloat16)
    attn = SimpleNamespace(
        in_proj=SimpleNamespace(weight=tensor),
        f_b_proj=SimpleNamespace(weight=tensor),
        conv_weight=tensor,
        A_log=a_log,
        dt_bias=torch.zeros(12, 128, dtype=torch.bfloat16),
        o_norm=SimpleNamespace(weight=tensor),
        o_proj=SimpleNamespace(weight=tensor),
    )
    experts = SimpleNamespace(
        quant_method=SimpleNamespace(is_guinterleave=False),
        global_num_experts=module.KIMI_K3_CONFIG.n_experts,
        intermediate_size_per_partition=module.KIMI_K3_CONFIG.inter,
        w13_weight=tensor,
        w2_weight=tensor,
        w13_weight_scale=tensor,
        w2_weight_scale=tensor,
    )
    moe = SimpleNamespace(
        experts=experts,
        routed_expert_up_proj=SimpleNamespace(
            weight=torch.zeros(module.KIMI_K3_CONFIG.hidden // 8, 1, dtype=torch.bfloat16)
        ),
        routed_expert_down_proj=SimpleNamespace(weight=tensor),
        routed_expert_norm=SimpleNamespace(weight=tensor),
        gate=SimpleNamespace(weight=tensor, e_score_correction_bias=tensor),
        shared_experts=SimpleNamespace(
            gate_up_proj=SimpleNamespace(weight=tensor),
            down_proj=SimpleNamespace(weight=tensor),
        ),
    )
    layer = SimpleNamespace(
        layer_idx=1,
        is_linear_attn=True,
        self_attn=attn,
        block_sparse_moe=moe,
        input_layernorm=SimpleNamespace(weight=tensor),
        post_attention_layernorm=SimpleNamespace(weight=tensor),
        self_attention_res_norm=SimpleNamespace(weight=tensor),
        self_attention_res_proj=SimpleNamespace(weight=tensor),
        mlp_res_norm=SimpleNamespace(weight=tensor),
        mlp_res_proj=SimpleNamespace(weight=tensor),
    )
    monkeypatch.setattr(module, "linear_bf16", lambda *_args, **_kwargs: tensor)
    monkeypatch.setattr(module, "atom_mxfp4_storage_view", lambda *_args, **_kwargs: tensor)

    weights = module._layer_weights(layer, rank=0, npes=8)

    assert a_log.dtype == torch.bfloat16
    assert weights.t["kda_a_log"].dtype == torch.float32
    assert weights.t["kda_a_log"].is_contiguous()
    assert weights.t["w_ug"] is experts.w13_weight
    assert weights.t["s_ug"] is experts.w13_weight_scale
    assert weights.mxfp4_weight_layout is module.Mxfp4WeightLayout.ATOM
    assert weights.mxfp4_scale_layout is module.Mxfp4ScaleLayout.ATOM


@pytest.mark.skipif(
    not hasattr(__import__("torch"), "float4_e2m1fn_x2"),
    reason="MXFP4 dtype is unavailable",
)
def test_shared_mxfp4_linear_round_trip():
    import copy

    import torch

    from atom.model_ops.monokernel.dispatch import MonoUnsupported
    from atom.model_ops.monokernel.weights import linear_bf16

    source = torch.randn(256, 256, dtype=torch.bfloat16)
    linear, expected = _preshuffled_mxfp4_linear(source)

    restored = linear_bf16(
        linear,
        name="routed_expert_down_proj",
        logical_rows=256,
        logical_cols=256,
    )

    assert torch.equal(restored, expected)
    linear.weight.is_shuffled = False
    with pytest.raises(MonoUnsupported, match="must be preshuffled"):
        linear_bf16(
            linear,
            name="routed_expert_down_proj",
            logical_rows=256,
            logical_cols=256,
        )
    linear.weight.is_shuffled = True
    bad_scale = copy.copy(linear)
    bad_scale.weight_scale = linear.weight_scale.reshape(-1)[:-1]
    with pytest.raises(MonoUnsupported, match="weight_scale shape .* expected"):
        linear_bf16(
            bad_scale,
            name="routed_expert_down_proj",
            logical_rows=256,
            logical_cols=256,
        )


@pytest.mark.skipif(
    not hasattr(__import__("torch"), "float4_e2m1fn_x2"),
    reason="MXFP4 dtype is unavailable",
)
def test_shared_mxfp4_linear_dequantizes_only_requested_rows(monkeypatch):
    import torch

    from atom.model_ops.monokernel import weights

    source = torch.randn(256, 256, dtype=torch.bfloat16)
    linear, expected = _preshuffled_mxfp4_linear(source)
    dequantize = weights.dequantize_mxfp4
    seen = []

    def record_dequantize(packed, scale):
        seen.append((packed.shape, scale.shape))
        return dequantize(packed, scale)

    monkeypatch.setattr(weights, "dequantize_mxfp4", record_dequantize)
    restored = weights.linear_bf16(
        linear,
        name="routed_expert_up_proj",
        logical_rows=256,
        logical_cols=256,
        row_start=96,
        row_count=32,
    )

    assert seen == [(torch.Size([32, 128]), torch.Size([32, 8]))]
    assert torch.equal(restored, expected[96:128])


def test_shared_bf16_linear_is_unchanged():
    import torch

    from atom.model_ops.monokernel.weights import linear_bf16

    weight = torch.randn(32, 64, dtype=torch.bfloat16)
    linear = SimpleNamespace(
        input_size=64,
        output_size=32,
        quant_type=SimpleNamespace(name="No"),
        params_dtype=torch.bfloat16,
        weight=weight,
    )

    full = linear_bf16(
        linear,
        name="routed_expert_down_proj",
        logical_rows=32,
        logical_cols=64,
    )
    shard = linear_bf16(
        linear,
        name="routed_expert_up_proj",
        logical_rows=32,
        logical_cols=64,
        row_start=16,
        row_count=16,
    )

    assert full is weight
    assert torch.equal(shard, weight[16:])


def test_atom_expert_storage_matches_consumer_abi():
    import torch

    from atom.model_ops.monokernel.config import Mxfp4ScaleLayout, Mxfp4WeightLayout
    from atom.model_ops.monokernel.weights import (
        LayerWeights,
        prepare_aiter_mxfp4_expert_storage,
        prepare_mxfp4_expert_storage,
    )
    from atom.model_ops.monokernel import packing

    config = SimpleNamespace(name="test", n_experts=2, inter=128, routed_hidden=128)
    tensors = {
        "w_ug": torch.zeros(2 * 2 * 128 * 128 // 2, dtype=torch.uint8),
        "s_ug": torch.zeros(512, 8, dtype=torch.uint8),
        "w_dn": torch.zeros(2 * 128 * 128 // 2, dtype=torch.uint8),
        "s_dn": torch.zeros(256, 8, dtype=torch.uint8),
    }
    tensors["w_ug"].is_shuffled = True
    tensors["w_dn"].is_shuffled = True
    weights = LayerWeights(
        heads=1,
        t=tensors,
        config=config,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
    )

    prepared = prepare_mxfp4_expert_storage(weights)

    for output, name in zip(prepared, ("w_ug", "s_ug", "w_dn", "s_dn")):
        assert output.data_ptr() != tensors[name].data_ptr()
        if name.startswith("w_"):
            assert output.numel() == tensors[name].numel()
        else:
            assert output.numel() <= tensors[name].numel()
    aiter_prepared = prepare_aiter_mxfp4_expert_storage(weights)
    for output, name in zip(aiter_prepared, ("w_ug", "s_ug", "w_dn", "s_dn")):
        assert output.data_ptr() == tensors[name].data_ptr()
    defaults = LayerWeights(heads=1, t={}, config=config)
    assert defaults.mxfp4_weight_layout is Mxfp4WeightLayout.NATIVE
    assert defaults.mxfp4_scale_layout is Mxfp4ScaleLayout.NATIVE


def test_glm_fused_shared_expert_storage_is_canonicalized():
    import torch

    from atom.model_ops.monokernel.config import Mxfp4ScaleLayout, Mxfp4WeightLayout
    from atom.model_ops.monokernel.weights import LayerWeights, prepare_mxfp4_expert_storage

    config = SimpleNamespace(
        name="glm-test",
        n_experts=2,
        num_shared_experts=1,
        hidden=128,
        inter=128,
        routed_hidden=None,
    )
    physical_experts = 3
    tensors = {
        "w_ug": torch.zeros(physical_experts * 2 * 128 * 128 // 2, dtype=torch.uint8),
        "s_ug": torch.zeros(768, 8, dtype=torch.uint8),
        "w_dn": torch.zeros(physical_experts * 128 * 128 // 2, dtype=torch.uint8),
        "s_dn": torch.zeros(512, 8, dtype=torch.uint8),
    }
    tensors["w_ug"].is_shuffled = True
    tensors["w_dn"].is_shuffled = True
    weights = LayerWeights(
        heads=1,
        t=tensors,
        config=config,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
        physical_experts=physical_experts,
    )

    prepared = prepare_mxfp4_expert_storage(weights)

    for output, name in zip(prepared, ("w_ug", "s_ug", "w_dn", "s_dn")):
        assert output.data_ptr() != tensors[name].data_ptr()
        if name.startswith("w_"):
            assert output.numel() == tensors[name].numel()
        else:
            assert output.numel() <= tensors[name].numel()


def test_kimi_preparation_is_backend_specific(monkeypatch):
    import torch

    from atom.model_ops.monokernel.config import KIMI_K3_CONFIG
    from atom.model_ops.monokernel.k3 import prepared as prepared_module
    from atom.model_ops.monokernel.weights import LayerWeights

    cfg = KIMI_K3_CONFIG
    projection = cfg.local_heads * cfg.v_dim
    fused_width = 4 * projection + cfg.local_heads + cfg.v_dim
    tensors = {
        "w_kda_in": torch.empty(fused_width, cfg.hidden, dtype=torch.bfloat16, device="meta"),
        "w_kda_o": torch.empty(cfg.hidden, projection, dtype=torch.bfloat16, device="meta"),
        "w_r": torch.empty(cfg.n_experts, cfg.hidden, dtype=torch.bfloat16, device="meta"),
        "w_latent_down": torch.empty(cfg.routed_hidden, cfg.hidden, dtype=torch.bfloat16, device="meta"),
        "w_shared_ug": torch.empty(2 * cfg.shared_inter, cfg.hidden, dtype=torch.bfloat16, device="meta"),
        "w_shared_dn": torch.empty(cfg.hidden, cfg.shared_inter, dtype=torch.bfloat16, device="meta"),
        "w_latent_up": torch.empty(cfg.hidden // 8, cfg.routed_hidden, dtype=torch.bfloat16, device="meta"),
        "w_ug": torch.empty(1, device="meta"),
        "s_ug": torch.empty(1, device="meta"),
        "w_dn": torch.empty(1, device="meta"),
        "s_dn": torch.empty(1, device="meta"),
    }
    weights = LayerWeights(cfg.local_heads, tensors, cfg, rank=0, npes=8)
    packed_shapes = []
    expert_preparations = []

    def pack_bf16(tensor):
        packed_shapes.append(tuple(tensor.shape))
        return torch.zeros(1, dtype=torch.uint8)

    monkeypatch.setattr(prepared_module, "pack_bf16", pack_bf16)
    monkeypatch.setattr(
        prepared_module,
        "quantize_mxfp8",
        lambda tensor: (
            torch.zeros(1, dtype=torch.uint8),
            torch.zeros(1, dtype=torch.uint8),
        ),
    )
    monkeypatch.setattr(prepared_module, "pack_mxfp8_weight", lambda _tensor: torch.zeros(1, dtype=torch.uint8))
    monkeypatch.setattr(prepared_module, "pack_mxfp8_scale", lambda _tensor: torch.zeros(1, dtype=torch.uint8))
    monkeypatch.setattr(
        prepared_module,
        "prepare_aiter_mxfp4_expert_storage",
        lambda _weights: (
            expert_preparations.append("aiter")
            or tuple(torch.zeros(1, dtype=torch.uint8) for _ in range(4))
        ),
    )

    staged = prepared_module.prepare_kimi_k3_weights(weights, "staged")
    assert packed_shapes == [(cfg.n_experts, cfg.hidden)]
    assert staged.w_kda_in_packed is None
    assert staged.w_kda_o_packed is None

    packed_shapes.clear()
    tail = prepared_module.prepare_kimi_k3_tail_weights(weights)
    assert packed_shapes == [(cfg.n_experts, cfg.hidden)]
    assert tail.backend == "tail"
    assert not hasattr(tail, "w_kda_in_packed")

    packed_shapes.clear()
    staged_mtp = prepared_module.prepare_kimi_k3_weights(
        weights, "staged", mtp=True
    )
    assert packed_shapes == [
        (prepared_module.MONOKERNEL_INPUT_ROWS, cfg.hidden),
        (cfg.hidden, projection),
        (cfg.n_experts, cfg.hidden),
    ]
    assert staged_mtp.w_kda_in_packed is not None
    assert staged_mtp.w_kda_o_packed is not None

    packed_shapes.clear()
    mono = prepared_module.prepare_kimi_k3_weights(weights, "mono")
    assert packed_shapes == [
        (prepared_module.MONOKERNEL_INPUT_ROWS, cfg.hidden),
        (cfg.hidden, projection),
        (cfg.n_experts, cfg.hidden),
    ]
    assert mono.w_kda_in_packed is not None
    assert mono.w_kda_o_packed is not None
    assert expert_preparations == ["aiter", "aiter", "aiter", "aiter"]
    with pytest.raises(ValueError, match="backend"):
        staged.validate_source(weights, "mono")

def test_kimi_geometry_is_grouped_not_concurrency_specific():
    geometry = KimiDecodeGeometry(
        groups=16,
        q=8,
        conv_state_rows=10,
        state_dtype="fp16",
        replay_mode=False,
    )

    assert geometry.tokens == 128
    with pytest.raises(ValueError, match="rollback window"):
        KimiDecodeGeometry(1, 8, 9, "fp16", False)


@pytest.mark.parametrize(
    ("num_spec", "dcp", "replay", "q", "conv_rows"),
    (
        (7, 1, False, 8, 10),
        (3, 8, True, 4, 6),
        (None, 8, False, 1, 3),
    ),
)
def test_kimi_recipe_geometry_modes(
    monkeypatch, num_spec, dcp, replay, q, conv_rows
):
    module = _kimi_mono_module()
    monkeypatch.setattr(
        module,
        "envs",
        SimpleNamespace(ATOM_ENABLE_REPLAYSSM=replay),
    )
    speculative = (
        None
        if num_spec is None
        else SimpleNamespace(method="dspark", num_speculative_tokens=num_spec)
    )

    geometry = module.resolve_kimi_decode_geometry(
        SimpleNamespace(
            speculative_config=speculative,
            decode_context_parallel_size=dcp,
        )
    )

    assert geometry == KimiDecodeGeometry(1, q, conv_rows, "fp16", replay)


def test_kimi_geometry_rejects_cross_band_flag_mix(monkeypatch):
    module = _kimi_mono_module()
    monkeypatch.setattr(
        module,
        "envs",
        SimpleNamespace(ATOM_ENABLE_REPLAYSSM=True),
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            num_speculative_tokens=7,
        ),
        decode_context_parallel_size=1,
    )

    with pytest.raises(module.MonoUnsupported, match="q=8, DCP=1, ReplaySSM=1"):
        module.resolve_kimi_decode_geometry(config)


@pytest.mark.parametrize(
    ("geometry", "widths"),
    (
        (KimiDecodeGeometry(3, 8, 10, "fp16", False), [8, 8, 8]),
        (KimiDecodeGeometry(3, 4, 6, "fp16", True), [4, 4, 4]),
        (KimiDecodeGeometry(14, 1, 3, "fp16", False), [8, 4, 2]),
        (KimiDecodeGeometry(160, 1, 3, "fp16", False), [8] * 20),
    ),
)
def test_kimi_parameterized_launch_slices(geometry, widths):
    module = _kimi_mono_module()
    slices = module.kimi_launch_slices(geometry)

    assert [end - start for start, end in slices] == widths
    assert slices[0][0] == 0
    assert slices[-1][1] == geometry.tokens


def test_kimi_state_chains_select_resume_slots_on_device():
    import torch

    module = _kimi_mono_module()
    slots = torch.tensor(
        [[10, 11, 12, 13, 14, 15, 16, 17], [-1, -1, -1, -1, -1, -1, -1, -1]],
        dtype=torch.int32,
    )
    accepted = torch.tensor([3, 1], dtype=torch.int32)
    output = torch.empty(2, 9, dtype=torch.int32)
    resume_columns = torch.empty(2, dtype=torch.int64)

    result = module.build_kimi_state_chains(
        output, resume_columns, slots, accepted
    )

    assert result is output
    assert output.tolist() == [
        [12, 10, 11, 12, 13, 14, 15, 16, 17],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]


def test_kimi_c1_runs_one_q8_chain_per_request_and_publishes_aux(monkeypatch):
    import torch

    module = _kimi_mono_module()
    monkeypatch.setattr(module.torch.cuda, "is_current_stream_capturing", lambda: False)
    groups = 2
    q = 8
    samples = groups * q
    slots = torch.tensor(
        [list(range(8)), [-1] * q],
        dtype=torch.int32,
    )
    accepted = torch.tensor([1, 1], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=groups,
        num_actual_tokens=q,
        replayssm=False,
        spec_state_indices_tensor=slots,
        non_spec_state_indices_tensor=None,
        num_accepted_tokens=accepted,
    )
    conv_state = torch.zeros(16, 10, 3, dtype=torch.bfloat16)
    recurrent_state = torch.zeros(16, 1, 1, 1, dtype=torch.float16)
    calls = []

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 1
            self.is_linear_attn = True
            self.block_sparse_moe = object()

        def forward(self, *_args, **_kwargs):
            raise AssertionError("expected C1 KDA must not whole-layer fallback")

    layer = Layer()
    aux = []
    layer.register_forward_hook(
        lambda _module, _args, output: aux.append(layer.aux_hidden_state(output))
    )
    layer.aux_hidden_state = lambda output: output[0]

    class Op:
        block_write_idx = 1

        @staticmethod
        def forward(
            hidden,
            blocks,
            indices,
            conv,
            recurrent,
            *,
            num_accepted_tokens,
            x_out,
            **_kwargs,
        ):
            calls.append(
                (
                    indices.clone(),
                    num_accepted_tokens.clone(),
                    blocks.shape[1],
                    conv.shape,
                    recurrent.dtype,
                )
            )
            x_out.copy_(hidden)
            return x_out

    owned = SimpleNamespace(
        op=Op(),
        output=lambda _geometry, hidden: torch.empty_like(hidden),
    )
    model = SimpleNamespace(
        get_input_embeddings=lambda _ids: torch.zeros(
            samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16
        ),
        layers=[layer],
        start_layer=0,
        end_layer=1,
        output_attn_res=lambda hidden, *_args: (hidden, None),
    )
    runner = object.__new__(module.KimiMonoDecode)
    runner._lm = SimpleNamespace(model=model)
    runner._geometry_template = KimiDecodeGeometry(1, 8, 10, "fp16", False)
    runner._chains = {}
    runner._op = lambda *_args: owned
    runner._output = lambda *_args: torch.empty_like(_args[-1])
    context = SimpleNamespace(
        context=SimpleNamespace(is_prefill=False),
        ubatch_slices=None,
        attn_metadata=SimpleNamespace(kda_metadata=metadata),
        kv_cache_data={
            "layer_1": SimpleNamespace(
                k_cache=conv_state,
                v_cache=recurrent_state,
            )
        },
    )
    monkeypatch.setattr(module, "get_forward_context", lambda: context)

    hidden = runner.forward(
        torch.arange(samples),
        torch.arange(samples),
        torch.zeros(samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16),
    )

    assert hidden.shape[0] == samples
    assert len(calls) == groups
    assert calls[0][0].tolist() == [0, *range(8)]
    assert calls[1][0].tolist() == [-1] * 9
    assert all(call[2] == 2 for call in calls)
    assert all(call[3][1] == 10 for call in calls)
    assert all(call[4] is torch.float16 for call in calls)
    assert len(aux) == 1
    assert torch.equal(aux[0], hidden)


def test_kimi_mla_uses_baseline_attention_and_fused_tail(monkeypatch):
    import torch

    module = _kimi_mono_module()
    samples = 2
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=samples,
        num_spec_decodes=0,
        num_actual_tokens=samples,
        replayssm=False,
        non_spec_state_indices_tensor=torch.arange(samples, dtype=torch.int32),
    )
    events = []

    class PreAttn:
        def __call__(self, hidden, blocks, pending, pending2):
            events.append(("pre", pending, pending2))
            return hidden + 1, hidden

        @staticmethod
        def maybe_close_block(prefix, blocks):
            return torch.cat((blocks, prefix[:, None]), dim=1), None

    class MlaLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 12
            self.is_linear_attn = False
            self.block_sparse_moe = object()
            self.self_attention_attn_res = PreAttn()
            self.self_attn = lambda _positions, hidden: hidden + 2

        def forward(self, *_args, **_kwargs):
            raise AssertionError("MLA+MoE must use the hybrid tail")

    class DenseLayer(torch.nn.Module):
        layer_idx = 13
        is_linear_attn = True

        def forward(
            self,
            _positions,
            hidden,
            blocks,
            *,
            pending_add,
            pending_add2,
        ):
            events.append(("dense", pending_add, pending_add2))
            return hidden + 1, None, None, blocks

    mla = MlaLayer()
    dense = DenseLayer()
    captures = []
    mla.register_forward_hook(
        lambda _module, _args, output: captures.append(output[0].clone())
    )

    class Tail:
        @staticmethod
        def forward_from_attention(
            prefix,
            blocks,
            attention_delta,
            *,
            x_out,
            **_kwargs,
        ):
            events.append(("tail", prefix, blocks.shape[1]))
            x_out.copy_(attention_delta + 3)
            return x_out

    owned = SimpleNamespace(
        op=Tail(),
        output=lambda _geometry, hidden: torch.empty_like(hidden),
    )
    model = SimpleNamespace(
        get_input_embeddings=lambda _ids: pytest.fail("inputs_embeds must be used"),
        layers=[mla, dense],
        start_layer=0,
        end_layer=2,
        output_attn_res=lambda hidden, *_args: (hidden, None),
    )
    runner = object.__new__(module.KimiMonoDecode)
    runner._lm = SimpleNamespace(model=model)
    runner._geometry_template = KimiDecodeGeometry(1, 1, 3, "fp16", False)
    runner._chains = {}

    def get_op(_layer, _geometry, kind, width):
        assert kind == "tail"
        assert width in {1, 2}
        return owned

    runner._op = get_op
    runner._output = lambda *_args: torch.empty_like(inputs)
    context = SimpleNamespace(
        context=SimpleNamespace(is_prefill=False),
        ubatch_slices=None,
        attn_metadata=SimpleNamespace(kda_metadata=metadata),
        kv_cache_data={},
    )
    monkeypatch.setattr(module, "get_forward_context", lambda: context)
    inputs = torch.zeros(samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16)

    output = runner.forward(torch.arange(samples), torch.arange(samples), inputs)

    assert torch.equal(output, torch.full_like(output, 7))
    assert events[-2:] == [("tail", None, 1), ("dense", None, None)]
    assert len(captures) == 1
    assert torch.equal(captures[0], torch.full_like(output, 6))


def test_kimi_replayssm_uses_baseline_kda_and_fused_tail(monkeypatch):
    import torch

    module = _kimi_mono_module()
    groups = 2
    q = 4
    samples = groups * q
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=groups,
        num_actual_tokens=samples,
        replayssm=True,
        spec_state_indices_tensor=torch.arange(
            samples, dtype=torch.int32
        ).view(groups, q),
        non_spec_state_indices_tensor=None,
        num_accepted_tokens=torch.ones(groups, dtype=torch.int32),
    )
    events = []

    class PreAttn:
        def __call__(self, hidden, blocks, pending, pending2):
            assert pending is None and pending2 is None
            return hidden + 1, hidden

        @staticmethod
        def maybe_close_block(prefix, blocks):
            return torch.cat((blocks, prefix[:, None]), dim=1), None

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 12
            self.is_linear_attn = True
            self.block_sparse_moe = object()
            self.self_attention_attn_res = PreAttn()

        def self_attn(self, hidden):
            events.append(("baseline_kda", hidden.shape[0]))
            return hidden + 2

        def forward(self, *_args, **_kwargs):
            raise AssertionError("ReplaySSM KDA must use the hybrid tail")

    layer = Layer()
    aux = []
    layer.register_forward_hook(
        lambda _module, _args, output: aux.append(output[0].clone())
    )

    class Tail:
        @staticmethod
        def forward_from_attention(
            prefix,
            blocks,
            attention_delta,
            *,
            x_out,
            **_kwargs,
        ):
            events.append(("tail", attention_delta.shape[0], prefix))
            assert blocks.shape[1] == 1
            x_out.copy_(attention_delta + 3)
            return x_out

    owned = SimpleNamespace(op=Tail())
    model = SimpleNamespace(
        get_input_embeddings=lambda _ids: pytest.fail("inputs_embeds must be used"),
        layers=[layer],
        start_layer=0,
        end_layer=1,
        output_attn_res=lambda hidden, *_args: (hidden, None),
    )
    runner = object.__new__(module.KimiMonoDecode)
    runner._lm = SimpleNamespace(model=model)
    runner._geometry_template = KimiDecodeGeometry(1, 4, 6, "fp16", True)
    runner._spec_chains = lambda *_args: pytest.fail(
        "ReplaySSM reconstruction belongs to baseline KDA"
    )
    runner._op = lambda _layer, _geometry, kind, width: (
        owned
        if kind == "tail" and width == 4
        else pytest.fail("unexpected ReplaySSM native op")
    )
    runner._output = lambda *_args: torch.empty_like(inputs)
    context = SimpleNamespace(
        context=SimpleNamespace(is_prefill=False),
        ubatch_slices=None,
        attn_metadata=SimpleNamespace(kda_metadata=metadata),
        kv_cache_data={},
    )
    monkeypatch.setattr(module, "get_forward_context", lambda: context)
    inputs = torch.zeros(samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16)

    output = runner.forward(torch.arange(samples), torch.arange(samples), inputs)

    assert events[0] == ("baseline_kda", samples)
    assert [(name, width) for name, width, _ in events[1:]] == [
        ("tail", 4),
        ("tail", 4),
    ]
    assert all(prefix is None for _, _, prefix in events[1:])
    assert torch.equal(output, torch.full_like(output, 6))
    assert len(aux) == 1 and torch.equal(aux[0], output)


def test_kimi_q1_chunks_arbitrary_graph_batch(monkeypatch):
    import torch

    module = _kimi_mono_module()
    samples = 14
    state_indices = torch.arange(samples, dtype=torch.int32)
    state_indices[-1] = -1
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=samples - 1,
        num_spec_decodes=0,
        num_actual_tokens=samples - 1,
        replayssm=False,
        non_spec_state_indices_tensor=state_indices,
    )
    calls = []

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 1
            self.is_linear_attn = True
            self.block_sparse_moe = object()

        def forward(self, *_args, **_kwargs):
            raise AssertionError("q1 KDA must not whole-layer fallback")

    layer = Layer()

    class Op:
        block_write_idx = 0

        def __init__(self, width):
            self.width = width

        def forward(
            self,
            hidden,
            _blocks,
            indices,
            _conv,
            _recurrent,
            *,
            num_accepted_tokens,
            x_out,
            **_kwargs,
        ):
            assert num_accepted_tokens is None
            calls.append((self.width, indices.clone()))
            x_out.copy_(hidden)
            return x_out

    model = SimpleNamespace(
        get_input_embeddings=lambda _ids: pytest.fail("inputs_embeds must be used"),
        layers=[layer],
        start_layer=0,
        end_layer=1,
        output_attn_res=lambda hidden, *_args: (hidden, None),
    )
    runner = object.__new__(module.KimiMonoDecode)
    runner._lm = SimpleNamespace(model=model)
    runner._geometry_template = KimiDecodeGeometry(1, 1, 3, "fp16", False)
    runner._op = lambda _layer, _geometry, kind, width: SimpleNamespace(
        op=Op(width)
    )
    runner._output = lambda *_args: torch.empty_like(inputs)
    context = SimpleNamespace(
        context=SimpleNamespace(is_prefill=False),
        ubatch_slices=None,
        attn_metadata=SimpleNamespace(kda_metadata=metadata),
        kv_cache_data={
            "layer_1": SimpleNamespace(
                k_cache=torch.zeros(16, 3, 3, dtype=torch.bfloat16),
                v_cache=torch.zeros(16, 1, dtype=torch.float16),
            )
        },
    )
    monkeypatch.setattr(module, "get_forward_context", lambda: context)
    inputs = torch.zeros(samples, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16)

    output = runner.forward(torch.arange(samples), torch.arange(samples), inputs)

    assert [width for width, _ in calls] == [8, 4, 2]
    assert torch.equal(torch.cat([indices for _, indices in calls]), state_indices)
    assert torch.equal(output, inputs)


def test_kimi_block_close_prefix_handoff():
    import torch

    from atom.model_ops.attention_residual_contract import resolve_attn_res_prefix

    first = torch.ones(2, 3)
    second = torch.full((2, 3), 2.0)

    prefix, pending, pending2 = resolve_attn_res_prefix(None, first, second)

    assert prefix is first
    assert pending is second
    assert pending2 is None


def test_kimi_fold_pending_values():
    import torch

    module = _kimi_mono_module()
    first = torch.ones(2, 3)
    second = torch.full((2, 3), 2.0)
    third = torch.full((2, 3), 3.0)
    assert torch.equal(module.fold_kimi_pending(first, second, third), first + second + third)
    assert torch.equal(module.fold_kimi_pending(None, first, second), first + second)


def test_kimi_attn_res_adds_delta_on_gfx950():
    import torch

    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")

    from atom.model_ops.monokernel.k3.attn_res import KimiK3AttnRes

    samples, hidden = 1, 7168
    generator = torch.Generator(device="cuda").manual_seed(0)
    make = lambda *shape: torch.randn(
        *shape,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    prefix = make(samples, hidden)
    delta = make(samples, hidden)
    blocks = make(samples, 1, hidden)
    weights = tuple(make(hidden) for _ in range(3))
    updated = torch.empty_like(prefix)
    output = torch.empty_like(prefix)

    KimiK3AttnRes(samples, hidden, 1, True, -1)(
        prefix,
        delta,
        blocks,
        *weights,
        updated,
        output,
    )
    torch.cuda.synchronize()

    assert torch.equal(updated, (prefix.float() + delta.float()).to(torch.bfloat16))


def test_atom_mxfp4_indices_cover_kimi_expert_storage():
    from atom.model_ops.monokernel.layout import (
        atom_mxfp4_scale_index,
        atom_mxfp4_weight_index,
    )

    for rows, k_size in ((768, 3584), (3584, 384)):
        weight_indices = {
            atom_mxfp4_weight_index(row_group, k_chunk, lane, step, k_size)
            for row_group in range(rows // 16)
            for k_chunk in range(k_size // 128)
            for lane in range(64)
            for step in range(4)
        }
        assert weight_indices == set(range(rows * k_size // 8))

        groups = k_size // 32
        padded_groups = (groups + 7) // 8 * 8
        scale_indices = {
            atom_mxfp4_scale_index(row, group, groups)
            for row in range(rows)
            for group in range(groups)
        }
        assert len(scale_indices) == rows * groups
        assert max(scale_indices) < rows * padded_groups
