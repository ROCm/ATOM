# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ast
import re
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.model_ops.monokernel.config import (
    GLM5_AGENTX_BATCHES,
    GLM5_GRAPH_BATCHES,
    glm5_kernel_samples,
    glm5_tp_config,
)
from atom.model_ops.monokernel.dispatch import glm52_native_config
from atom.model_ops.monokernel.glm import layout as glm_layout


def _mono_module(name):
    import importlib
    from unittest.mock import MagicMock

    from tests.aiter_stub import stubbed_aiter

    with stubbed_aiter():
        parallel_state = sys.modules["aiter.dist.parallel_state"]
        for attr in (
            "get_tensor_model_parallel_rank",
            "get_tensor_model_parallel_world_size",
            "get_tp_group",
        ):
            parallel_state.__dict__.setdefault(attr, MagicMock())
        return importlib.import_module(name)


def _kimi_mono_module():
    return _mono_module("atom.models.kimi_k3_mono")


def _glm_mono_module():
    return _mono_module("atom.models.glm52_mono")


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


def _kimi_c1_context():
    import torch

    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=1,
        num_actual_tokens=8,
        replayssm=False,
        spec_state_indices_tensor=torch.arange(8, dtype=torch.int32).view(1, 8),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32),
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
assert 'atom.model_ops.monokernel.k3.staged' not in sys.modules
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
    with pytest.raises(
        RuntimeError, match="allocation/export failed.*rank 0 allocation failed"
    ):
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
        lambda *_args, **_kwargs: pytest.fail(
            "failed open must skip address publication"
        ),
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
        root / "k3" / "router_projection.py",
    )
    scaled_calls = []
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
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
        dtype = next(
            keyword.value for keyword in load.keywords if keyword.arg == "dtype"
        )
        assert isinstance(dtype, ast.Attribute)
        assert isinstance(dtype.value, ast.Name) and dtype.value.id == "T"
        assert dtype.attr == "i32"


def test_native_decode_flag(monkeypatch):
    from atom.utils import envs

    name = "ATOM_NATIVE_DECODE_MONOKERNEL"
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) == "off"
    for value in ("off", "staged_c1"):
        monkeypatch.setenv(name, value.upper())
        assert getattr(envs, name) == value
    monkeypatch.setenv(name, "on")
    with pytest.raises(ValueError, match="off, staged_c1"):
        getattr(envs, name)


def test_glm_native_fp4_mfma_flag(monkeypatch):
    from atom.utils import envs

    name = "ATOM_GLM_NATIVE_FP4_MFMA"
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) is False
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name) is True


def test_glm_native_fp4_agentx_concurrency_contract():
    assert GLM5_AGENTX_BATCHES == (1, 2)


@pytest.mark.parametrize(
    "override",
    [
        {"samples": 4},
        {"tp_size": 8},
        {"kv_cache_dtype": "auto"},
        {"kv_cache_dtype": "bf16"},
        {"mtp": False},
        {"query_length": 1},
        {"query_length": 4},
    ],
)
def test_unsupported_glm_forward_falls_back(override):
    args = {
        "samples": 6,
        "tp_size": 4,
        "kv_cache_dtype": "fp8",
        "mtp": True,
        "query_length": 6,
    }
    args.update(override)
    assert glm52_native_config(**args) is None


def test_glm_announces_samples_once_on_rank_zero(monkeypatch):
    import torch

    module = _glm_mono_module()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
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
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(
        module, "get_tp_group", lambda: SimpleNamespace(cpu_group=object())
    )
    monkeypatch.setattr(module, "_bf16_vector", lambda tensor, *_args: tensor)
    monkeypatch.setattr(module, "_layer_weights", lambda *_args: object())
    monkeypatch.setattr(module, "_GlmLayerOp", lambda *_args: FakeOwned())
    monkeypatch.setattr(
        module, "tp_uniform_local_validation", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(module.logger, "info", lambda *args: messages.append(args))

    def runner():
        value = object.__new__(module.Glm52MonoDecode)
        value._lm = SimpleNamespace(
            model=SimpleNamespace(norm=SimpleNamespace(weight=object()))
        )
        value._atom_config = SimpleNamespace(
            hf_config=SimpleNamespace(index_topk=2048),
            tensor_parallel_size=4,
            kv_cache_dtype="fp8",
            speculative_config=SimpleNamespace(method="mtp"),
        )
        value._shard = glm5_tp_config(4)
        value._ops, value._weights, value._prepared, value._runtimes = {}, {}, {}, {}
        value._refused, value._announced = set(), set()
        value._mono_layers = lambda: layers
        return value

    rank_zero = runner()
    assert rank_zero._prepare(6, 6)
    assert rank_zero._prepare(6, 6)
    assert rank_zero._prepare(12, 6)
    rank[0] = 1
    assert runner()._prepare(6, 6)
    assert messages == [
        ("GLM-5.2 MonoKernel on: S=%d chunk=%d", 6, 6),
        ("GLM-5.2 MonoKernel on: S=%d chunk=%d", 12, 12),
    ]
    rank_zero.close()
    assert rank_zero._announced == set()


def test_glm_scaled_fp4_dispatch_contract():
    for query_length in (5, 6):
        config = glm52_native_config(
            samples=query_length,
            tp_size=4,
            kv_cache_dtype="fp8",
            mtp=True,
            query_length=query_length,
        )
        assert config == glm5_tp_config(4)


def test_glm_tp4_geometry_and_padded_graph_ladder():
    tp8, tp4 = glm5_tp_config(8), glm5_tp_config(4)
    assert (tp8.local_heads, tp8.inter) == (8, 256)
    assert (tp4.local_heads, tp4.inter) == (16, 512)
    assert tp8.local_heads * 8 == tp4.local_heads * 4
    assert tp8.inter * 8 == tp4.inter * 4
    assert tp4.n_experts + tp4.num_shared_experts == 257
    for batch in GLM5_GRAPH_BATCHES:
        samples = batch * 6
        assert (
            glm52_native_config(
                samples=samples,
                tp_size=4,
                kv_cache_dtype="fp8",
                mtp=True,
                query_length=6,
            )
            == tp4
        )
        chunk = glm5_kernel_samples(samples, 6)
        assert chunk in (6, 12) and chunk % 6 == 0 and samples % chunk == 0
        samples = batch * 5
        assert (
            glm52_native_config(
                samples=samples,
                tp_size=4,
                kv_cache_dtype="fp8",
                mtp=True,
                query_length=5,
            )
            == tp4
        )
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
        actual = glm_layout.sparse_cache_rows(
            flat, sample=sample, topk=2048, sparse_kv_indptr=indptr
        )
        assert actual == row
        request, token = divmod(sample, 6)
        assert set(actual).isdisjoint(
            slots[request * 6 + token + 1 : (request + 1) * 6]
        )


def test_glm_tp4_symmetric_and_split_schedule_contracts():
    cfg, samples = glm5_tp_config(4), 12
    scratch, symmetric = glm_layout.layout(
        samples, cfg.local_heads, 4, 2048, inter=cfg.inter
    )
    part = 4 * samples * cfg.hidden * 8
    assert scratch["mid"] >= 0
    assert symmetric["_part_stride"] == part and symmetric["_bytes"] == 4 * part
    stages = dict(
        glm_layout.stage_tasks(
            samples, cfg.local_heads, 2048, expert_mxfp4=True, inter=cfg.inter
        )
    )
    assert stages["split"] == samples * 2 * (
        2048 // glm_layout.sparse_keys_per_task(samples, cfg.local_heads)
    )
    assert stages["ug"] >= samples * 9 * (cfg.inter // glm_layout.UG_TILE)
    assert [glm_layout.down_prefetch_batch(s, True) for s in (4, 5, 6, 8, 12)] == [
        9,
        4,
        4,
        3,
        3,
    ]


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
    assert [glm_layout.split_score_column(wave, 0) for wave in range(8)] == list(
        range(8)
    )
    assert [glm_layout.fp8_kv_upper_pair_lane(lane) for lane in range(0, 16, 4)] == [
        2,
        6,
        10,
        14,
    ]
    assert [
        glm_layout.fp8_pe_upper_pair_lane(lane) for lane in range(0, 16, 2)
    ] == list(range(1, 16, 2))
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
        assert (
            glm_layout.sparse_cache_rows(
                physical,
                sample=sample,
                topk=4,
                sparse_kv_indptr=indptr,
            )
            == []
        )


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


def test_glm_bf16_linear_mapping():
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
    from atom.model_ops.monokernel.packing import pack_ptpc_fp8
    from atom.model_ops.monokernel.weights import linear_ptpc_fp8

    source = torch.randn(16, 64, dtype=torch.bfloat16)
    linear, _ = _preshuffled_per_token_fp8_linear(source)
    weight, scale = linear_ptpc_fp8(
        linear,
        name="q_b_proj",
        logical_rows=16,
        logical_cols=64,
    )

    assert weight.data_ptr() == linear.weight.data_ptr()
    assert scale.data_ptr() == linear.weight_scale.data_ptr()
    native = torch.randn(16, 64).to(torch.float8_e4m3fn)
    assert torch.equal(
        pack_ptpc_fp8(native).view(native.dtype).view_as(native),
        _preshuffle_linear_weight(native),
    )

    batched = torch.randn(2, 6, 64).to(torch.float8_e4m3fn)
    batched_scale = torch.ones(1, dtype=torch.float32)
    flat, head_scale = module._batched_fp8(
        batched,
        batched_scale,
        name="W_V",
        shape=(2, 6, 64),
    )
    assert flat.shape == (12, 64)
    assert flat.data_ptr() == batched.data_ptr()
    assert head_scale.data_ptr() == batched_scale.data_ptr()


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
        tensor_parallel_size=4,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_dp_attention=False,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=1,
        enable_expert_parallel=False,
        enable_tbo=False,
        enable_tbo_decode=False,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=5),
        kv_cache_dtype="fp8",
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
    assert module.Glm52MonoDecode(causal_lm, atom_config)._enabled

    module.envs.ATOM_MLA_PAGE_SIZE = 2
    assert not module.Glm52MonoDecode(causal_lm, atom_config)._enabled


def test_glm_c1_c2_padded_graph_dispatches_q6(monkeypatch):
    import torch

    module = _glm_mono_module()
    samples, active = 16 * 6, 2 * 6
    runner = object.__new__(module.Glm52MonoDecode)
    runner._enabled = True
    runner._shard = glm5_tp_config(4)
    runner._atom_config = SimpleNamespace(
        tensor_parallel_size=4,
        kv_cache_dtype="fp8",
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=5),
        enable_dp_attention=False,
        decode_context_parallel_size=1,
    )
    runner._lm = SimpleNamespace(
        model=SimpleNamespace(aux_hidden_state_layers=[], layers=[])
    )
    runner._mono_layers = list
    seen = []
    runner._prepare = (
        lambda rows, query_length: seen.append((rows, query_length)) or True
    )
    metadata = SimpleNamespace(
        max_seqlen_q=6,
        slot_mapping=torch.cat(
            (
                torch.arange(active, dtype=torch.int64),
                torch.full((samples - active,), -1, dtype=torch.int64),
            )
        ),
        sparse_kv_indptr=torch.cat(
            (
                torch.arange(active + 1, dtype=torch.int32),
                torch.full((samples - active,), active, dtype=torch.int32),
            )
        ),
    )
    context = SimpleNamespace(is_prefill=False, scheduled_bs=2, running_tokens=samples)
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
        torch.arange(samples), torch.arange(samples, dtype=torch.int64), None, None
    )
    assert seen == [(samples, 6)]

    context.scheduled_bs = 1
    assert runner.supports(
        torch.arange(samples), torch.arange(samples, dtype=torch.int64), None, None
    )
    assert seen == [(samples, 6), (samples, 6)]

    context.scheduled_bs = 4
    assert not runner.supports(
        torch.arange(samples), torch.arange(samples, dtype=torch.int64), None, None
    )


def test_glm_fp8_fused_576_cache_has_explicit_device_io():
    kernel = (
        Path(__file__).parents[1]
        / "atom"
        / "model_ops"
        / "monokernel"
        / "glm"
        / "kernel.py"
    ).read_text()
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


def test_kimi_staged_c1_config_is_explicit(monkeypatch):
    module = _kimi_mono_module()
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark", num_speculative_tokens=7
        ),
        decode_context_parallel_size=1,
    )
    monkeypatch.setattr(
        module, "envs", SimpleNamespace(ATOM_ENABLE_REPLAYSSM=False)
    )
    module.validate_kimi_c1_config(config)

    config.speculative_config.num_speculative_tokens = 3
    with pytest.raises(module.MonoUnsupported, match="DSpark7"):
        module.validate_kimi_c1_config(config)


def test_kimi_default_off_does_not_inspect_runtime_config():
    runner = _kimi_mono_module().KimiStagedC1Decode(None, object(), "off")
    assert runner._enabled is False
    assert runner._ops == {}
    assert runner._prepared == {}
    assert runner._outputs == {}
    assert runner._prebuilt is False


def test_kimi_staged_c1_supports_only_one_q8_request(monkeypatch):
    import torch

    module = _kimi_mono_module()
    runner = object.__new__(module.KimiStagedC1Decode)
    runner._enabled = True
    monkeypatch.setattr(module, "get_forward_context", _kimi_c1_context)
    inputs = torch.zeros(8, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16)

    assert runner.supports(torch.arange(8), torch.arange(8), None, inputs)
    assert not runner.supports(torch.arange(16), torch.arange(16), None, None)
    assert not runner.supports(torch.arange(8), torch.arange(8), object(), inputs)
    assert not runner.supports(torch.arange(8), torch.arange(8), None, inputs.float())


def test_kimi_staged_c1_prebuilds_only_tail_ops():
    module = _kimi_mono_module()
    layers = [
        SimpleNamespace(layer_idx=0),
        SimpleNamespace(layer_idx=1, block_sparse_moe=object()),
        SimpleNamespace(layer_idx=2, block_sparse_moe=object()),
    ]
    runner = object.__new__(module.KimiStagedC1Decode)
    runner._enabled = True
    runner._prebuilt = False
    runner._lm = SimpleNamespace(
        model=SimpleNamespace(layers=layers, start_layer=0, end_layer=3)
    )
    seen = []
    runner._op = lambda layer: seen.append(layer.layer_idx)

    runner.prepare()
    runner.prepare()

    assert seen == [1, 2]
    assert runner._prebuilt


def test_kimi_tail_preparation_has_no_attention_weights(monkeypatch):
    import torch

    from atom.model_ops.monokernel.config import KIMI_K3_CONFIG
    from atom.model_ops.monokernel.k3 import prepared as prepared_module
    from atom.model_ops.monokernel.weights import LayerWeights

    cfg = KIMI_K3_CONFIG
    tensors = {
        "w_r": torch.empty(
            cfg.n_experts, cfg.hidden, dtype=torch.bfloat16, device="meta"
        ),
        "w_latent_down": torch.empty(
            cfg.routed_hidden, cfg.hidden, dtype=torch.bfloat16, device="meta"
        ),
        "w_shared_ug": torch.empty(
            2 * cfg.shared_inter, cfg.hidden, dtype=torch.bfloat16, device="meta"
        ),
        "w_shared_dn": torch.empty(
            cfg.hidden, cfg.shared_inter, dtype=torch.bfloat16, device="meta"
        ),
        "w_latent_up": torch.empty(
            cfg.hidden // 8, cfg.routed_hidden, dtype=torch.bfloat16, device="meta"
        ),
        "w_ug": torch.empty(1, device="meta"),
        "s_ug": torch.empty(1, device="meta"),
        "w_dn": torch.empty(1, device="meta"),
        "s_dn": torch.empty(1, device="meta"),
    }
    weights = LayerWeights(cfg.local_heads, tensors, cfg, rank=0, npes=8)
    packed_shapes = []
    monkeypatch.setattr(
        prepared_module,
        "pack_bf16",
        lambda tensor: packed_shapes.append(tuple(tensor.shape))
        or torch.zeros(1, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        prepared_module,
        "quantize_mxfp8",
        lambda _tensor: (
            torch.zeros(1, dtype=torch.uint8),
            torch.zeros(1, dtype=torch.uint8),
        ),
    )
    monkeypatch.setattr(
        prepared_module,
        "pack_mxfp8_weight",
        lambda _tensor: torch.zeros(1, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        prepared_module,
        "pack_mxfp8_scale",
        lambda _tensor: torch.zeros(1, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        prepared_module,
        "prepare_aiter_mxfp4_expert_storage",
        lambda _weights: tuple(
            torch.zeros(1, dtype=torch.uint8) for _ in range(4)
        ),
    )

    prepared = prepared_module.prepare_kimi_k3_tail_weights(weights)

    assert packed_shapes == [(cfg.n_experts, cfg.hidden)]
    assert not hasattr(prepared, "w_kda_in_packed")
    prepared.validate_source(weights, "tail")


def test_kimi_staged_c1_keeps_production_attention_and_attnres():
    import torch

    module = _kimi_mono_module()
    events = []

    class PreAttn:
        def __call__(self, hidden, blocks, pending, pending2):
            events.append(("pre_attn_res", pending, pending2))
            return hidden + 1, hidden

        @staticmethod
        def maybe_close_block(prefix, blocks):
            return torch.cat((blocks, prefix[:, None]), dim=1), prefix

    class PostAttn:
        def __call__(self, prefix, blocks, attention_delta):
            events.append(("mlp_attn_res", blocks.shape[1]))
            return attention_delta + 2, prefix + attention_delta

    class Layer:
        layer_idx = 1
        is_linear_attn = True
        block_sparse_moe = object()
        self_attention_attn_res = PreAttn()
        mlp_attn_res = PostAttn()

        @staticmethod
        def self_attn(hidden):
            events.append(("production_kda", hidden.shape[0]))
            return hidden + 1

    inputs = torch.zeros(8, module.KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16)
    layer = Layer()
    model = SimpleNamespace(
        get_input_embeddings=lambda _ids: pytest.fail("inputs_embeds must be used"),
        layers=[layer],
        start_layer=0,
        end_layer=1,
        output_attn_res=lambda hidden, *_args: (hidden, None),
    )
    runner = object.__new__(module.KimiStagedC1Decode)
    runner._lm = SimpleNamespace(model=model)

    def run_tail(_layer, updated_prefix, moe_input):
        events.append(("fused_tail", updated_prefix, moe_input))
        return updated_prefix + moe_input

    runner._run_tail = run_tail
    output = runner.forward(torch.arange(8), torch.arange(8), inputs)

    assert [event[0] for event in events] == [
        "pre_attn_res",
        "production_kda",
        "mlp_attn_res",
        "fused_tail",
    ]
    assert output.shape == inputs.shape


def test_kimi_runner_close_is_idempotent():
    module = _kimi_mono_module()

    class Owned:
        def __init__(self):
            self.calls = 0

        def close(self):
            self.calls += 1

    runner = object.__new__(module.KimiStagedC1Decode)
    owned = Owned()
    runner._ops = {1: owned}
    runner._weights = {1: object()}
    runner._prepared = {1: object()}
    runner._outputs = {1: object()}
    runner._refused = {2}
    runner._announced = True
    runner._prebuilt = True

    runner.close()
    runner.close()

    assert owned.calls == 1
    assert runner._prepared == {}
    assert runner._outputs == {}
    assert runner._refused == set()
    assert runner._announced is False


def test_rejected_kimi_whole_layer_modules_are_absent():
    root = Path(__file__).parents[1] / "atom" / "model_ops" / "monokernel" / "k3"
    rejected = {
        "attn_res.py",
        "kda.py",
        "kda_recurrence.py",
        "kernel.py",
        "mla.py",
        "mla_kernel.py",
        "op.py",
        "torch_fusions.py",
    }
    assert not any((root / name).exists() for name in rejected)


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
    from atom.model_ops.monokernel.weights import (
        LayerWeights,
        prepare_mxfp4_expert_storage,
    )

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


def test_glm_kernel_uses_atom_bf16_rope_cache_abi():
    kernel = (
        Path(__file__).parents[1]
        / "atom"
        / "model_ops"
        / "monokernel"
        / "glm"
        / "kernel.py"
    ).read_text()
    assert "ld_f32(_rsrc(rope_cos)" not in kernel
    assert "ld_f32(_rsrc(rope_sin)" not in kernel
    assert len(re.findall(r"ld_bf16\(\s*_rsrc\(rope_cos\)", kernel)) == 3
    assert len(re.findall(r"ld_bf16\(\s*_rsrc\(rope_sin\)", kernel)) == 3


def test_glm_native_boundary_reduces_deferred_tp_partial():
    adapter = (
        Path(__file__).parents[1] / "atom" / "models" / "glm52_mono.py"
    ).read_text()
    assert (
        "if residual is not None and layer.input_layernorm.fused_allreduce:" in adapter
    )
    assert "get_tp_group().all_reduce(hidden, ca_fp8_quant=False)" in adapter
