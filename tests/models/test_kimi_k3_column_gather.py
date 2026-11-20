# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Typed column ownership and the real target-to-drafter tap path."""

import os
from types import MethodType, SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter
from torch import nn
from torch.library import Library

_LIB = Library("atom_kimi_column_dataflow_test", "FRAGMENT")


@pytest.fixture(scope="module")
def modules():
    if os.getenv("ATOM_TEST_COLUMN_GATHER_DISTRIBUTED") == "1":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            model=load("atom.models.kimi_k3.model"),
            ops=load("atom.model_ops.kimi_k3.attention_residual"),
            drafter=load("atom.spec_decode.drafter"),
            dspark=load("atom.spec_decode.dspark_proposer"),
            config=load("atom.config"),
            plugin=load("atom.plugin"),
            runner=load("atom.model_engine.model_runner"),
            peer_memory=load("atom.mono.runtime.peer_memory"),
        )


@pytest.fixture
def context(modules, monkeypatch):
    ctx = SimpleNamespace(
        is_draft=False, is_prefill=False, ubatch_token_offset=0, ubatch_token_order=None
    )
    monkeypatch.setattr(
        modules.drafter, "get_forward_context", lambda: SimpleNamespace(context=ctx)
    )
    monkeypatch.setattr(modules.drafter, "pcp_aux_rows", lambda n: None)
    monkeypatch.setattr(
        modules.model, "get_pp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(modules.plugin, "is_plugin_mode", lambda: False)
    return ctx


def test_columns_cannot_be_read_as_a_full_tensor(modules):
    columns = torch.ones((3, 4), dtype=torch.bfloat16)
    pending = modules.ops.ColumnTail(columns, "gather", 2)
    assert pending.columns is columns
    with pytest.raises(TypeError):
        torch.ones((3, 8), dtype=torch.bfloat16) + pending


@pytest.mark.parametrize(
    "hidden,world,fits",
    [
        (7168, 8, True),
        (8, 2, True),
        (7, 2, False),
        (7168, 3, False),
        (7168, 1, False),
        (8, 0, False),
    ],
)
def test_peer_geometry_is_explicit(modules, hidden, world, fits):
    assert modules.ops.PeerColumnGather.fits(hidden, world) is fits


def test_peer_owner_rejects_wrong_or_closed_shares(modules):
    owner = object.__new__(modules.ops.PeerColumnGather)
    owner.closed, owner.rows, owner.cols = False, 64, 4
    owner.scratch = torch.zeros(1, dtype=torch.int64)
    with pytest.raises(ValueError):
        owner.kernel_args(torch.zeros((65, 4), dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        owner.kernel_args(torch.zeros((2, 5), dtype=torch.bfloat16)[:, :4])
    owner.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        owner.kernel_args(torch.zeros((2, 4), dtype=torch.bfloat16))


def test_typed_column_producer_survives_aot_without_input_clones(modules, monkeypatch):
    from atom.utils.custom_register import direct_register_custom_op

    seen = []

    def compute(x):
        seen.append(x.untyped_storage().data_ptr())
        return x[:, :896] + 1

    block = SimpleNamespace(
        column_gather_name="gather",
        prefix="layer",
        tp_size=8,
        _can_defer_column_tail=lambda x: True,
        _column_moe_forward=compute,
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"layer": block})
    )
    monkeypatch.setattr(modules.config, "get_current_atom_config", lambda: config)
    producer = direct_register_custom_op(
        "produce",
        modules.ops._column_moe_op,
        [],
        fake_impl=modules.ops._column_moe_fake,
        target_lib=_LIB,
        dispatch_key="CPU",
    )

    def gather_cpu(columns: torch.Tensor, world: int) -> torch.Tensor:
        return torch.cat([columns] * world, dim=-1)

    gather = direct_register_custom_op(
        "gather",
        gather_cpu,
        [],
        fake_impl=modules.ops._gather_columns_fake,
        target_lib=_LIB,
        dispatch_key="CPU",
    )
    monkeypatch.setattr(torch.ops.aiter, "kimi_k3_moe_columns", producer)
    monkeypatch.setattr(torch.ops.aiter, "kimi_k3_gather_columns", gather)
    owner = torch.arange(2 * 7200, dtype=torch.float32).reshape(2, 7200)
    x = owner[:, :7168]

    def run(x):
        pending, shared = modules.model.KimiSparseMoeBlock.forward(block, x)
        assert shared is None
        return modules.ops.materialize_column_output(pending)

    compiled = torch.compile(run, backend="aot_eager", fullgraph=True)
    try:
        expected = run(x)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.shape == x.shape
        assert seen and all(ptr == owner.untyped_storage().data_ptr() for ptr in seen)
    finally:
        torch._dynamo.reset()


class _ResidualSite(nn.Module):
    def __init__(self, columns_type):
        super().__init__()
        self.columns_type = columns_type

    def forward(self, prefix, bank, first=None, second=None):
        if isinstance(first, self.columns_type):
            first = torch.cat([first.columns] * first.world, dim=-1)
        summed = prefix.float()
        if first is not None:
            summed = summed + first.float()
        if second is not None:
            summed = summed + second.float()
        summed = summed.to(prefix.dtype)
        return torch.full_like(summed, -99), summed


class _Layer(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.self_attention_attn_res = _ResidualSite(modules.ops.ColumnTail)
        self.aux_hidden_state = modules.model.KimiDecoderLayer.aux_hidden_state

    def forward(self, prefix, first, second, bank):
        return prefix, first, second, bank


def _armed(modules, *, ids=(0,), extract=None, capture="output"):
    target = nn.Module()
    target.start_layer, target.end_layer = 0, 3
    target.layers = nn.ModuleList([_Layer(modules) for _ in range(3)])
    target.embed_tokens = nn.Identity()
    target.aux_tap_site = MethodType(modules.model.KimiLinearModel.aux_tap_site, target)
    spec = modules.drafter.AuxCaptureSpec(
        ids, 8, extract or modules.dspark.DSparkProposer._extract_layer_hidden, capture
    )
    drafter = object.__new__(modules.dspark.DSparkProposer)
    drafter.max_num_tokens, drafter.device, drafter.dtype = 8, "cpu", torch.bfloat16
    drafter._aux_capture_spec = lambda: spec
    drafter.arm_aux_capture(target)
    return target, drafter


def test_moved_tap_waits_for_full_unnormalized_sum(modules, context, monkeypatch):
    target, drafter = _armed(modules)
    prefix = torch.full((2, 8), 2, dtype=torch.bfloat16)
    pending = modules.ops.ColumnTail(
        torch.full((2, 4), 3, dtype=torch.bfloat16), "gather", 2
    )
    bank = torch.empty((2, 0, 8), dtype=torch.bfloat16)
    monkeypatch.setattr(
        torch.ops.aiter,
        "kimi_k3_gather_columns",
        lambda *a: pytest.fail("the tap launched another gather"),
    )
    target.layers[0](prefix, pending, None, bank)
    assert torch.count_nonzero(drafter._aux_buffers[0]) == 0
    mixed, settled = target.layers[1].self_attention_attn_res(
        prefix, bank, pending, None
    )
    actual = drafter.aux_for(settled)[0]
    assert torch.equal(actual, torch.full_like(prefix, 5))
    assert not torch.equal(actual, mixed)


def test_moved_tap_preserves_prefill_double_add_rounding(modules, context):
    target, drafter = _armed(modules)
    context.is_prefill = True
    prefix = torch.ones((2, 8), dtype=torch.bfloat16)
    first = torch.full_like(prefix, 1 / 256)
    second = torch.full_like(prefix, 1 / 256)
    bank = torch.empty((2, 0, 8), dtype=torch.bfloat16)
    expected = modules.model.KimiDecoderLayer.aux_hidden_state(
        (prefix, first, second, bank)
    )
    _, settled = target.layers[1].self_attention_attn_res(prefix, bank, first, second)
    assert not torch.equal(expected, settled)
    assert torch.equal(drafter.aux_for(settled)[0], expected)


def test_tap_mapping_and_microbatch_rows_stay_with_the_source_layer(modules, context):
    target, drafter = _armed(modules, ids=(0, 1, 2))
    context.ubatch_token_offset = 3
    bank = torch.empty((2, 0, 8), dtype=torch.bfloat16)
    prefix = torch.ones((2, 8), dtype=torch.bfloat16)
    for lid in range(3):
        add = torch.full_like(prefix, lid + 2)
        target.layers[lid](prefix, add, None, bank)
        if lid < 2:
            assert torch.count_nonzero(drafter._aux_buffers[lid]) == 0
            target.layers[lid + 1].self_attention_attn_res(prefix, bank, add, None)
        assert torch.equal(drafter._aux_buffers[lid][3:5], prefix + add)
        assert torch.count_nonzero(drafter._aux_buffers[lid][:3]) == 0
        assert torch.count_nonzero(drafter._aux_buffers[lid][5:]) == 0


@pytest.mark.parametrize("capture", ["input", "output"])
def test_custom_extractors_keep_the_original_site(modules, context, capture):
    extract = lambda output, layer: output[0] * 7
    target, drafter = _armed(modules, extract=extract, capture=capture)
    prefix = torch.ones((2, 8), dtype=torch.bfloat16)
    bank = torch.empty((2, 0, 8), dtype=torch.bfloat16)
    target.layers[0](prefix, prefix, None, bank)
    assert torch.equal(drafter.aux_for(prefix)[0], prefix * 7)
    assert not target.layers[1].self_attention_attn_res._forward_hooks


def test_draft_steps_do_not_overwrite_moved_target_taps(modules, context):
    target, drafter = _armed(modules)
    context.is_draft = True
    prefix = torch.ones((2, 8), dtype=torch.bfloat16)
    target.layers[1].self_attention_attn_res(prefix, None, prefix, None)
    assert torch.count_nonzero(drafter._aux_buffers[0]) == 0


def test_engine_closes_model_resources_before_process_groups(modules, monkeypatch):
    events = []

    class Model(nn.Module):
        def close_inference_resources(self):
            events.append("model_close")

    runner = SimpleNamespace(
        still_running=True,
        model=Model(),
        vision_embeddings=SimpleNamespace(clear=lambda: None),
        structured_compile_executor=None,
    )
    monkeypatch.setattr(modules.runner, "get_kvconnector", lambda: None)
    monkeypatch.setattr(
        modules.runner, "destroy_dist_env", lambda: events.append("dist_close")
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setenv("ATOM_MORI_V2", "0")
    assert modules.runner.ModelRunner.exit(runner)
    assert events == ["model_close", "dist_close"]


def test_peer_close_quiesces_and_retries_only_unclosed_mappings(modules, monkeypatch):
    events, failures = [], [17]
    buffer = object.__new__(modules.peer_memory.SymmetricPeerBuffer)
    buffer.storage = torch.empty(1, dtype=torch.uint8)
    buffer.npes, buffer.group = 2, object()
    buffer._remote_bases = [17, 23]
    buffer._safety_barrier_complete = False
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed, "barrier", lambda group: events.append("barrier")
    )

    def close_mapping(base):
        events.append(base)
        if base in failures:
            failures.remove(base)
            raise RuntimeError("close failed")

    monkeypatch.setattr(modules.peer_memory, "close_ipc_handle", close_mapping)
    owner = object.__new__(modules.ops.PeerColumnGather)
    owner.buffer, owner.closed = buffer, False
    with pytest.raises(RuntimeError, match="close failed"):
        owner.close()
    assert not owner.closed and buffer._remote_bases == [17]
    assert events == ["sync", "barrier", 17, 23]
    owner.close()
    owner.close()
    assert owner.closed and buffer._remote_bases == []
    assert events == ["sync", "barrier", 17, 23, 17]


def test_model_close_keeps_ownership_until_close_succeeds(modules):
    calls = []

    def close():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("peer close failed")

    owner = SimpleNamespace(close=close)
    block = SimpleNamespace(column_gather_name="gather")
    target = SimpleNamespace(
        _column_gather=owner,
        _column_gather_blocks=[block],
        _column_gather_context={"gather": owner},
        _column_gather_name="gather",
    )
    with pytest.raises(RuntimeError, match="peer close failed"):
        modules.model.KimiLinearModel.close_inference_resources(target)
    assert target._column_gather is owner and block.column_gather_name == "gather"
    assert target._column_gather_context["gather"] is owner
    modules.model.KimiLinearModel.close_inference_resources(target)
    modules.model.KimiLinearModel.close_inference_resources(target)
    assert len(calls) == 2 and target._column_gather is None
    assert block.column_gather_name == "" and not target._column_gather_context


def test_failed_model_cleanup_still_reports_failure_and_tears_down_dist(
    modules, monkeypatch
):
    events = []

    class Model(nn.Module):
        def close_inference_resources(self):
            events.append("model_close")
            raise RuntimeError("peer close failed")

    runner = SimpleNamespace(
        still_running=True,
        model=Model(),
        vision_embeddings=SimpleNamespace(clear=lambda: None),
        structured_compile_executor=None,
    )
    monkeypatch.setattr(modules.runner, "get_kvconnector", lambda: None)
    monkeypatch.setattr(
        modules.runner, "destroy_dist_env", lambda: events.append("dist_close")
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    with pytest.raises(RuntimeError, match="peer close failed"):
        modules.runner.ModelRunner.exit(runner)
    assert events == ["model_close", "dist_close"]
    assert not runner.still_running


def test_fused_consumer_receives_columns_without_materializing(modules, monkeypatch):
    owner = object.__new__(modules.ops.PeerColumnGather)
    owner.closed, owner.rows, owner.cols, owner.world, owner.rank = False, 64, 4, 2, 0
    owner.scratch = torch.zeros(2 * 2 * 64 * 2, dtype=torch.int64)
    owner.epochs = torch.zeros(64, dtype=torch.int32)
    owner.buffer = SimpleNamespace(addresses=torch.zeros(2, dtype=torch.int64))
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"gather": owner})
    )
    monkeypatch.setattr(modules.config, "get_current_atom_config", lambda: config)
    columns = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    prefix = torch.ones((2, 8), dtype=torch.bfloat16)
    bank = torch.zeros((2, 2, 8), dtype=torch.bfloat16)
    full = torch.cat([columns, columns + 10], dim=-1)
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, kwargs))
                assert kwargs["GATHER"] and not kwargs["DO_ADD"]
                assert kwargs["share_ptr"] is columns
                assert kwargs["scratch_ptr"] is owner.scratch
                args[3].copy_(prefix + full)
                args[7].copy_(prefix + full)

            return launch

    monkeypatch.setattr(modules.ops, "_attn_res_fused_kernel", Kernel())
    monkeypatch.setattr(
        torch.ops.aiter,
        "kimi_k3_apply_attn_res_add",
        modules.ops._apply_attn_res_add_op,
    )
    monkeypatch.setattr(
        torch.ops.aiter,
        "kimi_k3_gather_columns",
        lambda *a: pytest.fail("separate gather"),
    )
    pending = modules.ops.ColumnTail(columns, "gather", 2)
    _, settled = modules.ops.apply_attn_res(prefix, bank, torch.ones(8), 1e-6, pending)
    assert torch.equal(settled, prefix + full)
    assert len(launches) == 1 and launches[0][0] == (2,)


@pytest.mark.parametrize(
    "fallback", [None, "pp", "plugin", "tbo", "comm", "next_disabled"]
)
def test_exchange_binds_only_safe_local_consumers(modules, monkeypatch, fallback):
    blocks = [
        SimpleNamespace(
            front_proj=object(),
            hidden_dim=7168,
            moe_hidden_size=3584,
            routed_expert_norm=object(),
            prefix=f"layer{i}",
            column_gather_name="",
        )
        for i in range(3)
    ]
    layers = [
        SimpleNamespace(
            block_sparse_moe=block,
            self_attention_attn_res=SimpleNamespace(enabled=True),
        )
        for block in blocks
    ]
    if fallback == "next_disabled":
        layers[1].self_attention_attn_res.enabled = False
        layers[2].self_attention_attn_res.enabled = False
    config = SimpleNamespace(
        enable_tbo=fallback == "tbo",
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    pp = SimpleNamespace(world_size=2 if fallback == "pp" else 1)
    group = SimpleNamespace(
        cpu_group=object(),
        device_communicator=SimpleNamespace(
            ca_comm=SimpleNamespace(disabled=fallback == "comm")
        ),
    )
    monkeypatch.setattr(modules.model, "get_pp_group", lambda: pp)
    monkeypatch.setattr(modules.model, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        modules.model, "get_tensor_model_parallel_world_size", lambda: 8
    )
    monkeypatch.setattr(modules.model, "get_tensor_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(modules.plugin, "is_plugin_mode", lambda: fallback == "plugin")
    monkeypatch.setenv("ATOM_USE_CUSTOM_ALL_GATHER", "1")
    calls = []

    def build(*args):
        calls.append(args)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(modules.ops, "PeerColumnGather", build)
    target = SimpleNamespace(start_layer=0, end_layer=3, layers=layers)
    modules.model.KimiLinearModel._init_column_gather(target, config, "model")
    if fallback is None:
        assert calls == [(7168, 64, 3, 8, group.cpu_group)]
        assert (
            blocks[0].column_gather_name
            == blocks[1].column_gather_name
            == "model.column_gather"
        )
        assert config.compilation_config.static_forward_context["layer0"] is blocks[0]
    else:
        assert not calls and all(not block.column_gather_name for block in blocks)
    assert blocks[-1].column_gather_name == ""


def _gpu_mix_inputs(tokens, banks, hidden=7168):
    prefix = torch.randn((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    full = torch.randn((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    bank = torch.randn((tokens, banks, hidden), dtype=torch.bfloat16, device="cuda")
    score = torch.randn((hidden,), dtype=torch.float32, device="cuda") * 0.01
    norm = torch.ones((hidden,), dtype=torch.bfloat16, device="cuda")
    return prefix, full, bank, score, norm


def _assert_mix_bytes(actual, expected):
    for got, want in zip(actual, expected):
        if isinstance(got, tuple):
            _assert_mix_bytes(got, want)
        else:
            assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


@pytest.mark.parametrize("tokens,banks", [(1, 1), (7, 2), (32, 3), (32, 4), (64, 11)])
@pytest.mark.parametrize("quant", [False, True])
def test_gpu_peer_word_layout_and_replay(modules, monkeypatch, tokens, banks, quant):
    """One GPU emulates remote publishers; this does not qualify peer concurrency."""
    if not torch.cuda.is_available():
        pytest.skip("GPU word-layout and graph qualification")
    from aiter import dtypes

    torch.manual_seed(1729)
    world, rank, rows, cols = 8, 3, 64, 896
    buffers = [
        torch.zeros((2, world, rows, cols // 2), dtype=torch.int64, device="cuda")
        for _ in range(world)
    ]
    owner = object.__new__(modules.ops.PeerColumnGather)
    owner.closed, owner.world, owner.rank, owner.rows, owner.cols = (
        False,
        world,
        rank,
        rows,
        cols,
    )
    owner.scratch, owner.epochs = buffers[rank], torch.zeros(
        rows, dtype=torch.int32, device="cuda"
    )
    owner.buffer = SimpleNamespace(
        addresses=torch.tensor(
            [b.data_ptr() for b in buffers], dtype=torch.int64, device="cuda"
        )
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"test": owner})
    )
    monkeypatch.setattr(modules.config, "get_current_atom_config", lambda: config)
    prefix, full, bank, score, norm = _gpu_mix_inputs(tokens, banks)
    local = full[:, rank * cols : (rank + 1) * cols].contiguous()
    pending = modules.ops.ColumnTail(local, "test", world)
    quant_dtype = dtypes.fp8 if quant else None

    def mix(add):
        return modules.ops.apply_attn_res(
            prefix, bank, score, 1e-6, add, norm, 1e-6, quant_dtype=quant_dtype
        )

    def publish(epoch):
        local.copy_(full[:, rank * cols : (rank + 1) * cols])
        bits = (
            full.reshape(tokens, world, cols)
            .transpose(0, 1)
            .contiguous()
            .view(torch.int32)
        )
        words = (bits.to(torch.int64) & 0xFFFFFFFF) | (epoch << 32)
        buffers[rank][epoch & 1, :, :tokens].copy_(words)

    publish(1)
    _assert_mix_bytes(mix(pending), mix(full))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mix(pending)
    for epoch in range(2, 7):
        full.normal_()
        prefix.add_(0.125)
        publish(epoch)
        expected = mix(full)
        graph.replay()
        torch.cuda.synchronize()
        _assert_mix_bytes(captured, expected)
        assert torch.all(owner.epochs[:tokens] == epoch)


def test_distributed_column_gather_stress(modules, monkeypatch):
    """Run with torchrun and an outer timeout; every rank must enter this test."""
    if os.getenv("ATOM_TEST_COLUMN_GATHER_DISTRIBUTED") != "1":
        pytest.skip("requires an explicitly coordinated multi-GPU run")
    dist = torch.distributed
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("gloo")
    world, rank = dist.get_world_size(), dist.get_rank()
    owner = None
    try:
        if world != 8:
            raise ValueError("the production column gather stress requires eight ranks")
        owner = modules.ops.PeerColumnGather(7168, 64, rank, world, dist.group.WORLD)
        config = SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={"stress": owner})
        )
        monkeypatch.setattr(modules.config, "get_current_atom_config", lambda: config)
        for tokens, banks in ((1, 1), (7, 2), (32, 3), (32, 4), (64, 11)):
            torch.manual_seed(2718 + tokens + banks)
            prefix, full, bank, score, norm = _gpu_mix_inputs(tokens, banks)
            local = full[:, rank * 896 : (rank + 1) * 896].contiguous()
            pending = modules.ops.ColumnTail(local, "stress", world)

            def mix(add, prefix=prefix, bank=bank, score=score, norm=norm):
                return modules.ops.apply_attn_res(prefix, bank, score, 1e-6, add, norm)

            _assert_mix_bytes(mix(pending), mix(full))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = mix(pending)
            for iteration in range(50):
                # Each rank constructs the same full oracle without a reference collective.
                full.fill_((iteration - 17) / 32)
                local.copy_(full[:, rank * 896 : (rank + 1) * 896])
                expected = mix(full)
                graph.replay()
                torch.cuda.synchronize()
                _assert_mix_bytes(captured, expected)
    finally:
        if owner is not None:
            owner.close()
        if created:
            dist.destroy_process_group()
