# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Real materialize kernels, pointer lifetimes and the four-buffer commit path."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("ReplaySSM materialize requires a GPU", allow_module_level=True)

from atom.model_ops.fla_ops import replayssm as replay
from atom.plugin.sglang.attention_backend import gdn_replayssm as plugin

SLOTS, CAP, T = 8, 16, 4
H, HV, K, V = 8, 24, 128, 128


def _index(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _random(shape, dtype=torch.float32):
    return torch.randn(shape, dtype=torch.float32, device="cuda").mul_(0.03).to(dtype)


def _layer(hv=HV, kda=False, k=K, v=V):
    shape = (hv, v, k) if kda else (hv, k, v)
    sk, su, sg = replay.replayssm_buffer_shapes(CAP, hv, k, v, kda)
    return (
        _random((SLOTS,) + shape),
        _random((SLOTS,) + sk, torch.bfloat16),
        _random((SLOTS,) + su, torch.bfloat16),
        -_random((SLOTS,) + sg).abs(),
    )


def _single_reference(rows, cursor, src, dst, steps):
    expected = []
    for ckpt, buf_k, buf_u, buf_g in rows:
        reference = ckpt.clone()
        replay.replayssm_materialize(
            reference, buf_k, buf_u, buf_g, cursor, src, dst, steps, T
        )
        expected.append(reference)
    return expected


def _assert_same(rows, expected):
    torch.cuda.synchronize()
    for row, want in zip(rows, expected):
        assert torch.isfinite(row[0]).all()
        assert torch.equal(row[0], want)


@pytest.fixture(autouse=True)
def _reset_runtime():
    torch.manual_seed(308)
    plugin.reset_replay_runtime_for_tests()
    yield
    torch.cuda.synchronize()
    plugin.reset_replay_runtime_for_tests()


class _Pool:
    def __init__(self, rows, cursor):
        self.rows = rows
        self.reads = []
        runtime = plugin.replay_runtime()
        runtime.write_pos = cursor
        runtime.max_query_len = T
        runtime.cache_len = CAP
        runtime.bufs = {
            i: (
                *row[1:],
                torch.zeros(SLOTS, row[0].shape[1], dtype=torch.int32, device="cuda"),
            )
            for i, row in enumerate(rows)
        }
        self.linear = SimpleNamespace(req_to_token_pool=self)

    def mamba2_layer_cache(self, layer):
        self.reads.append(layer)
        return SimpleNamespace(temporal=self.rows[layer][0])

    def materialize(self, src, dst, steps):
        plugin._materialize_track_states(self.linear, src, dst, steps)


@pytest.mark.parametrize("kda", [False, True])
def test_single_native_matches_independent_recurrence(kda):
    ckpt, bk, bu, bg = _layer(kda=kda, k=64 if kda else K)
    before = ckpt.clone()
    cursor = _index([8] + [0] * (SLOTS - 1))
    replay.replayssm_materialize(
        ckpt, bk, bu, bg, cursor, _index([0]), _index([4]), _index([2]), T
    )
    state = before[0].float()
    if kda:
        state = state.transpose(1, 2)
    # Independent sequential recurrence, rather than the kernel's replay GEMM.
    for pos in range(11):
        decay = bg[0, :, pos].float()
        if not kda:
            decay = decay[:, None]
        state = state * decay.exp()[:, :, None]
        state += bk[0, :, pos].float()[:, :, None] * bu[0, :, pos].float()[:, None, :]
    got = ckpt[4].float().transpose(1, 2) if kda else ckpt[4].float()
    assert (got - state).norm() / state.norm().clamp_min(1e-12) < 1e-2
    before[4] = ckpt[4]
    assert torch.equal(ckpt, before)
    assert cursor.tolist() == [8] + [0] * (SLOTS - 1)


@pytest.mark.parametrize("layers", [1, 3, 36])
def test_batch_api_matches_all_single_layer_outputs(layers):
    rows = [_layer() for _ in range(layers)]
    cursor = _index([0, 8, 12, 0, 0, 0, 0, 0])
    src, dst, steps = (
        _index([0, 1, 2, -1, 0]),
        _index([4, 5, 6, 7, -1]),
        _index([0, 3, -1, 1, 0]),
    )
    expected = _single_reference(rows, cursor, src, dst, steps)
    saved = [[tensor.clone() for tensor in row[1:]] for row in rows]
    ptrs = torch.tensor(
        [[t.data_ptr() for t in row] for row in rows], device="cuda", dtype=torch.int64
    )
    replay.replayssm_materialize_batch(ptrs, *rows[0], cursor, src, dst, steps, T)
    _assert_same(rows, expected)
    assert cursor.tolist() == [0, 8, 12, 0, 0, 0, 0, 0]
    for row, records in zip(rows, saved):
        assert all(torch.equal(actual, old) for actual, old in zip(row[1:], records))


def test_live_pool_and_record_replacement_changes_device_result():
    rows = [_layer() for _ in range(3)]
    cursor = _index([8] + [0] * (SLOTS - 1))
    pool = _Pool(rows, cursor)
    src, dst, steps = _index([0]), _index([4]), _index([3])
    pool.materialize(src, dst, steps)
    generation = plugin.replay_runtime().materialize_layers
    pool.reads.clear()
    pool.materialize(src, dst, steps)
    assert pool.reads == [0, 1, 2]
    assert plugin.replay_runtime().materialize_layers is generation
    rows[1] = tuple(t.clone() * 0.5 for t in rows[1])
    runtime = plugin.replay_runtime()
    runtime.bufs[1] = (*rows[1][1:], runtime.bufs[1][3])
    expected = _single_reference(rows, cursor, src, dst, steps)
    pool.materialize(src, dst, steps)
    _assert_same(rows, expected)
    assert runtime.materialize_layers is not generation
    # The plugin also accepts strided indices before launching either route.
    pool.materialize(
        src, dst.repeat_interleave(2)[::2], steps.repeat_interleave(2)[::2]
    )
    _assert_same(rows, expected)


def test_heterogeneous_layout_uses_single_layer_outputs():
    rows = [_layer(), _layer(hv=12), _layer()]
    cursor = _index([8] + [0] * (SLOTS - 1))
    pool = _Pool(rows, cursor)
    src, dst, steps = _index([0]), _index([4]), _index([3])
    expected = _single_reference(rows, cursor, src, dst, steps)
    pool.materialize(src, dst, steps)
    _assert_same(rows, expected)
    assert plugin.replay_runtime().materialize_layers.ptrs is None


def test_capture_replay_uses_changed_inputs_without_a_pointer_table():
    rows = [_layer() for _ in range(3)]
    cursor = _index([8] + [0] * (SLOTS - 1))
    pool = _Pool(rows, cursor)
    src, dst, steps = _index([0]), _index([4]), _index([3])
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for row in rows:
            replay.replayssm_materialize(*row, cursor, src, dst, steps, T)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pool.materialize(src, dst, steps)
    assert plugin.replay_runtime().materialize_layers is None
    for row in rows:
        row[0].mul_(0.75)
        row[2].add_(0.001)
    steps.fill_(1)
    expected = _single_reference(rows, cursor, src, dst, steps)
    graph.replay()
    _assert_same(rows, expected)


def test_retired_generation_keeps_queued_raw_pointer_inputs_alive():
    rows = [_layer() for _ in range(3)]
    cursor = _index([8] + [0] * (SLOTS - 1))
    pool = _Pool(rows, cursor)
    src, dst, steps = _index([0]), _index([4]), _index([3])
    expected = _single_reference(rows, cursor, src, dst, steps)
    # Compile before queuing the lifetime-sensitive use.
    pool.materialize(src, dst, steps)
    torch.cuda.synchronize()
    outputs = [row[0] for row in rows]
    work = _random((1024, 1024))
    consumer = torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(consumer):
        for _ in range(8):
            work = work @ work
        pool.materialize(src, dst, steps)
    assert plugin.replay_runtime().materialize_layers.stream == consumer.cuda_stream
    # Drop the generation and every Python reference to its record tensors
    # while work may still be queued. Same-sized allocations stress reuse.
    record_shapes = [(t.shape, t.dtype) for row in rows for t in row[1:]]
    plugin.reset_replay_runtime_for_tests()
    rows.clear()
    pressure = [
        torch.empty(shape, dtype=dtype, device="cuda").fill_(7)
        for shape, dtype in record_shapes
    ]
    consumer.synchronize()
    assert all(torch.equal(actual, want) for actual, want in zip(outputs, expected))
    assert pressure


@pytest.mark.parametrize("h", [0, 8, 12])
def test_four_buffer_forward_then_real_plugin_commit(h, monkeypatch):
    monkeypatch.setenv("ATOM_REPLAYSSM_CACHE_LEN", str(CAP))
    checkpoints = [_random((SLOTS, HV, K, V)) for _ in range(3)]
    q, k = (_random((1, T, H, K), torch.bfloat16) for _ in range(2))
    v = _random((1, T, HV, V), torch.bfloat16)
    g = -_random((1, T, HV)).abs()
    beta = torch.full((1, T, HV), 0.5, dtype=torch.bfloat16, device="cuda")
    rows = []
    for i, checkpoint in enumerate(checkpoints):
        prepared = plugin.prepare_layer(
            i,
            checkpoint,
            num_v_heads=HV,
            head_k_dim=K,
            head_v_dim=V,
            max_query_len=T,
            activation=q,
        )
        assert len(prepared) == 6
        bk, bu, bg, arrivals, cursor, wait = prepared
        bk.copy_(_random(bk.shape, bk.dtype))
        bu.copy_(_random(bu.shape, bu.dtype))
        bg.copy_(-_random(bg.shape).abs())
        cursor[0] = h
        output = replay.replayssm_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            checkpoint,
            bk,
            bu,
            bg,
            cursor,
            _index([0]),
            _index([0, T]),
            T,
            use_qk_l2norm_in_kernel=True,
            flush_arrivals=arrivals,
            flush_wait=wait,
        )
        assert torch.isfinite(output).all()
        assert not arrivals.any()
        rows.append((checkpoint, bk, bu, bg))
    runtime = plugin.replay_runtime()
    src, dst, steps = _index([0]), _index([4]), _index([1])
    expected = _single_reference(rows, runtime.write_pos, src, dst, steps)
    conv = _random((3, SLOTS, 6, 3), torch.bfloat16)
    windows = _random((3, 1, T, 6, 3), torch.bfloat16)
    expected_conv = conv.clone()
    expected_conv[:, 0] = windows[:, 0, 2]
    expected_conv[:, 4] = windows[:, 0, 1]
    pool = SimpleNamespace(
        mamba2_layer_cache=lambda i: SimpleNamespace(temporal=checkpoints[i]),
        get_speculative_mamba2_params_all_layers=lambda: SimpleNamespace(
            conv=[conv], intermediate_conv_window=[windows]
        ),
    )
    backend = SimpleNamespace(
        linear_attn_backend=SimpleNamespace(
            req_to_token_pool=pool,
            forward_metadata=SimpleNamespace(mamba_cache_indices=src),
        )
    )
    plugin._commit_accepted(backend, _index([2]), dst, steps)
    _assert_same(rows, expected)
    assert torch.equal(conv, expected_conv)
    assert runtime.write_pos.tolist() == [3 if h == 12 else h + 3] + [0] * (SLOTS - 1)
    assert runtime.materialize_layers.ptrs.shape == (3, 4)
