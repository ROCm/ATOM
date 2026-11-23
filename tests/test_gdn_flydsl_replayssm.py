# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL GDN prefill under ReplaySSM, without a GPU.

ReplaySSM needs the state pool to stay a contiguous KV checkpoint, and a
prefill to rewrite that checkpoint wholesale without touching the record ring.
FlyDSL decode needs a physical VK pool, so it stays off. FlyDSL prefill gathers
the initial state through the pool's strides and returns the final state as a
logical KV view, so it may run; it is enabled on gfx942 only. These tests pin
that contract on the Qwen4Exp cache binding and prefill metadata, and the
Native GDN layer. The policy matrix is in tests/test_gdn_flydsl_policy.py,
which runs without AITER; the kernel numerics are compared on a GPU in
tests/test_aiter_flydsl_gdn.py.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip(
    "atom.model_ops.attentions.qwen4_exp_attn",
    reason="requires AITER to import the attention backends",
    exc_type=ImportError,
)

from atom.model_ops import attention_gdn as gdn
from atom.model_ops.attentions.gdn_attn import (
    LINEAR_STATE_ROWS,
    GDNAttentionMetadataBuilder,
)
from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpMetadataBuilder
from atom.model_ops.fla_ops import chunk, state_checkpoint
from atom.model_ops.fla_ops import gdn_flydsl as fly
from atom.model_ops.fla_ops.gdn_flydsl import GDNFlyDSLPolicy
from atom.utils import envs

HK, HV, D = 2, 4, 8
CONV_DIM = 2 * HK * D + HV * D
LENGTHS = (3, 5)
T = sum(LENGTHS)
SLOTS = 6


@pytest.mark.parametrize("gfx942", [False, True])
@pytest.mark.parametrize("replayssm", [False, True])
def test_qwen_binding_keeps_the_replayssm_checkpoint_contiguous(
    monkeypatch, replayssm, gfx942
):
    raw = torch.zeros(4, 24, 128, 128, dtype=torch.bfloat16)
    monkeypatch.setattr(
        GDNAttentionMetadataBuilder,
        "build_kv_cache_tensor",
        lambda self, module: SimpleNamespace(v_cache=raw),
    )
    monkeypatch.setattr(envs, "ATOM_ENABLE_GDN_DECODE_LOSSY_FAST", False)
    monkeypatch.setattr(fly, "backend", lambda stage: "auto")
    monkeypatch.setattr(fly, "ops", lambda: object())
    monkeypatch.setattr(fly, "_validated_on_gfx942", lambda: gfx942)
    builder = object.__new__(Qwen4ExpMetadataBuilder)
    builder.replayssm = replayssm
    builder._state_shape_for_runner = lambda: ((), tuple(raw.shape[1:]))
    builder._state_dtypes = lambda: (raw.dtype, raw.dtype)
    attention = SimpleNamespace(allow_aiter_flydsl=True, dt_bias=raw)
    module = Mock(
        spec=["base_linear_attention", "impl"],
        base_linear_attention=None,
        impl=attention,
    )
    builder._pool_rows = {LINEAR_STATE_ROWS: {module: 0}}
    cache = builder.build_kv_cache_tensor(module)
    policy = attention.gdn_flydsl_policy
    # Off gfx942, ReplaySSM keeps the Triton prefill and no FlyDSL schedule.
    prefill = gfx942 or not replayssm
    assert policy.prefill is prefill
    assert builder._flydsl_prefill_enabled is prefill
    assert policy.decode is (gfx942 and not replayssm)
    if replayssm:
        # The exact tensor ReplaySSM addresses: no VK reinterpretation.
        assert cache.v_cache is raw
    else:
        assert cache.v_cache.data_ptr() == raw.data_ptr()
        assert cache.v_cache.stride()[-2:] == ((1, 128) if gfx942 else (128, 1))


def _finish_prefill(monkeypatch, *, replayssm, enabled):
    monkeypatch.setattr(
        GDNAttentionMetadataBuilder, "finish_prefill", lambda self, *args: None
    )
    monkeypatch.setattr(
        Qwen4ExpMetadataBuilder, "_finish_qsa", lambda self, *args, **kw: "qsa"
    )
    monkeypatch.setattr(
        Qwen4ExpMetadataBuilder, "_finish_ple", lambda self, *args, **kw: "ple"
    )
    builds = []
    monkeypatch.setattr(
        fly,
        "build_prefill_metadata",
        lambda lengths, cu_seqlens: builds.append((list(lengths), cu_seqlens))
        or "schedule",
    )
    builder = object.__new__(Qwen4ExpMetadataBuilder)
    builder.replayssm = replayssm
    builder._flydsl_prefill_enabled = enabled
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        gdn_metadata=SimpleNamespace(
            non_spec_query_start_loc=cu, flydsl_prefill_metadata=None
        )
    )
    batch = SimpleNamespace(
        block_tables=[[0], [1]],
        total_seqs_num_prefill=2,
        num_scheduled_tokens=[3, 5],
        total_tokens_num_prefill=8,
    )
    builder.finish_prefill(batch, attn_metadata, 2, 8)
    assert attn_metadata.qsa_metadata == "qsa"
    assert attn_metadata.ple_metadata == "ple"
    return builds, attn_metadata.gdn_metadata.flydsl_prefill_metadata, cu


def test_finish_prefill_builds_the_flydsl_schedule_under_replayssm(monkeypatch):
    builds, schedule, cu = _finish_prefill(monkeypatch, replayssm=True, enabled=True)
    assert builds == [([3, 5], cu)]
    assert schedule == "schedule"


@pytest.mark.parametrize("replayssm", [False, True])
def test_finish_prefill_builds_no_schedule_without_flydsl_prefill(
    monkeypatch, replayssm
):
    builds, schedule, _ = _finish_prefill(
        monkeypatch, replayssm=replayssm, enabled=False
    )
    assert builds == []
    assert schedule is None


def _gdn_layer(policy):
    layer = gdn.GatedDeltaNet.__new__(gdn.GatedDeltaNet)
    torch.nn.Module.__init__(layer)
    for name, value in {
        "layer_num": 0,
        "tp_size": 1,
        "num_k_heads": HK,
        "num_v_heads": HV,
        "head_k_dim": D,
        "head_v_dim": D,
        "key_dim": HK * D,
        "value_dim": HV * D,
        "A_log": torch.zeros(HV),
        "dt_bias": torch.zeros(HV),
        "allow_aiter_flydsl": True,
        "activation": "silu",
        "conv1d": SimpleNamespace(weight=torch.zeros(CONV_DIM, 1, 4), bias=None),
        "gdn_flydsl_policy": policy,
    }.items():
        setattr(layer, name, value)
    return layer


def _replayssm_prefill(monkeypatch, *, flydsl, keep, checkpoint):
    """One Native GDN prefill forward under ReplaySSM metadata.

    The convolution, gating and both prefill kernels are stand-ins that return
    the same tensors, so the two dispatches differ only in which kernel runs.
    The FlyDSL stand-in returns its final state as aiter does: a transposed
    view over FP32 VK storage.
    """
    torch.manual_seed(0)
    q = torch.randn(T, HK * D, dtype=torch.bfloat16)
    k = torch.randn(T, HK * D, dtype=torch.bfloat16)
    v = torch.randn(T, HV * D, dtype=torch.bfloat16)
    g = torch.randn(1, T, HV)
    beta = torch.rand(1, T, HV, dtype=torch.bfloat16)
    out = torch.randn(1, T, HV, D, dtype=torch.bfloat16)
    final_vk = torch.randn(len(LENGTHS), HV, D, D)
    h_kv = torch.randn(1, len(LENGTHS), HV, D, D, dtype=torch.bfloat16)

    pool = torch.randn(SLOTS, HV, D, D)
    original = pool.clone()
    replay = {
        "write_pos": torch.full((SLOTS,), 3, dtype=torch.int32),
        "replay_buf_k": torch.randn(SLOTS, HV, 8, D, dtype=torch.bfloat16),
        "replay_buf_u": torch.randn(SLOTS, HV, 8, D, dtype=torch.bfloat16),
        "replay_buf_g": torch.randn(SLOTS, HV, 8),
    }
    replay_before = {name: t.clone() for name, t in replay.items()}
    conv_state = torch.zeros(SLOTS, CONV_DIM, 3, dtype=torch.bfloat16)
    cache = SimpleNamespace(
        k_cache=conv_state,
        v_cache=pool,
        replay_buf_k=replay["replay_buf_k"],
        replay_buf_u=replay["replay_buf_u"],
        replay_buf_g=replay["replay_buf_g"],
    )
    # Row 0 forks from slot 2 into slot 4; row 1 is a cold slot.
    writes = torch.tensor([4, 1], dtype=torch.int32)
    reads = torch.tensor([2, 1], dtype=torch.int32)
    has_initial = torch.tensor([True, False])
    schedule = object()
    checkpoints = {
        name: torch.zeros(1, dtype=torch.int32)
        for name in ("rows", "slots", "offs", "is_end", "runtime")
    }
    metadata = SimpleNamespace(
        replayssm=True,
        write_pos=replay["write_pos"],
        slot_idx=writes,
        replayssm_max_query_len=4,
        replayssm_route="serial",
        replayssm_fork_slots=None,
        has_initial_state=has_initial,
        no_initial_state=~has_initial,
        spec_query_start_loc=None,
        non_spec_query_start_loc=torch.tensor([0, 3, 8], dtype=torch.int32),
        spec_sequence_masks=None,
        spec_token_indx=None,
        non_spec_token_indx=None,
        spec_state_indices_tensor=None,
        spec_state_indices_in_tensor=None,
        non_spec_state_indices_tensor=writes,
        non_spec_state_indices_in_tensor=reads,
        non_spec_state_indices_i64=writes.long(),
        non_spec_state_indices_in_i64=reads.long(),
        num_actual_tokens=T,
        num_accepted_tokens=None,
        num_prefills=len(LENGTHS),
        num_decodes=0,
        ssm_checkpoints=checkpoints if checkpoint else None,
        ssm_chunk_offsets=torch.tensor([0, 1], dtype=torch.int32),
        ssm_conv_state_len=3,
        keep_intermediate_states=keep,
        flydsl_prefill_metadata=schedule,
    )
    context = SimpleNamespace(
        attn_metadata=SimpleNamespace(gdn_metadata=metadata),
        kv_cache_data={"layer_0": cache},
    )
    monkeypatch.setattr(gdn, "get_forward_context", lambda: context)
    monkeypatch.setattr(gdn, "causal_conv1d_fn", lambda *args, **kw: (q, k, v))
    monkeypatch.setattr(gdn, "fused_gdn_gating", lambda *args: (g, beta))

    def flydsl_prefill(
        q_,
        k_,
        v_,
        g_,
        beta_,
        initial_state,
        cu_seqlens,
        metadata_,
        keep_intermediate_states=False,
        *,
        state_indices=None,
        has_initial_state=None,
        snapshot_layout="kv",
    ):
        assert metadata_ is schedule and snapshot_layout == "kv"
        # The pool itself, read through its strides inside the kernel.
        assert initial_state is pool
        assert state_indices is reads and has_initial_state is has_initial
        snapshot = h_kv.clone() if keep_intermediate_states else None
        return out.clone(), final_vk.transpose(-1, -2), snapshot

    def triton_prefill(*args, initial_state, o, keep_intermediate_states, **kw):
        expected_initial = original.index_select(0, reads.long())
        expected_initial[~has_initial] = 0
        assert torch.equal(initial_state, expected_initial)
        o.copy_(out)
        chunk.ChunkGatedDeltaRuleFunction.last_h = (
            h_kv.clone() if keep_intermediate_states else None
        )
        return o, final_vk.transpose(-1, -2).contiguous()

    prefill = Mock(side_effect=flydsl_prefill)
    chunked = Mock(side_effect=triton_prefill)
    written = Mock()
    replay_kernel = Mock()
    monkeypatch.setattr(fly, "prefill_supported", lambda *args: args[-1] is schedule)
    monkeypatch.setattr(fly, "prefill", prefill)
    monkeypatch.setattr(gdn, "chunk_gated_delta_rule", chunked)
    monkeypatch.setattr(gdn, "replayssm_gated_delta_rule", replay_kernel)
    monkeypatch.setattr(state_checkpoint, "write_state_checkpoints", written)

    monkeypatch.setattr(chunk.ChunkGatedDeltaRuleFunction, "last_h", None)

    layer = _gdn_layer(GDNFlyDSLPolicy(prefill=flydsl, decode=False))
    core_attn_out = torch.zeros(T + 2, HV, D, dtype=torch.bfloat16)
    result = layer(
        torch.zeros(T, CONV_DIM, dtype=torch.bfloat16),
        torch.zeros(T, HV, dtype=torch.bfloat16),
        torch.zeros(T, HV, dtype=torch.bfloat16),
        core_attn_out,
        "layer_0",
    )

    assert prefill.call_count == int(flydsl)
    assert chunked.call_count == int(not flydsl)
    replay_kernel.assert_not_called()
    if flydsl:
        keep_snapshots = prefill.call_args.kwargs["keep_intermediate_states"]
        assert keep_snapshots == (checkpoint or keep)
    assert torch.equal(result[:T], out[0])
    assert not result[T:].any()
    # The checkpoint stays the contiguous pool ReplaySSM addresses; only the
    # written slots change, to the logical KV final state.
    assert cache.v_cache is pool and pool.is_contiguous()
    assert torch.equal(pool[writes.long()], final_vk.transpose(-1, -2))
    untouched = [s for s in range(SLOTS) if s not in writes.tolist()]
    assert torch.equal(pool[untouched], original[untouched])
    # A prefill writes no records and leaves the cursor to the metadata.
    for name, tensor in replay.items():
        assert torch.equal(tensor, replay_before[name]), name
    if checkpoint:
        written.assert_called_once()
        h, ssm_state = written.call_args.args[:2]
        assert ssm_state is pool
        assert h.shape == h_kv.shape and torch.equal(h, h_kv)
    else:
        written.assert_not_called()
    published = chunk.pop_last_intermediate_states()
    if keep and not checkpoint:
        assert torch.equal(published, h_kv)


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("flydsl", [False, True])
def test_replayssm_prefill_writes_the_logical_kv_final_state_into_the_pool(
    monkeypatch, flydsl, keep, checkpoint
):
    _replayssm_prefill(monkeypatch, flydsl=flydsl, keep=keep, checkpoint=checkpoint)
