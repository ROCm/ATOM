"""ReplaySSM verify matches the snapshot recurrent, then a partial accept."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.fla_ops.replayssm import replayssm_commit
from atom.plugin.sglang.attention_backend.attention_gdn import SGLangGatedDeltaNet
from atom.plugin.sglang.attention_backend.gdn_replayssm import (
    replay_runtime,
    reset_replay_runtime_for_tests,
)
from atom.plugin.sglang.runtime import bind_current_forward_batch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GDN kernels require a GPU"
)

NUM_K_HEADS = 4
NUM_V_HEADS = 8
TP_SIZE = 1
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_KERNEL = 4
STATE_LEN = CONV_KERNEL - 1
K_DIM = (NUM_K_HEADS // TP_SIZE) * HEAD_K_DIM
V_DIM = (NUM_V_HEADS // TP_SIZE) * HEAD_V_DIM
CONV_DIM = 2 * K_DIM + V_DIM
NUM_SLOTS = 8
LAYER_NUM = 3


class _TargetVerifyMode:
    @staticmethod
    def is_target_verify():
        return True


def _dedup_conv_window(num_slots: int, draft: int) -> torch.Tensor:
    shared_win = draft + STATE_LEN - 1
    phys = torch.zeros(
        num_slots, CONV_DIM, shared_win, device="cuda", dtype=torch.bfloat16
    )
    return phys.as_strided(
        (num_slots, draft, CONV_DIM, STATE_LEN),
        (phys.stride(0), phys.stride(2), phys.stride(1), phys.stride(2)),
    )


def _impl() -> SGLangGatedDeltaNet:
    impl = SGLangGatedDeltaNet.__new__(SGLangGatedDeltaNet)
    torch.nn.Module.__init__(impl)
    impl.layer_num = LAYER_NUM
    impl.tp_size = TP_SIZE
    impl.num_k_heads = NUM_K_HEADS
    impl.num_v_heads = NUM_V_HEADS
    impl.head_k_dim = HEAD_K_DIM
    impl.head_v_dim = HEAD_V_DIM
    impl.activation = "silu"
    impl.A_log = torch.randn(NUM_V_HEADS // TP_SIZE, device="cuda", dtype=torch.float32)
    impl.dt_bias = torch.randn(
        NUM_V_HEADS // TP_SIZE, device="cuda", dtype=torch.float32
    )
    impl.conv1d = SimpleNamespace(
        weight=torch.randn(CONV_DIM, 1, CONV_KERNEL, device="cuda", dtype=torch.bfloat16),
        bias=torch.randn(CONV_DIM, device="cuda", dtype=torch.bfloat16),
    )
    return impl


def _cache(bs: int, draft: int, gen: torch.Generator):
    return SimpleNamespace(
        conv=[
            torch.randn(
                NUM_SLOTS,
                CONV_DIM,
                STATE_LEN,
                device="cuda",
                dtype=torch.bfloat16,
                generator=gen,
            )
        ],
        temporal=torch.randn(
            NUM_SLOTS,
            NUM_V_HEADS // TP_SIZE,
            HEAD_K_DIM,
            HEAD_V_DIM,
            device="cuda",
            dtype=torch.bfloat16,
            generator=gen,
        ),
        intermediate_ssm=torch.zeros(
            NUM_SLOTS,
            draft,
            NUM_V_HEADS // TP_SIZE,
            HEAD_K_DIM,
            HEAD_V_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        intermediate_conv_window=[_dedup_conv_window(NUM_SLOTS, draft)],
    )


def _run(impl, layer_cache, cache_indices, mixed_qkv, a, b, replay: bool):
    bs = int(cache_indices.shape[0])
    draft = mixed_qkv.shape[0] // bs
    os.environ["ATOM_ENABLE_REPLAYSSM"] = "1" if replay else "0"
    linear_backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(mamba_cache_indices=cache_indices),
        req_to_token_pool=SimpleNamespace(
            mamba2_layer_cache=lambda _layer_id: layer_cache
        ),
    )
    forward_batch = SimpleNamespace(
        forward_mode=_TargetVerifyMode(),
        spec_info=SimpleNamespace(draft_token_num=draft),
        batch_size=bs,
        attn_backend=SimpleNamespace(linear_attn_backend=linear_backend),
    )
    out = torch.empty(
        mixed_qkv.shape[0],
        NUM_V_HEADS // TP_SIZE,
        HEAD_V_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with bind_current_forward_batch(forward_batch):
        impl.forward(mixed_qkv, b, a, out, f"layers.{LAYER_NUM}")
    return out


def test_replay_matches_recurrent_across_a_partial_accept():
    gen = torch.Generator(device="cuda")
    gen.manual_seed(11)
    bs, draft, accept = 2, 3, 2
    reset_replay_runtime_for_tests()
    impl = _impl()
    layer_cache = _cache(bs, draft, gen)
    cache_indices = torch.tensor([5, 2], device="cuda", dtype=torch.int32)
    temporal0 = layer_cache.temporal.clone()
    conv0 = layer_cache.conv[0].clone()

    def inputs():
        tokens = bs * draft
        return (
            torch.randn(
                tokens, CONV_DIM, device="cuda", dtype=torch.bfloat16, generator=gen
            ),
            torch.randn(
                tokens,
                NUM_V_HEADS // TP_SIZE,
                device="cuda",
                dtype=torch.float32,
                generator=gen,
            ),
            torch.randn(
                tokens,
                NUM_V_HEADS // TP_SIZE,
                device="cuda",
                dtype=torch.float32,
                generator=gen,
            ),
        )

    mixed, a, b = inputs()
    legacy = _run(impl, layer_cache, cache_indices, mixed, a, b, replay=False)
    accepted_state = layer_cache.intermediate_ssm[:bs, accept - 1].clone()
    layer_cache.temporal.copy_(temporal0)
    layer_cache.conv[0].copy_(conv0)
    reset_replay_runtime_for_tests()
    replay = _run(impl, layer_cache, cache_indices, mixed, a, b, replay=True)
    torch.testing.assert_close(replay, legacy, rtol=1e-2, atol=1e-2)
    # No flush on an empty ring: the checkpoint stays put.
    torch.testing.assert_close(layer_cache.temporal, temporal0, rtol=0, atol=0)
    torch.testing.assert_close(layer_cache.conv[0], conv0, rtol=0, atol=0)

    runtime = replay_runtime()
    replayssm_commit(
        runtime.write_pos,
        cache_indices,
        torch.full((bs,), accept, device="cuda", dtype=torch.int32),
        runtime.max_query_len,
        runtime.cache_len,
    )
    torch.testing.assert_close(
        runtime.write_pos[cache_indices.long()],
        torch.full((bs,), accept, device="cuda", dtype=torch.int32),
    )

    mixed2, a2, b2 = inputs()
    replay2 = _run(impl, layer_cache, cache_indices, mixed2, a2, b2, replay=True)
    layer_cache.temporal[cache_indices] = accepted_state
    layer_cache.conv[0].copy_(conv0)
    legacy2 = _run(impl, layer_cache, cache_indices, mixed2, a2, b2, replay=False)
    torch.testing.assert_close(replay2, legacy2, rtol=2e-2, atol=2e-2)
    os.environ.pop("ATOM_ENABLE_REPLAYSSM", None)
    reset_replay_runtime_for_tests()
