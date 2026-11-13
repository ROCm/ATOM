"""ReplaySSM under the mamba radix cache: the track slot gets the tracked state.

The snapshot path (`_verify_batched_ssm`) leaves every verify step's state in
`intermediate_ssm`, and SGLang scatters the tracked one into the track slot.
With the ring there is no per-step state, so `_commit_accepted` rebuilds it
from the checkpoint and the records. These tests run both verify paths on the
same inputs and require the track slot to hold the snapshot path's state, over
two verify rounds, with and without a flush in the second.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from atom.plugin.sglang.attention_backend import gdn_replayssm
from atom.plugin.sglang.attention_backend.gdn_replayssm import (
    replay_runtime,
    reset_replay_runtime_for_tests,
)
from tests.plugin.test_sglang_gdn_replayssm_verify import (
    CONV_DIM,
    NUM_V_HEADS,
    TP_SIZE,
    _cache,
    _impl,
    _run,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GDN kernels require a GPU"
)

BS, DRAFT = 2, 3
LIVE = [5, 2]
TRACK = [7, 1]


def _backend(layer_cache, cache_indices):
    # SGLang's all-layers view stacks a leading layer axis; one layer here.
    all_layers = SimpleNamespace(
        conv=[layer_cache.conv[0].unsqueeze(0)],
        intermediate_conv_window=[layer_cache.intermediate_conv_window[0].unsqueeze(0)],
    )
    pool = SimpleNamespace(
        mamba2_layer_cache=lambda _layer_id: layer_cache,
        get_speculative_mamba2_params_all_layers=lambda: all_layers,
    )
    return SimpleNamespace(
        linear_attn_backend=SimpleNamespace(
            forward_metadata=SimpleNamespace(mamba_cache_indices=cache_indices),
            req_to_token_pool=pool,
        )
    )


def _inputs(gen):
    tokens = BS * DRAFT
    heads = NUM_V_HEADS // TP_SIZE
    return (
        torch.randn(
            tokens, CONV_DIM, device="cuda", dtype=torch.bfloat16, generator=gen
        ),
        torch.randn(tokens, heads, device="cuda", dtype=torch.float32, generator=gen),
        torch.randn(tokens, heads, device="cuda", dtype=torch.float32, generator=gen),
    )


def _snapshot_states(impl, layer_cache, cache_indices, live_state, conv, inputs):
    """Per-step states of the snapshot path from `live_state` / `conv`."""
    saved_t = layer_cache.temporal.clone()
    saved_c = layer_cache.conv[0].clone()
    layer_cache.temporal[cache_indices.long()] = live_state
    layer_cache.conv[0].copy_(conv)
    _run(impl, layer_cache, cache_indices, *inputs, replay=False)
    states = layer_cache.intermediate_ssm[:BS].clone()
    layer_cache.temporal.copy_(saved_t)
    layer_cache.conv[0].copy_(saved_c)
    return states


def _commit(layer_cache, cache_indices, accept, track_steps):
    dev = cache_indices.device
    gdn_replayssm._commit_accepted(
        _backend(layer_cache, cache_indices),
        torch.full((BS,), accept - 1, device=dev, dtype=torch.int64),
        torch.tensor(TRACK, device=dev, dtype=torch.int64),
        torch.tensor(track_steps, device=dev, dtype=torch.int64),
    )


@pytest.mark.parametrize("cache_len", [6, 16], ids=["flush-in-round-2", "no-flush"])
def test_track_slot_holds_the_snapshot_paths_state(monkeypatch, cache_len):
    monkeypatch.setenv("ATOM_REPLAYSSM_CACHE_LEN", str(cache_len))
    gen = torch.Generator(device="cuda")
    gen.manual_seed(23)
    reset_replay_runtime_for_tests()
    impl = _impl()
    layer_cache = _cache(BS, DRAFT, gen)
    cache_indices = torch.tensor(LIVE, device="cuda", dtype=torch.int32)
    live = cache_indices.long()
    track = torch.tensor(TRACK, device="cuda").long()
    try:
        # Round 1: accept 2 of 3; request 0 tracks step 1, request 1 none.
        inputs1 = _inputs(gen)
        state0 = layer_cache.temporal[live].clone()
        conv0 = layer_cache.conv[0].clone()
        ref1 = _snapshot_states(
            impl, layer_cache, cache_indices, state0, conv0, inputs1
        )
        untouched = layer_cache.temporal[track[1]].clone()
        _run(impl, layer_cache, cache_indices, *inputs1, replay=True)
        assert replay_runtime().cache_len == cache_len
        _commit(layer_cache, cache_indices, accept=2, track_steps=[1, -1])
        torch.testing.assert_close(
            layer_cache.temporal[track[0]], ref1[0, 1], rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            layer_cache.temporal[track[1]], untouched, rtol=0, atol=0
        )
        assert replay_runtime().write_pos[live].tolist() == [2, 2]

        # Round 2 continues from the accepted state, through the records the
        # first round committed (or the checkpoint they were flushed into).
        inputs2 = _inputs(gen)
        conv1 = layer_cache.conv[0].clone()
        _run(impl, layer_cache, cache_indices, *inputs2, replay=True)
        _commit(layer_cache, cache_indices, accept=3, track_steps=[2, 0])
        ref2 = _snapshot_states(
            impl, layer_cache, cache_indices, ref1[:, 1], conv1, inputs2
        )
        torch.testing.assert_close(
            layer_cache.temporal[track[0]], ref2[0, 2], rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            layer_cache.temporal[track[1]], ref2[1, 0], rtol=2e-2, atol=2e-2
        )
        flushed = 2 + 2 * DRAFT > cache_len
        assert replay_runtime().write_pos[live].tolist() == (
            [3, 3] if flushed else [5, 5]
        )
    finally:
        os.environ.pop("ATOM_ENABLE_REPLAYSSM", None)
        reset_replay_runtime_for_tests()
