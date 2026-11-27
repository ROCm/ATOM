from types import SimpleNamespace
from unittest.mock import patch

import torch

from atom.plugin.sglang.attention_backend.attention_gdn import (
    SGLangGatedDeltaNet,
    SGLangGDNForwardContext,
)
from atom.plugin.sglang.runtime import bind_current_forward_batch


class _TargetVerifyMode:
    @staticmethod
    def is_target_verify():
        return True


class _FakeLinearBackend:
    def __init__(self, layer_cache):
        self.forward_metadata = SimpleNamespace(
            mamba_cache_indices=torch.tensor([0], dtype=torch.int32)
        )
        self.req_to_token_pool = SimpleNamespace(
            mamba2_layer_cache=lambda _layer_id: layer_cache
        )


def _make_impl():
    impl = SGLangGatedDeltaNet.__new__(SGLangGatedDeltaNet)
    torch.nn.Module.__init__(impl)
    impl.layer_num = 7
    impl.tp_size = 1
    impl.num_k_heads = 1
    impl.num_v_heads = 1
    impl.head_k_dim = 1
    impl.head_v_dim = 1
    impl.A_log = torch.zeros(1)
    impl.dt_bias = torch.zeros(1)
    impl.activation = "silu"
    impl.conv1d = SimpleNamespace(
        weight=torch.ones(3, 1, 2),
        bias=None,
    )
    return impl


def test_target_verify_populates_batched_snapshots_without_touching_live_state():
    conv_phys = torch.zeros(1, 3, 2)
    conv_view = conv_phys.as_strided((1, 2, 3, 1), (6, 1, 2, 1))
    layer_cache = SimpleNamespace(
        conv=[torch.zeros(1, 3, 1)],
        temporal=torch.zeros(1, 1, 1, 1),
        intermediate_ssm=torch.zeros(1, 2, 1, 1, 1),
        intermediate_conv_window=[conv_view],
    )
    linear_backend = _FakeLinearBackend(layer_cache)
    active_backend = SimpleNamespace(linear_attn_backend=linear_backend)
    forward_batch = SimpleNamespace(
        forward_mode=_TargetVerifyMode(),
        spec_info=SimpleNamespace(draft_token_num=2),
        batch_size=1,
    )
    mixed_qkv = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = torch.ones(2, 1)
    a = torch.ones(2, 1)

    def fake_conv(x, conv_state, *_args, **_kwargs):
        for step in range(x.shape[0]):
            conv_state[0, :, step] = step + 1
        return x[:, :1], x[:, 1:2], x[:, 2:3]

    def fake_gating(_A_log, a_step, b_step, _dt_bias):
        return a_step.unsqueeze(0), b_step.unsqueeze(0)

    def fake_recurrent(*, q, initial_state, ssm_state_indices, **_kwargs):
        for step, index in enumerate(ssm_state_indices[0]):
            initial_state[index] += step + 1
        return q, None

    with (
        bind_current_forward_batch(forward_batch),
        patch.object(
            SGLangGDNForwardContext,
            "_resolve_attn_backend",
            return_value=active_backend,
        ),
        patch(
            "atom.plugin.sglang.attention_backend.attention_gdn.causal_conv1d_update",
            side_effect=fake_conv,
        ),
        patch(
            "atom.plugin.sglang.attention_backend.attention_gdn.fused_gdn_gating",
            side_effect=fake_gating,
        ),
        patch(
            "atom.plugin.sglang.attention_backend.attention_gdn.fused_recurrent_gated_delta_rule",
            side_effect=fake_recurrent,
        ),
    ):
        output = _make_impl().forward(
            mixed_qkv,
            b,
            a,
            torch.empty(2, 1, 1),
            "layer_7",
        )

    torch.testing.assert_close(output[:, 0, 0], torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(
        layer_cache.intermediate_ssm[0, :, 0, 0, 0],
        torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(
        layer_cache.intermediate_conv_window[0][0, :, 0, 0],
        torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(
        layer_cache.temporal, torch.zeros_like(layer_cache.temporal)
    )
    torch.testing.assert_close(
        layer_cache.conv[0], torch.zeros_like(layer_cache.conv[0])
    )


_GDN = "atom.plugin.sglang.attention_backend.attention_gdn"


def _verify_once(*, replay, cache_indices, core_attn_out, conv=None, replay_fn=None):
    """One TARGET_VERIFY forward of `_make_impl()` (bs=1, 2 drafts) with the
    GPU kernels faked; returns (output, layer_cache, conv kwargs)."""
    conv_phys = torch.zeros(1, 3, 2)
    layer_cache = SimpleNamespace(
        conv=[torch.arange(12.0).reshape(4, 3, 1) if conv is None else conv],
        temporal=torch.zeros(4, 1, 1, 1),
        intermediate_ssm=torch.zeros(1, 2, 1, 1, 1),
        intermediate_conv_window=[conv_phys.as_strided((1, 2, 3, 1), (6, 1, 2, 1))],
    )
    linear_backend = _FakeLinearBackend(layer_cache)
    linear_backend.forward_metadata.mamba_cache_indices = cache_indices
    forward_batch = SimpleNamespace(
        forward_mode=_TargetVerifyMode(),
        spec_info=SimpleNamespace(draft_token_num=2),
        batch_size=1,
    )
    seen = {}

    def fake_conv(x, conv_state, *_args, **kwargs):
        seen.update(kwargs, window_before=conv_state.clone())
        conv_state[0] = 7.0
        return x[:, :1], x[:, 1:2], x[:, 2:3]

    def fake_recurrent(*, q, **_kwargs):
        return q, None

    with (
        bind_current_forward_batch(forward_batch),
        patch.object(
            SGLangGDNForwardContext,
            "_resolve_attn_backend",
            return_value=SimpleNamespace(linear_attn_backend=linear_backend),
        ),
        patch(f"{_GDN}.replayssm_enabled", return_value=replay),
        patch(f"{_GDN}.causal_conv1d_update", side_effect=fake_conv),
        patch(
            f"{_GDN}.fused_gdn_gating",
            side_effect=lambda _A, a, b, _dt: (a.unsqueeze(0), b.unsqueeze(0)),
        ),
        patch(f"{_GDN}.fused_recurrent_gated_delta_rule", side_effect=fake_recurrent),
        patch(
            f"{_GDN}.prepare_layer",
            return_value=(None, None, None, None, None, "s_waitcnt vmcnt(0)"),
        ),
        patch(f"{_GDN}.replayssm_gated_delta_rule", side_effect=replay_fn),
    ):
        output = _make_impl().forward(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            torch.ones(2, 1),
            torch.ones(2, 1),
            core_attn_out,
            "layer_7",
        )
    return output, layer_cache, seen


def test_target_verify_conv_reads_the_live_state_in_place():
    """The conv kernel reads the history from the live pool by slot; nothing
    is gathered into SGLang's window first and the live state is never written."""
    live = torch.arange(12.0).reshape(4, 3, 1)
    for replay in (False, True):
        cache_indices = torch.tensor([2], dtype=torch.int32)
        _, layer_cache, seen = _verify_once(
            replay=replay,
            cache_indices=cache_indices,
            core_attn_out=torch.empty(2, 1, 1),
            conv=live.clone(),
            replay_fn=lambda *, o, **_kw: o,
        )
        assert seen["conv_state_in"] is layer_cache.conv[0]
        reads = seen["conv_state_indices_in"]
        writes = seen["conv_state_indices"]
        assert reads.tolist() == [[2]]
        assert reads.data_ptr() == cache_indices.data_ptr()
        assert reads.shape == writes.shape and reads.stride() == writes.stride()
        assert torch.count_nonzero(seen["window_before"]) == 0
        torch.testing.assert_close(layer_cache.conv[0], live, rtol=0, atol=0)
        torch.testing.assert_close(
            layer_cache.intermediate_conv_window[0][0, :, :, 0],
            torch.full((2, 3), 7.0),
        )


def test_replayssm_verify_writes_the_layer_output_in_place():
    calls = []

    def fake_replay(*, o, **_kwargs):
        calls.append(o)
        o.fill_(3.0)
        return o.unsqueeze(0)

    out = torch.empty(2, 1, 1)
    result, _, _ = _verify_once(
        replay=True,
        cache_indices=torch.tensor([1], dtype=torch.int32),
        core_attn_out=out,
        replay_fn=fake_replay,
    )
    assert calls[-1].data_ptr() == out.data_ptr()
    assert result is out
    torch.testing.assert_close(out, torch.full((2, 1, 1), 3.0))


def test_target_verify_rejects_non_speculative_state_pool():
    expected = torch.empty(2, 1, 1)
    linear_backend = _FakeLinearBackend(SimpleNamespace())
    active_backend = SimpleNamespace(linear_attn_backend=linear_backend)
    forward_batch = SimpleNamespace(
        forward_mode=_TargetVerifyMode(),
        spec_info=SimpleNamespace(draft_token_num=2),
        batch_size=1,
    )

    with (
        bind_current_forward_batch(forward_batch),
        patch.object(
            SGLangGDNForwardContext,
            "_resolve_attn_backend",
            return_value=active_backend,
        ),
    ):
        try:
            _make_impl().forward(
                torch.empty(2, 3),
                torch.empty(2, 1),
                torch.empty(2, 1),
                expected,
                "layer_7",
            )
        except RuntimeError as exc:
            assert "intermediate-state buffers" in str(exc)
        else:
            raise AssertionError("Expected missing speculative state to fail")
