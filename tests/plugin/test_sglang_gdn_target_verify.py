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


def test_target_verify_snapshots_each_step_and_restores_live_state():
    layer_cache = SimpleNamespace(
        conv=[torch.zeros(1, 3, 1)],
        temporal=torch.zeros(1, 1, 1, 1),
        intermediate_ssm=torch.zeros(1, 2, 1, 1, 1),
        intermediate_conv_window=[torch.zeros(1, 2, 3, 1)],
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

    def fake_conv(x, conv_state, *_args, conv_state_indices, **_kwargs):
        conv_state[conv_state_indices] += 1
        return x[:, :1], x[:, 1:2], x[:, 2:3]

    def fake_gating(_A_log, a_step, b_step, _dt_bias):
        return a_step.unsqueeze(0), b_step.unsqueeze(0)

    def fake_recurrent(*, q, initial_state, ssm_state_indices, **_kwargs):
        initial_state[ssm_state_indices] += 1
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
    torch.testing.assert_close(layer_cache.temporal, torch.zeros_like(layer_cache.temporal))
    torch.testing.assert_close(layer_cache.conv[0], torch.zeros_like(layer_cache.conv[0]))


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
