"""FlyDSL GDN prefill under the SGLang ReplaySSM ring, on CPU tensors.

ReplaySSM verify replays the raw contiguous checkpoint and refuses a strided
one (`SGLangGatedDeltaNet._verify_replayssm`), so under the ring the plugin
keeps FlyDSL decode and its VK view off while the layout-agnostic FlyDSL
prefill still runs on gfx942.
"""

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.fla_ops import gdn_flydsl as fly
from atom.plugin.sglang.attention_backend import attention_gdn
from atom.plugin.sglang.attention_backend.attention_gdn import (
    SGLangGDNForwardContext,
)


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    # The conv1d program map is a Triton kernel; these tests run on CPU tensors.
    monkeypatch.setattr(
        attention_gdn,
        "compute_causal_conv1d_metadata",
        lambda query_start_loc, num_tokens: (None, None, None),
    )
    monkeypatch.setattr(attention_gdn, "_PREFILL_SCHEDULES", OrderedDict())


class _ExtendMode:
    @staticmethod
    def is_target_verify():
        return False

    @staticmethod
    def is_decode_or_idle():
        return False

    @staticmethod
    def is_extend():
        return True

    @staticmethod
    def is_prefill():
        return True

    @staticmethod
    def is_idle():
        return False


class _MambaPool:
    """One mamba layer; per-layer views are taken from the pool's state
    tensors, as SGLang's `MambaPool.State.at_layer_idx` does."""

    def __init__(self, temporal):
        self.mamba_map = {0: 0}
        self.mamba_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(
                conv=[torch.zeros(1, 2, 4)], temporal=temporal[None]
            )
        )

    @staticmethod
    def get_mamba_indices(req_pool_indices):
        return req_pool_indices

    def mamba2_layer_index(self, layer_id):
        return self.mamba_map[layer_id]

    def mamba2_layer_cache(self, layer_id):
        state = self.mamba_pool.mamba_cache
        index = self.mamba2_layer_index(layer_id)
        return SimpleNamespace(
            conv=[state.conv[0][index]], temporal=state.temporal[index]
        )


def _qwen4_impl(monkeypatch):
    impl = SimpleNamespace(
        allow_aiter_flydsl=True,
        dt_bias=torch.zeros(1, dtype=torch.bfloat16),
        gdn_flydsl_policy=None,
    )
    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"Linear_0": SimpleNamespace(impl=impl)}
        ),
        enable_dp_attention=False,
    )
    monkeypatch.setattr(attention_gdn, "get_current_atom_config", lambda: cfg)
    return impl


@pytest.mark.parametrize("replayssm", [False, True])
def test_ring_keeps_the_raw_checkpoint_and_runs_flydsl_prefill(monkeypatch, replayssm):
    impl = _qwen4_impl(monkeypatch)
    monkeypatch.setattr(attention_gdn, "replayssm_enabled", lambda: replayssm)
    monkeypatch.setattr(fly, "backend", lambda stage: "auto")
    monkeypatch.setattr(fly, "ops", lambda: object())
    # Decode would be eligible on this device; only the ring rules it out.
    monkeypatch.setattr(fly, "_validated_on_gfx942", lambda: True)
    schedule = object()
    monkeypatch.setattr(
        fly, "build_prefill_metadata", lambda lengths, cu_seqlens: schedule
    )

    temporal = torch.zeros(4, 24, 128, 128)
    pool = _MambaPool(temporal)
    caches = SGLangGDNForwardContext._build_kv_cache_tensors(
        SimpleNamespace(req_to_token_pool=pool),
        SimpleNamespace(req_to_token_pool=pool),
    )
    policy = impl.gdn_flydsl_policy
    assert policy.prefill
    assert policy.decode is (not replayssm)
    state = caches["layer_0"].v_cache
    if replayssm:
        # The tensor `_verify_replayssm` replays, untransposed.
        assert state.data_ptr() == temporal.data_ptr()
        assert state.shape == temporal.shape and state.is_contiguous()
    else:
        assert state.data_ptr() == temporal.data_ptr()
        assert state.stride()[-2:] == (1, 128)

    forward_batch = SimpleNamespace(
        forward_mode=_ExtendMode(),
        batch_size=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
        extend_seq_lens_cpu=[8],
    )
    linear = SimpleNamespace(
        forward_metadata=SimpleNamespace(
            query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
            mamba_cache_indices=torch.tensor([3], dtype=torch.int32),
            logical_num_tokens=8,
        ),
        req_to_token_pool=pool,
    )
    metadata = SGLangGDNForwardContext._build_gdn_metadata(forward_batch, linear)
    assert metadata.flydsl_prefill_metadata is schedule
    # Extend metadata never turns on the Native layer's ReplaySSM kernels.
    assert metadata.replayssm is False
