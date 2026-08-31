from types import SimpleNamespace

import torch

from atom.plugin.sglang.attention_backend import attention_gdn
from atom.plugin.sglang.attention_backend.attention_gdn import (
    SGLangGDNForwardContext,
)


class _DecodeMode:
    @staticmethod
    def is_prefill():
        return False

    @staticmethod
    def is_idle():
        return False

    @staticmethod
    def is_target_verify():
        return False

    @staticmethod
    def is_decode_or_idle():
        return True

    @staticmethod
    def is_extend():
        return False


class _MambaPool:
    @staticmethod
    def get_mamba_indices(req_pool_indices):
        return req_pool_indices + 10


def test_build_gdn_metadata_reconstructs_missing_forward_metadata():
    pool = _MambaPool()
    forward_batch = SimpleNamespace(
        forward_mode=_DecodeMode(),
        batch_size=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        req_to_token_pool=pool,
    )
    linear_backend = SimpleNamespace(
        forward_metadata=None,
        req_to_token_pool=pool,
    )

    metadata = SGLangGDNForwardContext._build_gdn_metadata(
        forward_batch, linear_backend
    )

    assert metadata is not None
    assert torch.equal(
        metadata.non_spec_query_start_loc,
        torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    assert torch.equal(
        metadata.non_spec_state_indices_tensor,
        torch.tensor([10, 11], dtype=torch.int32),
    )


def test_build_context_falls_back_to_local_decode_mode(monkeypatch):
    monkeypatch.setattr(
        attention_gdn,
        "get_current_atom_config",
        lambda: SimpleNamespace(enable_dp_attention=True),
    )
    forward_batch = SimpleNamespace(
        forward_mode=_DecodeMode(),
        positions=torch.zeros(2, dtype=torch.long),
        batch_size=2,
    )

    context, num_tokens = SGLangGDNForwardContext._build_context(forward_batch)

    assert context.running_tokens_are_unified
    assert num_tokens == 2


def test_bind_preserves_outer_attention_metadata(monkeypatch):
    slot_mapping = object()
    outer_metadata = SimpleNamespace(max_seqlen_k=128, slot_mapping=slot_mapping)
    outer_kv = {"outer": object()}
    current_context = SimpleNamespace(
        context=object(),
        attn_metadata=outer_metadata,
        kv_cache_data=outer_kv,
    )
    gdn_metadata = object()
    inner_kv = {"inner": object()}
    forward_context = SimpleNamespace(
        gdn_metadata=gdn_metadata,
        kv_cache_data=inner_kv,
    )
    kv_updates = []

    def fail_if_forward_context_is_rebuilt(**_kwargs):
        raise AssertionError("outer forward context must be reused")

    monkeypatch.setattr(
        SGLangGDNForwardContext,
        "build",
        classmethod(lambda cls, _metadata: forward_context),
    )
    monkeypatch.setattr(attention_gdn, "get_forward_context", lambda: current_context)
    monkeypatch.setattr(attention_gdn, "set_kv_cache_data", kv_updates.append)
    monkeypatch.setattr(
        attention_gdn, "set_forward_context", fail_if_forward_context_is_rebuilt
    )

    with SGLangGDNForwardContext.bind(object()):
        bound_metadata = current_context.attn_metadata
        assert bound_metadata is not outer_metadata
        assert bound_metadata.max_seqlen_k == 128
        assert bound_metadata.slot_mapping is slot_mapping
        assert bound_metadata.gdn_metadata is gdn_metadata
        assert current_context.kv_cache_data is inner_kv

    assert current_context.attn_metadata is outer_metadata
    assert current_context.kv_cache_data is outer_kv
    assert kv_updates[0] is inner_kv
