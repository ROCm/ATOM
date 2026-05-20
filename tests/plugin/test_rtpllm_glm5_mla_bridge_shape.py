"""Shape-level tests for the GLM5 RTP MLA bridge."""

from types import SimpleNamespace

import torch

from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention


def test_mla_attention_legacy_boundary_shape_stays_executable_during_migration():
    q = torch.empty(2, 4, 256)
    compressed_kv = torch.empty(2, 512)
    k_pe = torch.empty(2, 64)
    positions = torch.arange(2, dtype=torch.int32)
    attention = RTPMLAAttention(mla_modules=SimpleNamespace(v_head_dim=128))

    output = attention(q, compressed_kv, k_pe, positions=positions)

    assert output.shape == (2, 4, 128)


def test_mla_attention_is_marked_as_mla_adapter():
    assert RTPMLAAttention.use_mla is True

