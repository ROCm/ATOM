from types import SimpleNamespace

import pytest
from torch import nn

pytest.importorskip("aiter")

from atom.models import eagle3_llama


class _Attention(nn.Module):
    def __init__(self, *, reduce_results, **_):
        super().__init__()
        self.reduce_results = reduce_results


class _MLP(nn.Module):
    def __init__(self, *, reduce_results, **_):
        super().__init__()
        self.reduce_results = reduce_results


class _Norm(nn.Module):
    def __init__(self, _dim, *, eps, fused_allreduce=False):
        super().__init__()
        self.eps = eps
        self.fused_allreduce = fused_allreduce


def test_eagle3_keeps_attention_reduce_separate(monkeypatch):
    monkeypatch.setattr(eagle3_llama, "ENABLE_ALLREDUCE_RMSNORM_FUSION", True)
    monkeypatch.setattr(eagle3_llama, "Eagle3LlamaAttention", _Attention)
    monkeypatch.setattr(eagle3_llama, "Eagle3LlamaMLP", _MLP)
    monkeypatch.setattr(eagle3_llama, "RMSNorm", _Norm)

    config = SimpleNamespace(
        hidden_size=6144,
        num_attention_heads=64,
        num_key_value_heads=4,
        intermediate_size=18432,
        rms_norm_eps=1e-6,
    )
    layer = eagle3_llama.Eagle3LlamaDecoderLayer(config, norm_output=True)

    assert layer.self_attn.reduce_results is True
    assert layer.post_attention_layernorm.fused_allreduce is False
    assert layer.mlp.reduce_results is False
