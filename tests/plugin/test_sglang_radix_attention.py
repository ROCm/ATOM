"""Regression tests for ATOM's SGLang RadixAttention wrapper."""

from pathlib import Path


def test_radix_attention_uses_forward_context_backend_not_forward_batch():
    """SGLang 0.5.15 resolves attention backend from the forward context."""
    source = Path(
        "atom/plugin/sglang/attention_backend/full_attention/radix_attention.py"
    ).read_text()

    assert "forward_batch.attn_backend" not in source
    assert 'getattr(forward_batch, "attn_backend"' not in source
    assert "_resolve_forward_context_input_dtype()" in source
