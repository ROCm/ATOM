"""Regression tests for ATOM's SGLang RadixAttention wrapper."""

from pathlib import Path


def test_radix_attention_does_not_directly_require_forward_batch_attn_backend():
    """Current SGLang ForwardBatch may not expose attn_backend directly."""
    source = Path(
        "atom/plugin/sglang/attention_backend/full_attention/radix_attention.py"
    ).read_text()

    assert "forward_batch.attn_backend" not in source
