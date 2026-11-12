# SPDX-License-Identifier: MIT
"""ATOM-side QSA cache format selection and KV pool layout."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter")

from atom.model_ops.qwen4_exp.ops import qsa


def _config(kv, index=None):
    return SimpleNamespace(kv_cache_dtype=kv, index_cache_dtype=index)


@pytest.mark.parametrize(
    ("kv", "index", "expected"),
    [
        ("bf16", None, ("bf16", "bf16")),
        ("auto", None, ("bf16", "bf16")),
        ("fp8", None, ("fp8", "fp8")),
        ("fp8", "bf16", ("fp8", "bf16")),
        ("fp8", "fp4", ("fp8", "fp4")),
        ("bf16", "fp8", ("bf16", "fp8")),
    ],
)
def test_cache_formats_follow_the_kv_dtype(kv, index, expected):
    assert qsa.qsa_cache_formats(_config(kv, index)) == expected


@pytest.mark.parametrize(("kv", "index"), [("int8", None), ("fp4", None)])
def test_cache_formats_reject_unserved(kv, index):
    with pytest.raises(ValueError, match="Qwen QSA serves"):
        qsa.qsa_cache_formats(_config(kv, index))


def _builder(kv, index, draft_rows, block=16, rope=False):
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpMetadataBuilder

    builder = object.__new__(Qwen4ExpMetadataBuilder)
    hf = SimpleNamespace(head_dim=256, indexer_head_dim=128, indexer_compress_ratio=4)
    config = SimpleNamespace(kv_cache_dtype=kv, index_cache_dtype=index, hf_config=hf)
    builder.model_runner = SimpleNamespace(config=config, _get_num_kv_heads=lambda: 2)
    builder.block_size = block
    builder.cache_rope_positions = rope
    counts = {"qsa": 12, "qsa_draft": draft_rows}
    builder.row_counts = lambda: {k: v for k, v in counts.items() if v}
    return builder


def test_bf16_fields_are_unchanged():
    fields = _builder("bf16", None, 0)._qsa_fields()
    assert [(f.name, f.layers, f.shape, f.dtype) for f in fields] == [
        ("k", 12, (16, 2, 256), torch.bfloat16),
        ("v", 12, (16, 2, 256), torch.bfloat16),
        ("index_raw", 12, (16, 1, 128), torch.bfloat16),
        ("index_compressed", 12, (4, 1, 128), torch.bfloat16),
    ]


def test_quantized_fields_and_bf16_draft_rows():
    builder = _builder("fp8", "fp4", 1, block=64)
    fields = {f.name: f for f in builder._qsa_fields()}
    fp8 = qsa.FP8_DTYPE
    expected = {
        "k": (12, (64, 2, 256), fp8),
        "v": (12, (64, 2, 256), fp8),
        "k_scale": (12, (64, 2), torch.float32),
        "v_scale": (12, (64, 2), torch.float32),
        "index_raw": (12, (64, 1, 128), torch.bfloat16),
        "index_compressed": (12, (16, 1, 64), torch.uint8),
        "index_compressed_scale": (12, (16, 1, 4), torch.uint8),
        "k_draft": (1, (64, 2, 256), torch.bfloat16),
        "v_draft": (1, (64, 2, 256), torch.bfloat16),
        "index_raw_draft": (1, (64, 1, 128), torch.bfloat16),
        "index_compressed_draft": (1, (16, 1, 128), torch.bfloat16),
    }
    assert {n: (f.layers, f.shape, f.dtype) for n, f in fields.items()} == expected


def test_fp8_pool_fits_more_blocks():
    bf16 = _builder("bf16", None, 0, rope=True)
    bf16.row_counts = lambda: {"qsa": 13}
    # Block bytes of Qwen3.8-Flash-Next's 12 target + 1 draft QSA layers, as
    # served (multimodal, so mRoPE positions are cached).
    assert bf16._page_bytes() == 497664
    assert _builder("fp8", "fp8", 1, rope=True)._page_bytes() == 298240
    assert _builder("fp8", "bf16", 1, 64, rope=True)._page_bytes() == 1216000
