# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Draft fallback dispatch must respect optional text-only RoPE caches (CPU)."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from atom.models.qwen38.mono import draft_fallback


def _context(requests, *, prefill=False):
    meta = SimpleNamespace(block_tables=SimpleNamespace(shape=(requests, 4096)))
    return SimpleNamespace(
        attn_metadata=SimpleNamespace(qsa_metadata=meta),
        context=SimpleNamespace(is_prefill=prefill, is_draft=True),
    )


def _attention(rope_cache):
    module = Mock()
    module.rope_position_cache = rope_cache
    module.compressed_key_scale = None
    module.k_scale_cache = None
    module.indexer.compress_ratio = 4
    module.indexer.index_head_dim = 128
    module.qkv_proj.side_effect = RuntimeError("entered fused QSA")
    return module


class TestDraftFallbackAttention(unittest.TestCase):
    def test_text_pool_uses_native_qsa_before_fused_state_changes(self):
        # Six requests reproduce the serving failure after leaving the <=4
        # request draft Mono program. Check both q1 and q4 fallback layouts.
        for requests in (1, 6):
            for query_length in (1, 4):
                with self.subTest(requests=requests, query_length=query_length):
                    module = _attention(None)
                    positions = object()
                    mixed = SimpleNamespace(shape=(requests * query_length, 2560))
                    with patch.object(
                        draft_fallback,
                        "get_forward_context",
                        return_value=_context(requests),
                    ):
                        result = draft_fallback.attention(module, positions, mixed)
                    self.assertIs(result, module.return_value)
                    module.assert_called_once_with(positions, mixed)
                    module.qkv_proj.assert_not_called()

    def test_position_cache_keeps_supported_fused_fallback(self):
        for query_length in (1, 4):
            with self.subTest(query_length=query_length):
                module = _attention(object())
                mixed = SimpleNamespace(shape=(6 * query_length, 2560))
                with (
                    patch.object(
                        draft_fallback, "get_forward_context", return_value=_context(6)
                    ),
                    self.assertRaisesRegex(RuntimeError, "entered fused QSA"),
                ):
                    draft_fallback.attention(module, object(), mixed)
                module.qkv_proj.assert_called_once_with(mixed)
                module.assert_not_called()

    def test_existing_native_dispatch_cases_are_preserved(self):
        contexts = [_context(6, prefill=True), _context(6), _context(6)]
        contexts[1].attn_metadata.qsa_metadata = None
        for context, rows in zip(contexts, (6, 6, 7)):
            with self.subTest(prefill=context.context.is_prefill, rows=rows):
                module = _attention(object())
                positions = object()
                mixed = SimpleNamespace(shape=(rows, 2560))
                with patch.object(
                    draft_fallback, "get_forward_context", return_value=context
                ):
                    result = draft_fallback.attention(module, positions, mixed)
                self.assertIs(result, module.return_value)
                module.assert_called_once_with(positions, mixed)
                module.qkv_proj.assert_not_called()


if __name__ == "__main__":
    unittest.main()
