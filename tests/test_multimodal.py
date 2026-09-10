# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for atom/model_engine/multimodal.py.

Covers the pieces the prefix cache depends on: where each media item's tokens
land in a prompt, and the content hash that tells two images apart.
"""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from atom.model_engine.multimodal import (
    MediaPlaceholder,
    build_media_placeholders,
    expand_media_placeholders,
    hash_media,
    locate_media_placeholders,
    resolve_placeholder_token_id,
)

PAD = 999


def _config(architecture: str | None, multimodal_config):
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[architecture] if architecture else []),
        multimodal_config=multimodal_config,
    )


# ── locating already-expanded placeholders ─────────────────────────────────


class TestLocateMediaPlaceholders:
    def test_single_run(self):
        ids = [1, 2, PAD, PAD, PAD, 3]
        assert locate_media_placeholders(ids, PAD, 1) == [(2, 3)]

    def test_two_runs_keep_prompt_order(self):
        ids = [1, PAD, PAD, 2, 3, PAD, 4]
        assert locate_media_placeholders(ids, PAD, 2) == [(1, 2), (5, 1)]

    def test_run_at_the_end_of_the_prompt(self):
        ids = [1, 2, PAD, PAD]
        assert locate_media_placeholders(ids, PAD, 1) == [(2, 2)]

    def test_count_mismatch_raises(self):
        # Two adjacent images with no separator read as one run. Guessing which
        # half belongs to which image would hand the prefix cache a range
        # covering an image it does not name, so this has to fail loudly.
        ids = [1, PAD, PAD, 2]
        with pytest.raises(ValueError, match="placeholder run"):
            locate_media_placeholders(ids, PAD, 2)

    def test_no_placeholders_for_a_media_request_raises(self):
        with pytest.raises(ValueError, match="placeholder run"):
            locate_media_placeholders([1, 2, 3], PAD, 1)


# ── expanding one-per-item placeholders (Kimi-K3) ──────────────────────────


class TestExpandMediaPlaceholders:
    def test_ranges_match_the_expansion(self):
        expanded, ranges = expand_media_placeholders([1, PAD, 2, PAD, 3], [3, 2], PAD)
        assert expanded == [1, PAD, PAD, PAD, 2, PAD, PAD, 3]
        assert ranges == [(1, 3), (5, 2)]
        # The ranges have to address the expanded ids, not the originals.
        for offset, length in ranges:
            assert expanded[offset : offset + length] == [PAD] * length

    def test_text_only_prompt_is_untouched(self):
        expanded, ranges = expand_media_placeholders([1, 2, 3], [], PAD)
        assert expanded == [1, 2, 3]
        assert ranges == []

    def test_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="media placeholder tokens"):
            expand_media_placeholders([1, PAD, 2], [3, 2], PAD)


# ── media content hashing ──────────────────────────────────────────────────


class TestHashMedia:
    def test_deterministic(self):
        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        assert hash_media(image) == hash_media(image.copy())

    def test_different_pixels_differ(self):
        first = np.zeros((2, 2, 3), dtype=np.uint8)
        second = first.copy()
        second[0, 0, 0] = 1
        assert hash_media(first) != hash_media(second)

    def test_shape_participates(self):
        # tobytes() drops the shape, so without it these two collide.
        flat = np.arange(12, dtype=np.uint8)
        assert hash_media(flat.reshape(2, 2, 3)) != hash_media(flat.reshape(3, 2, 2))

    def test_dtype_participates(self):
        values = [0, 1, 2, 3]
        assert hash_media(np.array(values, dtype=np.uint8)) != hash_media(
            np.array(values, dtype=np.int32)
        )


class TestBuildMediaPlaceholders:
    def test_pairs_ranges_with_hashes_in_order(self):
        first = np.zeros((2, 2, 3), dtype=np.uint8)
        second = np.ones((2, 2, 3), dtype=np.uint8)
        items = build_media_placeholders([(4, 2), (9, 3)], [first, second])

        assert [(i.offset, i.length) for i in items] == [(4, 2), (9, 3)]
        assert [i.modality for i in items] == ["image", "image"]
        assert items[0].identifier == hash_media(first)
        assert items[1].identifier != items[0].identifier

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="media range"):
            build_media_placeholders([(0, 1)], [np.zeros(1), np.zeros(1)])

    def test_placeholder_is_hashable_and_frozen(self):
        item = MediaPlaceholder(identifier=1, modality="image", offset=0, length=2)
        assert {item}  # shared across n>1 siblings, so it must not be mutable
        with pytest.raises(FrozenInstanceError):
            item.offset = 5


# ── placeholder token id resolution ────────────────────────────────────────


class TestResolvePlaceholderTokenId:
    def test_qwen_reads_image_token_id(self):
        config = _config(
            "Qwen3_5ForConditionalGeneration", SimpleNamespace(image_token_id=248056)
        )
        assert resolve_placeholder_token_id(config) == 248056

    def test_kimi_reads_media_placeholder_token_id(self):
        config = _config(
            "KimiK3ForConditionalGeneration",
            SimpleNamespace(media_placeholder_token_id=163605),
        )
        assert resolve_placeholder_token_id(config) == 163605

    def test_unregistered_architecture_raises(self):
        config = _config("SomeOtherForCausalLM", SimpleNamespace())
        with pytest.raises(ValueError, match="no media placeholder token"):
            resolve_placeholder_token_id(config)

    def test_missing_multimodal_config_raises(self):
        # `--trust-remote-code` off leaves this None. Falling back to a literal
        # id would point the media ranges at text and key blocks on offsets
        # that describe nothing, so it has to raise.
        config = _config("Qwen3_5ForConditionalGeneration", None)
        with pytest.raises(ValueError, match="full HF config"):
            resolve_placeholder_token_id(config)

    def test_missing_attribute_raises(self):
        config = _config("Qwen3_5ForConditionalGeneration", SimpleNamespace())
        with pytest.raises(ValueError, match="image_token_id"):
            resolve_placeholder_token_id(config)
