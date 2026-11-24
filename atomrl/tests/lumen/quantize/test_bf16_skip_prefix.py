# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Matching of the BF16-skip prefixes against module names.

The prefixes are layer module paths ending in an index, so a plain
``startswith`` makes ``decoder.layers.1`` match ``decoder.layers.10`` through
``.19`` as well. Nothing raises when that happens -- the run simply keeps ten
more layers in BF16 than asked for, and the only symptom is a step time that
looks like a slow kernel. Both ``_patch_linear_layers`` and
``enable_fp8_for_parallel_linear`` select layers through this predicate.
"""

from atomrl.lumen.quantize import is_under_bf16_prefix


class TestIsUnderBF16Prefix:
    def test_exact_name_matches(self):
        assert is_under_bf16_prefix("decoder.layers.1", {"decoder.layers.1"})

    def test_submodule_of_a_bf16_layer_matches(self):
        prefixes = {"decoder.layers.1"}
        assert is_under_bf16_prefix("decoder.layers.1.mlp.linear_fc1", prefixes)
        assert is_under_bf16_prefix(
            "decoder.layers.1.self_attention.linear_qkv", prefixes
        )

    def test_sibling_index_sharing_the_prefix_does_not_match(self):
        """The regression: .1 must not capture .10 through .19."""
        prefixes = {"decoder.layers.1"}
        for index in range(10, 20):
            name = f"decoder.layers.{index}"
            assert not is_under_bf16_prefix(name, prefixes), name
            assert not is_under_bf16_prefix(f"{name}.mlp.linear_fc1", prefixes), name

    def test_no_prefixes_matches_nothing(self):
        assert not is_under_bf16_prefix("decoder.layers.0", set())

    def test_start_of_model_skip_holds_only_its_own_layers(self):
        """A start-of-2 skip on 20 layers: exactly layers 0 and 1 stay BF16."""
        prefixes = {"decoder.layers.0", "decoder.layers.1"}
        held = [
            i
            for i in range(20)
            if is_under_bf16_prefix(f"decoder.layers.{i}", prefixes)
        ]
        assert held == [0, 1]

    def test_tail_skip_holds_only_its_own_layers(self):
        """The default recipe, which the collision happens to spare."""
        prefixes = {f"decoder.layers.{i}" for i in range(31, 36)}
        held = [
            i
            for i in range(36)
            if is_under_bf16_prefix(f"decoder.layers.{i}", prefixes)
        ]
        assert held == [31, 32, 33, 34, 35]


class TestBf16SkipWithoutALayerCount:
    """num_layers=0 must not turn into "every layer stays BF16".

    ``total - bf16_end`` goes non-positive, so the skip predicate answers True
    for every index and the caller keeps the whole model in BF16 -- a run that
    reports quantization enabled and trains none of it. The Megatron native
    path guards this at its call site; the generic path did not.
    """

    @staticmethod
    def _model():
        from torch import nn

        model = nn.Module()
        model.layers = nn.ModuleList(nn.Linear(8, 8) for _ in range(4))
        return model

    def _config(self, num_layers):
        from atomrl.lumen.quantize.config import QuantConfig

        return QuantConfig(
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=1,
            num_layers_at_end_in_bf16=1,
            num_layers=num_layers,
        )

    def test_no_layer_count_quantizes_everything(self, caplog):
        import logging

        from atomrl.lumen.quantize import _build_bf16_skip_prefixes

        with caplog.at_level(logging.WARNING):
            prefixes = _build_bf16_skip_prefixes(self._model(), self._config(0))

        assert prefixes == set(), "kept the whole model in BF16 instead of none of it"
        assert any("num_layers" in r.message for r in caplog.records), "said nothing"

    def test_a_layer_count_still_selects_the_ends(self):
        from atomrl.lumen.quantize import _build_bf16_skip_prefixes

        prefixes = _build_bf16_skip_prefixes(self._model(), self._config(4))
        assert prefixes == {"layers.0", "layers.3"}
