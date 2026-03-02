# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the per-layer quantization config refactor.

Tests cover:
- LayerQuantSpec: frozen dataclass, no_quant, predicates
- ParsedQuantConfig: dataclass construction
- Parser registry: registration, dispatch, fallback
- QuarkParser / CompressedTensorsParser / GenericParser: parsing logic
- QuantizationConfig.resolve(): exclude-list resolution, per-layer overrides
- Backward compatibility: dict access, should_ignore_layer, get_quant_config_for_layer
- LinearBase: layer_spec parameter plumbing
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# ── conftest.py stubs atom.config → clean up before we import ──
for mod_name in list(sys.modules):
    if mod_name.startswith("atom"):
        del sys.modules[mod_name]

import torch
from aiter import QuantType
from aiter.utility.dtypes import d_dtypes


# ====================================================================
# 1. LayerQuantSpec
# ====================================================================
class TestLayerQuantSpec(unittest.TestCase):
    def test_default_is_no_quant(self):
        from atom.quant_spec import LayerQuantSpec
        spec = LayerQuantSpec()
        self.assertEqual(spec.quant_type, QuantType.No)
        self.assertEqual(spec.quant_dtype, torch.bfloat16)
        self.assertFalse(spec.is_quantized)
        self.assertFalse(spec.needs_online_quant)

    def test_no_quant_singleton(self):
        from atom.quant_spec import LayerQuantSpec
        a = LayerQuantSpec.no_quant()
        b = LayerQuantSpec.no_quant()
        self.assertIs(a, b)

    def test_quantized_spec(self):
        from atom.quant_spec import LayerQuantSpec
        spec = LayerQuantSpec(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            quant_method="quark",
        )
        self.assertTrue(spec.is_quantized)
        self.assertFalse(spec.needs_online_quant)

    def test_online_quant(self):
        from atom.quant_spec import LayerQuantSpec
        spec = LayerQuantSpec(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            checkpoint_dtype=torch.bfloat16,
        )
        self.assertTrue(spec.needs_online_quant)

    def test_frozen(self):
        from atom.quant_spec import LayerQuantSpec
        spec = LayerQuantSpec()
        with self.assertRaises(AttributeError):
            spec.quant_type = QuantType.per_Token  # type: ignore

    def test_equality(self):
        from atom.quant_spec import LayerQuantSpec
        a = LayerQuantSpec(quant_type=QuantType.per_Token, quant_dtype=d_dtypes["fp8"])
        b = LayerQuantSpec(quant_type=QuantType.per_Token, quant_dtype=d_dtypes["fp8"])
        self.assertEqual(a, b)

    def test_flags(self):
        from atom.quant_spec import LayerQuantSpec
        spec = LayerQuantSpec(flags={"use_triton": True})
        self.assertTrue(spec.flags["use_triton"])


# ====================================================================
# 2. ParsedQuantConfig
# ====================================================================
class TestParsedQuantConfig(unittest.TestCase):
    def test_default_fields(self):
        from atom.quant_spec import LayerQuantSpec, ParsedQuantConfig
        pqc = ParsedQuantConfig(global_spec=LayerQuantSpec())
        self.assertEqual(pqc.layer_specs, {})
        self.assertEqual(pqc.exclude_layers, [])

    def test_with_overrides(self):
        from atom.quant_spec import LayerQuantSpec, ParsedQuantConfig
        global_spec = LayerQuantSpec(quant_type=QuantType.per_1x32, quant_dtype=torch.float4_e2m1fn_x2)
        layer_specs = {"model.layers.0.self_attn.qkv_proj": LayerQuantSpec.no_quant()}
        pqc = ParsedQuantConfig(
            global_spec=global_spec,
            layer_specs=layer_specs,
            exclude_layers=["lm_head"],
        )
        self.assertEqual(pqc.layer_specs["model.layers.0.self_attn.qkv_proj"].quant_type, QuantType.No)
        self.assertEqual(pqc.exclude_layers, ["lm_head"])


# ====================================================================
# 3. Parser registry
# ====================================================================
class TestParserRegistry(unittest.TestCase):
    def test_builtin_parsers_registered(self):
        from atom.quant_spec import _QUANT_PARSERS
        self.assertIn("quark", _QUANT_PARSERS)
        self.assertIn("compressed-tensors", _QUANT_PARSERS)
        self.assertIn("__generic__", _QUANT_PARSERS)

    def test_get_quant_parser_known(self):
        from atom.quant_spec import get_quant_parser, QuarkParser
        parser = get_quant_parser("quark")
        self.assertIsInstance(parser, QuarkParser)

    def test_get_quant_parser_unknown_falls_back(self):
        from atom.quant_spec import get_quant_parser, GenericParser
        parser = get_quant_parser("some_unknown_method")
        self.assertIsInstance(parser, GenericParser)

    def test_register_custom_parser(self):
        from atom.quant_spec import (
            QuantConfigParser,
            ParsedQuantConfig,
            LayerQuantSpec,
            register_quant_parser,
            get_quant_parser,
            _QUANT_PARSERS,
        )

        @register_quant_parser("test_method")
        class TestParser(QuantConfigParser):
            def parse(self, raw_config: dict) -> ParsedQuantConfig:
                return ParsedQuantConfig(global_spec=LayerQuantSpec())

        parser = get_quant_parser("test_method")
        self.assertIsInstance(parser, TestParser)

        # Clean up
        del _QUANT_PARSERS["test_method"]


# ====================================================================
# 4. QuarkParser
# ====================================================================
class TestQuarkParser(unittest.TestCase):
    def test_parse_mxfp4(self):
        from atom.quant_spec import QuarkParser
        raw = {
            "quant_method": "quark",
            "global_quant_config": {
                "weight": {"dtype": "fp4", "group_size": 32},
            },
            "exclude": ["lm_head", "model.layers.0.self_attn.q_proj"],
        }
        parsed = QuarkParser().parse(raw)
        self.assertEqual(parsed.global_spec.quant_type, QuantType.per_1x32)
        self.assertEqual(parsed.global_spec.quant_dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(parsed.global_spec.quant_method, "quark")
        self.assertIn("lm_head", parsed.exclude_layers)

    def test_parse_fp8(self):
        from atom.quant_spec import QuarkParser
        raw = {
            "quant_method": "quark",
            "global_quant_config": {
                "weight": {"dtype": "fp8", "group_size": 128},
            },
            "exclude": [],
        }
        parsed = QuarkParser().parse(raw)
        self.assertEqual(parsed.global_spec.quant_type, QuantType.per_1x128)
        self.assertEqual(parsed.global_spec.quant_dtype, d_dtypes["fp8"])


# ====================================================================
# 5. CompressedTensorsParser
# ====================================================================
class TestCompressedTensorsParser(unittest.TestCase):
    def test_parse_ct(self):
        from atom.quant_spec import CompressedTensorsParser
        raw = {
            "quant_method": "compressed-tensors",
            "weight_dtype": "fp8",
            "ignore": ["lm_head"],
        }
        parsed = CompressedTensorsParser().parse(raw)
        self.assertEqual(parsed.global_spec.quant_type, QuantType.per_Token)
        self.assertIn("lm_head", parsed.exclude_layers)


# ====================================================================
# 6. QuantizationConfig.resolve()
# ====================================================================
class TestQuantizationConfigResolve(unittest.TestCase):
    def _make_qc(self, exclude_layers=None):
        from atom.config import QuantizationConfig
        return QuantizationConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            quant_method="quark",
            exclude_layers=exclude_layers or [],
        )

    def test_resolve_normal_layer(self):
        qc = self._make_qc()
        spec = qc.resolve("model.layers.0.mlp.down_proj")
        self.assertTrue(spec.is_quantized)
        self.assertEqual(spec.quant_type, QuantType.per_1x32)

    def test_resolve_excluded_exact(self):
        qc = self._make_qc(exclude_layers=["model.layers.0.self_attn.q_proj"])
        spec = qc.resolve("model.layers.0.self_attn")  # prefix match
        self.assertFalse(spec.is_quantized)

    def test_resolve_excluded_suffix(self):
        qc = self._make_qc(exclude_layers=["lm_head"])
        spec = qc.resolve("lm_head")
        self.assertFalse(spec.is_quantized)

    def test_resolve_excluded_regex(self):
        qc = self._make_qc(exclude_layers=["re:model.layers.*self_attn.*"])
        spec = qc.resolve("model.layers.5.self_attn.qkv_proj")
        self.assertFalse(spec.is_quantized)

    def test_resolve_per_layer_override(self):
        from atom.config import QuantizationConfig
        from atom.quant_spec import LayerQuantSpec, ParsedQuantConfig

        global_spec = LayerQuantSpec(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            quant_method="quark",
        )
        custom_spec = LayerQuantSpec(
            quant_type=QuantType.per_Token,
            quant_dtype=d_dtypes["fp8"],
            quant_method="custom",
        )
        parsed = ParsedQuantConfig(
            global_spec=global_spec,
            layer_specs={"model.layers.0.mlp.down_proj": custom_spec},
        )
        qc = QuantizationConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            parsed=parsed,
        )
        # Per-layer override takes priority
        spec = qc.resolve("model.layers.0.mlp.down_proj")
        self.assertEqual(spec.quant_type, QuantType.per_Token)
        self.assertEqual(spec.quant_method, "custom")

        # Non-overridden layer gets global spec
        spec2 = qc.resolve("model.layers.1.mlp.down_proj")
        self.assertEqual(spec2.quant_type, QuantType.per_1x32)

    def test_resolve_no_quant_config(self):
        from atom.config import QuantizationConfig
        qc = QuantizationConfig()
        spec = qc.resolve("model.layers.0.self_attn")
        self.assertFalse(spec.is_quantized)


# ====================================================================
# 7. Backward compatibility
# ====================================================================
class TestBackwardCompat(unittest.TestCase):
    def test_dict_access(self):
        from atom.config import QuantizationConfig
        qc = QuantizationConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            quant_method="quark",
            exclude_layers=["lm_head"],
        )
        self.assertEqual(qc["quant_type"], QuantType.per_1x32)
        self.assertEqual(qc["quant_dtype"], torch.float4_e2m1fn_x2)
        self.assertEqual(qc["quant_method"], "quark")
        self.assertEqual(qc["exclude_layers"], ["lm_head"])

    def test_should_ignore_layer_uses_resolve(self):
        from atom.config import QuantizationConfig
        from atom.models.utils import should_ignore_layer
        qc = QuantizationConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            exclude_layers=["lm_head"],
        )
        self.assertTrue(should_ignore_layer(qc, "lm_head"))
        self.assertFalse(should_ignore_layer(qc, "model.layers.0.mlp.down_proj"))

    def test_get_quant_config_for_layer(self):
        from atom.config import QuantizationConfig
        from atom.models.utils import get_quant_config_for_layer
        qc = QuantizationConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            exclude_layers=["lm_head"],
        )
        self.assertIsNone(get_quant_config_for_layer(qc, "lm_head"))
        self.assertIs(get_quant_config_for_layer(qc, "model.layers.0.mlp.down_proj"), qc)

    def test_parsed_property(self):
        from atom.config import QuantizationConfig
        from atom.quant_spec import ParsedQuantConfig
        qc = QuantizationConfig(quant_type=QuantType.per_Token)
        self.assertIsInstance(qc.parsed, ParsedQuantConfig)

    def test_global_spec_property(self):
        from atom.config import QuantizationConfig
        from atom.quant_spec import LayerQuantSpec
        qc = QuantizationConfig(quant_type=QuantType.per_Token, quant_dtype=d_dtypes["fp8"])
        self.assertIsInstance(qc.global_spec, LayerQuantSpec)
        self.assertEqual(qc.global_spec.quant_type, QuantType.per_Token)

    def test_compute_hash_unchanged(self):
        from atom.config import QuantizationConfig
        qc1 = QuantizationConfig(quant_type=QuantType.per_1x32, quant_dtype=torch.float4_e2m1fn_x2)
        qc2 = QuantizationConfig(quant_type=QuantType.per_1x32, quant_dtype=torch.float4_e2m1fn_x2)
        self.assertEqual(qc1.compute_hash(), qc2.compute_hash())


# ====================================================================
# 8. LinearBase with layer_spec
# ====================================================================
class TestLinearBaseLayerSpec(unittest.TestCase):
    @patch("atom.model_ops.linear.get_tp_group")
    def test_layer_spec_overrides_quant_config(self, mock_tp):
        """When layer_spec is provided, LinearBase uses it instead of
        dict-based quant_config fields."""
        from atom.quant_spec import LayerQuantSpec

        mock_group = MagicMock()
        mock_group.rank_in_group = 0
        mock_group.world_size = 1
        mock_tp.return_value = mock_group

        from atom.model_ops.linear import LinearBase
        from atom.config import QuantizationConfig

        spec = LayerQuantSpec(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            quant_method="quark",
        )
        # Pass a no-quant QuantizationConfig but override with layer_spec
        qc = QuantizationConfig()
        lb = LinearBase(1024, 512, layer_spec=spec, quant_config=qc)
        self.assertEqual(lb._layer_spec, spec)
        self.assertEqual(lb.quant_type, QuantType.per_1x32)
        self.assertEqual(lb.params_dtype, torch.float4_e2m1fn_x2)

    @patch("atom.model_ops.linear.get_tp_group")
    def test_layer_spec_with_checkpoint_dtype(self, mock_tp):
        """LayerQuantSpec.checkpoint_dtype flows into source_quant_dtype."""
        from atom.quant_spec import LayerQuantSpec

        mock_group = MagicMock()
        mock_group.rank_in_group = 0
        mock_group.world_size = 1
        mock_tp.return_value = mock_group

        from atom.model_ops.linear import LinearBase
        from atom.config import QuantizationConfig

        spec = LayerQuantSpec(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            checkpoint_dtype=torch.bfloat16,
        )
        lb = LinearBase(1024, 512, layer_spec=spec)
        self.assertEqual(lb.source_quant_dtype, torch.bfloat16)
        self.assertTrue(lb._layer_spec.needs_online_quant)

    @patch("atom.model_ops.linear.get_tp_group")
    def test_no_layer_spec_builds_from_dict(self, mock_tp):
        """When layer_spec is not provided, LinearBase builds one from the dict."""
        mock_group = MagicMock()
        mock_group.rank_in_group = 0
        mock_group.world_size = 1
        mock_tp.return_value = mock_group

        from atom.model_ops.linear import LinearBase
        from atom.config import QuantizationConfig

        qc = QuantizationConfig(
            quant_type=QuantType.per_Token,
            quant_dtype=d_dtypes["fp8"],
        )
        lb = LinearBase(1024, 512, quant_config=qc)
        self.assertEqual(lb._layer_spec.quant_type, QuantType.per_Token)
        self.assertEqual(lb._layer_spec.quant_dtype, d_dtypes["fp8"])


# ====================================================================
# 9. QKVParallelLinear resolve integration
# ====================================================================
class TestQKVResolve(unittest.TestCase):
    @patch("atom.model_ops.linear.get_tp_group")
    def test_qkv_excluded_layer_gets_no_quant(self, mock_tp):
        mock_group = MagicMock()
        mock_group.rank_in_group = 0
        mock_group.world_size = 1
        mock_tp.return_value = mock_group

        from atom.model_ops.linear import QKVParallelLinear
        from atom.config import QuantizationConfig

        qc = QuantizationConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            quant_method="quark",
            exclude_layers=["re:.*self_attn.*"],
        )
        qkv = QKVParallelLinear(
            hidden_size=1024,
            head_size=128,
            total_num_heads=8,
            total_num_kv_heads=2,
            quant_config=qc,
            prefix="model.layers.0.self_attn",
        )
        # Should have QuantType.No since self_attn is excluded
        self.assertEqual(qkv.quant_type, QuantType.No)

    @patch("atom.model_ops.linear.get_tp_group")
    def test_qkv_non_excluded_layer_gets_quant(self, mock_tp):
        mock_group = MagicMock()
        mock_group.rank_in_group = 0
        mock_group.world_size = 1
        mock_tp.return_value = mock_group

        from atom.model_ops.linear import QKVParallelLinear
        from atom.config import QuantizationConfig

        qc = QuantizationConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            quant_method="quark",
            exclude_layers=["lm_head"],
        )
        qkv = QKVParallelLinear(
            hidden_size=1024,
            head_size=128,
            total_num_heads=8,
            total_num_kv_heads=2,
            quant_config=qc,
            prefix="model.layers.0.self_attn",
        )
        self.assertEqual(qkv.quant_type, QuantType.per_1x32)


# ====================================================================
# 10. RowParallelLinear resolve integration
# ====================================================================
class TestRowResolve(unittest.TestCase):
    @patch("atom.model_ops.linear.get_tp_group")
    def test_row_excluded_gets_no_quant(self, mock_tp):
        mock_group = MagicMock()
        mock_group.rank_in_group = 0
        mock_group.world_size = 1
        mock_tp.return_value = mock_group

        from atom.model_ops.linear import RowParallelLinear
        from atom.config import QuantizationConfig

        qc = QuantizationConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            exclude_layers=["lm_head"],
        )
        row = RowParallelLinear(
            input_size=1024,
            output_size=512,
            quant_config=qc,
            prefix="lm_head",
        )
        self.assertEqual(row.quant_type, QuantType.No)


if __name__ == "__main__":
    unittest.main()
