# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Regression test for FP8 MoE block_n/block_k alignment auto-degradation (#442).

Root cause:
  Models with intermediate_size=1536 at TP=8 produce
  intermediate_size_per_partition=192, which is not divisible by
  block_n=128 or block_k=128 (the defaults for per_1x128 quantisation).
  Both CompressedTensorsFp8MoEMethod and Fp8MoEMethod raise ValueError.

Fix (Option B from #442 analysis):
  When intermediate_size_per_partition is not divisible by 128 but
  *is* divisible by 64, auto-degrade block_n and/or block_k to 64
  and emit a logger warning.  The ValueErrors are preserved when
  alignment is still impossible (e.g. 190 % 64 != 0).
"""

import logging
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

# ---------- saved-modules snapshots for isolation ----------
_saved_atom_modules: dict[str, object] = {}


def setUpModule():
    global _saved_atom_modules
    _saved_atom_modules = {
        name: mod for name, mod in sys.modules.items() if name.startswith("atom")
    }
    for name in list(_saved_atom_modules):
        del sys.modules[name]


def tearDownModule():
    for name in [n for n in sys.modules if n.startswith("atom")]:
        del sys.modules[name]
    sys.modules.update(_saved_atom_modules)


# ---------- helpers ----------


def _make_compressed_tensors_method(quant_type=None):
    """Return a CompressedTensorsFp8MoEMethod with default per_1x128."""
    import torch
    from aiter import QuantType
    from atom.config import LayerQuantConfig
    from atom.model_ops.moe import CompressedTensorsFp8MoEMethod

    qt = quant_type if quant_type is not None else QuantType.per_1x128
    qc = LayerQuantConfig(
        quant_type=qt,
        quant_dtype=torch.float8_e4m3fn,
        quant_method="compressed-tensors",
        is_dynamic=True,
    )
    moe_config = MagicMock()
    return CompressedTensorsFp8MoEMethod(qc, moe_config)


def _make_fp8_method(quant_type=None):
    """Return a Fp8MoEMethod with default per_1x128."""
    import torch
    from aiter import QuantType
    from atom.config import LayerQuantConfig
    from atom.model_ops.moe import Fp8MoEMethod

    qt = quant_type if quant_type is not None else QuantType.per_1x128
    qc = LayerQuantConfig(
        quant_type=qt,
        quant_dtype=torch.float8_e4m3fn,
        quant_method="online_quant",
        is_dynamic=True,
    )
    moe_config = MagicMock()
    return Fp8MoEMethod(qc, moe_config)


def _make_mock_layer(hidden_size=7168):
    """Create a mock layer with register_parameter spy."""
    layer = MagicMock()
    layer.hidden_size = hidden_size
    layer.intermediate_size_per_partition = 192
    layer.activation = "silu"
    layer.has_bias = False
    registered = {}

    def _reg(name, param):
        registered[name] = param

    layer.register_parameter = _reg
    return layer, registered


def _tp_mock(world_size=1):
    """Return a mock tensor-parallel group."""
    mg = MagicMock()
    mg.world_size = world_size
    return mg


# ---------- CompressedTensorsFp8MoEMethod tests ----------


class TestCompressedTensorsFp8MoEAutoDegrade(unittest.TestCase):
    """block_n=128 → 64 when intermediate % 128 != 0 and % 64 == 0."""

    @patch("atom.model_ops.moe.get_tp_group")
    def test_degrades_to_64_on_alignment_mismatch(self, mock_tp):
        import torch

        mock_tp.return_value = _tp_mock(world_size=1)

        method = _make_compressed_tensors_method()
        self.assertEqual(method.block_n, 128)
        self.assertEqual(method.block_k, 128)

        layer, _registered = _make_mock_layer()

        with self.assertLogs(logger="atom", level="WARNING") as log_ctx:
            method.create_weights(
                layer=layer,
                num_experts=8,
                hidden_size=7168,
                intermediate_size_per_partition=192,
                params_dtype=torch.float8_e4m3fn,
                weight_loader=lambda *a: None,
            )

        self.assertEqual(method.block_n, 64)
        self.assertGreaterEqual(len(log_ctx.output), 1)
        warning_text = log_ctx.output[0].lower()
        self.assertIn("block_n", warning_text)
        self.assertIn("64", warning_text)

    @patch("atom.model_ops.moe.get_tp_group")
    def test_degrade_block_k_at_tp_gt_1(self, mock_tp):
        """block_k also degrades when tp_size > 1."""
        import torch

        mock_tp.return_value = _tp_mock(world_size=8)

        method = _make_compressed_tensors_method()
        layer, _registered = _make_mock_layer()

        with self.assertLogs(logger="atom", level="WARNING") as log_ctx:
            method.create_weights(
                layer=layer,
                num_experts=8,
                hidden_size=7168,
                intermediate_size_per_partition=192,
                params_dtype=torch.float8_e4m3fn,
                weight_loader=lambda *a: None,
            )

        self.assertEqual(method.block_n, 64)
        self.assertEqual(method.block_k, 64)
        # Should see warnings for both block_n and block_k
        self.assertGreaterEqual(len(log_ctx.output), 2)

    @patch("atom.model_ops.moe.get_tp_group")
    def test_keeps_128_when_aligned(self, mock_tp):
        import torch

        mock_tp.return_value = _tp_mock(world_size=1)

        method = _make_compressed_tensors_method()
        layer, _registered = _make_mock_layer()

        with self.assertNoLogs(logger="atom", level="WARNING"):
            method.create_weights(
                layer=layer,
                num_experts=8,
                hidden_size=7168,
                intermediate_size_per_partition=256,
                params_dtype=torch.float8_e4m3fn,
                weight_loader=lambda *a: None,
            )

        self.assertEqual(method.block_n, 128)
        self.assertEqual(method.block_k, 128)

    @patch("atom.model_ops.moe.get_tp_group")
    def test_raises_when_no_safe_degradation_possible(self, mock_tp):
        import torch

        mock_tp.return_value = _tp_mock(world_size=1)

        method = _make_compressed_tensors_method()
        layer, _registered = _make_mock_layer()

        with self.assertRaises(ValueError):
            method.create_weights(
                layer=layer,
                num_experts=8,
                hidden_size=7168,
                intermediate_size_per_partition=190,
                params_dtype=torch.float8_e4m3fn,
                weight_loader=lambda *a: None,
            )

    @patch("atom.model_ops.moe.get_tp_group")
    def test_per_1x32_not_affected(self, mock_tp):
        import torch
        from aiter import QuantType

        mock_tp.return_value = _tp_mock(world_size=8)

        method = _make_compressed_tensors_method(quant_type=QuantType.per_1x32)
        self.assertEqual(method.block_n, 1)

        layer, _registered = _make_mock_layer()

        method.create_weights(
            layer=layer,
            num_experts=8,
            hidden_size=7168,
            intermediate_size_per_partition=192,
            params_dtype=torch.float4_e2m1fn_x2,
            weight_loader=lambda *a: None,
        )
        self.assertEqual(method.block_n, 1)

    @patch("atom.model_ops.moe.get_tp_group")
    def test_scale_shapes_after_degrade(self, mock_tp):
        """Scale tensor shapes must use the degraded block_n=64 (tp=1)."""
        import torch

        mock_tp.return_value = _tp_mock(world_size=1)

        method = _make_compressed_tensors_method()
        layer, registered = _make_mock_layer()

        method.create_weights(
            layer=layer,
            num_experts=8,
            hidden_size=7168,
            intermediate_size_per_partition=192,
            params_dtype=torch.float8_e4m3fn,
            weight_loader=lambda *a: None,
        )

        w13_scale = registered["w13_weight_scale"]
        w2_scale = registered["w2_weight_scale"]

        # block_n=64, block_k=128 (tp=1 skips block_k check):
        # w13_scale: (E, 2 * ceil(192/64), ceil(7168/128)) = (8, 6, 56)
        # w2_scale:  (E, ceil(7168/64), ceil(192/128))      = (8, 112, 2)
        self.assertEqual(w13_scale.shape, (8, 6, 56))
        self.assertEqual(w2_scale.shape, (8, 112, 2))


# ---------- Fp8MoEMethod tests ----------


class TestFp8MoEAutoDegrade(unittest.TestCase):
    """block_n=128 → 64 when intermediate % 128 != 0 and % 64 == 0."""

    @patch("atom.model_ops.moe.get_tp_group")
    def test_degrades_to_64_on_alignment_mismatch(self, mock_tp):
        import torch
        from aiter import QuantType

        mock_tp.return_value = _tp_mock(world_size=1)

        method = _make_fp8_method()
        self.assertEqual(method.quant_type, QuantType.per_1x128)

        layer, _registered = _make_mock_layer()

        with self.assertLogs(logger="atom", level="WARNING") as log_ctx:
            method.create_weights(
                layer=layer,
                num_experts=8,
                hidden_size=7168,
                intermediate_size_per_partition=192,
                params_dtype=torch.float8_e4m3fn,
                weight_loader=lambda *a: None,
            )

        self.assertGreaterEqual(len(log_ctx.output), 1)
        warning_text = log_ctx.output[0].lower()
        self.assertIn("block_n", warning_text)
        self.assertIn("64", warning_text)

    @patch("atom.model_ops.moe.get_tp_group")
    def test_keeps_128_when_aligned(self, mock_tp):
        import torch

        mock_tp.return_value = _tp_mock(world_size=1)

        method = _make_fp8_method()
        layer, _registered = _make_mock_layer()

        with self.assertNoLogs(logger="atom", level="WARNING"):
            method.create_weights(
                layer=layer,
                num_experts=8,
                hidden_size=7168,
                intermediate_size_per_partition=256,
                params_dtype=torch.float8_e4m3fn,
                weight_loader=lambda *a: None,
            )

    @patch("atom.model_ops.moe.get_tp_group")
    def test_raises_when_no_safe_degradation_possible(self, mock_tp):
        import torch

        mock_tp.return_value = _tp_mock(world_size=1)

        method = _make_fp8_method()
        layer, _registered = _make_mock_layer()

        with self.assertRaises(ValueError):
            method.create_weights(
                layer=layer,
                num_experts=8,
                hidden_size=7168,
                intermediate_size_per_partition=190,
                params_dtype=torch.float8_e4m3fn,
                weight_loader=lambda *a: None,
            )

    @patch("atom.model_ops.moe.get_tp_group")
    def test_scale_shapes_after_degrade(self, mock_tp):
        """Scale tensor shapes must use the degraded block_n=64 (tp=1)."""
        import torch

        mock_tp.return_value = _tp_mock(world_size=1)

        method = _make_fp8_method()
        layer, registered = _make_mock_layer()

        method.create_weights(
            layer=layer,
            num_experts=8,
            hidden_size=7168,
            intermediate_size_per_partition=192,
            params_dtype=torch.float8_e4m3fn,
            weight_loader=lambda *a: None,
        )

        w13_scale = registered["w13_weight_scale"]
        w2_scale = registered["w2_weight_scale"]

        # block_n=64, block_k=128 (tp=1 skips block_k check):
        self.assertEqual(w13_scale.shape, (8, 6, 56))
        self.assertEqual(w2_scale.shape, (8, 112, 2))


# ---------- Real-world scenario tests ----------


class TestTP8Scenarios(unittest.TestCase):
    """Simulate the TP=8 scenario from #442 report."""

    INTERMEDIATE_PER_TP8 = 192  # 1536 / 8

    @patch("atom.model_ops.moe.get_tp_group")
    def test_compressed_tensors_method_tp8_1536(self, mock_tp):
        """CompressedTensorsFp8MoEMethod must not raise at TP=8, 1536."""
        import torch

        mock_tp.return_value = _tp_mock(world_size=8)

        method = _make_compressed_tensors_method()
        layer, _registered = _make_mock_layer()

        try:
            method.create_weights(
                layer=layer,
                num_experts=128,
                hidden_size=7168,
                intermediate_size_per_partition=self.INTERMEDIATE_PER_TP8,
                params_dtype=torch.float8_e4m3fn,
                weight_loader=lambda *a: None,
            )
        except ValueError:
            self.fail(
                "CompressedTensorsFp8MoEMethod raised ValueError at "
                "TP=8, intermediate=1536 — auto-degradation failed"
            )
        self.assertEqual(method.block_n, 64)
        self.assertEqual(method.block_k, 64)

    @patch("atom.model_ops.moe.get_tp_group")
    def test_fp8_method_tp8_1536(self, mock_tp):
        """Fp8MoEMethod must not raise at TP=8, 1536."""
        import torch

        mock_tp.return_value = _tp_mock(world_size=8)

        method = _make_fp8_method()
        layer, _registered = _make_mock_layer()

        try:
            method.create_weights(
                layer=layer,
                num_experts=128,
                hidden_size=7168,
                intermediate_size_per_partition=self.INTERMEDIATE_PER_TP8,
                params_dtype=torch.float8_e4m3fn,
                weight_loader=lambda *a: None,
            )
        except ValueError:
            self.fail(
                "Fp8MoEMethod raised ValueError at "
                "TP=8, intermediate=1536 — auto-degradation failed"
            )

    @patch("atom.model_ops.moe.get_tp_group")
    def test_scale_shapes_tp8(self, mock_tp):
        """Scale shapes after both block_n and block_k degrade at TP=8."""
        import torch

        mock_tp.return_value = _tp_mock(world_size=8)

        method = _make_compressed_tensors_method()
        layer, registered = _make_mock_layer()

        method.create_weights(
            layer=layer,
            num_experts=8,
            hidden_size=7168,
            intermediate_size_per_partition=192,
            params_dtype=torch.float8_e4m3fn,
            weight_loader=lambda *a: None,
        )

        w13_scale = registered["w13_weight_scale"]
        w2_scale = registered["w2_weight_scale"]

        # Both block_n and block_k degrade to 64:
        # w13_scale: (8, 2 * ceil(192/64), ceil(7168/64)) = (8, 6, 112)
        # w2_scale:  (8, ceil(7168/64), ceil(192/64))      = (8, 112, 3)
        self.assertEqual(w13_scale.shape, (8, 6, 112))
        self.assertEqual(w2_scale.shape, (8, 112, 3))


if __name__ == "__main__":
    unittest.main()
