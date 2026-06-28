# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Regression test for the MXFP4 MoE uninitialized bias bug.

Root cause:
  FusedMoE defaulted has_bias=True, but Qwen3MoE experts have no bias
  in the checkpoint. Mxfp4MoEMethod.create_weights allocated bias
  parameters with torch.empty() that never got loaded, causing the
  kernel to add garbage bias to every expert output.

Fix:
  - FusedMoE default changed to has_bias=False
  - Qwen3MoeSparseMoeBlock and Qwen3NextSparseMoeBlock explicitly
    pass has_bias=False
"""

import sys
import unittest

import pytest

# These tests load the real atom.config / atom.model_ops.moe, which import the
# AITER GPU kernel library (e.g. `from aiter import QuantType`). AITER has no
# CPU/PyPI build, so skip this module visibly on the non-GPU unit gate; it runs
# in the GPU CI where AITER is present.
pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

# This test needs to inspect real atom source (not conftest.py stubs), so it
# wipes any cached `atom.*` modules at module-import time. Previously this
# also wiped the conftest stubs and never restored them, polluting later
# tests (test_arg_utils_spec / test_scheduler / test_sequence) that depend
# on the stubs. setUpModule / tearDownModule now snapshots-and-restores
# sys.modules so this file's effect is local to its own collection.
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


class TestFusedMoEDefaultHasBias(unittest.TestCase):
    """FusedMoE must default to has_bias=False."""

    def test_default_is_false(self):
        import inspect
        from atom.model_ops.moe import FusedMoE

        sig = inspect.signature(FusedMoE.__init__)
        default = sig.parameters["has_bias"].default
        self.assertFalse(
            default,
            "FusedMoE default has_bias must be False to prevent "
            "uninitialized bias when checkpoint has no expert bias",
        )


class TestQwen3MoeExplicitHasBias(unittest.TestCase):
    """Qwen3 MoE models must explicitly pass has_bias=False."""

    def _check_source_has_bias_false(self, module_path: str, class_name: str):
        import importlib
        import inspect

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        source = inspect.getsource(cls.__init__)
        self.assertIn(
            "has_bias=False",
            source,
            f"{class_name} must pass has_bias=False to FusedMoE",
        )

    def test_qwen3_moe_sparse_block(self):
        self._check_source_has_bias_false(
            "atom.models.qwen3_moe", "Qwen3MoeSparseMoeBlock"
        )

    def test_qwen3_next_sparse_block(self):
        self._check_source_has_bias_false(
            "atom.models.qwen3_next", "Qwen3NextSparseMoeBlock"
        )


class TestGptOssKeepsBias(unittest.TestCase):
    """gpt_oss explicitly uses has_bias=True and must not be affected."""

    def test_gpt_oss_has_bias_true(self):
        import inspect
        from atom.models.gpt_oss import MLPBlock as SparseMoeBlock

        source = inspect.getsource(SparseMoeBlock.__init__)
        self.assertIn(
            "has_bias=True",
            source,
            "gpt_oss SparseMoeBlock must keep has_bias=True",
        )


class TestMxfp4NoBiasCreated(unittest.TestCase):
    """When has_bias=False, Mxfp4MoEMethod must not create bias parameters."""

    def test_no_bias_when_has_bias_false(self):
        import torch
        from unittest.mock import MagicMock

        from atom.model_ops.moe import Mxfp4MoEMethod
        from atom.config import LayerQuantConfig
        from aiter import QuantType

        qc = LayerQuantConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            quant_method="quark",
        )
        moe_config = MagicMock()
        method = Mxfp4MoEMethod(qc, moe_config)

        # Create a mock layer with has_bias=False
        layer = MagicMock()
        layer.has_bias = False
        layer.hidden_size = 6144
        layer.intermediate_size_per_partition = 2560
        layer.activation = "silu"

        # Track what register_parameter is called with
        registered = {}

        def mock_register(name, param):
            registered[name] = param

        layer.register_parameter = mock_register

        method.create_weights(
            layer=layer,
            num_experts=8,
            hidden_size=6144,
            intermediate_size_per_partition=2560,
            params_dtype=torch.float4_e2m1fn_x2,
            weight_loader=lambda *a: None,
        )

        # Bias should be None when has_bias=False
        self.assertIsNone(
            registered.get("w13_bias"),
            "w13_bias must be None when has_bias=False",
        )
        self.assertIsNone(
            registered.get("w2_bias"),
            "w2_bias must be None when has_bias=False",
        )

    def test_bias_created_when_has_bias_true(self):
        import torch
        from unittest.mock import MagicMock

        from atom.model_ops.moe import Mxfp4MoEMethod
        from atom.config import LayerQuantConfig
        from aiter import QuantType

        qc = LayerQuantConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2,
            quant_method="quark",
        )
        moe_config = MagicMock()
        method = Mxfp4MoEMethod(qc, moe_config)

        # Create a mock layer with has_bias=True
        layer = MagicMock()
        layer.has_bias = True
        layer.hidden_size = 6144
        layer.intermediate_size_per_partition = 2560
        layer.activation = "silu"

        registered = {}

        def mock_register(name, param):
            registered[name] = param

        layer.register_parameter = mock_register

        method.create_weights(
            layer=layer,
            num_experts=8,
            hidden_size=6144,
            intermediate_size_per_partition=2560,
            params_dtype=torch.float4_e2m1fn_x2,
            weight_loader=lambda *a: None,
        )

        # Bias should be a Parameter when has_bias=True
        self.assertIsNotNone(registered.get("w13_bias"))
        self.assertIsInstance(registered["w13_bias"], torch.nn.Parameter)
        self.assertIsNotNone(registered.get("w2_bias"))
        self.assertIsInstance(registered["w2_bias"], torch.nn.Parameter)


class TestSwiGLUInterleavingWithoutBias(unittest.TestCase):
    """SwiGLU weight interleaving must happen regardless of has_bias.

    Root cause:
      process_weights_after_loading guarded the SwiGLU interleaving branch
      on ``layer.w13_bias is not None``.  When has_bias=False (no bias),
      it fell through to the generic 'else' branch that uses different
      shuffling functions (shuffle_weights + e8m0_shuffle) which do NOT
      interleave gate/up weights.  The a16w4 kernel still expects
      interleaved weights -> garbage output.

    Fix:
      Change condition from
        ``layer.activation == ActivationType.Swiglu and layer.w13_bias is not None``
      to
        ``layer.activation == ActivationType.Swiglu``
      and guard only the bias interleaving on ``layer.w13_bias is not None``.
    """

    def test_swiglu_branch_does_not_couple_bias_and_shuffle(self):
        """Ensure the old coupled condition is gone."""
        import inspect
        from atom.model_ops.moe import Mxfp4MoEMethod

        source = inspect.getsource(Mxfp4MoEMethod.process_weights_after_loading)

        self.assertNotIn(
            "Swiglu and layer.w13_bias is not None",
            source,
            "Old coupled condition (Swiglu AND bias) must be removed",
        )


class TestQwen3MoeQKNormShape(unittest.TestCase):
    """Qwen3MoeAttention must apply q/k norm per-head, not on flattened vectors."""

    def test_qk_norm_is_per_head(self):
        import inspect
        from atom.models.qwen3_moe import Qwen3MoeAttention

        source = inspect.getsource(Qwen3MoeAttention.forward)
        self.assertIn(
            "self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view",
            source,
            "q_norm must reshape q to [tokens, num_heads, head_dim] before RMSNorm",
        )
        self.assertIn(
            "self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view",
            source,
            "k_norm must reshape k to [tokens, num_kv_heads, head_dim] before RMSNorm",
        )


if __name__ == "__main__":
    unittest.main()
