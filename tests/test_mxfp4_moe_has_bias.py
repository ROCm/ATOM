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

import unittest

import pytest

# These tests load the real atom.config / atom.model_ops.moe, which import the
# AITER GPU kernel library (e.g. `from aiter import QuantType`). AITER has no
# CPU/PyPI build, so skip this module visibly on the non-GPU unit gate; it runs
# in the GPU CI where AITER is present.
pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")


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
