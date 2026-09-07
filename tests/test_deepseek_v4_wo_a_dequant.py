# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Regression test for DeepseekV4Attention.process_weights_after_loading's wo_a
FP8 -> BF16 dequant.

wo_a ships FP8 + UE8M0 block-scale on disk for V4-Pro/V4-Flash-Base; the
grouped-LoRA einsum in forward_impl needs it as BF16 (aiter has no FP8 grouped
einsum), so process_weights_after_loading dequants it in place after loading.

AITER's FP8 dtype resolves to torch.float8_e4m3fn on gfx950/NV but
torch.float8_e4m3fnuz on gfx942/MI300X (same bit layout, different exponent
bias). Commit 6a2638b5 widened the dtype gate to cover both dialects after a
fn-only check silently skipped the dequant on gfx942, leaving wo_a as FP8 at
forward time (`RuntimeError: expected scalar type BFloat16 but found
Float8_e4m3fnuz`). This test locks in that fix for both dialects.
"""

import sys
import unittest

import pytest

# process_weights_after_loading lives on DeepseekV4Attention, whose import
# chain pulls atom.model_ops -> AITER (GPU-only). Skip on the non-GPU unit
# gate; runs in GPU CI (and locally on the box) where AITER is present.
pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

# Loading the real atom source wipes the conftest.py stubs; snapshot and
# restore sys.modules so this file's effect stays local to its own collection
# (mirrors test_dummy_weight_init.py / test_mxfp4_moe_has_bias.py).
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


def _make_wo_a(dtype, out_features=256, in_features=128, block=128):
    import torch
    from torch import nn

    class FakeWoA(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.full((out_features, in_features), 2.0, dtype=torch.float32).to(
                    dtype
                ),
                requires_grad=False,
            )
            self.weight_scale = nn.Parameter(
                torch.ones(
                    out_features // block, in_features // block, dtype=torch.float32
                ),
                requires_grad=False,
            )

    return FakeWoA()


class _FakeAttention:
    """The ``self`` that ``process_weights_after_loading`` runs against.

    Refuses an attribute it does not model, by name. A plain
    ``SimpleNamespace`` let a missing one surface as ``'types.SimpleNamespace'
    object has no attribute 'n_local_groups'``, raised from inside the method
    with nothing to say about where to fix it -- which is how all three tests
    below sat red after production grew reads of ``n_local_groups`` and
    ``o_lora_rank``, on every machine that can run them. That is only a
    machine with aiter, because this module ``importorskip``s it and CI has
    none, so nothing reported it.

    Runtime, not a scan of the method's source. A source scan sees only the
    literal ``self.X`` in that one body -- not a read through a helper, not
    one through a base class -- and it re-breaks on a refactor that changed
    nothing. This fails at the read, wherever the read is.

    Values coherent with ``_make_wo_a``'s 256 x 128 weight: the gfx950
    batched-GEMM path wants ``out_dim == n_local_groups * o_lora_rank``, so
    2 x 128. ``_is_gfx950`` and ``_is_preshuffle`` are False because this is
    the gfx942 dtype gate -- the BF16 fallback is the path under test.
    """

    def __init__(self, wo_a):
        self.wo_a = wo_a
        self.n_local_groups = 2
        self.o_lora_rank = 128
        self._is_gfx950 = False
        self._is_preshuffle = False

    def __getattr__(self, name):
        raise AttributeError(
            f"process_weights_after_loading read self.{name}, which this "
            f"double does not model -- add it to _FakeAttention with a value "
            f"coherent with _make_wo_a's weight"
        )


class TestWoADequantFnFnuzGate(unittest.TestCase):
    def _run(self, dtype):
        from atom.models.deepseek_v4 import DeepseekV4Attention

        fake_self = _FakeAttention(_make_wo_a(dtype))
        DeepseekV4Attention.process_weights_after_loading(fake_self)
        return fake_self.wo_a

    def test_dequants_ocp_e4m3fn(self):
        import torch

        wo_a = self._run(torch.float8_e4m3fn)
        self.assertEqual(wo_a.weight.dtype, torch.bfloat16)
        self.assertFalse(hasattr(wo_a, "weight_scale"))

    def test_dequants_amd_e4m3fnuz(self):
        import torch

        wo_a = self._run(torch.float8_e4m3fnuz)
        self.assertEqual(wo_a.weight.dtype, torch.bfloat16)
        self.assertFalse(hasattr(wo_a, "weight_scale"))

    def test_noop_when_already_bf16(self):
        # Idempotency: a second call (e.g. from load_model's generic post-load
        # sweep running after DeepseekV4ForCausalLM.load_weights already
        # dequanted it) must not touch an already-BF16 wo_a.
        import torch

        wo_a = self._run(torch.float8_e4m3fnuz)
        weight_before = wo_a.weight
        from atom.models.deepseek_v4 import DeepseekV4Attention

        fake_self = _FakeAttention(wo_a)
        DeepseekV4Attention.process_weights_after_loading(fake_self)
        self.assertIs(wo_a.weight, weight_before)


if __name__ == "__main__":
    unittest.main()
