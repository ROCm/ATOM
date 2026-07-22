# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Unit tests for --load_dummy weight init modes (zero / xavier).

initialize_dummy_weights fills the params that --load_dummy skips loading with
finite values instead of leaving them as torch.empty garbage:
  - zero:   every param byte-zeroed.
  - xavier: xavier_uniform_ for bf16 2D weights, constant target magnitude for
            fp4 (uint8 fp4x2 weight + uint8 e8m0 scale) and fp8 packed weights.
"""

import sys
import unittest

import pytest

# initialize_dummy_weights lives in atom.model_loader.loader, whose import chain
# pulls atom.model_ops -> AITER (GPU-only). Skip on the non-GPU unit gate; runs
# in GPU CI (and locally on the box) where AITER is present.
pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

# Loading the real atom source wipes the conftest.py stubs; snapshot and restore
# sys.modules so this file's effect stays local to its own collection (mirrors
# test_mxfp4_moe_has_bias.py).
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


def _build_module():
    import torch
    from torch import nn

    def P(t):
        # ATOM params are non-grad; uint8/fp8 params cannot require grad.
        return nn.Parameter(t, requires_grad=False)

    m = nn.Module()
    # bf16 dense weight + bias (unquantized path)
    m.lin = nn.Linear(8, 4, bias=True, dtype=torch.bfloat16)
    # 1D norm weight (non-bias)
    m.register_parameter("norm", P(torch.empty(4, dtype=torch.bfloat16)))
    # MXFP4 MoE params: packed fp4x2 weight + e8m0 block scale (both uint8)
    m.register_parameter("w13_weight", P(torch.empty(2, 4, dtype=torch.uint8)))
    m.register_parameter("w13_weight_scale", P(torch.empty(2, 4, dtype=torch.uint8)))
    # static activation scale (must stay ~1.0, not shrunk to weight magnitude)
    m.register_parameter("w13_input_scale", P(torch.empty(2, dtype=torch.float32)))
    return m


class TestDummyWeightInit(unittest.TestCase):
    def test_xavier_mode(self):
        import torch
        from atom.model_loader.loader import (
            _E8M0_UNIT_CODE,
            _FP4_UNIT_BYTE,
            initialize_dummy_weights,
        )

        m = _build_module()
        initialize_dummy_weights(m, "xavier")

        # bf16 2D weight -> xavier_uniform_ (finite, not all zero)
        self.assertTrue(torch.isfinite(m.lin.weight).all())
        self.assertGreater(m.lin.weight.abs().sum().item(), 0.0)
        # bias -> 0, norm (1D weight) -> 1.0
        self.assertTrue((m.lin.bias == 0).all())
        self.assertTrue((m.norm == 1).all())
        # fp4x2 packed weight -> unit byte; e8m0 scale -> unit code
        self.assertTrue((m.w13_weight == _FP4_UNIT_BYTE).all())
        self.assertTrue((m.w13_weight_scale == _E8M0_UNIT_CODE).all())
        # input_scale stays 1.0 (not shrunk to weight magnitude)
        self.assertTrue((m.w13_input_scale == 1.0).all())

    def test_zero_mode(self):
        from atom.model_loader.loader import initialize_dummy_weights

        m = _build_module()
        initialize_dummy_weights(m, "zero")

        for name, p in m.named_parameters():
            self.assertTrue((p == 0).all(), f"{name} not all-zero in zero mode")

    def test_e8m0_unit_code_matches_target_std(self):
        # _E8M0_UNIT_CODE must decode (2^(code-127)) to _DUMMY_WEIGHT_STD so the
        # fp4 effective magnitude (fp4=1.0 * 2^(code-127)) matches the intended
        # weight scale.
        from atom.model_loader.loader import _DUMMY_WEIGHT_STD, _E8M0_UNIT_CODE

        self.assertEqual(2.0 ** (_E8M0_UNIT_CODE - 127), _DUMMY_WEIGHT_STD)


if __name__ == "__main__":
    unittest.main()
