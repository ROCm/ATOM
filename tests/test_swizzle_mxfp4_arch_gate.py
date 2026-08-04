# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Regression test for _swizzle_mxfp4's arch gate on aiter's shuffle_scale_moe.

shuffle_scale_moe only implements the native preshuffled FP4 tensor-core
scale layout for CDNA4 (gfx950) and gfx1250; other archs (e.g. gfx942/
MI300X, which has no such hardware layout to shuffle for) fall through
with an UnboundLocalError. The Triton MXFP4 GEMM kernels already have a
complete, arch-general fallback for swizzle_mx_scale=None (unshuffled
row-major scale reads), so _swizzle_mxfp4 must skip the shuffle call
entirely on archs outside that set instead of calling into it.
"""

import os
import sys
import unittest
from unittest import mock

import pytest

# fused_moe_triton pulls atom.model_ops.moe -> AITER (GPU-only). Skip on the
# non-GPU unit gate; runs in GPU CI (and locally on the box) where AITER is
# present.
#
# `pytest.importorskip("aiter")` alone isn't enough: test_pp.py / test_pd_pp.py
# install a bare `sys.modules.setdefault("aiter", types.ModuleType("aiter"))`
# stub with no teardown, which leaks into every later-collected module on the
# non-GPU gate (this file sorts after both alphabetically). That stub has no
# `ActivationType`, so importorskip "succeeds" against the fake package and
# `from aiter import ActivationType` below fails hard instead of skipping.
_aiter = pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
if not hasattr(_aiter, "ActivationType"):
    pytest.skip(
        "sys.modules['aiter'] is a fake stub leaked by another test module, "
        "not the real AITER package",
        allow_module_level=True,
    )

# Loading the real atom source wipes the conftest.py stubs; snapshot and
# restore sys.modules so this file's effect stays local to its own collection
# (mirrors test_dummy_weight_init.py / test_deepseek_v4_wo_a_dequant.py).
_saved_atom_modules: dict[str, object] = {}
_saved_env: dict[str, str | None] = {}

_TRITON_MOE_ENV = "ATOM_USE_TRITON_MOE"


def setUpModule():
    global _saved_atom_modules
    _saved_atom_modules = {
        name: mod for name, mod in sys.modules.items() if name.startswith("atom")
    }
    for name in list(_saved_atom_modules):
        del sys.modules[name]
    # fused_moe_triton's aiter Triton-kernel imports are gated at module-import
    # time on this flag; set it before the first import below.
    _saved_env[_TRITON_MOE_ENV] = os.environ.get(_TRITON_MOE_ENV)
    os.environ[_TRITON_MOE_ENV] = "1"


def tearDownModule():
    for name in [n for n in sys.modules if n.startswith("atom")]:
        del sys.modules[name]
    sys.modules.update(_saved_atom_modules)
    if _saved_env[_TRITON_MOE_ENV] is None:
        os.environ.pop(_TRITON_MOE_ENV, None)
    else:
        os.environ[_TRITON_MOE_ENV] = _saved_env[_TRITON_MOE_ENV]


def _fake_weights():
    import torch

    # N=32, K=256 satisfies N % 32 == 0 and K % (32 * 8) == 0 for both w1/w2
    # so the shape gate always passes -- isolates the test to the arch gate.
    w1 = torch.zeros(32, 256, dtype=torch.uint8)
    w1_scale = torch.zeros(32, 8, dtype=torch.uint8)
    w2 = torch.zeros(32, 256, dtype=torch.uint8)
    w2_scale = torch.zeros(32, 8, dtype=torch.uint8)
    return w1, w1_scale, w2, w2_scale


class TestSwizzleMxfp4ArchGate(unittest.TestCase):
    def test_skips_shuffle_on_gfx942(self):
        from atom.model_ops import fused_moe_triton as fmt

        w1, w1_scale, w2, w2_scale = _fake_weights()
        with (
            mock.patch.object(fmt, "get_arch", return_value="gfx942"),
            mock.patch.object(fmt, "shuffle_scale_moe") as mock_shuffle,
        ):
            (
                _,
                _,
                w1_swizzle_layout,
                _,
                _,
                w2_swizzle_layout,
            ) = fmt._swizzle_mxfp4(w1, w1_scale, w2, w2_scale, "mx4", 32, 256, 32, 256)

        mock_shuffle.assert_not_called()
        self.assertIsNone(w1_swizzle_layout)
        self.assertIsNone(w2_swizzle_layout)

    def test_shuffles_on_gfx950(self):
        from atom.model_ops import fused_moe_triton as fmt

        w1, w1_scale, w2, w2_scale = _fake_weights()
        with (
            mock.patch.object(fmt, "get_arch", return_value="gfx950"),
            mock.patch.object(
                fmt,
                "shuffle_scale_moe",
                return_value=(w1_scale, "CDNA4_SCALE"),
            ) as mock_shuffle,
        ):
            (
                _,
                _,
                w1_swizzle_layout,
                _,
                _,
                w2_swizzle_layout,
            ) = fmt._swizzle_mxfp4(w1, w1_scale, w2, w2_scale, "mx4", 32, 256, 32, 256)

        self.assertEqual(mock_shuffle.call_count, 2)
        self.assertEqual(w1_swizzle_layout, "CDNA4_SCALE")
        self.assertEqual(w2_swizzle_layout, "CDNA4_SCALE")


if __name__ == "__main__":
    unittest.main()
