# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""MXFP4's support matrix is gfx950, enforced at enable time.

Off gfx950 the FP4 conversion falls to a Triton path whose rounding is wrong in
four ways that no test covers, so the failure mode is a quiet accuracy loss
rather than an error. The gate turns that into a refusal at the entry point.
"""

import atomrl.lumen.quantize as quant
import pytest
import torch
from atomrl.lumen.quantize import QuantConfig, QuantFormat, ScalingType


@pytest.fixture
def unsupported_arch(monkeypatch):
    """Make this box look like an architecture without hardware FP4."""
    monkeypatch.setattr(quant, "_MXFP4_SUPPORTED_ARCHS", ("gfx_nonexistent",))
    monkeypatch.delenv(quant._MXFP4_ARCH_OVERRIDE_ENV, raising=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="arch needs a GPU to read")
class TestMXFP4ArchGate:
    def test_supported_arch_passes(self, monkeypatch):
        from atomrl.lumen.ops.quantize.ops import triton_arch

        monkeypatch.setattr(quant, "_MXFP4_SUPPORTED_ARCHS", (triton_arch(),))
        quant.assert_mxfp4_arch_supported()

    def test_unsupported_arch_is_refused(self, unsupported_arch):
        with pytest.raises(RuntimeError, match="MXFP4 is supported on"):
            quant.assert_mxfp4_arch_supported()

    def test_refusal_names_the_reason(self, unsupported_arch):
        """A bare 'unsupported' would send the reader to the GEMM dispatch."""
        with pytest.raises(RuntimeError, match="rounding is known to be wrong"):
            quant.assert_mxfp4_arch_supported()

    def test_override_opts_in(self, unsupported_arch, monkeypatch):
        monkeypatch.setenv(quant._MXFP4_ARCH_OVERRIDE_ENV, "1")
        quant.assert_mxfp4_arch_supported()

    def test_enable_refuses_mxfp4_on_an_unsupported_arch(self, unsupported_arch):
        cfg = QuantConfig(format=QuantFormat.MXFP4, scaling=ScalingType.BLOCKWISE)
        with pytest.raises(RuntimeError, match="MXFP4 is supported on"):
            quant.enable(torch.nn.Linear(32, 32).cuda(), config=cfg)

    def test_enable_leaves_fp8_alone_on_the_same_arch(self, unsupported_arch):
        """The gate is MXFP4's; FP8 runs on every architecture Lumen targets."""
        cfg = QuantConfig(format=QuantFormat.FP8_E4M3, scaling=ScalingType.DELAYED)
        quant.enable(torch.nn.Linear(32, 32).cuda(), config=cfg)


class TestMXFP4ArchGateWithoutGPU:
    def test_no_gpu_is_not_an_unsupported_arch(self, monkeypatch):
        """With no device the arch is unknowable, and nothing will run either."""
        monkeypatch.setattr(quant, "_MXFP4_SUPPORTED_ARCHS", ("gfx_nonexistent",))
        monkeypatch.delenv(quant._MXFP4_ARCH_OVERRIDE_ENV, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        quant.assert_mxfp4_arch_supported()
