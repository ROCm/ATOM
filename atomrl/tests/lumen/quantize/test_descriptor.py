# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# tests/quantize/test_descriptor.py
import pytest
import torch
from atomrl.lumen.ops.quantize.linear import quantize_input
from atomrl.lumen.quantize.config import _get_float8_e4m3
from atomrl.lumen.quantize.descriptor import FP8Descriptor


class TestFP8Descriptor:
    def _make_desc(self, m=4, k=8):
        fp8_dtype = torch.float8_e4m3fnuz
        data = torch.randn(m, k, device="cpu").to(fp8_dtype)
        scale = torch.tensor(0.5, dtype=torch.float32)
        return FP8Descriptor(data=data, scale=scale, fp8_dtype=fp8_dtype)

    def test_creation(self):
        desc = self._make_desc()
        assert desc.data.dtype == torch.float8_e4m3fnuz
        assert desc.scale.item() == pytest.approx(0.5)
        assert desc.fp8_dtype == torch.float8_e4m3fnuz

    def test_transpose_cached(self):
        desc = self._make_desc(m=4, k=8)
        t1 = desc.transpose_cached
        assert t1.shape == (8, 4)
        assert t1.is_contiguous()
        t2 = desc.transpose_cached
        assert t1.data_ptr() == t2.data_ptr()  # same cache

    def test_invalidate_transpose(self):
        desc = self._make_desc()
        t1 = desc.transpose_cached
        desc.invalidate_transpose()
        t2 = desc.transpose_cached
        assert t1.data_ptr() != t2.data_ptr()  # new allocation

    def test_tensors_roundtrip(self):
        desc = self._make_desc()
        data, scale = desc.tensors()
        reconstructed = FP8Descriptor.from_tensors(data, scale, desc.fp8_dtype)
        # Same tensor objects: torch.equal(x, x) errors on float8 CPU (PyTorch).
        assert (desc.data == reconstructed.data).all().item()
        assert torch.equal(desc.scale, reconstructed.scale)
        assert desc.fp8_dtype == reconstructed.fp8_dtype

    def test_none_for_passthrough(self):
        """None represents scaling_type='none' passthrough."""
        desc = None
        assert desc is None  # quantize_input returns None for BF16

    def test_quantize_input_none_returns_none(self):
        x = torch.randn(2, 4, dtype=torch.bfloat16)
        fp8_dtype = torch.float8_e4m3fnuz
        out = quantize_input(x, "none", fp8_dtype)
        assert out is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_quantize_input_dynamic_returns_fp8_descriptor(self):
        device = torch.device("cuda")
        x = torch.randn(8, 16, dtype=torch.bfloat16, device=device)
        fp8_dtype = _get_float8_e4m3()
        desc = quantize_input(x, "dynamic", fp8_dtype)
        assert isinstance(desc, FP8Descriptor)
        assert desc.data.shape == x.shape
        assert desc.data.dtype == fp8_dtype
        assert desc.fp8_dtype == fp8_dtype
        assert desc.scale is not None
        d, s = desc.tensors()
        assert d is desc.data
        assert s is desc.scale

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_quantize_input_per_token_returns_fp8_descriptor(self):
        device = torch.device("cuda")
        x = torch.randn(4, 32, dtype=torch.bfloat16, device=device)
        fp8_dtype = _get_float8_e4m3()
        desc = quantize_input(x, "per_token", fp8_dtype)
        assert isinstance(desc, FP8Descriptor)
        assert desc.data.shape == x.shape
        assert desc.fp8_dtype == fp8_dtype


class TestDispatchGemmDescriptor:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_accepts_descriptors(self):
        from atomrl.lumen.ops.quantize.linear import dispatch_gemm

        fp8_dtype = _get_float8_e4m3()
        a = FP8Descriptor(
            data=torch.randn(4, 64, device="cuda").to(fp8_dtype),
            scale=torch.tensor(1.0, device="cuda"),
            fp8_dtype=fp8_dtype,
        )
        w = FP8Descriptor(
            data=torch.randn(32, 64, device="cuda").to(fp8_dtype),
            scale=torch.tensor(1.0, device="cuda"),
            fp8_dtype=fp8_dtype,
        )
        result = dispatch_gemm(a, w, scaling_type="dynamic")
        assert result.shape == (4, 32)
        assert result.dtype == torch.bfloat16
