# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops"))
from atomrl.lumen.quantize.comm_tensor import FP8CommTensor, MXFP4CommTensor
from conftest import compute_snr


class TestFP8CommTensor:
    def _make(self, rows=16, cols=64):
        data = torch.randn(rows, cols, dtype=torch.bfloat16)
        return FP8CommTensor(data, fp8_dtype=torch.float8_e4m3fnuz)

    def test_creation(self):
        t = self._make()
        assert t.shape == (16, 64)
        assert t.dtype == torch.bfloat16

    def test_view_preserves_subclass(self):
        t = self._make()
        v = t.view(4, 4, 64)
        assert isinstance(v, FP8CommTensor)

    def test_clone_preserves_subclass(self):
        t = self._make()
        c = t.clone()
        assert isinstance(c, FP8CommTensor)

    def test_unknown_op_unwraps(self):
        t = self._make()
        result = t + 1.0
        assert not isinstance(result, FP8CommTensor)
        assert result.dtype == torch.bfloat16

    def test_flatten_unflatten_roundtrip(self):
        t = self._make()
        _names, metadata = t.__tensor_flatten__()
        inner = {"_data": t._data}
        t2 = FP8CommTensor.__tensor_unflatten__(inner, metadata, t.shape, t.stride())
        assert isinstance(t2, FP8CommTensor)
        assert torch.equal(t._data, t2._data)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
    def test_pre_post_all_gather_roundtrip(self):
        data = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        t = FP8CommTensor(data, fp8_dtype=torch.float8_e4m3fnuz)
        tensors, meta = FP8CommTensor.fsdp_pre_all_gather(t)
        result = FP8CommTensor.fsdp_post_all_gather(tensors, meta, torch.bfloat16)
        torch.testing.assert_close(result, data, atol=0.1, rtol=0.05)


class TestMXFP4CommTensor:
    def _make(self, rows=32, cols=64):
        data = torch.randn(rows, cols, dtype=torch.bfloat16)
        return MXFP4CommTensor(data, block_size=32)

    def test_creation(self):
        t = self._make()
        assert t.shape == (32, 64)
        assert t.dtype == torch.bfloat16

    def test_view_preserves_subclass(self):
        t = self._make()
        v = t.view(4, 8, 64)
        assert isinstance(v, MXFP4CommTensor)

    def test_clone_preserves_subclass(self):
        t = self._make()
        c = t.clone()
        assert isinstance(c, MXFP4CommTensor)

    def test_split_preserves_subclass(self):
        t = self._make(rows=64)
        shards = t.split(32, dim=0)
        assert len(shards) == 2
        assert all(isinstance(shard, MXFP4CommTensor) for shard in shards)

    def test_chunk_preserves_subclass(self):
        t = self._make(rows=64)
        shards = t.chunk(2, dim=0)
        assert len(shards) == 2
        assert all(isinstance(shard, MXFP4CommTensor) for shard in shards)

    def test_unknown_op_unwraps(self):
        t = self._make()
        result = t + 1.0
        assert not isinstance(result, MXFP4CommTensor)
        assert result.dtype == torch.bfloat16

    def test_flatten_unflatten_roundtrip(self):
        t = self._make()
        _names, metadata = t.__tensor_flatten__()
        inner = {"_data": t._data}
        t2 = MXFP4CommTensor.__tensor_unflatten__(inner, metadata, t.shape, t.stride())
        assert isinstance(t2, MXFP4CommTensor)
        assert torch.equal(t._data, t2._data)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
    def test_pre_post_all_gather_roundtrip(self):
        data = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")
        t = MXFP4CommTensor(data, block_size=32)
        tensors, meta = t.fsdp_pre_all_gather(mesh=None)
        result, inner_tensors = t.fsdp_post_all_gather(tensors, meta, torch.bfloat16)
        # MXFP4 2D block quant is lossy; the hook must reconstruct a BF16 tensor
        # close enough that FSDP2 can treat it as the unsharded master.
        assert result.shape == data.shape
        assert inner_tensors == (result,)
        assert compute_snr(data.float(), result.float()) >= 8.0

        out = torch.empty_like(data)
        assert t.fsdp_post_all_gather(tensors, meta, torch.bfloat16, out=out) is None
        torch.testing.assert_close(out, result)

        wrong_out = torch.empty((64, 64), dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="output buffer mismatch"):
            t.fsdp_post_all_gather(tensors, meta, torch.bfloat16, out=wrong_out)
