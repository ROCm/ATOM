# SPDX-License-Identifier: MIT
"""When `fused_allreduce_gemma_rms_norm_mxfp8` takes aiter's fused epilogue.

The fused MXFP8 epilogue is only an exact substitute for AR + RMSNorm followed
by `quantize_fp8` where its RMSNorm is the unquantized kernel's bit for bit:
where the unquantized path takes aiter's 1-stage kernel (at TP4,
128 * 7168 * 2 / 4 = 458,752 B, 37 rows of 6144). Past that (the unquantized
2-stage and split kernels reduce the squares in another order) and on an
aiter without the per-group `mx_scale` mode, the separate path must run. These
cases pin that decision with a stand-in communicator; the kernel itself is
checked against the separate path on 4 GPUs in aiter's op test.
"""

import pytest
import torch

pytest.importorskip("aiter")

from atom.model_ops.layernorm import _custom_ar_mxfp8

H = 6144


class _CA:
    def __init__(self, disabled=False, fuse=True):
        self.disabled = disabled
        self._fuse = fuse

    def should_fuse_ar_rms(self, inp, n, out_n):
        return self._fuse

    def custom_fused_ar_rms_per_group_quant(
        self,
        input,
        residual_inp,
        weight,
        eps,
        group_size=128,
        use_1stage=False,
        emit_bf16=False,
        transpose_scale=False,
        gemma_norm=False,
        mx_scale=False,
    ):
        raise AssertionError("not called by the dispatch check")


class _CANoMx(_CA):
    def custom_fused_ar_rms_per_group_quant(
        self,
        input,
        residual_inp,
        weight,
        eps,
        group_size=128,
        use_1stage=False,
        emit_bf16=False,
        transpose_scale=False,
        gemma_norm=False,
    ):
        raise AssertionError("not called by the dispatch check")


class _Comm:
    def __init__(self, ca, world_size=4):
        self.ca_comm = ca
        self.world_size = world_size


def _rows(n, k=H):
    return torch.empty(n, k, dtype=torch.bfloat16)


@pytest.mark.parametrize("rows", [1, 16, 32, 37])
def test_takes_fused_epilogue_in_the_one_stage_range(rows):
    # 37 rows x 6144 x 2 B = 454,656 B <= 458,752 B
    assert _custom_ar_mxfp8(_Comm(_CA()), _rows(rows))


def test_one_stage_overrides():
    comm = _Comm(_CA())
    comm._ar_1stage_override = False
    assert not _custom_ar_mxfp8(comm, _rows(1))
    comm._ar_1stage_override = True
    assert _custom_ar_mxfp8(comm, _rows(64))
    comm = _Comm(_CA())
    comm._ar_1stage_max_bytes = 2 * H * 2
    assert _custom_ar_mxfp8(comm, _rows(2)) and not _custom_ar_mxfp8(comm, _rows(3))


@pytest.mark.parametrize("rows", [38, 42, 43, 1355, 4120])
def test_larger_inputs_keep_the_separate_path(rows):
    assert not _custom_ar_mxfp8(_Comm(_CA()), _rows(rows))


def test_aiter_without_mx_scale_keeps_the_separate_path():
    assert not _custom_ar_mxfp8(_Comm(_CANoMx()), _rows(16))


def test_no_custom_all_reduce():
    assert not _custom_ar_mxfp8(_Comm(None), _rows(16))
    assert not _custom_ar_mxfp8(_Comm(_CA(disabled=True)), _rows(16))
    assert not _custom_ar_mxfp8(_Comm(_CA(fuse=False)), _rows(16))


def test_shape_and_group_limits():
    assert not _custom_ar_mxfp8(_Comm(_CA()), _rows(4, k=6144 + 16))
    assert not _custom_ar_mxfp8(_Comm(_CA(), world_size=6), _rows(4))
