# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tests for atomrl.lumen.ops.quantize.padding."""

import pytest
import torch
from atomrl.lumen.ops.quantize.padding import get_fp8_align_size, pad_to_block

# ---------------------------------------------------------------------------
# get_fp8_align_size
# ---------------------------------------------------------------------------


class TestGetFp8AlignSize:
    @pytest.mark.parametrize(
        "scaling_type", ["delayed", "dynamic", "per_token", "none"]
    )
    def test_per_tensor_types_return_one(self, scaling_type):
        assert get_fp8_align_size(scaling_type) == 1
        assert get_fp8_align_size(scaling_type, block_size=64) == 1

    @pytest.mark.parametrize("scaling_type", ["blockwise", "blockwise2d"])
    def test_blockwise_returns_block_size(self, scaling_type):
        assert get_fp8_align_size(scaling_type, block_size=128) == 128
        assert get_fp8_align_size(scaling_type, block_size=64) == 64

    def test_mxfp8_default_block_size_128(self):
        assert get_fp8_align_size("mxfp8", block_size=128) == 32

    def test_mxfp8_block_size_64(self):
        assert get_fp8_align_size("mxfp8", block_size=64) == 32

    def test_mxfp8_block_size_32(self):
        assert get_fp8_align_size("mxfp8", block_size=32) == 32

    @pytest.mark.parametrize("block_size", [65, 96, 127])
    def test_mxfp8_block_size_between_65_and_127(self, block_size):
        assert get_fp8_align_size("mxfp8", block_size=block_size) == 32

    def test_unknown_scaling_type_raises(self):
        with pytest.raises(ValueError, match="Unknown scaling_type"):
            get_fp8_align_size("unknown_recipe")


# ---------------------------------------------------------------------------
# pad_to_block
# ---------------------------------------------------------------------------


class TestPadToBlock:
    def test_already_aligned_is_zero_copy(self):
        t = torch.randn(4, 128)
        padded, orig = pad_to_block(t, 128, dim=-1)
        assert padded is t
        assert orig == 128

    def test_align_size_one_is_always_zero_copy(self):
        t = torch.randn(7, 13)
        padded, orig = pad_to_block(t, 1, dim=-1)
        assert padded is t
        assert orig == 13

    def test_pad_last_dim_2d(self):
        t = torch.randn(4, 100)
        padded, orig = pad_to_block(t, 32, dim=-1)
        assert padded.shape == (4, 128)
        assert orig == 100
        torch.testing.assert_close(padded[:, :100], t)
        assert (padded[:, 100:] == 0).all()

    def test_pad_first_dim_2d(self):
        t = torch.randn(100, 64)
        padded, orig = pad_to_block(t, 32, dim=0)
        assert padded.shape == (128, 64)
        assert orig == 100
        torch.testing.assert_close(padded[:100, :], t)
        assert (padded[100:, :] == 0).all()

    def test_pad_middle_dim_3d(self):
        t = torch.randn(2, 5, 8)
        padded, orig = pad_to_block(t, 4, dim=1)
        assert padded.shape == (2, 8, 8)
        assert orig == 5
        torch.testing.assert_close(padded[:, :5, :], t)
        assert (padded[:, 5:, :] == 0).all()

    def test_pad_4d_tensor(self):
        t = torch.randn(2, 3, 7, 16)
        padded, orig = pad_to_block(t, 4, dim=2)
        assert padded.shape == (2, 3, 8, 16)
        assert orig == 7

    def test_negative_dim_equivalent(self):
        t = torch.randn(4, 100)
        p1, o1 = pad_to_block(t, 32, dim=-1)
        p2, o2 = pad_to_block(t, 32, dim=1)
        assert o1 == o2
        torch.testing.assert_close(p1, p2)

    def test_orig_size_round_trip(self):
        t = torch.randn(3, 50)
        padded, orig = pad_to_block(t, 32, dim=-1)
        recovered = padded[:, :orig]
        torch.testing.assert_close(recovered, t)

    def test_align_size_zero_raises(self):
        t = torch.randn(4, 8)
        with pytest.raises(ValueError, match="align_size must be >= 1"):
            pad_to_block(t, 0, dim=-1)

    def test_align_size_negative_raises(self):
        t = torch.randn(4, 8)
        with pytest.raises(ValueError, match="align_size must be >= 1"):
            pad_to_block(t, -1, dim=-1)

    def test_pad_preserves_dtype(self):
        t = torch.randn(4, 100, dtype=torch.bfloat16)
        padded, _ = pad_to_block(t, 32, dim=-1)
        assert padded.dtype == torch.bfloat16

    def test_pad_preserves_device(self):
        t = torch.randn(4, 100)
        padded, _ = pad_to_block(t, 32, dim=-1)
        assert padded.device == t.device


# ---------------------------------------------------------------------------
# Integration regression: _round_to_mxfp8
# ---------------------------------------------------------------------------


class TestRoundToMxfp8Regression:
    """Verify _round_to_mxfp8 produces identical results after refactoring."""

    def _reference_round_to_mxfp8_padding(self, tensor, block_size=32):
        """Original inline-padding logic (pre-refactor) for comparison."""
        orig_shape = tensor.shape
        flat = tensor.reshape(-1, orig_shape[-1]).contiguous()
        _M, N = flat.shape
        pad_n = (block_size - N % block_size) % block_size
        if pad_n > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_n))
        return flat, N

    def test_aligned_shape_zero_copy(self):
        flat = torch.randn(4, 64).contiguous()
        padded, orig = pad_to_block(flat, 32, dim=-1)
        assert padded is flat, "aligned tensor should be returned as-is (zero-copy)"
        assert orig == 64

    def test_unaligned_shape_matches_reference(self):
        t = torch.randn(4, 50)
        padded, orig = pad_to_block(t, 32, dim=-1)
        ref_padded, ref_n = self._reference_round_to_mxfp8_padding(t, 32)
        assert padded.shape == ref_padded.shape
        assert orig == ref_n
        torch.testing.assert_close(padded, ref_padded)

    def test_end_to_end_round_to_mxfp8(self):
        """Call the real _round_to_mxfp8 and verify it returns correct shape
        for both aligned and unaligned inputs."""
        from atomrl.lumen.quantize.scaling_manager import _round_to_mxfp8

        for N in [64, 50]:
            t = torch.randn(4, N, dtype=torch.bfloat16, device="cuda")
            out = _round_to_mxfp8(t, block_size=32)
            assert out.shape == t.shape, f"expected shape {t.shape}, got {out.shape}"
            assert out.dtype == t.dtype


# ---------------------------------------------------------------------------
# Integration regression: _quant_blockwise_2d padding
# ---------------------------------------------------------------------------


class TestQuantBlockwise2dPaddingRegression:
    """Verify pad_to_block produces same padded tensor as manual torch.zeros+copy."""

    def _manual_2d_pad(self, we, bs):
        """Reference implementation: manual torch.zeros padding for K and N."""
        K, N = we.shape
        K_pad = ((K + bs - 1) // bs) * bs
        N_pad = ((N + bs - 1) // bs) * bs
        manual = torch.zeros(K_pad, N_pad, dtype=we.dtype, device=we.device)
        manual[:K, :N] = we
        return manual, K, N

    def test_2d_pad_matches_manual(self):
        K, N, bs = 100, 200, 128
        we = torch.randn(K, N)

        we_pad_k, orig_k = pad_to_block(we, bs, dim=0)
        we_pad_kn, orig_n = pad_to_block(we_pad_k, bs, dim=1)

        manual, ref_k, ref_n = self._manual_2d_pad(we, bs)

        assert we_pad_kn.shape == manual.shape
        assert orig_k == ref_k
        assert orig_n == ref_n
        torch.testing.assert_close(we_pad_kn, manual)

    @pytest.mark.parametrize("K,N", [(128, 256), (100, 200), (1, 1), (127, 129)])
    def test_2d_pad_various_shapes(self, K, N):
        bs = 128
        we = torch.randn(K, N)
        we_pad_k, orig_k = pad_to_block(we, bs, dim=0)
        we_pad_kn, orig_n = pad_to_block(we_pad_k, bs, dim=1)
        manual, _, _ = self._manual_2d_pad(we, bs)

        assert we_pad_kn.shape == manual.shape
        assert orig_k == K
        assert orig_n == N
        torch.testing.assert_close(we_pad_kn, manual)

    def test_end_to_end_quant_blockwise_2d(self):
        """Call the real _quant_blockwise_2d with unaligned expert weights.
        Requires triton + GPU."""
        from atomrl.lumen.ops.gemm.grouped_gemm import _quant_blockwise_2d
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        E, K, N, bs = 2, 100, 200, 128
        w = torch.randn(E, K, N, device="cuda")
        fp8_dtype = _get_float8_e4m3()
        w_fp8, _w_scales = _quant_blockwise_2d(w, fp8_dtype, block_size=bs)

        assert w_fp8.shape == (
            E,
            K,
            N,
        ), f"output should be sliced back to original: {w_fp8.shape}"
        assert w_fp8.dtype == fp8_dtype


# ---------------------------------------------------------------------------
# Integration regression: quantize_input mxfp8 padding
# ---------------------------------------------------------------------------


class TestQuantizeInputMxfp8Padding:
    """Verify that quantize_input('mxfp8', ...) no longer asserts on
    unaligned shapes after the pad_to_block fix."""

    def test_unaligned_k_is_padded(self):
        """K=50 is not a multiple of 32; should be auto-padded."""
        from unittest.mock import patch

        mock_result = (
            torch.zeros(64, 64, dtype=torch.float8_e4m3fn),
            torch.ones(64, 2, dtype=torch.uint8),
        )

        with patch(
            "atomrl.lumen.ops.quantize.ops.convert_to_mxfp8", return_value=mock_result
        ) as mock_fn:
            from atomrl.lumen.ops.quantize.linear import quantize_input

            x = torch.randn(64, 50, dtype=torch.bfloat16)
            desc = quantize_input(x, "mxfp8", torch.float8_e4m3fn, block_size=128)

            call_args = mock_fn.call_args
            input_tensor = call_args[0][0]
            assert (
                input_tensor.size(-1) % 32 == 0
            ), f"K={input_tensor.size(-1)} should be aligned to 32"
            assert desc.unpadded_shape == x.shape

    def test_unaligned_m_is_padded(self):
        """M=50 is not a multiple of 32; should be auto-padded."""
        from unittest.mock import patch

        mock_result = (
            torch.zeros(64, 64, dtype=torch.float8_e4m3fn),
            torch.ones(64, 2, dtype=torch.uint8),
        )

        with patch(
            "atomrl.lumen.ops.quantize.ops.convert_to_mxfp8", return_value=mock_result
        ) as mock_fn:
            from atomrl.lumen.ops.quantize.linear import quantize_input

            x = torch.randn(50, 64, dtype=torch.bfloat16)
            desc = quantize_input(x, "mxfp8", torch.float8_e4m3fn, block_size=128)

            call_args = mock_fn.call_args
            input_tensor = call_args[0][0]
            assert (
                input_tensor.size(0) % 32 == 0
            ), f"M={input_tensor.size(0)} should be aligned to 32"
            assert desc.unpadded_shape == x.shape

    def test_both_axes_unaligned_padded_shape_consistency(self):
        """Both M=50 and K=50 unaligned; verify output tensor dimensions are
        multiples of the mxfp8 block alignment (32)."""
        from unittest.mock import patch

        def fake_convert_to_mxfp8(tensor, block_size=32, axis=-1, float8_dtype_pt=None):
            return (
                torch.zeros_like(tensor, dtype=torch.float8_e4m3fn),
                torch.ones(
                    tensor.size(0), tensor.size(1) // block_size, dtype=torch.uint8
                ),
            )

        with patch(
            "atomrl.lumen.ops.quantize.ops.convert_to_mxfp8",
            side_effect=fake_convert_to_mxfp8,
        ) as mock_fn:
            from atomrl.lumen.ops.quantize.linear import quantize_input

            x = torch.randn(50, 50, dtype=torch.bfloat16)
            desc = quantize_input(x, "mxfp8", torch.float8_e4m3fn, block_size=128)

            call_args = mock_fn.call_args
            input_tensor = call_args[0][0]
            assert (
                input_tensor.size(0) % 32 == 0
            ), f"M={input_tensor.size(0)} not aligned"
            assert (
                input_tensor.size(-1) % 32 == 0
            ), f"K={input_tensor.size(-1)} not aligned"
            assert input_tensor.size(0) == 64
            assert input_tensor.size(-1) == 64
            assert desc.unpadded_shape == x.shape

    def test_gfx1250_mxfp8_k_is_padded_to_128(self):
        from unittest.mock import patch

        def fake_convert_to_mxfp8(tensor, block_size=32, axis=-1, float8_dtype_pt=None):
            return (
                torch.zeros_like(tensor, dtype=torch.float8_e4m3fn),
                torch.ones(
                    tensor.size(0), tensor.size(1) // block_size, dtype=torch.uint8
                ),
            )

        with (
            patch(
                "atomrl.lumen.ops.quantize.linear._mxfp8_k_alignment",
                return_value=128,
            ),
            patch(
                "atomrl.lumen.ops.quantize.ops.convert_to_mxfp8",
                side_effect=fake_convert_to_mxfp8,
            ) as mock_fn,
        ):
            from atomrl.lumen.ops.quantize.linear import quantize_input

            x = torch.randn(50, 50, dtype=torch.bfloat16)
            desc = quantize_input(x, "mxfp8", torch.float8_e4m3fn, block_size=64)

        input_tensor = mock_fn.call_args[0][0]
        assert input_tensor.shape == (64, 128)
        assert desc.scale.shape == (64, 4)
        assert desc.unpadded_shape == x.shape

    def test_end_to_end_quantize_input_mxfp8(self):
        """Call the real quantize_input with unaligned shapes on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from atomrl.lumen.ops.quantize.linear import quantize_input
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        for M, K in [(64, 50), (50, 64), (50, 50)]:
            x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            desc = quantize_input(x, "mxfp8", fp8_dtype, block_size=128)
            assert desc.data.size(0) % 32 == 0
            assert desc.data.size(-1) % 32 == 0
            assert desc.unpadded_shape == x.shape
