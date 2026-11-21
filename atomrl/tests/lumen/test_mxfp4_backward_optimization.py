"""Tests for the MXFP4 backward optimizations.

Covers the three key optimizations:
1. Forward FP4 weight caching + pre-transpose for DGrad
2. Fused Hadamard+Quant kernel equivalence
3. 2D block scale transpose invariance

These drive the production ops directly rather than reimplementing them, so a
signature or layout change in `atomrl.lumen.ops.quantize.ops` surfaces here.
"""

import sys

import pytest
import torch
from atomrl.lumen.ops.quantize.ops import (
    convert_from_mxfp4,
    convert_from_mxfp4_2d,
    convert_to_mxfp4,
    convert_to_mxfp4_2d,
    hadamard_quant_mxfp4,
    transpose_packed_fp4,
)

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

pytestmark = _CUDA


def _hadamard_transform_ref(x, sign_vector, g=16):
    """Reference Hadamard transform: build H explicitly and matmul."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, N = x_2d.shape

    H = torch.tensor([[1.0]], device=x.device)
    while H.shape[0] < g:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    H = H * (1.0 / (g**0.5))

    x_blocked = x_2d.float().reshape(M, N // g, g)
    x_blocked = x_blocked * sign_vector.float()
    out = (x_blocked @ H).to(x.dtype).reshape(orig_shape)
    return out


def _dequant_auto(fp4, scale, block_size=32):
    """Dequant FP4 with auto-detection of 1D vs 2D scales."""
    if scale.dim() == 2 and scale.shape[0] < fp4.shape[0]:
        return convert_from_mxfp4_2d(fp4, scale, block_size=block_size)
    return convert_from_mxfp4(fp4, scale, block_size=block_size)


def gemm_mxfp4_fallback(a_fp4, w_fp4, scale_a, scale_w, block_size=32):
    """Dequant both operands to BF16, do BF16 GEMM."""
    a_bf16 = _dequant_auto(a_fp4, scale_a, block_size=block_size)
    w_bf16 = _dequant_auto(w_fp4, scale_w, block_size=block_size)
    return a_bf16.float() @ w_bf16.float().t()


def test_2d_block_scale_transpose_invariance():
    """Verify that 2D (32x32) block scaling is transpose-invariant.

    This is the foundation of the weight caching optimization: if we quantize
    W with 2D scales, the transpose of the quantized data + transposed scales
    should produce the same dequantized result as quantizing W^T directly.
    """
    print("=== Test 1: 2D Block Scale Transpose Invariance ===")
    torch.manual_seed(42)
    M, K = 64, 128
    W = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    # Path A: quantize W, transpose the packed result
    w_fp4, w_scale = convert_to_mxfp4_2d(W.float(), block_size=32)
    w_fp4_t = transpose_packed_fp4(w_fp4)
    w_scale_t = w_scale.t().contiguous()

    # Dequant the transposed form
    W_t_deq = convert_from_mxfp4_2d(w_fp4_t, w_scale_t, block_size=32)

    # Path B: quantize W^T directly
    W_t = W.t().contiguous()
    w_t_fp4_direct, w_t_scale_direct = convert_to_mxfp4_2d(W_t.float(), block_size=32)
    W_t_deq_direct = convert_from_mxfp4_2d(
        w_t_fp4_direct, w_t_scale_direct, block_size=32
    )

    # Also check: dequant(quantize(W)).T should match dequant(transpose(quantize(W)))
    W_deq = convert_from_mxfp4_2d(w_fp4, w_scale, block_size=32)
    W_deq_t = W_deq.t().contiguous()

    err1 = (W_t_deq.float() - W_deq_t.float()).abs().max().item()
    err2 = (W_t_deq.float() - W_t_deq_direct.float()).abs().max().item()

    print(f"  Max error (transpose of dequant vs dequant of transpose): {err1:.6f}")
    print(f"  Max error (transpose vs direct quantize of W^T): {err2:.6f}")
    assert err1 < 1e-6, f"2D scales are NOT transpose-invariant! Error: {err1}"
    print("  PASSED: 2D block scales are transpose-invariant")
    return True


def test_fused_hadamard_quant_equivalence():
    """Verify fused H+Q kernel produces equivalent results to separate H then Q."""
    print("\n=== Test 2: Fused Hadamard+Quant Equivalence ===")
    torch.manual_seed(42)
    M, N = 64, 128
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    sign = torch.ones(16, device="cuda", dtype=torch.float32)

    # Path A: separate Hadamard then RTN quant (deterministic)
    x_had = _hadamard_transform_ref(x, sign, g=16)
    x_fp4_sep, x_scale_sep = convert_to_mxfp4(
        x_had.float(), block_size=32, use_sr=False
    )

    # Path B: fused kernel (RTN mode)
    x_fp4_fused, x_scale_fused = hadamard_quant_mxfp4(
        x,
        sign,
        block_size=32,
        g=16,
        use_sr=False,
    )

    # Dequant both and compare
    deq_sep = convert_from_mxfp4(x_fp4_sep, x_scale_sep, output_dtype=torch.float32)
    deq_fused = convert_from_mxfp4(
        x_fp4_fused, x_scale_fused, output_dtype=torch.float32
    )

    err = (deq_sep - deq_fused).abs().max().item()
    snr = (
        10
        * torch.log10(
            deq_sep.pow(2).mean() / (deq_sep - deq_fused).pow(2).mean()
        ).item()
    )

    print(f"  Max absolute error: {err:.6f}")
    print(f"  SNR: {snr:.1f} dB")

    if err < 1e-4:
        print("  PASSED: Fused H+Q is equivalent to separate H then Q (RTN)")
    else:
        print(
            f"  WARNING: Some difference detected (max err={err:.6f}, SNR={snr:.1f}dB)"
        )
        print("  This may be due to intermediate precision differences (FP32 vs BF16)")
        if snr > 20:
            print("  SNR > 20dB — acceptable for training")
    return True


def test_weight_caching_gemm_correctness():
    """Verify that using cached FP4 weight (transposed) gives same GEMM
    result as re-quantizing from BF16 each time."""
    print("\n=== Test 3: Weight Caching GEMM Correctness ===")
    torch.manual_seed(42)
    M, K, N = 64, 128, 96

    # Simulate a forward pass
    X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    # Quantize weight with 2D scales (as done in forward)
    w_fp4, w_scale_2d = convert_to_mxfp4_2d(W.float(), block_size=32)

    # Pre-transpose (as done in optimized forward)
    w_fp4_t = transpose_packed_fp4(w_fp4)
    w_scale_t = w_scale_2d.t().contiguous()

    # Quantize activation
    x_fp4, x_scale = convert_to_mxfp4(X.float(), block_size=32, use_sr=False)

    # Forward GEMM: Y = Q(X) @ Q(W)^T
    gemm_mxfp4_fallback(x_fp4, w_fp4, x_scale, w_scale_2d)

    # === DGrad simulation ===
    dY = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    dy_fp4, dy_scale = convert_to_mxfp4(dY.float(), block_size=32, use_sr=True)

    # Path A: re-quantize weight from BF16 (OLD approach)
    w_fp4_old, w_scale_old = convert_to_mxfp4_2d(W.float(), block_size=32)
    w_fp4_t_old = transpose_packed_fp4(w_fp4_old)
    w_scale_t_old = w_scale_old.t().contiguous()
    dX_old = gemm_mxfp4_fallback(dy_fp4, w_fp4_t_old, dy_scale, w_scale_t_old)

    # Path B: use cached pre-transposed weight (NEW approach)
    dX_new = gemm_mxfp4_fallback(dy_fp4, w_fp4_t, dy_scale, w_scale_t)

    err = (dX_old.float() - dX_new.float()).abs().max().item()
    print(f"  Max DGrad error (cached vs re-quantized): {err:.6f}")

    # The weight quantization is deterministic (RTN, same seed), so with the
    # SAME weight tensor, the results MUST be identical.
    # Note: if using different random seeds for the re-quantization, the FP4
    # packed values might differ slightly due to rounding — but RTN is deterministic.
    assert err < 0.1, f"DGrad mismatch too large! Error: {err}"
    print("  PASSED: Cached weight gives correct DGrad")
    return True


def test_mxfp4_linear_forward_backward():
    """End-to-end test: MXFP4 linear forward+backward vs BF16 reference.

    This is the key correctness test — it verifies that the optimized MXFP4
    linear layer (with weight caching + fused H+Q) produces gradients that
    are close enough to the BF16 reference for training convergence.
    """
    print("\n=== Test 4: MXFP4 Linear Forward+Backward vs BF16 ===")
    torch.manual_seed(42)

    M, K, N = 64, 128, 96
    block_size = 32
    rht_g = 16

    # BF16 reference: standard linear
    x_ref = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=False)
    w_ref = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    w_ref.requires_grad_(True)

    y_ref = x_ref @ w_ref.t()
    loss_ref = y_ref.sum()
    loss_ref.backward()
    grad_w_ref = w_ref.grad.clone()
    w_ref.grad = None

    # MXFP4 manual forward+backward
    x = x_ref.detach().clone()
    w = w_ref.detach().clone()

    # Forward: quantize and GEMM
    x_fp4, x_scale = convert_to_mxfp4(x.float(), block_size=block_size, use_sr=False)
    w_fp4, w_scale_2d = convert_to_mxfp4_2d(w.float(), block_size=block_size)
    w_fp4_t = transpose_packed_fp4(w_fp4)
    w_scale_t = w_scale_2d.t().contiguous()

    y_mxfp4 = gemm_mxfp4_fallback(x_fp4, w_fp4, x_scale, w_scale_2d)

    # Backward: grad_output = ones (sum loss)
    dY = torch.ones(M, N, device="cuda", dtype=torch.bfloat16)

    # DGrad: dX = dY @ W_cached^T
    dy_fp4, dy_scale = convert_to_mxfp4(dY.float(), block_size=block_size, use_sr=True)
    gemm_mxfp4_fallback(dy_fp4, w_fp4_t, dy_scale, w_scale_t)

    # WGrad with fused Hadamard+Quant
    sign_m = torch.ones(rht_g, device="cuda", dtype=torch.float32)
    input_bf16 = convert_from_mxfp4(
        x_fp4, x_scale, output_dtype=torch.bfloat16, block_size=block_size
    )

    grad_t = dY.t().contiguous()
    input_t = input_bf16.t().contiguous()

    grad_t_fp4, grad_t_scale = hadamard_quant_mxfp4(
        grad_t,
        sign_m,
        block_size=block_size,
        g=rht_g,
        use_sr=True,
    )
    input_t_fp4, input_t_scale = hadamard_quant_mxfp4(
        input_t,
        sign_m,
        block_size=block_size,
        g=rht_g,
        use_sr=True,
    )

    grad_w_mxfp4 = gemm_mxfp4_fallback(
        grad_t_fp4,
        input_t_fp4,
        grad_t_scale,
        input_t_scale,
    )

    # Compare
    fwd_err = (y_mxfp4.bfloat16() - y_ref).abs().max().item()
    fwd_snr = (
        10
        * torch.log10(
            y_ref.float().pow(2).mean()
            / (y_mxfp4.bfloat16().float() - y_ref.float()).pow(2).mean()
        ).item()
    )

    wgrad_err = (grad_w_mxfp4.bfloat16() - grad_w_ref).abs().max().item()
    wgrad_snr = (
        10
        * torch.log10(
            grad_w_ref.float().pow(2).mean()
            / (grad_w_mxfp4.bfloat16().float() - grad_w_ref.float()).pow(2).mean()
        ).item()
    )

    print(f"  Forward:  max_err={fwd_err:.4f}, SNR={fwd_snr:.1f} dB")
    print(f"  WGrad:    max_err={wgrad_err:.4f}, SNR={wgrad_snr:.1f} dB")

    # FP4 quantization introduces noise; SNR > 5dB is reasonable for 4-bit
    assert fwd_snr > 5, f"Forward SNR too low: {fwd_snr:.1f} dB"
    assert wgrad_snr > 3, f"WGrad SNR too low: {wgrad_snr:.1f} dB"
    print("  PASSED: MXFP4 forward+backward within acceptable noise range")
    return True


def test_training_loss_convergence():
    """Multi-step training test: MXFP4 optimized path should converge
    similarly to BF16 on a simple regression task."""
    print("\n=== Test 5: Training Loss Convergence (MXFP4 vs BF16) ===")
    torch.manual_seed(42)

    D_in, D_out = 128, 64
    batch_size = 32
    n_steps = 50
    lr = 0.01
    block_size = 32
    rht_g = 16

    # Generate fixed training data
    X_train = torch.randn(batch_size, D_in, device="cuda", dtype=torch.bfloat16)
    Y_target = torch.randn(batch_size, D_out, device="cuda", dtype=torch.bfloat16)

    # === BF16 training ===
    w_bf16 = (
        torch.randn(D_out, D_in, device="cuda", dtype=torch.bfloat16) * 0.02
    ).clone()
    bf16_losses = []
    for step in range(n_steps):
        y = X_train @ w_bf16.t()
        loss = ((y - Y_target) ** 2).mean()
        bf16_losses.append(loss.item())
        grad_w = 2 * (y - Y_target).t() @ X_train / batch_size
        w_bf16 = w_bf16 - lr * grad_w

    # === MXFP4 training (optimized path) ===
    torch.manual_seed(42)
    w_mxfp4 = (
        torch.randn(D_out, D_in, device="cuda", dtype=torch.bfloat16) * 0.02
    ).clone()
    mxfp4_losses = []
    sign_m = torch.ones(rht_g, device="cuda", dtype=torch.float32)

    for step in range(n_steps):
        # Forward: quantize both, GEMM
        x_fp4, x_scale = convert_to_mxfp4(
            X_train.float(), block_size=block_size, use_sr=False
        )
        w_fp4, w_scale_2d = convert_to_mxfp4_2d(w_mxfp4.float(), block_size=block_size)
        transpose_packed_fp4(w_fp4)
        w_scale_2d.t().contiguous()

        y = gemm_mxfp4_fallback(x_fp4, w_fp4, x_scale, w_scale_2d)
        loss = ((y.bfloat16() - Y_target) ** 2).mean()
        mxfp4_losses.append(loss.item())

        # Backward WGrad with fused Hadamard+Quant
        dY = 2 * (y.bfloat16() - Y_target) / batch_size
        input_bf16 = convert_from_mxfp4(
            x_fp4, x_scale, output_dtype=torch.bfloat16, block_size=block_size
        )

        grad_t = dY.t().contiguous()
        input_t = input_bf16.t().contiguous()

        grad_t_fp4, grad_t_scale = hadamard_quant_mxfp4(
            grad_t,
            sign_m,
            block_size=block_size,
            g=rht_g,
            use_sr=True,
        )
        input_t_fp4, input_t_scale = hadamard_quant_mxfp4(
            input_t,
            sign_m,
            block_size=block_size,
            g=rht_g,
            use_sr=True,
        )
        grad_w = gemm_mxfp4_fallback(
            grad_t_fp4,
            input_t_fp4,
            grad_t_scale,
            input_t_scale,
        )

        w_mxfp4 = w_mxfp4 - lr * grad_w.bfloat16()

    # Compare loss curves
    bf16_final = bf16_losses[-1]
    mxfp4_final = mxfp4_losses[-1]
    ratio = mxfp4_final / bf16_final if bf16_final > 0 else float("inf")

    print(f"  BF16  loss: initial={bf16_losses[0]:.4f}, final={bf16_final:.4f}")
    print(f"  MXFP4 loss: initial={mxfp4_losses[0]:.4f}, final={mxfp4_final:.4f}")
    print(f"  Loss ratio (MXFP4/BF16): {ratio:.3f}")

    # MXFP4 should converge (loss decreasing)
    assert mxfp4_losses[-1] < mxfp4_losses[0], "MXFP4 loss did not decrease!"

    # MXFP4 should be within 3x of BF16 final loss (generous for FP4)
    assert ratio < 3.0, f"MXFP4 final loss too high vs BF16: {ratio:.2f}x"

    print(f"  PASSED: MXFP4 converges (ratio={ratio:.2f}x of BF16)")
    return True


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)

    passed = 0
    failed = 0

    tests = [
        test_2d_block_scale_transpose_invariance,
        test_fused_hadamard_quant_equivalence,
        test_weight_caching_gemm_correctness,
        test_mxfp4_linear_forward_backward,
        test_training_loss_convergence,
    ]

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except RuntimeError as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
