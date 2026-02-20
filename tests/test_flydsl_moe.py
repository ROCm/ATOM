#!/usr/bin/env python3
"""Tests for FlyDSL MOE backend (atom/model_ops/flydsl_moe.py).

Test tiers:
  1. Unit tests (CPU-only, no FlyDSL/AITER deps): sorting, quantization, detection
  2. Integration tests (GPU + FlyDSL): full flydsl_fp8_moe pipeline

Run:
  # Unit tests only (no GPU needed):
  python3 tests/test_flydsl_moe.py --unit

  # Full GPU test (needs FlyDSL on PYTHONPATH + GPU):
  ATOM_USE_FLYDSL_MOE=1 PYTHONPATH=/path/to/FlyDSL:$PYTHONPATH \
    python3 tests/test_flydsl_moe.py --gpu
"""

import os
import sys
import argparse
import torch

# Ensure ATOM root is on path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Unit tests (CPU-only)
# ---------------------------------------------------------------------------
def test_detection_disabled():
    """_has_flydsl_moe returns False when env var is disabled."""
    import atom.model_ops.flydsl_moe as mod

    # Reset cache
    mod._flydsl_moe_available = None
    old = os.environ.get("ATOM_USE_FLYDSL_MOE")
    os.environ["ATOM_USE_FLYDSL_MOE"] = "0"
    try:
        assert mod._has_flydsl_moe() is False, "Should be False when disabled"
    finally:
        mod._flydsl_moe_available = None
        if old is None:
            os.environ.pop("ATOM_USE_FLYDSL_MOE", None)
        else:
            os.environ["ATOM_USE_FLYDSL_MOE"] = old
    print("  PASS: test_detection_disabled")


def test_sorting_basic():
    """Test moe_sorting_torch_native produces correct shapes and encoding."""
    from atom.model_ops.flydsl_moe import moe_sorting_torch_native

    M, topk, E, block_size = 8, 2, 4, 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Deterministic routing
    torch.manual_seed(42)
    topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
    topk_weights = torch.rand(M, topk, device=device, dtype=torch.float32)

    sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad = (
        moe_sorting_torch_native(topk_ids, topk_weights, E, block_size)
    )

    # Shape checks
    assert sorted_ids.dtype == torch.int32
    assert sorted_weights.dtype == torch.float32
    assert sorted_expert_ids.dtype == torch.int32
    assert num_tokens_post_pad.shape == (2,)
    assert num_tokens_post_pad[1].item() == M, "num_tokens should be M"

    total_padded = num_tokens_post_pad[0].item()
    assert total_padded > 0
    assert total_padded % block_size == 0, "total should be block-aligned"

    # Verify packed encoding: (topk_slot << 24 | token_id)
    init_val = (topk << 24) | M
    for i in range(total_padded):
        val = sorted_ids[i].item()
        if val == init_val:
            continue  # padding sentinel
        token_id = val & 0xFFFFFF
        topk_slot = (val >> 24) & 0xFF
        assert 0 <= token_id < M, f"token_id={token_id} out of range"
        assert 0 <= topk_slot < topk, f"topk_slot={topk_slot} out of range"

    print("  PASS: test_sorting_basic")


def test_sorting_all_same_expert():
    """Edge case: all tokens routed to same expert."""
    from atom.model_ops.flydsl_moe import moe_sorting_torch_native

    M, topk, E, block_size = 16, 1, 8, 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    topk_ids = torch.zeros(M, topk, device=device, dtype=torch.int32)  # all expert 0
    topk_weights = torch.ones(M, topk, device=device, dtype=torch.float32)

    sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad = (
        moe_sorting_torch_native(topk_ids, topk_weights, E, block_size)
    )

    total_padded = num_tokens_post_pad[0].item()
    expected_blocks = (M + block_size - 1) // block_size
    expected_padded = expected_blocks * block_size
    assert (
        total_padded == expected_padded
    ), f"Expected {expected_padded} padded tokens, got {total_padded}"

    # All expert_ids should be 0 for the used blocks
    for i in range(expected_blocks):
        assert sorted_expert_ids[i].item() == 0

    print("  PASS: test_sorting_all_same_expert")


def test_pertoken_quant_fp8():
    """Test per-token FP8 quantization correctness."""
    from atom.model_ops.flydsl_moe import _pertoken_quant_fp8

    if not hasattr(torch, "float8_e4m3fnuz"):
        print("  SKIP: test_pertoken_quant_fp8 (torch version lacks fp8 support)")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp8_dtype = torch.float8_e4m3fnuz

    x = torch.randn(32, 128, device=device, dtype=torch.float32)
    x_fp8, scale_1d = _pertoken_quant_fp8(x, fp8_dtype)

    assert x_fp8.shape == x.shape
    assert x_fp8.dtype == fp8_dtype
    assert scale_1d.shape == (32,), f"Expected [32], got {scale_1d.shape}"
    assert scale_1d.dtype == torch.float32

    # Dequantize and check approximate reconstruction
    x_deq = x_fp8.to(torch.float32) * scale_1d.unsqueeze(1)
    rel_err = (x_deq - x).abs() / (x.abs() + 1e-6)
    mean_err = rel_err.mean().item()
    assert mean_err < 0.2, f"Mean relative error too high: {mean_err}"

    print(f"  PASS: test_pertoken_quant_fp8 (mean_rel_err={mean_err:.4f})")


def test_pertoken_quant_3d():
    """Test per-token FP8 quantization with 3D input."""
    from atom.model_ops.flydsl_moe import _pertoken_quant_fp8

    if not hasattr(torch, "float8_e4m3fnuz"):
        print("  SKIP: test_pertoken_quant_3d (torch version lacks fp8 support)")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp8_dtype = torch.float8_e4m3fnuz

    x = torch.randn(8, 2, 64, device=device, dtype=torch.float32)
    x_fp8, scale_1d = _pertoken_quant_fp8(x, fp8_dtype)

    assert x_fp8.shape == x.shape
    assert scale_1d.shape == (16,), f"Expected [16], got {scale_1d.shape}"

    print("  PASS: test_pertoken_quant_3d")


def run_unit_tests():
    print("=" * 60)
    print("Unit tests (CPU/GPU, no FlyDSL dependency)")
    print("=" * 60)
    test_detection_disabled()
    test_sorting_basic()
    test_sorting_all_same_expert()
    test_pertoken_quant_fp8()
    test_pertoken_quant_3d()
    print("\nAll unit tests passed!")


# ---------------------------------------------------------------------------
# GPU integration tests (requires FlyDSL + GPU)
# ---------------------------------------------------------------------------
def test_flydsl_fp8_moe_gpu():
    """Full end-to-end test: flydsl_fp8_moe on random FP8 weights."""
    from atom.model_ops.flydsl_moe import _has_flydsl_moe, flydsl_fp8_moe

    if not _has_flydsl_moe():
        print("  SKIP: FlyDSL MOE not available")
        return

    device = torch.device("cuda")
    torch.manual_seed(42)

    # Model dimensions (small for testing)
    M = 64  # tokens
    model_dim = 256
    inter_dim = 128
    E = 4  # experts
    topk = 2

    fp8_dtype = torch.float8_e4m3fnuz

    # Create random FP8 weights (preshuffled via aiter)
    w13_fp32 = torch.randn(E, 2 * inter_dim, model_dim, device=device) * 0.1
    w2_fp32 = torch.randn(E, model_dim, inter_dim, device=device) * 0.1

    # Quantize weights to FP8
    fp8_max = torch.finfo(fp8_dtype).max
    w13_amax = w13_fp32.abs().amax(dim=-1, keepdim=True)
    w13_scale_full = (w13_amax / fp8_max).clamp(min=1e-12)
    w13 = (w13_fp32 / w13_scale_full).clamp(-fp8_max, fp8_max).to(fp8_dtype)

    w2_amax = w2_fp32.abs().amax(dim=-1, keepdim=True)
    w2_scale_full = (w2_amax / fp8_max).clamp(min=1e-12)
    w2 = (w2_fp32 / w2_scale_full).clamp(-fp8_max, fp8_max).to(fp8_dtype)

    # Per-tensor scales [E]
    w13_scale = w13_scale_full.squeeze(-1).amax(dim=-1)  # [E]
    w2_scale = w2_scale_full.squeeze(-1).amax(dim=-1)  # [E]

    # Shuffle weights (FlyDSL expects preshuffled)
    try:
        from aiter.ops.shuffle import shuffle_weight

        w13 = shuffle_weight(w13)
        w2 = shuffle_weight(w2)
    except ImportError:
        print("  WARNING: aiter shuffle not available, using unshuffled weights")

    # Input and routing
    x = torch.randn(M, model_dim, device=device, dtype=torch.bfloat16)
    scores = torch.randn(M, E, device=device)
    topk_vals, topk_ids = torch.topk(scores, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)

    # Run FlyDSL MOE
    out = flydsl_fp8_moe(
        x=x,
        w13=w13,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        top_k=topk,
        block_quant=False,
        quant_type=None,
    )

    assert out.shape == (M, model_dim), f"Expected ({M}, {model_dim}), got {out.shape}"
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"

    print(
        f"  PASS: test_flydsl_fp8_moe_gpu (output shape={out.shape}, "
        f"mean={out.float().mean():.4f}, std={out.float().std():.4f})"
    )


def run_gpu_tests():
    print("=" * 60)
    print("GPU integration tests (requires FlyDSL + GPU)")
    print("=" * 60)
    if not torch.cuda.is_available():
        print("SKIP: No GPU available")
        return
    test_flydsl_fp8_moe_gpu()
    print("\nAll GPU tests passed!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--gpu", action="store_true", help="Run GPU integration tests")
    args = parser.parse_args()

    if not args.unit and not args.gpu:
        args.unit = True  # default to unit tests

    if args.unit:
        run_unit_tests()
    if args.gpu:
        run_gpu_tests()
