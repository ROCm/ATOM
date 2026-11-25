# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################
"""
Tests for atomrl.lumen.ops.cross_entropy: vocab-parallel cross-entropy.

Covers:
  - Forward — compare against cross_entropy_ref from conftest
  - Forward + backward — compare gradient on logits
  - Label smoothing
  - reduce_loss=True (scalar mean loss)
  - ignore_idx with actual ignored targets
  - seq_len > 1
  - is_cg_capturable=True backward path
  - Non-contiguous inputs, small batch edge case
"""

import pytest
import torch
from atomrl.lumen.ops.cross_entropy import parallel_cross_entropy
from conftest import compute_snr, cross_entropy_ref

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

BATCH_VOCAB = [(32, 32000), (128, 50000)]
BATCH_VOCAB_IDS = ["b32_v32k", "b128_v50k"]


# ===================================================================
# Forward
# ===================================================================


@pytest.mark.parametrize("batch,vocab", BATCH_VOCAB, ids=BATCH_VOCAB_IDS)
def test_cross_entropy_fwd(batch, vocab):
    """Compare cross-entropy forward against cross_entropy_ref."""
    seq_len = 1
    dtype = torch.bfloat16

    logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=dtype) * 0.1
    target = torch.randint(0, vocab, (batch, seq_len), device="cuda", dtype=torch.long)

    loss_ref = cross_entropy_ref(logits, target).reshape(batch, seq_len)
    loss_lumen = parallel_cross_entropy(
        logits,
        target,
        0.0,
        False,
        None,
        -100,
    )

    snr = compute_snr(loss_ref.float(), loss_lumen.float())
    assert snr > 20, f"Cross-entropy fwd SNR: {snr:.1f} dB (expected > 20)"


# ===================================================================
# Forward + backward
# ===================================================================


@pytest.mark.parametrize("batch,vocab", BATCH_VOCAB, ids=BATCH_VOCAB_IDS)
def test_cross_entropy_fwd_bwd(batch, vocab):
    """Forward + backward, compare gradient on logits."""
    seq_len = 1
    dtype = torch.bfloat16

    logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=dtype) * 0.1
    logits.requires_grad_(True)
    target = torch.randint(0, vocab, (batch, seq_len), device="cuda", dtype=torch.long)
    logits_ref = logits.detach().clone().requires_grad_(True)

    cross_entropy_ref(logits_ref, target).mean().backward()

    loss_lumen = parallel_cross_entropy(
        logits,
        target,
        0.0,
        False,
        None,
        -100,
    )
    loss_lumen.mean().backward()

    grad_snr = compute_snr(logits_ref.grad, logits.grad)
    assert (
        grad_snr > 15
    ), f"Cross-entropy bwd grad SNR: {grad_snr:.1f} dB (expected > 15)"


# ===================================================================
# Label smoothing
# ===================================================================


@pytest.mark.parametrize("batch,vocab", BATCH_VOCAB, ids=BATCH_VOCAB_IDS)
def test_cross_entropy_label_smoothing(batch, vocab):
    """Cross-entropy with label_smoothing > 0."""
    seq_len = 1
    dtype = torch.bfloat16
    label_smoothing = 0.1

    logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=dtype) * 0.1
    target = torch.randint(0, vocab, (batch, seq_len), device="cuda", dtype=torch.long)

    loss_ref = cross_entropy_ref(
        logits, target, label_smoothing=label_smoothing
    ).reshape(batch, seq_len)
    loss_lumen = parallel_cross_entropy(
        logits,
        target,
        label_smoothing,
        False,
        None,
        -100,
    )

    snr = compute_snr(loss_ref.float(), loss_lumen.float())
    assert snr > 15, f"Cross-entropy label_smoothing SNR: {snr:.1f} dB (expected > 15)"


# ===================================================================
# reduce_loss=True (scalar mean)
# ===================================================================


@pytest.mark.parametrize("batch,vocab", BATCH_VOCAB, ids=BATCH_VOCAB_IDS)
def test_cross_entropy_reduce_loss(batch, vocab):
    """reduce_loss=True returns scalar mean; kernel uses different gradient scaling."""
    seq_len = 1
    dtype = torch.bfloat16

    logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=dtype) * 0.1
    logits.requires_grad_(True)
    target = torch.randint(0, vocab, (batch, seq_len), device="cuda", dtype=torch.long)
    logits_ref = logits.detach().clone().requires_grad_(True)

    V = vocab
    loss_ref_scalar = torch.nn.functional.cross_entropy(
        logits_ref.reshape(-1, V),
        target.reshape(-1),
        reduction="mean",
    )
    loss_ref_scalar.backward()

    loss_lumen = parallel_cross_entropy(
        logits,
        target,
        0.0,
        True,
        None,
        -100,
    )
    assert loss_lumen.dim() == 0, "reduce_loss=True should return scalar"
    loss_lumen.backward()

    snr_loss = compute_snr(
        loss_ref_scalar.detach().float().unsqueeze(0),
        loss_lumen.detach().float().unsqueeze(0),
    )
    assert snr_loss > 15, f"reduce_loss fwd SNR: {snr_loss:.1f} dB"

    grad_snr = compute_snr(logits_ref.grad, logits.grad)
    assert grad_snr > 12, f"reduce_loss bwd grad SNR: {grad_snr:.1f} dB"


# ===================================================================
# ignore_idx with actual ignored targets
# ===================================================================


@pytest.mark.parametrize("batch,vocab", BATCH_VOCAB, ids=BATCH_VOCAB_IDS)
def test_cross_entropy_ignore_idx(batch, vocab):
    """Targets set to ignore_idx should produce zero loss and zero gradients."""
    seq_len = 1
    dtype = torch.bfloat16
    ignore_idx = -100

    logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=dtype) * 0.1
    logits.requires_grad_(True)
    target = torch.randint(0, vocab, (batch, seq_len), device="cuda", dtype=torch.long)
    # Set ~25% of targets to ignore_idx
    mask = torch.rand(batch, seq_len, device="cuda") < 0.25
    target[mask] = ignore_idx

    logits_ref = logits.detach().clone().requires_grad_(True)

    loss_ref = cross_entropy_ref(logits_ref, target, ignore_idx=ignore_idx).reshape(
        batch, seq_len
    )
    loss_ref.mean().backward()

    loss_lumen = parallel_cross_entropy(
        logits,
        target,
        0.0,
        False,
        None,
        ignore_idx,
    )
    loss_lumen.mean().backward()

    # Ignored positions should have zero loss
    ignored_loss = loss_lumen[mask]
    if ignored_loss.numel() > 0:
        assert (ignored_loss == 0).all(), "Ignored positions should have zero loss"

    snr = compute_snr(loss_ref.float(), loss_lumen.float())
    assert snr > 15, f"ignore_idx fwd SNR: {snr:.1f} dB"

    grad_snr = compute_snr(logits_ref.grad, logits.grad)
    assert grad_snr > 12, f"ignore_idx bwd grad SNR: {grad_snr:.1f} dB"


# ===================================================================
# seq_len > 1
# ===================================================================


@pytest.mark.parametrize("seq_len", [8, 128], ids=["sq8", "sq128"])
def test_cross_entropy_seq_len(seq_len):
    """Multi-token sequences: parallel_cross_entropy expects [B, SQ, V]."""
    batch, vocab = 32, 32000
    dtype = torch.bfloat16

    logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=dtype) * 0.1
    logits.requires_grad_(True)
    target = torch.randint(0, vocab, (batch, seq_len), device="cuda", dtype=torch.long)
    logits_ref = logits.detach().clone().requires_grad_(True)

    loss_ref = cross_entropy_ref(logits_ref, target).reshape(batch, seq_len)
    loss_ref.mean().backward()

    loss_lumen = parallel_cross_entropy(
        logits,
        target,
        0.0,
        False,
        None,
        -100,
    )
    loss_lumen.mean().backward()

    assert loss_lumen.shape == (batch, seq_len)
    snr = compute_snr(loss_ref.float(), loss_lumen.float())
    assert snr > 15, f"seq_len={seq_len} fwd SNR: {snr:.1f} dB"

    grad_snr = compute_snr(logits_ref.grad, logits.grad)
    assert grad_snr > 12, f"seq_len={seq_len} bwd grad SNR: {grad_snr:.1f} dB"


# ===================================================================
# is_cg_capturable=True
# ===================================================================


def test_cross_entropy_cg_capturable():
    """is_cg_capturable=True skips scalar grad_output optimization in backward."""
    batch, seq_len, vocab = 32, 1, 32000
    dtype = torch.bfloat16

    logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=dtype) * 0.1
    logits.requires_grad_(True)
    target = torch.randint(0, vocab, (batch, seq_len), device="cuda", dtype=torch.long)
    logits_ref = logits.detach().clone().requires_grad_(True)

    cross_entropy_ref(logits_ref, target).mean().backward()

    loss_lumen = parallel_cross_entropy(
        logits,
        target,
        0.0,
        False,
        None,
        -100,
        True,
    )
    loss_lumen.mean().backward()

    grad_snr = compute_snr(logits_ref.grad, logits.grad)
    assert grad_snr > 15, f"cg_capturable bwd grad SNR: {grad_snr:.1f} dB"


# ===================================================================
# Non-contiguous inputs
# ===================================================================


def test_cross_entropy_non_contiguous():
    """Non-contiguous logits — implementation has .contiguous() fallback."""
    batch, seq_len, vocab = 32, 2, 32000
    dtype = torch.bfloat16

    logits_base = torch.randn(seq_len, batch, vocab, device="cuda", dtype=dtype)
    logits = logits_base.transpose(0, 1).requires_grad_(True)
    assert not logits.is_contiguous()

    target = torch.randint(0, vocab, (batch, seq_len), device="cuda", dtype=torch.long)

    logits_ref = logits.detach().clone().contiguous().requires_grad_(True)
    cross_entropy_ref(logits_ref, target).mean().backward()

    loss_lumen = parallel_cross_entropy(
        logits,
        target,
        0.0,
        False,
        None,
        -100,
    )
    loss_lumen.mean().backward()

    grad_snr = compute_snr(logits_ref.grad, logits.grad)
    assert grad_snr > 15, f"non-contiguous bwd grad SNR: {grad_snr:.1f} dB"


# ===================================================================
# Small batch edge case
# ===================================================================


def test_cross_entropy_batch1():
    """batch=1 edge case."""
    batch, seq_len, vocab = 1, 1, 32000
    dtype = torch.bfloat16

    logits = torch.randn(batch, seq_len, vocab, device="cuda", dtype=dtype) * 0.1
    logits.requires_grad_(True)
    target = torch.randint(0, vocab, (batch, seq_len), device="cuda", dtype=torch.long)
    logits_ref = logits.detach().clone().requires_grad_(True)

    loss_ref = cross_entropy_ref(logits_ref, target).reshape(batch, seq_len)
    loss_ref.mean().backward()

    loss_lumen = parallel_cross_entropy(
        logits,
        target,
        0.0,
        False,
        None,
        -100,
    )
    loss_lumen.mean().backward()

    snr = compute_snr(loss_ref.float(), loss_lumen.float())
    assert snr > 15, f"batch=1 fwd SNR: {snr:.1f} dB"
    grad_snr = compute_snr(logits_ref.grad, logits.grad)
    assert grad_snr > 12, f"batch=1 bwd grad SNR: {grad_snr:.1f} dB"
