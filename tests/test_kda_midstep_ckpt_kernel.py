# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's midstep checkpoints through `write_state_checkpoints`.

One launch per layer writes every checkpoint of a step: an interior target's
state from its row of FlashKDA's `snapshot_out` (`h_rows`), an end target's
from the runtime slot, and each one's conv window from the step's conv input.
The pool's speculation-rollback columns are left as they were. GDN's use of
the same kernel, from a chunk kernel's per-chunk states, is checked alongside.

Runs on a GPU, or on the CPU under `TRITON_INTERPRET=1`; skipped without
Triton, as on ATOM's CPU job.
"""

import os

import pytest
import torch

from atom.models.kimi_k3.midstep import kda_snapshots

CKPT = pytest.importorskip(
    "atom.model_ops.fla_ops.state_checkpoint",
    reason="the checkpoint kernel needs Triton",
    exc_type=ImportError,
)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif os.environ.get("TRITON_INTERPRET") == "1":
    DEVICE = "cpu"
else:
    pytest.skip("needs a GPU or TRITON_INTERPRET=1", allow_module_level=True)

CONV_LEN, ROLLBACK = 3, 2
SLOTS, H, V, K, D = 9, 2, 4, 4, 6


def _pools(conv_transposed: bool):
    g = torch.Generator().manual_seed(0)
    ssm = torch.randn(SLOTS, H, V, K, generator=g)
    if conv_transposed:
        # As the KDA layer sees it: a view of a [slots, width, dim] allocation.
        conv = torch.randn(SLOTS, CONV_LEN + ROLLBACK, D, generator=g).transpose(1, 2)
    else:
        conv = torch.randn(SLOTS, D, CONV_LEN + ROLLBACK, generator=g)
    return ssm.to(DEVICE), conv.to(DEVICE)


@pytest.mark.parametrize("conv_transposed", [False, True])
def test_one_launch_writes_interior_and_end_targets(conv_transposed):
    """Row 0: interior targets at 32 and 64 and the anchor at its end, 96.
    Row 1, 48 tokens after it: one interior target at 32."""
    ssm, conv = _pools(conv_transposed)
    snaps = kda_snapshots(
        [[(4, 32), (5, 64), (6, 96)], [(7, 32)]],
        state_rows=[0, 1],
        num_cached_tokens=[0, 0],
        num_scheduled_tokens=[96, 48],
        runtime_slots=[1, 2],
        num_slots=SLOTS,
        conv_len=CONV_LEN,
        device=DEVICE,
    )
    assert snaps.targets == ((0, 32), (0, 64), (1, 32))
    g = torch.Generator().manual_seed(1)
    snap_out = torch.randn(len(snaps.targets), H, V, K, generator=g).to(DEVICE)
    mixed_qkv = torch.randn(144, D, generator=g).to(DEVICE)
    before_ssm, before_conv = ssm.clone(), conv.clone()

    CKPT.write_state_checkpoints(
        **snaps.checkpoint_args(snap_out, ssm, mixed_qkv, conv)
    )

    for i, slot in enumerate((4, 5, 7)):
        assert torch.equal(ssm[slot], snap_out[i])
    assert torch.equal(ssm[6], before_ssm[1]), "the anchor is the runtime slot's state"
    for slot, start in ((4, 32), (5, 64), (6, 96), (7, 96 + 32)):
        window = mixed_qkv[start - CONV_LEN : start].T
        assert torch.equal(conv[slot, :, :CONV_LEN], window), slot
        # Left as they were: a resumed prefill reads only the first CONV_LEN.
        assert torch.equal(conv[slot, :, CONV_LEN:], before_conv[slot, :, CONV_LEN:])
    for slot in (0, 1, 2, 3, 8):
        assert torch.equal(ssm[slot], before_ssm[slot])
        assert torch.equal(conv[slot], before_conv[slot])


def test_an_end_target_alone_needs_no_snapshot():
    ssm, conv = _pools(False)
    snaps = kda_snapshots(
        [[(4, 64)]],
        [0],
        [0],
        [64],
        [1],
        num_slots=SLOTS,
        conv_len=CONV_LEN,
        device=DEVICE,
    )
    mixed_qkv = torch.randn(64, D).to(DEVICE)
    before = ssm.clone()
    CKPT.write_state_checkpoints(**snaps.checkpoint_args(None, ssm, mixed_qkv, conv))
    assert torch.equal(ssm[4], before[1])
    assert torch.equal(conv[4, :, :CONV_LEN], mixed_qkv[61:64].T)


def test_gdn_chunk_states_still_read_by_chunk():
    """GDN's path, unchanged: target i reads h[chunk_offsets[row] + off // 64]."""
    ssm, conv = _pools(False)
    h = torch.randn(1, 6, H, V, K).to(DEVICE)
    x = torch.randn(200, D).to(DEVICE)
    i32 = {"dtype": torch.int32, "device": DEVICE}
    CKPT.write_state_checkpoints(
        h,
        ssm,
        x,
        conv,
        torch.tensor([0, 1], **i32),
        torch.tensor([4, 5], **i32),
        torch.tensor([64, 64], **i32),
        torch.tensor([0, 0], **i32),
        torch.tensor([1, 2], **i32),
        torch.tensor([0, 2, 6], **i32),
        torch.tensor([0, 128, 200], **i32),
        64,
        CONV_LEN,
    )
    assert torch.equal(ssm[4], h[0, 1])
    assert torch.equal(ssm[5], h[0, 3])
    assert torch.equal(conv[5, :, :CONV_LEN], x[128 + 61 : 128 + 64].T)
