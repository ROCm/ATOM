"""Check the FlyDSL intermediate state used by RTP prefix reuse."""

import pytest
import torch


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="ROCm GPU required",
)
def test_flydsl_snapshot_matches_prefix_final_state():
    from atom.model_ops.fla_ops.gdn_flydsl import (
        build_prefill_metadata,
        prefill,
    )

    device = torch.device("cuda")
    torch.manual_seed(7)
    length, prefix = 1062, 1024
    q = torch.randn(1, length, 8, 128, device=device, dtype=torch.bfloat16) / 16
    k = torch.randn_like(q) / 16
    v = torch.randn(1, length, 24, 128, device=device, dtype=torch.bfloat16) / 16
    g = -torch.rand(1, length, 24, device=device, dtype=torch.bfloat16) / 8
    beta = torch.rand(1, length, 24, device=device, dtype=torch.bfloat16)
    pool = torch.zeros(1, 24, 128, 128, device=device, dtype=torch.float32)
    cu_full = torch.tensor([0, length], device=device, dtype=torch.int32)
    _, _, snapshots = prefill(
        q,
        k,
        v,
        g,
        beta,
        pool,
        cu_full,
        build_prefill_metadata((length,), cu_full),
        True,
    )
    cu_prefix = torch.tensor([0, prefix], device=device, dtype=torch.int32)
    _, final, _ = prefill(
        q[:, :prefix],
        k[:, :prefix],
        v[:, :prefix],
        g[:, :prefix],
        beta[:, :prefix],
        pool,
        cu_prefix,
        build_prefill_metadata((prefix,), cu_prefix),
        False,
    )
    torch.testing.assert_close(
        snapshots[:, prefix // 64].transpose(-1, -2).float(),
        final.float(),
        rtol=1e-2,
        atol=2e-2,
    )
    from atom.model_ops.fla_ops.gdn_flydsl import prepare_prefill_state
    from atom.model_ops.fla_ops.state_checkpoint import write_state_checkpoints

    elements = 24 * 128 * 128
    padded = torch.full((4, elements + 64), -99.0, device=device)
    state = padded[:, :elements].view(4, 24, 128, 128).transpose(-1, -2)
    conv = torch.zeros(4, 16, 3, device=device, dtype=torch.bfloat16)
    values = lambda xs: torch.tensor(xs, device=device, dtype=torch.int32)
    write_state_checkpoints(
        snapshots,
        state,
        torch.zeros(length, 16, device=device, dtype=torch.bfloat16),
        conv,
        values([0]),
        values([1]),
        values([prefix]),
        values([0]),
        values([2]),
        values([0, 17]),
        cu_full,
        64,
        3,
        h_vk=True,
    )
    restored = prepare_prefill_state(
        state, values([1]), torch.ones(1, device=device, dtype=torch.bool)
    )
    torch.testing.assert_close(
        restored, final.transpose(-1, -2).float(), rtol=1e-2, atol=2e-2
    )
    assert torch.all(padded[:, elements:] == -99.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_checkpoint_writer_accepts_native_vk_snapshot_with_unequal_dims():
    from atom.model_ops.fla_ops.state_checkpoint import write_state_checkpoints

    device = "cuda"
    k, v = 64, 128
    logical = torch.arange(k * v, device=device, dtype=torch.float32).reshape(k, v)
    # AITER emits [V, K]; RTP's persistent pool is a logical [K, V] view
    # backed by VK-contiguous storage. Unequal dimensions catch swapped axes.
    snapshot = logical.transpose(-1, -2).contiguous().view(1, 1, 1, v, k)
    snapshot = snapshot.expand(1, 2, 1, v, k).contiguous()
    state = torch.zeros(2, 1, v, k, device=device).transpose(-1, -2)
    conv = torch.zeros(2, 4, 3, device=device)
    values = lambda items: torch.tensor(items, device=device, dtype=torch.int32)
    write_state_checkpoints(
        snapshot,
        state,
        torch.zeros(128, 4, device=device),
        conv,
        values([0]),
        values([1]),
        values([64]),
        values([0]),
        values([0]),
        values([0, 2]),
        values([0, 128]),
        64,
        3,
        h_vk=True,
    )
    torch.testing.assert_close(state[1, 0], logical)
