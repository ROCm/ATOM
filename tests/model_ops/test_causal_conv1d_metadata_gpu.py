"""GPU checks for the causal_conv1d program map built by Triton."""

import types

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

import atom.model_ops.mamba_ops.causal_conv1d as conv_mod
from atom.model_ops.mamba_ops.causal_conv1d import (
    PAD_SLOT_ID,
    causal_conv1d_fn,
    compute_causal_conv1d_metadata,
)

DEV = "cuda"
LENS = [0, 1, 7, 8, 9, 300, 1025, 4]


def _qsl(lens, dtype=torch.int32):
    return torch.tensor([0] + lens, dtype=dtype, device=DEV).cumsum(0).to(dtype)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_map_matches_host_reference(dtype):
    _, batch_ptr, offset_ptr = compute_causal_conv1d_metadata(
        _qsl(LENS, dtype), sum(LENS)
    )
    nums = [-(-n // 8) for n in LENS]
    seqs = [s for s, n in enumerate(nums) for _ in range(n)]
    chunks = [c for n in nums for c in range(n)]
    n = len(seqs)
    assert batch_ptr[:n].tolist() == seqs
    assert offset_ptr[:n].tolist() == chunks
    assert (batch_ptr[n:] == PAD_SLOT_ID).all()
    assert (offset_ptr[n:] == PAD_SLOT_ID).all()


def test_short_bound_stays_in_bounds(monkeypatch):
    # num_tokens=1 sizes cdiv(1, 8) + 1 = 2 programs for a 24-token sequence's
    # 3 chunks. The map is placed between sentinels, which must survive.
    sentinel = 12345
    buf = torch.full((2, 2 + 2), sentinel, dtype=torch.int32, device=DEV)
    monkeypatch.setattr(torch, "empty", lambda *a, **k: buf[:, 1:3])
    compute_causal_conv1d_metadata(_qsl([24]), 1)
    monkeypatch.undo()
    assert buf[:, 0].tolist() == [sentinel] * 2
    assert buf[:, 3].tolist() == [sentinel] * 2
    assert buf[0, 1:3].tolist() == [0, 0]
    assert buf[1, 1:3].tolist() == [0, 1]


@pytest.mark.parametrize("variant", ["none", "exact", "short", "pad_slot_id"])
def test_conv_matches_without_metadata(variant, monkeypatch):
    torch.manual_seed(0)
    kd, vd, width = 128, 256, 4
    dim, B, T = 2 * kd + vd, len(LENS), sum(LENS)
    qsl = _qsl(LENS)
    x = torch.randn(T + 5, dim, device=DEV, dtype=torch.bfloat16).T
    w = torch.randn(dim, width, device=DEV, dtype=torch.bfloat16)
    states0 = torch.randn(B + 3, dim, width - 1, device=DEV, dtype=torch.bfloat16)
    idx = torch.arange(B, dtype=torch.int32, device=DEV) + 2
    has_init = torch.arange(B, device=DEV) % 2 == 0

    def run(metadata, pad_slot_id=PAD_SLOT_ID):
        states = states0.clone()
        q, k, v = causal_conv1d_fn(
            x,
            w,
            None,
            activation="silu",
            conv_states=states,
            has_initial_state=has_init,
            cache_indices=idx,
            query_start_loc=qsl,
            k_dim_size=kd,
            v_dim_size=vd,
            metadata=metadata,
            pad_slot_id=pad_slot_id,
        )
        return q[:T], k[:T], v[:T], states

    def meta(num_tokens):
        d, b, o = compute_causal_conv1d_metadata(qsl, num_tokens)
        return types.SimpleNamespace(nums_dict=d, batch_ptr=b, token_chunk_offset_ptr=o)

    def reused(metadata):
        # A map built for x's full width must be used as given, never rebuilt.
        def no_rebuild(*args, **kwargs):
            raise AssertionError("causal_conv1d_fn rebuilt a sufficient map")

        monkeypatch.setattr(conv_mod, "compute_causal_conv1d_metadata", no_rebuild)
        return run(metadata)

    expected = run(None)
    got = {
        "none": lambda: run(None),
        "exact": lambda: reused(meta(x.shape[1])),
        # The launcher rebuilds a map whose bound is short of x.
        "short": lambda: run(meta(1)),
        # Surplus programs carry -1 and must exit whatever pad_slot_id is.
        "pad_slot_id": lambda: run(None, pad_slot_id=-2),
    }[variant]()
    for a, b in zip(expected, got):
        assert torch.equal(a, b)
