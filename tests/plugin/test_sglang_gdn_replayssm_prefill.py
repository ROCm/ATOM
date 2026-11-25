"""ReplaySSM cursor after a prefill: parked once per forward, no host read-back.

A prefill rewrites the GDN checkpoint of every request in the batch, so the
ring cursor of each slot is parked at 0 (``note_full_state_write``). The cursor
is one per slot, shared by every GDN layer, so the reset belongs to the
forward's GDN metadata rather than to each layer, and it must not read the
live-slot count back to the host.
"""

from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from atom.model_ops.attention_gdn import GatedDeltaNet
from atom.plugin.sglang.attention_backend import attention_gdn
from atom.plugin.sglang.attention_backend.attention_gdn import (
    SGLangGatedDeltaNet,
    SGLangGDNForwardContext,
)
from atom.plugin.sglang.attention_backend.gdn_replayssm import (
    note_full_state_write,
    replay_runtime,
    reset_replay_runtime_for_tests,
)
from atom.plugin.sglang.runtime import bind_current_forward_batch

NUM_SLOTS = 8


@pytest.fixture(autouse=True)
def _fresh_runtime(monkeypatch):
    # The conv1d program map is a Triton kernel; these tests run on CPU tensors.
    monkeypatch.setattr(
        attention_gdn,
        "compute_causal_conv1d_metadata",
        lambda query_start_loc, num_tokens: (None, None, None),
    )
    reset_replay_runtime_for_tests()
    yield
    reset_replay_runtime_for_tests()


_HOST_READ_OPS = (
    torch.ops.aten._local_scalar_dense.default,
    torch.ops.aten.nonzero.default,
    torch.ops.aten.masked_select.default,
)


def _reads_back(func, args) -> bool:
    if func in _HOST_READ_OPS:
        return True
    # Boolean-mask indexing sizes its output from a device-side count.
    return func is torch.ops.aten.index.Tensor and any(
        isinstance(index, torch.Tensor) and index.dtype == torch.bool
        for index in args[1]
    )


class _HostReadSpy(TorchDispatchMode):
    """Records the ops whose result size or value the host has to wait for."""

    def __init__(self):
        super().__init__()
        self.host_reads = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if _reads_back(func, args):
            self.host_reads.append(func)
        return func(*args, **(kwargs or {}))


def _boolean_mask_park(write_pos, slot_idx):
    """The per-layer version this replaced."""
    live = slot_idx[slot_idx >= 0].to(torch.int64)
    if live.numel():
        write_pos.index_fill_(0, live, 0)


def _cursor():
    # Nonzero everywhere, so a parked slot is visible.
    return torch.arange(1, NUM_SLOTS + 1, dtype=torch.int32)


@pytest.mark.parametrize(
    "slots",
    [[3, 0, 7], [5, -1, 2, -1], [-1, -1], [4, 4, 1], [NUM_SLOTS - 1]],
    ids=["live", "pad-rows", "all-pad", "repeated-slot", "last-slot"],
)
def test_park_matches_the_boolean_mask_version(slots):
    slot_idx = torch.tensor(slots, dtype=torch.int32)
    expected = _cursor()
    _boolean_mask_park(expected, slot_idx)
    replay_runtime().write_pos = _cursor()

    note_full_state_write(slot_idx)

    assert torch.equal(replay_runtime().write_pos, expected)


def test_park_does_not_read_back_to_the_host():
    slot_idx = torch.tensor([3, -1, 6], dtype=torch.int32)
    with _HostReadSpy() as old:
        _boolean_mask_park(_cursor(), slot_idx)
    # The spy does see the boolean-mask read-back...
    assert old.host_reads

    replay_runtime().write_pos = _cursor()
    with _HostReadSpy() as new:
        note_full_state_write(slot_idx)
    # ...and the park issues none.
    assert new.host_reads == []
    assert replay_runtime().write_pos.tolist() == [1, 2, 3, 0, 5, 6, 0, 8]


def test_park_is_a_no_op_before_the_ring_exists():
    note_full_state_write(torch.tensor([1, 2], dtype=torch.int32))
    assert replay_runtime().write_pos is None


def _batch(mode):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return SimpleNamespace(
        forward_mode=ForwardMode[mode],
        batch_size=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([0, 64], dtype=torch.int32),
        extend_seq_lens_cpu=[3, 2],
        mamba_track_mask=None,
    )


def _linear(cache_indices):
    return SimpleNamespace(
        forward_metadata=SimpleNamespace(
            query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
            mamba_cache_indices=cache_indices,
            logical_num_tokens=5,
        ),
        req_to_token_pool=None,
    )


def _run_gdn_layers(monkeypatch, forward_batch, num_layers):
    monkeypatch.setattr(
        GatedDeltaNet, "forward", lambda self, mixed_qkv, b, a, out, name: out
    )
    monkeypatch.setattr(
        SGLangGatedDeltaNet, "_track_mamba_states", lambda self, step, mixed_qkv: None
    )
    layer = SGLangGatedDeltaNet.__new__(SGLangGatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.dt_bias = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    tokens = 5
    with bind_current_forward_batch(forward_batch):
        for layer_id in range(num_layers):
            layer.forward(
                torch.zeros(tokens, 4),
                torch.zeros(tokens, 2),
                torch.zeros(tokens, 2),
                torch.zeros(tokens, 2, 4),
                f"layers.{layer_id}",
            )


def test_extend_parks_the_cursor_once_per_forward(monkeypatch):
    calls = []
    monkeypatch.setattr(attention_gdn, "replayssm_enabled", lambda: True)
    monkeypatch.setattr(attention_gdn, "note_full_state_write", calls.append)
    forward_batch = _batch("EXTEND")
    # The third row is past batch_size: not a slot of this batch.
    cache_indices = torch.tensor([6, 2, 0], dtype=torch.int32)

    metadata = SGLangGDNForwardContext._build_gdn_metadata(
        forward_batch, _linear(cache_indices)
    )
    _run_gdn_layers(monkeypatch, forward_batch, num_layers=36)

    assert metadata is not None
    assert len(calls) == 1
    # The batch's rows of SGLang's slot tensor, as the per-layer call read them.
    assert calls[0].data_ptr() == cache_indices.data_ptr()
    assert calls[0].tolist() == [6, 2]


@pytest.mark.parametrize(
    "mode, enabled, parked",
    [
        ("MIXED", True, [[6, 2]]),
        ("DECODE", True, []),
        ("TARGET_VERIFY", True, []),
        # Kept out by SGLang's is_extend() alone.
        ("DRAFT_EXTEND_V2", True, []),
        ("EXTEND", False, []),
    ],
    ids=["mixed", "decode", "target-verify", "draft-extend-v2", "replayssm-off"],
)
def test_only_a_replayssm_extend_parks_the_cursor(monkeypatch, mode, enabled, parked):
    calls = []
    monkeypatch.setattr(attention_gdn, "replayssm_enabled", lambda: enabled)
    monkeypatch.setattr(attention_gdn, "note_full_state_write", calls.append)

    SGLangGDNForwardContext._build_gdn_metadata(
        _batch(mode), _linear(torch.tensor([6, 2], dtype=torch.int32))
    )

    assert [slots.tolist() for slots in calls] == parked


def test_extend_metadata_parks_only_the_batch_slots(monkeypatch):
    monkeypatch.setattr(attention_gdn, "replayssm_enabled", lambda: True)
    replay_runtime().write_pos = _cursor()

    # Row 1 is a pad row; row 2 is past batch_size.
    SGLangGDNForwardContext._build_gdn_metadata(
        _batch("EXTEND"),
        _linear(torch.tensor([6, -1, 2], dtype=torch.int32)),
    )

    assert replay_runtime().write_pos.tolist() == [1, 2, 3, 4, 5, 6, 0, 8]


def test_reconstructed_extend_metadata_parks_the_cursor(monkeypatch):
    calls = []
    monkeypatch.setattr(attention_gdn, "replayssm_enabled", lambda: True)
    monkeypatch.setattr(attention_gdn, "note_full_state_write", calls.append)
    slots = torch.tensor([6, 2], dtype=torch.int32)
    monkeypatch.setattr(
        attention_gdn,
        "reconstruct_linear_metadata",
        lambda forward_batch, linear_backend: (
            torch.tensor([0, 3, 5], dtype=torch.int32),
            slots,
        ),
    )
    # SGLang published no GDN metadata: the slots come from the mamba pool.
    linear = SimpleNamespace(forward_metadata=None, req_to_token_pool=None)

    metadata = SGLangGDNForwardContext._build_gdn_metadata(_batch("EXTEND"), linear)

    assert metadata is not None
    assert len(calls) == 1
    assert calls[0].tolist() == [6, 2]
