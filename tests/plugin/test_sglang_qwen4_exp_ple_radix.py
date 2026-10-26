"""Flash PLE history under SGLang's mamba radix cache (extra_buffer).

The tree caches the ping-pong track slot, and SGLang moves a cached slot into
the working slot on a hit. Flash keeps its PLE history in plugin-owned buffers,
so it has to join both halves of that contract: ride the slot lifecycle, and be
snapshotted at the same boundary the GDN state is.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpPLEMetadata
from atom.model_ops.qwen4_exp.ple_layer import Qwen4ExpPLELayer
from atom.plugin.sglang.qwen4_exp_bridge import (
    _ple_decode_track_slots,
    _ple_track_indices,
    _PLESlotState,
)

CHANNELS = 6
STATE_LEN = 4
NGRAM_WIDTH = 2
EOS = 7


def _model(slots: int = 8):
    return SimpleNamespace(
        _atom_ple_conv_state=torch.zeros(slots, CHANNELS, STATE_LEN),
        _atom_ple_ngram_state=torch.full((slots, NGRAM_WIDTH), EOS, dtype=torch.int64),
    )


def _metadata(model, **track):
    return Qwen4ExpPLEMetadata(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        ngram_state=model._atom_ple_ngram_state,
        state_indices_in=torch.tensor([0], dtype=torch.int32),
        state_indices_out=torch.tensor([0], dtype=torch.int32),
        has_initial_state=torch.ones(1, dtype=torch.bool),
        conv_state=model._atom_ple_conv_state,
        **track,
    )


def test_slot_state_copies_both_buffers_on_a_hit():
    """A prefix hit hands a cached slot to the working slot; PLE must follow."""
    model = _model()
    model._atom_ple_conv_state[5] = 3.0
    model._atom_ple_ngram_state[5] = 11
    sibling = _PLESlotState(model, EOS)

    sibling.copy_slots(torch.tensor([5]), torch.tensor([2]))

    assert torch.equal(model._atom_ple_conv_state[2], model._atom_ple_conv_state[5])
    assert torch.equal(model._atom_ple_ngram_state[2], model._atom_ple_ngram_state[5])
    assert model._atom_ple_conv_state[1].abs().sum() == 0


def test_slot_state_reset_restores_the_cold_history():
    """A recycled slot must look unused, not like its previous tenant."""
    model = _model()
    model._atom_ple_conv_state[3] = 2.0
    model._atom_ple_ngram_state[3] = 11

    _PLESlotState(model, EOS).reset_slots(torch.tensor([3]))

    assert model._atom_ple_conv_state[3].abs().sum() == 0
    assert torch.equal(
        model._atom_ple_ngram_state[3], torch.full((NGRAM_WIDTH,), EOS).long()
    )


def test_slot_state_follows_a_regrown_buffer():
    """The buffers are allocated lazily; the sibling must not pin the old one."""
    model = _model(slots=4)
    sibling = _PLESlotState(model, EOS)
    model._atom_ple_conv_state = torch.zeros(16, CHANNELS, STATE_LEN)
    model._atom_ple_ngram_state = torch.full((16, NGRAM_WIDTH), EOS, dtype=torch.int64)
    model._atom_ple_conv_state[12] = 5.0

    sibling.copy_slots(torch.tensor([12]), torch.tensor([15]))

    assert model._atom_ple_conv_state[15].eq(5.0).all()


def test_slot_state_round_trips_through_host_memory():
    model = _model()
    model._atom_ple_conv_state[6] = 4.0
    model._atom_ple_ngram_state[6] = 9
    sibling = _PLESlotState(model, EOS)
    indices = torch.tensor([6])

    saved = sibling.get_cpu_slots(indices)
    sibling.reset_slots(indices)
    sibling.load_cpu_slots(saved, indices)

    assert model._atom_ple_conv_state[6].eq(4.0).all()
    assert model._atom_ple_ngram_state[6].eq(9).all()


def _extend_batch(aligned, track_indices, mask):
    return SimpleNamespace(
        mamba_track_mask=torch.tensor(mask),
        mamba_track_indices=torch.tensor(track_indices, dtype=torch.int32),
        mamba_track_aligned_lens=lambda: torch.tensor(aligned),
    )


def test_extend_track_window_ends_at_the_aligned_boundary():
    """The snapshot is the window ending at the boundary, not at the chunk end."""
    qsl = torch.tensor([0, 10, 20], dtype=torch.int32)
    fb = _extend_batch(aligned=[8, 6], track_indices=[4, 5], mask=[True, True])

    dst, conv_idx, ngram_idx = _ple_track_indices(fb, qsl, STATE_LEN, NGRAM_WIDTH)

    assert dst.tolist() == [4, 5]
    # Request 0 starts at 0 and is tracked at 8; request 1 starts at 10.
    assert conv_idx.tolist() == [[4, 5, 6, 7], [12, 13, 14, 15]]
    assert ngram_idx.tolist() == [[6, 7], [14, 15]]


def test_extend_track_skips_untracked_rows():
    qsl = torch.tensor([0, 10, 20], dtype=torch.int32)
    fb = _extend_batch(aligned=[8, 6], track_indices=[4, 5], mask=[False, True])

    dst, conv_idx, _ = _ple_track_indices(fb, qsl, STATE_LEN, NGRAM_WIDTH)

    assert dst.tolist() == [5]
    assert conv_idx.tolist() == [[12, 13, 14, 15]]


def test_extend_track_is_absent_without_a_mask():
    qsl = torch.tensor([0, 10], dtype=torch.int32)
    fb = SimpleNamespace(mamba_track_mask=None, mamba_track_indices=None)
    assert _ple_track_indices(fb, qsl, STATE_LEN, NGRAM_WIDTH) is None


def test_extend_snapshot_reads_the_boundary_window():
    model = _model()
    conv_input = torch.arange(10 * CHANNELS, dtype=torch.float32).view(10, CHANNELS)
    input_ids = torch.arange(100, 110)
    md = _metadata(
        model,
        track_dst=torch.tensor([5]),
        track_conv_indices=torch.tensor([[4, 5, 6, 7]]),
        track_ngram_indices=torch.tensor([[6, 7]]),
    )

    Qwen4ExpPLELayer._track_radix_state(md, input_ids, conv_input)

    # [rows, state_len, channels] stored as the pool's [slot, channels, state_len].
    torch.testing.assert_close(model._atom_ple_conv_state[5], conv_input[4:8].T)
    assert model._atom_ple_ngram_state[5].tolist() == [106, 107]
    assert model._atom_ple_conv_state[4].abs().sum() == 0


def test_decode_snapshot_copies_the_working_slot():
    """One token per step, so the working slot already is the boundary state."""
    model = _model()
    model._atom_ple_conv_state[1] = 6.0
    model._atom_ple_ngram_state[1] = 13
    md = _metadata(model, track_dst=torch.tensor([7]), track_src=torch.tensor([1]))

    Qwen4ExpPLELayer._track_radix_state(md, torch.zeros(1, dtype=torch.long), None)

    assert model._atom_ple_conv_state[7].eq(6.0).all()
    assert model._atom_ple_ngram_state[7].eq(13).all()


def test_snapshot_is_a_no_op_without_a_destination():
    model = _model()
    md = _metadata(model)
    Qwen4ExpPLELayer._track_radix_state(md, torch.zeros(1, dtype=torch.long), None)
    assert model._atom_ple_conv_state.abs().sum() == 0


@pytest.mark.parametrize("pad_slot", [-1])
def test_decode_track_slots_drop_graph_pad_rows(pad_slot):
    """Graph pads park dead rows on PAD_SLOT_ID; indexing with it would wrap."""
    fb = SimpleNamespace(
        mamba_track_mask=torch.tensor([True, True]),
        mamba_track_indices=torch.tensor([6, pad_slot], dtype=torch.int32),
    )
    dst, src = _ple_decode_track_slots(fb, torch.tensor([2, 3], dtype=torch.int32))
    assert dst.tolist() == [6] and src.tolist() == [2]


def _layer_stub(model, advance_conv_by):
    """Minimal stand-in for the parts of the PLE layer `forward_with_state` uses.

    `_short_conv` advances the working slot the way the real kernel does, so a
    snapshot taken before it would capture the previous token's window.
    """
    return SimpleNamespace(
        ple_embedding=SimpleNamespace(eos_token_id=EOS, ngram_size=NGRAM_WIDTH + 1),
        norm_conv=None,
        gated_memory=lambda *a, **k: torch.zeros(1, 1, CHANNELS),
        _apply_norm=lambda norm, x: x,
        _short_conv=lambda inputs, qsl, state, idx_in, idx_out, init, accepted: (
            state.__setitem__(idx_out.long(), advance_conv_by),
            torch.zeros(1, CHANNELS),
        )[1],
        _track_radix_state=Qwen4ExpPLELayer._track_radix_state,
    )


def test_decode_snapshot_runs_after_both_halves_advance(monkeypatch):
    """The window and the n-gram ids must describe the same token position."""
    from atom.model_ops.qwen4_exp import ple_layer as layer_mod

    model = _model()
    advanced_ngram = 21

    def fake_advance(input_ids, qsl, state, idx_in, idx_out, *a, **k):
        state[idx_out.long()] = advanced_ngram
        return torch.zeros(1, NGRAM_WIDTH)

    monkeypatch.setattr(layer_mod, "advance_ngram_state", fake_advance)
    md = _metadata(model, track_dst=torch.tensor([7]), track_src=torch.tensor([0]))

    Qwen4ExpPLELayer.forward_with_state(
        _layer_stub(model, advance_conv_by=5.0),
        torch.zeros(1, 1, CHANNELS),
        torch.zeros(1, dtype=torch.long),
        md,
    )

    assert model._atom_ple_conv_state[7].eq(5.0).all()
    assert model._atom_ple_ngram_state[7].eq(advanced_ngram).all()


def test_graph_epilogue_uses_live_slots_not_the_last_built_metadata():
    """Replay does not rebuild PLE metadata, so the stale one would name the
    slots of whatever forward ran eagerly before -- usually a prefill."""
    from atom.plugin.sglang import qwen4_exp_bridge as bridge

    model = _model()
    model._atom_ple_conv_state[3] = 8.0
    model._atom_ple_ngram_state[3] = 17
    live = SimpleNamespace(
        linear_attn_backend=SimpleNamespace(
            forward_metadata=SimpleNamespace(
                mamba_cache_indices=torch.tensor([3, 4], dtype=torch.int32)
            )
        )
    )
    fb = SimpleNamespace(
        batch_size=2,
        forward_mode=SimpleNamespace(is_decode_or_idle=lambda: True),
        mamba_track_mask=torch.tensor([True, False]),
        mamba_track_indices=torch.tensor([6, 7], dtype=torch.int32),
    )

    bridge.track_ple_radix_slots(fb, attn_backend=live, model=model)

    assert model._atom_ple_conv_state[6].eq(8.0).all()
    assert model._atom_ple_ngram_state[6].eq(17).all()
    assert model._atom_ple_conv_state[7].abs().sum() == 0


def test_decode_track_slots_absent_when_nothing_is_tracked():
    fb = SimpleNamespace(
        mamba_track_mask=torch.tensor([False, False]),
        mamba_track_indices=torch.tensor([6, 7], dtype=torch.int32),
    )
    assert _ple_decode_track_slots(fb, torch.tensor([2, 3], dtype=torch.int32)) == (
        None,
        None,
    )
