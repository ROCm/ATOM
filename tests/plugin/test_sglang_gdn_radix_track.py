"""SGLang mamba radix cache (extra_buffer) tracking on the plugin GDN path.

The plugin replaces SGLang's GDN kernels, so it has to make the track calls
SGLang's own backend makes around them. These tests drive
`SGLangGatedDeltaNet._track_mamba_states` against a stand-in linear backend and
check which helper receives which buffers, plus the ReplaySSM radix guard.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.fla_ops import chunk as chunk_mod
from atom.plugin.sglang.attention_backend import attention_gdn, gdn_replayssm
from atom.plugin.sglang.attention_backend.attention_gdn import (
    SGLangGatedDeltaNet,
    SGLangGDNForwardContext,
)

CONV_DIM = 6
STATE_LEN = 3
NUM_SLOTS = 8
LAYER_NUM = 5


class _Mode:
    def __init__(self, kind: str):
        self.kind = kind

    def is_decode_or_idle(self):
        return self.kind == "decode"

    def is_extend(self):
        return self.kind == "extend"


class _LinearBackend:
    def __init__(self, metadata):
        self.forward_metadata = metadata
        self.layer_cache = SimpleNamespace(
            conv=[torch.zeros(NUM_SLOTS, CONV_DIM, STATE_LEN)],
            temporal=torch.zeros(NUM_SLOTS, 2, 4, 4),
        )
        self.req_to_token_pool = SimpleNamespace(
            mamba2_layer_cache=lambda layer: self.layer_cache
        )
        self.calls = []

    def _track_mamba_state_decode(self, fb, conv, ssm, idx, layer_id):
        self.calls.append(("decode", conv, ssm, idx, layer_id))

    def _track_mamba_state_extend(self, fb, h, ssm, metadata):
        self.calls.append(("extend", h, ssm, metadata))


def _impl() -> SGLangGatedDeltaNet:
    impl = SGLangGatedDeltaNet.__new__(SGLangGatedDeltaNet)
    torch.nn.Module.__init__(impl)
    impl.layer_num = LAYER_NUM
    impl.gdn_flydsl_policy = None
    return impl


@pytest.fixture
def backend(monkeypatch):
    holder = {}

    def resolve(_fb):
        return SimpleNamespace(linear_attn_backend=holder["backend"])

    monkeypatch.setattr(
        SGLangGDNForwardContext, "_resolve_attn_backend", staticmethod(resolve)
    )

    def install(metadata):
        holder["backend"] = _LinearBackend(metadata)
        return holder["backend"]

    return install


def test_no_track_mask_is_a_no_op(backend):
    lb = backend(SimpleNamespace())
    fb = SimpleNamespace(mamba_track_mask=None, forward_mode=_Mode("decode"))
    _impl()._track_mamba_states(fb, torch.zeros(4, CONV_DIM))
    assert lb.calls == []


def test_decode_tracks_through_sglang_with_layer_id(backend):
    idx = torch.tensor([2, 3])
    lb = backend(SimpleNamespace(mamba_cache_indices=idx))
    fb = SimpleNamespace(
        mamba_track_mask=torch.tensor([True, False]), forward_mode=_Mode("decode")
    )
    _impl()._track_mamba_states(fb, torch.zeros(2, CONV_DIM))
    (call,) = lb.calls
    assert call[0] == "decode"
    assert call[1] is lb.layer_cache.conv[0]
    assert call[2] is lb.layer_cache.temporal
    assert call[3] is idx
    assert call[4] == LAYER_NUM


def _extend_metadata(track_conv_indices, dst, h_src_rows):
    return SimpleNamespace(
        has_mamba_track_mask=True,
        track_conv_indices=track_conv_indices,
        conv_states_mask_indices=dst,
        mamba_track_indices=torch.tensor([6, 7]),
        track_ssm_h_src=torch.arange(h_src_rows),
    )


def test_extend_copies_conv_window_from_pre_conv_input(backend, monkeypatch):
    h = torch.randn(1, 3, 2, 4, 4)
    monkeypatch.setattr(chunk_mod.ChunkGatedDeltaRuleFunction, "last_h", h)
    # Two tracked rows whose windows end at tokens 4 and 9.
    track = torch.tensor([[2, 3, 4], [7, 8, 9]])
    lb = backend(_extend_metadata(track, torch.tensor([6, 1]), 1))
    mixed_qkv = torch.arange(10 * CONV_DIM, dtype=torch.float32).view(10, CONV_DIM)
    fb = SimpleNamespace(
        mamba_track_mask=torch.tensor([True, True]), forward_mode=_Mode("extend")
    )

    _impl()._track_mamba_states(fb, mixed_qkv)

    conv = lb.layer_cache.conv[0]
    torch.testing.assert_close(conv[6], mixed_qkv[2:5].T)
    torch.testing.assert_close(conv[1], mixed_qkv[7:10].T)
    assert conv[[0, 2, 3, 4, 5, 7]].abs().sum() == 0
    (call,) = lb.calls
    assert call[0] == "extend" and call[1] is h
    assert call[2] is lb.layer_cache.temporal
    # Consumed, so a later forward cannot read a stale `h`.
    assert chunk_mod.ChunkGatedDeltaRuleFunction.last_h is None


def test_extend_derives_conv_destination_when_backend_did_not(backend, monkeypatch):
    monkeypatch.setattr(
        chunk_mod.ChunkGatedDeltaRuleFunction, "last_h", torch.zeros(1, 1, 2, 4, 4)
    )
    lb = backend(_extend_metadata(torch.tensor([[1, 2, 3]]), None, 0))
    mixed_qkv = torch.randn(4, CONV_DIM)
    fb = SimpleNamespace(
        mamba_track_mask=torch.tensor([False, True]), forward_mode=_Mode("extend")
    )

    _impl()._track_mamba_states(fb, mixed_qkv)

    conv = lb.layer_cache.conv[0]
    torch.testing.assert_close(conv[7], mixed_qkv[1:4].T)
    assert conv[:7].abs().sum() == 0


def test_extend_without_kept_states_fails_closed(backend, monkeypatch):
    monkeypatch.setattr(chunk_mod.ChunkGatedDeltaRuleFunction, "last_h", None)
    backend(_extend_metadata(torch.empty(0, 3, dtype=torch.long), None, 1))
    fb = SimpleNamespace(
        mamba_track_mask=torch.tensor([True]), forward_mode=_Mode("extend")
    )
    with pytest.raises(RuntimeError, match="intermediate states"):
        _impl()._track_mamba_states(fb, torch.zeros(4, CONV_DIM))


def test_extend_metadata_keeps_states_only_when_tracking(monkeypatch):
    monkeypatch.setattr(attention_gdn, "_flydsl_prefill_metadata", lambda *a: None)
    monkeypatch.setattr(
        attention_gdn,
        "apply_gdn_pad_sentinels",
        lambda fb, idx, qsl, mode, bs: (idx, qsl),
    )

    def build(track: bool):
        fm = SimpleNamespace(
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            mamba_cache_indices=torch.tensor([1], dtype=torch.int32),
            has_mamba_track_mask=track,
        )

        class _ExtendMode:
            def is_target_verify(self):
                return False

            def is_decode_or_idle(self):
                return False

            def is_extend(self):
                return True

        fb = SimpleNamespace(
            forward_mode=_ExtendMode(),
            batch_size=1,
            extend_prefix_lens=torch.tensor([0]),
        )
        return SGLangGDNForwardContext._build_gdn_metadata(
            fb, SimpleNamespace(forward_metadata=fm)
        )

    assert build(True).keep_intermediate_states is True
    assert build(False).keep_intermediate_states is False


def _server(monkeypatch, **fields):
    args = SimpleNamespace(**fields)
    monkeypatch.setattr(gdn_replayssm, "_server_args", lambda: args)


def test_replayssm_follows_speculation_without_radix(monkeypatch):
    monkeypatch.delenv("ATOM_ENABLE_REPLAYSSM", raising=False)
    _server(monkeypatch, disable_radix_cache=True, speculative_algorithm="DFLASH")
    assert gdn_replayssm.replayssm_enabled() is True


def test_replayssm_is_off_under_radix_cache(monkeypatch):
    monkeypatch.delenv("ATOM_ENABLE_REPLAYSSM", raising=False)
    _server(monkeypatch, disable_radix_cache=False, speculative_algorithm="DFLASH")
    assert gdn_replayssm.replayssm_enabled() is False


def test_forcing_replayssm_under_radix_cache_is_rejected(monkeypatch):
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "1")
    _server(monkeypatch, disable_radix_cache=False, speculative_algorithm="DFLASH")
    with pytest.raises(RuntimeError, match="radix cache"):
        gdn_replayssm.replayssm_enabled()
