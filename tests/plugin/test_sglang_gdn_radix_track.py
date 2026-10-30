"""SGLang mamba radix cache (extra_buffer) tracking on the plugin GDN path.

The plugin replaces SGLang's GDN kernels, so it has to make the track calls
SGLang's own backend makes around them. These tests drive
`SGLangGatedDeltaNet._track_mamba_states` against a stand-in linear backend and
check which helper receives which buffers, that extend and decode each have a
single owner, plus ReplaySSM staying on under the radix cache.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

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


def test_decode_tracks_through_the_radix_slot_helper(backend, monkeypatch):
    lb = backend(SimpleNamespace(mamba_cache_indices=torch.tensor([2, 3])))
    seen = []
    monkeypatch.setattr(
        attention_gdn,
        "_track_mamba_radix_slots",
        lambda fb, layer_id: seen.append((fb, layer_id)),
    )
    fb = SimpleNamespace(
        mamba_track_mask=torch.tensor([True, False]), forward_mode=_Mode("decode")
    )
    _impl()._track_mamba_states(fb, torch.zeros(2, CONV_DIM))
    # One owner for decode: the helper that also backs the post-replay
    # epilogue, never SGLang's in-graph per-layer call.
    assert seen == [(fb, LAYER_NUM)]
    assert lb.calls == []


def test_radix_slot_helper_leaves_extend_to_chunk_states(backend):
    """A finished-state row copy on extend would overwrite the chunk-boundary
    snapshot an unaligned track point needs."""
    lb = backend(
        SimpleNamespace(
            has_mamba_track_mask=True,
            mamba_cache_indices=torch.tensor([1, 2]),
            mamba_track_indices=torch.tensor([6, 7]),
            track_ssm_h_src=torch.empty(0, dtype=torch.long),
        )
    )
    lb.layer_cache.conv[0][1:3] = 1.0
    lb.layer_cache.temporal[1:3] = 1.0
    fb = SimpleNamespace(
        mamba_track_mask=torch.tensor([True, True]),
        forward_mode=_Mode("extend"),
        batch_size=2,
    )
    attention_gdn._track_mamba_radix_slots(fb, LAYER_NUM)
    assert lb.calls == []
    assert lb.layer_cache.conv[0][6:].abs().sum() == 0
    assert lb.layer_cache.temporal[6:].abs().sum() == 0


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


def test_replayssm_follows_speculation_under_radix_cache(monkeypatch):
    # The commit hook materializes the tracked state, so the radix cache no
    # longer turns the ring off.
    monkeypatch.delenv("ATOM_ENABLE_REPLAYSSM", raising=False)
    _server(monkeypatch, disable_radix_cache=False, speculative_algorithm="EAGLE")
    assert gdn_replayssm.replayssm_enabled() is True
    _server(monkeypatch, disable_radix_cache=False, speculative_algorithm=None)
    assert gdn_replayssm.replayssm_enabled() is False


def test_replayssm_env_override_holds_under_radix_cache(monkeypatch):
    _server(monkeypatch, disable_radix_cache=False, speculative_algorithm="EAGLE")
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "1")
    assert gdn_replayssm.replayssm_enabled() is True
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "0")
    assert gdn_replayssm.replayssm_enabled() is False


def _patched_decode_runner(monkeypatch):
    """Install the plugin's DecodeCudaGraphRunner patch on a stand-in class."""
    from atom.plugin.sglang import register

    class _Runner:
        def __init__(self):
            self.model_runner = SimpleNamespace(attn_backend="backend")

        def capture_one_shape(self, *args, **kwargs):
            return None

        def execute(self, forward_batch, *args, **kwargs):
            return ("replayed", forward_batch)

    forward_context = ModuleType("sglang.srt.model_executor.forward_context")
    forward_context.ForwardContext = object
    forward_context.forward_context = None
    forward_context.has_forward_context = lambda: True
    runner_module = ModuleType(
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner"
    )
    runner_module.DecodeCudaGraphRunner = _Runner
    monkeypatch.setitem(sys.modules, forward_context.__name__, forward_context)
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    register._install_decode_graph_forward_context_patch()
    assert _Runner._atom_mamba_track_epilogue
    return _Runner()


def test_decode_graph_epilogue_snapshots_after_replay(monkeypatch):
    seen = []
    monkeypatch.setattr(
        attention_gdn,
        "track_mamba_radix_after_decode_graph",
        lambda fb, attn_backend: seen.append((fb, attn_backend)),
    )
    runner = _patched_decode_runner(monkeypatch)
    assert runner.execute("fb") == ("replayed", "fb")
    assert seen == [("fb", "backend")]


def test_decode_graph_epilogue_failure_is_not_swallowed(monkeypatch):
    """The epilogue is the only snapshot writer for a replayed decode; a
    failure must not return the step as if the track slot had been written."""

    def fail(fb, attn_backend):
        raise RuntimeError("track launch failed")

    monkeypatch.setattr(attention_gdn, "track_mamba_radix_after_decode_graph", fail)
    runner = _patched_decode_runner(monkeypatch)
    with pytest.raises(RuntimeError, match="track launch failed"):
        runner.execute("fb")
