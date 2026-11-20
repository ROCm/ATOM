# SPDX-License-Identifier: MIT
"""Native hook activation, layer ownership and prefill/width fallback."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from atom.models.deepseek_v41.mono import indexer_dispatch as module
from atom.models.deepseek_v41.mono.config import indexer_enabled


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [SimpleNamespace(attn=SimpleNamespace(indexer=SimpleNamespace()))]
        self.calls = []
        self.layer = False
        self.fail = False

    def uses_layer_mono(self, *args):
        return self.layer

    def forward(self, ids, positions, **kwargs):
        self.calls.append(self.layers[0].attn.indexer._mono.runner.active)
        if self.fail:
            raise RuntimeError("forward failed")
        return ids + 1


def setup(monkeypatch):
    model = Model()
    runner = SimpleNamespace(active=False, begin_step=Mock())
    lazy = SimpleNamespace(runner=runner, ready=Mock(return_value=True))
    monkeypatch.setattr(module, "LazyRunner", lambda *args: lazy)
    md = SimpleNamespace(cache=object(), step=object(), engram_embeddings=None)
    monkeypatch.setattr(
        module, "get_forward_context", lambda: SimpleNamespace(attn_metadata=md)
    )
    monkeypatch.setattr(module, "step_supported", lambda *args, **kwargs: True)
    config = SimpleNamespace(
        max_model_len=8192,
        index_cache_dtype="fp4",
        index_cache_layout="mfma",
        enable_dp_attention=False,
    )
    return model, module.MonoIndexerModel(model, model, config), lazy, md


def test_hook_is_active_only_for_the_supported_forward(monkeypatch):
    raw, wrapped, lazy, md = setup(monkeypatch)
    ids = torch.arange(8)
    assert torch.equal(wrapped(ids, ids), ids + 1)
    assert raw.calls == [True]
    lazy.runner.begin_step.assert_called_once_with(8, md.cache, md.step)
    assert not lazy.runner.active
    raw.fail = True
    with pytest.raises(RuntimeError, match="forward failed"):
        wrapped(ids, ids)
    assert not lazy.runner.active


def test_layer_runner_owns_its_indexer_without_preparing_native_buffers(monkeypatch):
    raw, wrapped, lazy, _ = setup(monkeypatch)
    raw.layer = True
    ids = torch.arange(16)
    wrapped(ids, ids)
    lazy.ready.assert_not_called()
    lazy.runner.begin_step.assert_not_called()
    assert raw.calls == [False]


def test_unsupported_step_keeps_native_route(monkeypatch):
    raw, wrapped, lazy, _ = setup(monkeypatch)
    monkeypatch.setattr(module, "step_supported", lambda *args, **kwargs: False)
    ids = torch.arange(65)
    wrapped(ids, ids)
    lazy.ready.assert_not_called()
    assert raw.calls == [False]


def test_default_on_with_explicit_opt_out_and_fp8_fallback(monkeypatch):
    monkeypatch.delenv("ATOM_DSV41_MONO_INDEXER", raising=False)
    assert indexer_enabled(True, 4)
    monkeypatch.setenv("ATOM_DSV41_MONO_INDEXER", "1")
    assert indexer_enabled(True, 4)
    assert not indexer_enabled(False, 4)
    monkeypatch.setenv("ATOM_DSV41_MONO_INDEXER", "0")
    assert not indexer_enabled(True, 4)


@pytest.mark.parametrize("dp_ranks", [2, 4])
def test_dp_local_indexer_is_explicit_opt_in(monkeypatch, dp_ranks):
    monkeypatch.delenv("ATOM_DSV41_MONO_INDEXER", raising=False)
    assert not indexer_enabled(True, 1, dp_ranks=dp_ranks)
    monkeypatch.setenv("ATOM_DSV41_MONO_INDEXER", "1")
    assert indexer_enabled(True, 1, dp_ranks=dp_ranks)
    assert not indexer_enabled(True, 1, dp_ranks=8)
    assert not indexer_enabled(False, 1, dp_ranks=dp_ranks)
    monkeypatch.setenv("ATOM_DSV41_MONO_INDEXER", "0")
    assert not indexer_enabled(True, 1, dp_ranks=dp_ranks)


def test_native_wrapper_does_not_route_layers_to_mono(monkeypatch):
    raw, _, lazy, _ = setup(monkeypatch)
    wrapper = module.NativeDecodeModel(raw)
    ids = torch.arange(8)
    assert not wrapper.uses_layer_mono(ids)
    assert torch.equal(wrapper(ids, ids), ids + 1)
    lazy.ready.assert_not_called()


def test_native_staging_is_joined_once_before_persistent_launch(gpu_module):
    runner = gpu_module.MonoIndexer.__new__(gpu_module.MonoIndexer)
    staged = SimpleNamespace(stage=Mock(), join=Mock())
    runner.engram_staged = staged
    runner._join_staging()
    staged.join.assert_called_once()
    assert runner.engram_staged is None
    runner._join_staging()
    staged.join.assert_called_once()


def test_hi_fallback_does_not_enable_standalone_indexer(monkeypatch, dispatch_module):
    dispatch = dispatch_module

    monkeypatch.setenv("ATOM_DSV41_MONO_INDEXER", "1")
    raw = Model()
    create = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(dispatch, "V41MonoDecodeRunner", create)
    wrapped = dispatch.MonoDecodeModel(
        raw, None, 8192, True, 0, 2, standalone_indexer=False
    )
    assert wrapped.layer_rows_supported(40)
    wrapped._mono._create()
    create.assert_called_once_with(
        raw, None, 8192, True, standalone_indexer=False, indexer_workspace=None
    )


def test_dp_attention_keeps_rank_local_indexer_work(monkeypatch, gpu_module):
    # Attention DP ranks carry different requests and caches, so they cannot
    # exchange selections even if a TP process group exists for the FFN.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 4)
    monkeypatch.setattr(gpu_module, "bind_agreed", lambda bind, group: None)
    monkeypatch.setattr(gpu_module, "WidthBuilds", lambda *args: object())
    peers = Mock()
    monkeypatch.setattr(gpu_module, "owned_peer_buffer", peers)
    r = gpu_module.MonoIndexer(None, 163840, True, "mfma", object(), replicated=False)
    assert r.world == 1 and r.rank == 0 and r._key(8).world == 1
    peers.assert_not_called()


def test_bounded_small_width_uses_local_key_without_peer_offsets(gpu_module):
    r = gpu_module.MonoIndexer.__new__(gpu_module.MonoIndexer)
    r.world, r.bound_max, r.peer_plain = 4, 163968, 4096
    r.fp4, r.layout = True, "mfma"
    assert r._key(8).world == 1 and r._key(8).peer_plain == 0
    assert r._key(16).world == 4 and r._key(16).peer_plain == 4096
    r.bound_max = 262272
    assert r._key(8).world == 4


def test_reindex_launches_local_binary_with_its_own_scratch(gpu_module):
    r = gpu_module.MonoIndexer.__new__(gpu_module.MonoIndexer)
    r.widths = {8: object()}
    r.local_widths = {8: object()}
    r.full_widths = None
    r.scratch, r.local_scratch = object(), object()
    assert r._launch_context(SimpleNamespace(candidate_source=None), 8) == (
        r.widths[8],
        r.scratch,
    )
    assert r._launch_context(SimpleNamespace(candidate_source=20), 8) == (
        r.local_widths[8],
        r.local_scratch,
    )
    r.local_widths = None
    assert r._launch_context(SimpleNamespace(candidate_source=20), 8) == (
        r.widths[8],
        r.scratch,
    )


def test_full_binary_is_separate_from_candidate_and_reindex_buffers(gpu_module):
    r = gpu_module.MonoIndexer.__new__(gpu_module.MonoIndexer)
    r.widths, r.local_widths, r.full_widths = ({8: object()} for _ in range(3))
    r.scratch, r.local_scratch, r.full_scratch = object(), object(), object()
    for source, produces, expected in (
        (None, False, (r.full_widths[8], r.full_scratch)),
        (None, True, (r.widths[8], r.scratch)),
        (20, False, (r.local_widths[8], r.local_scratch)),
    ):
        spec = SimpleNamespace(candidate_source=source, produces_candidates=produces)
        assert r._launch_context(spec, 8) == expected


def test_layer_reset_preserves_candidate_data_and_separates_split_histograms(
    gpu_module,
):
    from atom.models.deepseek_v41.mono import indexer_build as ib
    from atom.models.deepseek_v41.mono.runner import V41MonoDecodeRunner

    key = ib.IndexerBuild(8, 163968, world=4)
    r = V41MonoDecodeRunner.__new__(V41MonoDecodeRunner)
    r._indexer_key = lambda rows: key
    r.index_scratch = torch.full((ib.scratch_bytes(key),), 7, dtype=torch.uint8)
    local_key = ib.reindex_build(key)
    r.local_index_scratch = torch.full(
        (ib.scratch_bytes(local_key),), 7, dtype=torch.uint8
    )
    r.indexer_full_ratio = 2
    full_key = ib.full_build(key, 2)
    r.full_index_scratch = torch.full(
        (ib.scratch_bytes(full_key),), 7, dtype=torch.uint8
    )
    resets, split_resets = r._index_reset_buffers(8)
    assert resets and len(split_resets) == 1
    assert (
        sum(x.numel() * x.element_size() for x in resets) < r.index_scratch.numel() // 2
    )
    for flags in resets:
        flags.zero_()
    for build, scratch in (
        (key, r.index_scratch),
        (local_key, r.local_index_scratch),
        (full_key, r.full_index_scratch),
    ):
        layout = ib.scratch_layout(build)
        # Retained scores/IDs are overwritten by their producers, not cleared.
        for name in ("iq", "isel0"):
            if name in layout:
                off, size = layout[name]
                assert torch.all(scratch[off : off + size] == 7)
        off, size = layout["irdy"]
        assert not torch.any(scratch[off : off + size])
    assert torch.all(split_resets[0].view(torch.uint8) == 7)
    split_resets[0].zero_()
    assert not torch.any(split_resets[0])


@pytest.mark.parametrize("tp,rows", [(2, 40), (2, 48)])
def test_tp2_large_batch_keeps_native_layers_and_mono_indexer(
    monkeypatch, tp, rows, dispatch_module
):
    dispatch = dispatch_module

    raw, _, lazy, md = setup(monkeypatch)
    monkeypatch.setenv("ATOM_DSV41_MONO_INDEXER", "1")
    monkeypatch.setattr(dispatch, "step_supported", lambda *args, **kwargs: True)
    layer = dispatch.MonoDecodeModel(raw, None, 8192, True, 0, tp)
    layer._mono = SimpleNamespace(ready=Mock(), runner=SimpleNamespace(forward=Mock()))
    cfg = SimpleNamespace(
        max_model_len=8192,
        index_cache_dtype="fp4",
        index_cache_layout="mfma",
        enable_dp_attention=False,
    )
    wrapped = module.MonoIndexerModel(layer, raw, cfg)
    ids = torch.arange(rows)
    assert torch.equal(wrapped(ids, ids), ids + 1)
    assert raw.calls == [True]
    layer._mono.ready.assert_not_called()
    layer._mono.runner.forward.assert_not_called()
    lazy.runner.begin_step.assert_called_once_with(rows, md.cache, md.step)
    assert not lazy.runner.active


@pytest.mark.parametrize(
    "tp,rows,fp4,enabled",
    [
        (2, 8, True, "1"),
        (4, 40, True, "1"),
        (8, 40, True, "1"),
        (2, 40, False, "1"),
        (2, 40, True, "0"),
    ],
)
def test_other_layer_routes_do_not_prepare_native_indexer(
    monkeypatch, tp, rows, fp4, enabled, dispatch_module
):
    dispatch = dispatch_module

    raw, _, lazy, _ = setup(monkeypatch)
    monkeypatch.setenv("ATOM_DSV41_MONO_INDEXER", enabled)
    monkeypatch.setattr(dispatch, "step_supported", lambda *args, **kwargs: True)
    layer = dispatch.MonoDecodeModel(raw, None, 8192, fp4, 0, tp)
    layer_run = Mock(side_effect=lambda ids, positions: ids + 2)
    layer._mono = SimpleNamespace(
        ready=Mock(return_value=True), runner=SimpleNamespace(forward=layer_run)
    )
    cfg = SimpleNamespace(
        max_model_len=8192,
        index_cache_dtype="fp4",
        index_cache_layout="mfma",
        enable_dp_attention=False,
    )
    wrapped = module.MonoIndexerModel(layer, raw, cfg)
    ids = torch.arange(rows)
    assert torch.equal(wrapped(ids, ids), ids + 2)
    assert raw.calls == []
    layer_run.assert_called_once()
    lazy.ready.assert_not_called()
    lazy.runner.begin_step.assert_not_called()


@pytest.fixture
def gpu_module():
    pytest.importorskip("aiter")
    pytest.importorskip("flydsl")
    from atom.models.deepseek_v41.mono import indexer

    return indexer


@pytest.fixture
def dispatch_module(monkeypatch):
    import sys
    import types

    from aiter_stub import atom_under_stubs

    with atom_under_stubs() as load:
        for mod, cls in (
            ("runner", "V41MonoDecodeRunner"),
            ("draft_runner", "DraftMonoRunner"),
        ):
            stub = types.ModuleType(f"atom.models.deepseek_v41.mono.{mod}")
            setattr(stub, cls, object)
            monkeypatch.setitem(sys.modules, stub.__name__, stub)
        yield load("atom.models.deepseek_v41.mono.dispatch")


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_tp_default_and_explicit_override(monkeypatch, tp):
    monkeypatch.delenv("ATOM_DSV41_MONO_INDEXER", raising=False)
    assert indexer_enabled(True, tp) == (tp != 8)
    monkeypatch.setenv("ATOM_DSV41_MONO_INDEXER", "1")
    assert indexer_enabled(True, tp)
    monkeypatch.setenv("ATOM_DSV41_MONO_INDEXER", "0")
    assert not indexer_enabled(True, tp)


@pytest.mark.parametrize("tp,dp", [(4, 0), (1, 2)])
def test_workspace_is_shared_at_install_before_any_decode(
    monkeypatch, dispatch_module, tp, dp
):
    import sys
    import types

    dispatch = dispatch_module
    raw = Model()
    workspace = object()
    create = Mock(return_value=workspace)
    stub = types.ModuleType("atom.models.deepseek_v41.mono.indexer_workspace")
    stub.IndexerWorkspace = create
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    layer = Mock(return_value=SimpleNamespace(uses_layer_mono=Mock()))
    hook = Mock()
    monkeypatch.setenv("ATOM_DSV41_MONO_INDEXER", "1")
    monkeypatch.setattr(dispatch, "config_refusal", lambda cfg: None)
    monkeypatch.setattr(dispatch, "dp_attention_ranks", lambda cfg: dp)
    monkeypatch.setattr(dispatch.hi_gate, "decide", lambda *args, **kwargs: False)
    monkeypatch.setattr(dispatch, "MonoDecodeModel", layer)
    monkeypatch.setattr(module, "MonoIndexerModel", hook)
    # atom_under_stubs imports a fresh copy, so patch that copy too.
    import atom.models.deepseek_v41.mono.indexer_dispatch as fresh

    monkeypatch.setattr(fresh, "MonoIndexerModel", hook)
    cfg = SimpleNamespace(
        max_model_len=163840,
        index_cache_dtype="fp4",
        tensor_parallel_size=tp,
        enable_dp_attention=bool(dp),
    )
    dispatch.install_mono_decode(raw, cfg, None)
    create.assert_called_once_with(raw, 163840, tp)
    assert layer.call_args.kwargs["indexer_workspace"] is workspace
    assert hook.call_args.kwargs["workspace"] is workspace


@pytest.mark.parametrize(
    "flag,dp,refusal,hi,installed",
    [
        ("1", 4, None, "0", True),
        (None, 4, None, "0", False),
        ("0", 4, None, "0", False),
        ("1", 4, "unsupported geometry", "0", False),
        ("1", 4, None, "1", False),
        ("1", 8, None, "0", False),
    ],
)
def test_dp_native_install_is_local_and_explicit(
    monkeypatch, dispatch_module, flag, dp, refusal, hi, installed
):
    import sys
    import types

    import atom.models.deepseek_v41.mono.indexer_dispatch as fresh

    dispatch = dispatch_module
    raw = Model()
    workspace = object()
    create = Mock(return_value=workspace)
    stub = types.ModuleType("atom.models.deepseek_v41.mono.indexer_workspace")
    stub.IndexerWorkspace = create
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    hook = Mock(return_value="local indexer")
    monkeypatch.setattr(fresh, "MonoIndexerModel", hook)
    old = Mock(return_value="existing layer route")
    monkeypatch.setattr(dispatch, "_install_layer_decode", old)
    monkeypatch.setattr(dispatch, "dp_attention_ranks", lambda cfg: dp)
    monkeypatch.setattr(dispatch.hi_gate, "mode", lambda: hi)
    monkeypatch.setattr(
        dispatch,
        "config_refusal",
        lambda cfg, supported_dpa=(2,): refusal if dp in supported_dpa else "DP ranks",
    )
    if flag is None:
        monkeypatch.delenv("ATOM_DSV41_MONO_INDEXER", raising=False)
    else:
        monkeypatch.setenv("ATOM_DSV41_MONO_INDEXER", flag)
    cfg = SimpleNamespace(
        max_model_len=163840,
        index_cache_dtype="fp4",
        tensor_parallel_size=1,
        enable_dp_attention=True,
    )
    got = dispatch.install_mono_decode(raw, cfg, None)
    if installed:
        assert got == "local indexer"
        create.assert_called_once_with(raw, 163840, 1)
        wrapped, actual_model, actual_config = hook.call_args.args
        assert isinstance(wrapped, fresh.NativeDecodeModel)
        assert actual_model is raw and actual_config is cfg
        assert not wrapped.uses_layer_mono(None, None)
        assert hook.call_args.kwargs["workspace"] is workspace
        old.assert_not_called()
    else:
        assert got == "existing layer route"
        old.assert_called_once()
        create.assert_not_called()
        hook.assert_not_called()
