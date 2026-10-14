# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 MTP on the vLLM plugin: the draft lives in the target's slot."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

pytest.importorskip("aiter", reason="the V4 bridge reaches AITER dtypes")

from atom.plugin.vllm import deepseek_v4_bridge as B
from atom.plugin.vllm.deepseek_v4_state_checkpoint import v4_state_checkpoint_sizing

RATIOS = [0, 0, 4, 128, 4, 128, 4, 0]
MTP_LAYER = len(RATIOS) - 1
DRAFT = f"model.layers.{MTP_LAYER}.{B.ATOM_DEEPSEEK_V4_DRAFT_PROXY_LAYER_PREFIX}"
TARGET = B.ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME


def _vcfg(max_num_seqs=4):
    hf = NS(
        compress_ratios=RATIOS,
        num_hidden_layers=MTP_LAYER,
        head_dim=512,
        index_head_dim=128,
        qk_rope_head_dim=64,
        sliding_window=128,
        index_topk=512,
    )
    return NS(
        model_config=NS(
            hf_config=hf, max_model_len=4096, architectures=["DeepseekV4ForCausalLM"]
        ),
        scheduler_config=NS(max_num_seqs=max_num_seqs),
        cache_config=NS(cache_dtype="fp8", enable_prefix_caching=False),
        speculative_config=NS(num_speculative_tokens=1, method="mtp"),
        compilation_config=NS(static_forward_context={}),
    )


def _args():
    return NS(
        compress_ratios=RATIOS,
        window_size=128,
        head_dim=512,
        index_head_dim=128,
        index_topk=512,
    )


def _attn(layer_id, ratio):
    return NS(
        layer_id=layer_id,
        compress_ratio=ratio,
        compressor=NS(),
        indexer=NS(compressor=NS()),
    )


def _models():
    target = NS(
        args=_args(),
        model=NS(
            layers=[NS(attn=_attn(i, r)) for i, r in enumerate(RATIOS[:MTP_LAYER])]
        ),
    )
    draft = NS(args=_args(), model=NS(mtp=[NS(attn=_attn(MTP_LAYER, 0))]))
    return target, draft


def _register_both(vc):
    B.register_deepseek_v4_proxy_layer(vc, TARGET)
    B.register_deepseek_v4_proxy_layer(vc, DRAFT, kv_sharing_target_layer_name=TARGET)
    return vc.compilation_config.static_forward_context


def _shared_pool(vc, sfc):
    """The target proxy's 0.31 view over the schedulable blocks, tail after it;
    the draft proxy gets that same tensor, as vLLM's KV sharing hands it."""
    s = v4_state_checkpoint_sizing(vc)
    tensor_blocks = -(-s.num_slots * s.slot_bytes // s.page_bytes) + 64
    usable = s.usable_blocks(tensor_blocks)
    assert usable > 0
    head = s.page_bytes // (2 * B.ATOM_DEEPSEEK_V4_BLOCK_SIZE)
    raw = torch.zeros(tensor_blocks * s.page_bytes, dtype=torch.uint8)
    view = raw[: usable * s.page_bytes].view(
        usable, 1, B.ATOM_DEEPSEEK_V4_BLOCK_SIZE, 2 * head
    )
    sfc[TARGET].kv_cache = view
    sfc[DRAFT].kv_cache = view
    return raw


@pytest.fixture
def bound(monkeypatch):
    # The decode-metadata buffers are pinned host + device staging; not what
    # these tests are about.
    monkeypatch.setattr(B, "_ensure_v4_decode_bufs", lambda *a, **k: None)
    vc = _vcfg()
    sfc = _register_both(vc)
    raw = _shared_pool(vc, sfc)
    target, draft = _models()
    return vc, sfc, raw, target, draft


def test_draft_proxy_has_no_kv_spec_and_joins_the_targets_group():
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.utils import add_kv_sharing_layers_to_kv_cache_groups

    vc = _vcfg()
    sfc = _register_both(vc)
    # Registering a sharing proxy installs the runner patch.
    assert getattr(
        GPUModelRunner.get_kv_cache_spec, "_atom_v4_draft_kv_sharing_patched", False
    )
    runner = NS(vllm_config=vc, compilation_config=vc.compilation_config)
    runner.shared_kv_cache_layers = {}

    def original(self):
        # What vLLM's own method returns for two bare AttentionLayerBase
        # layers: a spec for each.
        return {name: layer.get_kv_cache_spec(vc) for name, layer in sfc.items()}

    specs = _call_wrapped(original, runner)
    # vLLM prices and allocates the target proxy only.
    assert list(specs) == [TARGET]
    assert runner.shared_kv_cache_layers == {DRAFT: TARGET}
    # What vLLM then does with the record: the draft joins the target's group,
    # where the drafter finds its layer for its own metadata builder.
    groups = [KVCacheGroupSpec(layer_names=[TARGET], kv_cache_spec=specs[TARGET])]
    add_kv_sharing_layers_to_kv_cache_groups(runner.shared_kv_cache_layers, groups)
    assert groups[0].layer_names == [TARGET, DRAFT]
    cfg = KVCacheConfig(num_blocks=8, kv_cache_tensors=[], kv_cache_groups=groups)
    # vLLM's check that every draft layer is in one KV cache group holds.
    SpecDecodeBaseProposer.validate_same_kv_cache_group(
        NS(_draft_attn_layer_names={DRAFT}), cfg
    )


def _call_wrapped(original, runner):
    """Run the patch's wrapper around a stand-in for vLLM's own method."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    saved = GPUModelRunner.get_kv_cache_spec
    try:
        GPUModelRunner.get_kv_cache_spec = original
        B.apply_vllm_v4_draft_kv_sharing_patch()
        return GPUModelRunner.get_kv_cache_spec(runner)
    finally:
        GPUModelRunner.get_kv_cache_spec = saved


def test_a_draft_sharing_kv_of_a_layer_without_spec_is_refused():
    vc = _vcfg()
    sfc = _register_both(vc)
    runner = NS(vllm_config=vc, compilation_config=vc.compilation_config)
    runner.shared_kv_cache_layers = {}
    with pytest.raises(RuntimeError, match="has no KV cache spec"):
        _call_wrapped(lambda self: {DRAFT: sfc[DRAFT].get_kv_cache_spec(vc)}, runner)


def test_draft_waits_for_the_target_then_binds_into_its_slot(bound):
    vc, sfc, _, target, draft = bound
    # The target forwards first in a step; before it has bound, neither does
    # the draft.
    assert not B.bind_deepseek_v4_proxy_cache_views(draft, vc, DRAFT)
    assert B.bind_deepseek_v4_proxy_cache_views(target, vc, TARGET)
    assert B.bind_deepseek_v4_proxy_cache_views(draft, vc, DRAFT)

    views = target._atom_v4_views
    d_attn = draft.model.mtp[0].attn
    # The draft's layer is the target carve's layer `n_layers`: same views.
    assert d_attn.unified_kv.data_ptr() == views["unified"][MTP_LAYER].data_ptr()
    assert d_attn.unified_kv_rope is None or (
        d_attn.unified_kv_rope.data_ptr() == views["unified_rope"][MTP_LAYER].data_ptr()
    )
    assert d_attn.swa_window == target.model.layers[0].attn.swa_window
    assert d_attn.swa_cache_size == v4_state_checkpoint_sizing(vc).win_with_spec == 129
    # One allocator, one geometry, one set of meta params; no state checkpoint path.
    assert draft._atom_v4_slot_allocator.owner is target._atom_v4_slot_allocator
    assert draft._atom_v4_geometry is target._atom_v4_geometry
    assert draft._atom_v4_meta_params is target._atom_v4_meta_params
    assert draft._atom_state_checkpoint_worker is None and not draft._atom_v4_is_target
    assert sfc[DRAFT]._atom_v4_model is draft and sfc[TARGET]._atom_v4_model is target
    assert target._atom_v4_slot_sharers == [d_attn]
    # Rebinding on the same tensor is a no-op.
    assert B.bind_deepseek_v4_proxy_cache_views(draft, vc, DRAFT)


def test_draft_ring_rows_are_the_targets_slot_rows_for_layer_n_layers(bound):
    vc, _, _, target, draft = bound
    B.bind_deepseek_v4_proxy_cache_views(target, vc, TARGET)
    B.bind_deepseek_v4_proxy_cache_views(draft, vc, DRAFT)
    geo = target._atom_v4_geometry
    plane = target._atom_v4_views["kv_plane"].view(torch.uint8)
    plane = plane.reshape(plane.shape[0], -1)
    d_attn = draft.model.mtp[0].attn
    d_plane = d_attn.swa_plane.view(torch.uint8).reshape(d_attn.swa_plane.shape[0], -1)
    slot = geo.physical_slot(2)
    start, stop = geo.slot_span(slot)
    runs = geo.entry_row_runs()
    before = plane.clone()
    other_layers = set()
    for lid, r in enumerate(RATIOS[:MTP_LAYER]):
        w = geo.window_params(r)
        base = geo.layer_base_row(lid)
        other_layers |= {base + w.index(slot, p) for p in range(geo.ring_slots)}
    for pos in (0, 1, 127, 128, 300):
        row = d_attn.swa_window.index(slot, pos)
        d_plane[row].fill_(0xA5)
        absolute = geo.layer_base_row(MTP_LAYER) + row
        # Written through the draft, landed in the target's plane, inside the
        # target's slot, in the window rows a state checkpoint carries.
        assert torch.all(plane[absolute] == 0xA5)
        assert start <= absolute < stop
        entry_row = absolute - start - geo.arena_rows
        assert any(s <= entry_row < s + n for s, n in runs)
        assert absolute not in other_layers
    # Nothing outside that slot changed.
    touched = (plane != before).any(dim=1).nonzero().flatten()
    assert touched.numel() == 5
    assert all(start <= int(r) < stop for r in touched)


def test_the_targets_reset_clears_the_drafts_rows_of_the_slot(bound):
    vc, _, _, target, draft = bound
    B.bind_deepseek_v4_proxy_cache_views(target, vc, TARGET)
    B.bind_deepseek_v4_proxy_cache_views(draft, vc, DRAFT)
    geo = target._atom_v4_geometry
    d_attn = draft.model.mtp[0].attn
    rows = {
        g: [d_attn.swa_window.index(geo.physical_slot(g), p) for p in range(129)]
        for g in (0, 1)
    }
    for slot_rows in rows.values():
        d_attn.swa_plane.view(torch.uint8)[slot_rows] = 7
    B.reset_deepseek_v4_state_slots(target, {geo.physical_slot(1)})
    assert not d_attn.swa_plane.view(torch.uint8)[rows[1]].any()
    assert (d_attn.swa_plane.view(torch.uint8)[rows[0]] == 7).all()


def test_draft_looks_slots_up_and_never_binds_resets_or_releases():
    owner = B._V4StateSlotAllocator(4, evict_lru=False)
    slots, reset = owner.assign(["a", "b"], np.array([0, 0]))
    assert reset == set(slots.tolist())
    view = B._V4SharedSlotLookup(owner)
    before = (dict(owner._key_to_slot), list(owner._free), owner._step)
    got, got_reset = view.assign(["b", "a"], np.array([0, 0]))
    # Same request, same slot; a first-chunk row (num_computed 0) resets
    # nothing on the draft's build: the target already did, before its forward.
    assert got.tolist() == [int(slots[1]), int(slots[0])]
    assert got_reset == set() and view.last_fresh_rows == []
    assert (dict(owner._key_to_slot), list(owner._free), owner._step) == before
    with pytest.raises(RuntimeError, match="no state slot"):
        view.assign(["a", "c"], np.array([5, 0]))
    # Lifecycle events go to the owner only, which the draft then follows.
    owner.release(["a"])
    with pytest.raises(RuntimeError, match="no state slot"):
        view.assign(["a"], np.array([0]))
