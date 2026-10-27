# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 prefix state checkpoints on the vLLM plugin: slots, sizing, round
trip."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

pytest.importorskip("aiter", reason="the V4 bridge reaches AITER dtypes")

from aiter.jit.utils import chip_info

from atom.plugin.vllm.deepseek_v4_bridge import (
    ATOM_DEEPSEEK_V4_BLOCK_SIZE,
    _proxy_carve_source,
    _proxy_through_slot_tail,
    _v4_kv_fp8,
    _v4_state_layout,
    slice_deepseek_v4_proxy_cache_views,
)
from atom.plugin.vllm.deepseek_v4_bridge import (
    _V4StateSlotAllocator as StateSlotAllocator,
)
from atom.plugin.vllm.deepseek_v4_state_checkpoint import (
    AtomDeepseekV4StateCheckpointLayer,
    V4StateCheckpointPlacement,
    deepseek_v4_state_checkpoint_copy_funcs,
    deepseek_v4_state_checkpoint_layer_names,
    make_v4_state_checkpoint_copier,
    v4_state_checkpoint_sizing,
)
from atom.plugin.vllm.paged_state_checkpoint import (
    execute_state_checkpoint_restores,
    execute_state_checkpoint_stores,
)

RATIOS = [0, 0, 4, 128, 4, 128, 4, 0]

_HOST_GET_GFX = chip_info.get_gfx


@pytest.fixture(autouse=True)
def _gfx950(monkeypatch):
    # `_v4_kv_fp8` grants the fp8 layout these configs ask for only on gfx950 /
    # gfx1250, so pin the chip; on gfx942, or when the probe fails, every fp8
    # config would size and route as bf16. Tests that need another chip set it
    # through `gfx`.
    monkeypatch.setattr(chip_info, "get_gfx", lambda: "gfx950")


def _vcfg(max_num_seqs=4):
    hf = NS(
        compress_ratios=RATIOS,
        num_hidden_layers=len(RATIOS) - 1,
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
        cache_config=NS(cache_dtype="fp8", enable_prefix_caching=True),
        speculative_config=None,
    )


def _hook_vcfg(arch, ratios, *, prefix=True, spec=None):
    vc = _vcfg()
    vc.model_config.architectures = [arch]
    vc.model_config.hf_config.compress_ratios = ratios
    vc.model_config.hf_config.num_hidden_layers = len(ratios) - 1
    vc.cache_config.enable_prefix_caching = prefix
    vc.cache_config.mamba_cache_mode = "none"
    vc.speculative_config = spec
    vc.scheduler_config.long_prefill_token_threshold = 0
    return vc


def _through_hook(vc, specs):
    from atom.plugin.vllm import paged_state_checkpoint as G
    from atom.plugin.vllm.deepseek_v4_state_checkpoint import (
        register_v4_state_checkpoint_adapter,
    )

    register_v4_state_checkpoint_adapter()
    sentinel, called = [object()], []

    def original(vc_, specs_, mem):
        called.append(specs_)
        return sentinel

    return G._kv_cache_configs_hook(original, vc, specs, [0]), sentinel, called


@pytest.mark.parametrize("prefix", [True, False])
def test_vllms_own_v41_is_left_to_vllm(prefix):
    # vLLM's native V4.1: ratios 1 and 2 are not V4 pool geometry.
    vc = _hook_vcfg("DeepseekV41ForCausalLM", [0, 1, 2, 4, 128, 0], prefix=prefix)
    out, sentinel, called = _through_hook(vc, [{"model.layers.0.attn": object()}])
    assert out is sentinel and len(called) == 1
    assert vc.cache_config.mamba_cache_mode == "none"


def test_v4_served_by_vllm_itself_is_left_to_vllm():
    # Without the plugin's registration the registry resolves V4 to vLLM.
    from atom.plugin.vllm.deepseek_v4_state_checkpoint import is_atom_deepseek_v4_config

    vc = _hook_vcfg("DeepseekV4ForCausalLM", RATIOS, spec=NS(num_speculative_tokens=1))
    assert not is_atom_deepseek_v4_config(vc)
    out, sentinel, called = _through_hook(vc, [{"model.layers.0.attn": object()}])
    assert out is sentinel and len(called) == 1
    assert vc.cache_config.mamba_cache_mode == "none"


def test_atoms_v4_is_matched_only_by_exact_arch_and_registry(monkeypatch):
    from vllm.model_executor.models.registry import ModelRegistry

    from atom.plugin.vllm.deepseek_v4_state_checkpoint import is_atom_deepseek_v4_config

    atom = NS(module_name="atom.plugin.vllm.model_wrapper", class_name="X")
    monkeypatch.setitem(ModelRegistry.models, "DeepseekV4ForCausalLM", atom)
    monkeypatch.setitem(ModelRegistry.models, "DeepseekV41ForCausalLM", atom)
    assert is_atom_deepseek_v4_config(_hook_vcfg("DeepseekV4ForCausalLM", RATIOS))
    assert not is_atom_deepseek_v4_config(_hook_vcfg("DeepseekV41ForCausalLM", RATIOS))
    # ATOM's V4 without its proxy in the specs (e.g. a pooling / non-plugin path)
    # is still not touched.
    vc = _hook_vcfg("DeepseekV4ForCausalLM", RATIOS, spec=NS(num_speculative_tokens=1))
    out, sentinel, _ = _through_hook(vc, [{"model.layers.0.attn": object()}])
    assert out is sentinel and vc.cache_config.mamba_cache_mode == "none"
    # With the proxy: state checkpoints on, and with MTP too, since the draft's ring is
    # in the target's slot: native's hit rule (vLLM's own drop switch) and align.
    from atom.plugin.vllm import paged_state_checkpoint as G
    from atom.plugin.vllm.deepseek_v4_bridge import ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME
    from atom.plugin.vllm.deepseek_v4_state_checkpoint import (
        register_v4_state_checkpoint_adapter,
    )

    register_v4_state_checkpoint_adapter()
    vc.speculative_config = NS(
        method="mtp",
        draft_model_config=NS(architectures=["DeepSeekV4MTPModel"]),
        num_speculative_tokens=1,
        use_eagle=lambda: True,
        disable_eagle_block_drop=False,
    )
    out = G._kv_cache_configs_hook(
        lambda *a: [], vc, [{ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME: object()}], [0]
    )
    assert out == [] and vc.cache_config.mamba_cache_mode == "align"
    assert vc.speculative_config.disable_eagle_block_drop is True


@pytest.fixture
def atom_v4(monkeypatch):
    from vllm.model_executor.models.registry import ModelRegistry

    from atom.plugin.vllm.deepseek_v4_state_checkpoint import (
        register_v4_state_checkpoint_adapter,
    )

    atom = NS(module_name="atom.plugin.vllm.model_wrapper", class_name="X")
    monkeypatch.setitem(ModelRegistry.models, "DeepseekV4ForCausalLM", atom)
    register_v4_state_checkpoint_adapter()


@pytest.fixture(scope="module")
def v4_target(tmp_path_factory):
    """vLLM's ModelConfig of a tiny DeepSeek-V4 checkpoint (config only)."""
    import json

    from vllm.config import ModelConfig

    d = tmp_path_factory.mktemp("v4")
    hf = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_nextn_predict_layers": 1,
        "compress_ratios": [0, 4, 0],
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "qk_rope_head_dim": 16,
        "q_lora_rank": 32,
        "o_lora_rank": 32,
        "o_groups": 1,
        "index_head_dim": 16,
        "index_n_heads": 2,
        "index_topk": 8,
        "sliding_window": 128,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "hc_mult": 1,
        "vocab_size": 1000,
        "max_position_embeddings": 4096,
    }
    (d / "config.json").write_text(json.dumps(hf))
    return ModelConfig(
        model=str(d), skip_tokenizer_init=True, max_model_len=4096, dtype="float16"
    )


def _real_spec(target, method):
    from vllm.config import ParallelConfig, SpeculativeConfig

    lookup = {"prompt_lookup_max": 2, "prompt_lookup_min": 1}
    return SpeculativeConfig(
        method=method,
        num_speculative_tokens=1,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
        **(lookup if method == "ngram" else {}),
    )


def _v4_hook(vc):
    from atom.plugin.vllm import paged_state_checkpoint as G
    from atom.plugin.vllm.deepseek_v4_bridge import ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME

    return G._kv_cache_configs_hook(
        lambda *a: [], vc, [{ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME: object()}], [0]
    )


@pytest.mark.parametrize("method", ["mtp", "deepseek_mtp"])
def test_v4_mtp_is_the_drafter_bound_into_the_targets_slot(atom_v4, v4_target, method):
    from atom.plugin.vllm.deepseek_v4_state_checkpoint import v4_drafter_in_target_slot

    spec = _real_spec(v4_target, method)
    # vLLM normalizes every MTP spelling; the V4 target gets the V4 MTP draft,
    # whose architecture is what the plugin binds into the target's slot.
    assert spec.method == "mtp"
    assert spec.draft_model_config.architectures == ["DeepSeekV4MTPModel"]
    assert spec.use_eagle_block_drop()
    vc = _hook_vcfg("DeepseekV4ForCausalLM", RATIOS, spec=spec)
    assert v4_drafter_in_target_slot(vc)
    assert _v4_hook(vc) == [] and vc.cache_config.mamba_cache_mode == "align"
    assert spec.disable_eagle_block_drop is True


def _drafters(v4_target):
    """Every other drafter: vLLM's ngram for real; those that need a draft
    checkpoint as vLLM normalizes them."""
    eagle = {"use_eagle": lambda: True, "num_speculative_tokens": 1}
    return [
        _real_spec(v4_target, "ngram"),
        NS(method="eagle3", draft_model_config=NS(architectures=["Eagle3X"]), **eagle),
        NS(
            method="dspark",
            draft_model_config=NS(architectures=["DeepseekV4ForCausalLM"]),
            **eagle,
        ),
        NS(method="eagle", draft_model_config=NS(architectures=["EagleX"]), **eagle),
        NS(method="suffix", use_eagle=lambda: False, num_speculative_tokens=1),
        # "mtp" with a draft the plugin does not bind into the slot.
        NS(
            method="mtp",
            draft_model_config=NS(architectures=["DeepSeekMTPModel"]),
            **eagle,
        ),
    ]


def test_v4_state_checkpoints_refuse_every_other_drafter_at_startup(atom_v4, v4_target):
    for spec in _drafters(v4_target):
        spec.disable_eagle_block_drop = False
        vc = _hook_vcfg("DeepseekV4ForCausalLM", RATIOS, spec=spec)
        with pytest.raises(ValueError) as e:
            _v4_hook(vc)
        msg = str(e.value)
        assert msg.startswith("DeepSeek-V4 prefix caching through state checkpoints")
        assert f"speculative method {spec.method!r}" in msg
        assert "the DeepSeek-V4 MTP draft" in msg
        assert "--no-enable-prefix-caching" in msg
        # Refused before vLLM's EAGLE drop or the state checkpoint groups are touched.
        assert spec.disable_eagle_block_drop is False
        assert vc.cache_config.mamba_cache_mode == "none"


def test_v4_without_state_checkpoints_takes_any_drafter_and_no_drafter_takes_state_checkpoints(
    atom_v4, v4_target
):
    for spec in [_real_spec(v4_target, "mtp"), *_drafters(v4_target)]:
        spec.disable_eagle_block_drop = False
        vc = _hook_vcfg("DeepseekV4ForCausalLM", RATIOS, prefix=False, spec=spec)
        assert _v4_hook(vc) == []
        assert vc.cache_config.mamba_cache_mode == "none"
        assert spec.disable_eagle_block_drop is False
    vc = _hook_vcfg("DeepseekV4ForCausalLM", RATIOS)
    assert _v4_hook(vc) == [] and vc.cache_config.mamba_cache_mode == "align"


@pytest.fixture
def gfx(monkeypatch):
    from aiter.jit.utils import chip_info

    def set_gfx(name):
        monkeypatch.setattr(chip_info, "get_gfx", lambda: name)

    return set_gfx


def test_v4_state_checkpoints_run_with_an_fp8_kv_cache(atom_v4, gfx):
    gfx("gfx950")
    for dtype in ("fp8", "fp8_e4m3"):
        vc = _hook_vcfg("DeepseekV4ForCausalLM", RATIOS)
        vc.cache_config.cache_dtype = dtype
        assert _v4_hook(vc) == [] and vc.cache_config.mamba_cache_mode == "align"


@pytest.mark.parametrize(
    "dtype,arch,why",
    [
        ("auto", "gfx950", "a bf16 KV cache"),
        ("bfloat16", "gfx950", "a bf16 KV cache"),
        # The same decision as the sizing: fp8 asked, degraded to bf16 off
        # gfx950 / gfx1250.
        ("fp8", "gfx942", "which this GPU runs as a bf16 KV cache"),
    ],
)
def test_v4_state_checkpoints_refuse_a_bf16_kv_cache_at_startup(
    atom_v4, gfx, monkeypatch, v4_target, dtype, arch, why
):
    from atom.plugin.vllm.deepseek_v4_bridge import _v4_kv_fp8
    from atom.plugin.vllm.deepseek_v4_state_checkpoint import (
        DeepseekV4StateCheckpointAdapter,
        register_deepseek_v4_state_checkpoint_layers,
    )

    gfx(arch)

    def no_sizing(self, vc):
        raise AssertionError("sized before the dtype check")

    for spec in (None, _real_spec(v4_target, "mtp")):
        vc = _hook_vcfg("DeepseekV4ForCausalLM", RATIOS, spec=spec)
        vc.cache_config.cache_dtype = dtype
        vc.compilation_config = NS(static_forward_context={})
        assert not _v4_kv_fp8(vc)
        # Both where a worker turns state checkpoints on (model construction, before the
        # model profiles) and in EngineCore's KV-config hook.
        for site in (register_deepseek_v4_state_checkpoint_layers, _v4_hook):
            with monkeypatch.context() as m:
                m.setattr(DeepseekV4StateCheckpointAdapter, "sizing", no_sizing)
                with pytest.raises(ValueError) as e:
                    site(vc)
            msg = str(e.value)
            assert msg.startswith(
                "DeepSeek-V4 prefix caching through state checkpoints is only "
                "validated with an fp8 KV cache"
            )
            assert f"--kv-cache-dtype {dtype!r}, {why}." in msg
            assert "--kv-cache-dtype fp8" in msg
            assert "--no-enable-prefix-caching" in msg
        assert vc.compilation_config.static_forward_context == {}
        assert vc.cache_config.mamba_cache_mode == "none"
        # Without prefix caching there are no state checkpoints to refuse.
        vc.cache_config.enable_prefix_caching = False
        assert _v4_hook(vc) == [] and vc.cache_config.mamba_cache_mode == "none"


def test_event_driven_slots_never_take_a_live_requests_slot():
    alloc = StateSlotAllocator(2, evict_lru=False)
    slots, reset = alloc.assign(["a", "b"], np.array([0, 256]))
    assert alloc.last_fresh_rows == [0, 1] and reset == set(slots.tolist())
    # Request "a" was not scheduled this step; it is still alive.
    with pytest.raises(RuntimeError, match="finished or preempted"):
        alloc.assign(["b", "c"], np.array([300, 0]))
    assert alloc.release(["a"]) == [int(slots[0])]
    again, _ = alloc.assign(["b", "c"], np.array([301, 0]))
    assert again[0] == slots[1] and again[1] == slots[0]
    assert alloc.last_fresh_rows == [1]
    # A preempted request comes back fresh, wherever it resumes.
    alloc.release(["b"])
    alloc.assign(["b"], np.array([128]))
    assert alloc.last_fresh_rows == [0]


def test_lru_default_is_unchanged_for_other_bridges():
    alloc = StateSlotAllocator(1)
    first, _ = alloc.assign(["a"], np.array([0]))
    second, reset = alloc.assign(["b"], np.array([0]))
    assert first[0] == second[0] and reset == {int(second[0])}


def test_sizing_prices_real_blocks_and_native_checkpoint_ranges():
    vc = _vcfg()
    s = v4_state_checkpoint_sizing(vc)
    assert s.page_bytes % 256 == 0 and s.page_bytes - s.page_unit_bytes < 256
    assert s.k == -(-s.image_bytes // s.page_unit_bytes)
    # HCA compressor state is not in a 128-aligned state checkpoint; the window rows
    # are.
    assert s.image_bytes < s.slot_bytes
    assert "nocopy=hca_main_kv,hca_main_score" in s.layout_id
    assert len(deepseek_v4_state_checkpoint_layer_names(s.k)) == s.k
    # The slot tail comes out of the tensor, not out of every page.
    tail = -(-s.num_slots * s.slot_bytes // s.page_bytes)
    for blocks in (tail + 64, tail + 1000):
        usable = s.usable_blocks(blocks)
        assert 0 < usable <= blocks - tail
        assert s.carve_bytes(usable) <= blocks * s.page_bytes
        assert s.carve_bytes(usable + 1) > blocks * s.page_bytes


def test_draft_proxy_carves_its_slots_from_its_own_withheld_tail():
    vc = _vcfg()
    vc.speculative_config = NS(num_speculative_tokens=1)
    s = v4_state_checkpoint_sizing(vc)
    tensor_blocks = -(-s.num_slots * s.slot_bytes // s.page_bytes) + 64
    usable = s.usable_blocks(tensor_blocks)
    head = s.page_bytes // (2 * ATOM_DEEPSEEK_V4_BLOCK_SIZE)
    region = tensor_blocks * s.page_bytes
    # vLLM's layer-outer placement of one group holding both proxies: target at
    # byte 0, MTP draft one layer stride later; each view stops at `usable`.
    raw = torch.zeros(2 * region, dtype=torch.uint8)
    draft = raw[region : region + usable * s.page_bytes].view(
        usable, 1, ATOM_DEEPSEEK_V4_BLOCK_SIZE, 2 * head
    )
    planes, arena_rows, widths = _v4_state_layout(vc, _v4_kv_fp8(vc))
    carve = {
        "num_slots": s.num_slots,
        "window_size": s.win_with_spec,
        "kv_fp8": True,
        "arena_planes": planes,
        "arena_rows": arena_rows,
        "row_widths": widths,
    }
    # The draft's slots do not fit inside its view.
    with pytest.raises(ValueError, match="proxy cache too small"):
        slice_deepseek_v4_proxy_cache_views(draft, compress_ratios=RATIOS, **carve)
    for ratios in (RATIOS, [0]):
        source, blocks = _proxy_carve_source(draft, s, ratios)
        assert blocks == usable and source.data_ptr() == draft.data_ptr()
        assert source.numel() <= region
        views = slice_deepseek_v4_proxy_cache_views(
            source, compress_ratios=ratios, num_blocks=blocks, **carve
        )
        views["kv_plane"].view(torch.uint8).fill_(0xFF)
        # The slot rows land in the draft's tail, and nothing reaches the target.
        assert raw[region + usable * s.page_bytes :].any()
        assert not raw[:region].any()
        raw.zero_()
    with pytest.raises(ValueError, match="more than the"):
        _proxy_carve_source(draft, s, RATIOS + [4, 4])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_state_checkpoint_round_trip_through_the_plugin_carve(monkeypatch):
    monkeypatch.setattr(chip_info, "get_gfx", _HOST_GET_GFX)
    vc = _vcfg()
    if not _v4_kv_fp8(vc):
        pytest.skip("this GPU does not run the fp8 KV layout (gfx950 / gfx1250 only)")
    s = v4_state_checkpoint_sizing(vc)
    tensor_blocks = -(-s.num_slots * s.slot_bytes // s.page_bytes) + 6 * s.k + 8
    usable = s.usable_blocks(tensor_blocks)
    head = s.page_bytes // (2 * ATOM_DEEPSEEK_V4_BLOCK_SIZE)
    # vLLM's per-layer view, [blocks, H=1, 128, K+V], covers the schedulable
    # blocks only; the withheld tail follows it in the allocation.
    raw = torch.zeros(tensor_blocks * s.page_bytes, dtype=torch.uint8, device="cuda")
    proxy = raw[: usable * s.page_bytes].view(
        usable, 1, ATOM_DEEPSEEK_V4_BLOCK_SIZE, 2 * head
    )
    planes, arena_rows, widths = _v4_state_layout(vc, _v4_kv_fp8(vc))
    views = slice_deepseek_v4_proxy_cache_views(
        _proxy_through_slot_tail(proxy, s.carve_bytes(usable)),
        compress_ratios=RATIOS,
        num_slots=s.num_slots,
        window_size=s.win_with_spec,
        kv_fp8=True,
        arena_planes=planes,
        arena_rows=arena_rows,
        row_widths=widths,
        num_blocks=usable,
    )
    copier = make_v4_state_checkpoint_copier(
        s, views, num_blocks=usable, device=raw.device
    )
    geo = views["geometry"]

    def slot_bytes(group):
        start, stop = geo.slot_span(geo.physical_slot(group))
        return [
            p.view(torch.uint8).reshape(p.shape[0], -1)[start:stop].clone()
            for p in copier._kv_planes()
        ]

    def fill_slot(group, value=None):
        start, stop = geo.slot_span(geo.physical_slot(group))
        for p in copier._kv_planes():
            rows = p.view(torch.uint8).reshape(p.shape[0], -1)[start:stop]
            if value is None:
                noise = torch.randint(0, 256, rows.shape, dtype=torch.uint8)
                rows.copy_(noise.to(rows.device))
            else:
                rows.fill_(value)

    fill_slot(2)
    source = slot_bytes(2)
    # Non-contiguous, out-of-order ids, as a block pool hands them out.
    units = tuple(range(usable - 1, usable - 1 - 2 * s.k, -2))
    untouched = raw.clone()
    execute_state_checkpoint_stores(copier, [2], [units])
    fill_slot(1, 0)
    execute_state_checkpoint_restores(copier, [1], [units])
    torch.cuda.synchronize()
    restored = slot_bytes(1)
    ranges = copier._checkpoint_slot_ranges()
    carried = 0
    for src, dst, plane_ranges in zip(source, restored, ranges):
        flat_src, flat_dst = src.reshape(-1), dst.reshape(-1)
        mask = torch.zeros_like(flat_dst, dtype=torch.bool)
        for off, n in plane_ranges:
            assert torch.equal(flat_dst[off : off + n], flat_src[off : off + n])
            mask[off : off + n] = True
            carried += n
        # Bytes the state checkpoint does not carry stay as the reset left them.
        assert int(flat_dst[~mask].count_nonzero()) == 0
    assert carried == s.image_bytes
    # Only the slots and the state checkpoint's own units changed in the pool.
    changed = (raw != untouched).nonzero().flatten()
    assert changed.numel() > 0
    region_bases, region_sizes = copier._page_unit_regions()
    base_ptr = raw.data_ptr()
    allowed = []
    for u in units:
        for b, n in zip(region_bases, region_sizes):
            lo = int(b) + u * int(n) - base_ptr
            allowed.append((lo, lo + int(n)))
    for g in (1,):
        start, stop = geo.slot_span(geo.physical_slot(g))
        for p, w in zip(copier._kv_planes(), widths):
            lo = p.data_ptr() - base_ptr + start * w
            allowed.append((lo, lo + (stop - start) * w))
    ok = torch.zeros(raw.numel(), dtype=torch.bool, device=raw.device)
    for lo, hi in allowed:
        ok[lo:hi] = True
    assert bool(ok[changed].all())


def test_copy_funcs_meet_vllms_by_type_contract():
    from vllm.v1.worker.mamba_utils import validate_mamba_state_copy_funcs

    s = v4_state_checkpoint_sizing(_vcfg())
    spec = AtomDeepseekV4StateCheckpointLayer("x", 0, s.page_bytes).get_kv_cache_spec(
        None
    )
    funcs = deepseek_v4_state_checkpoint_copy_funcs({spec.mamba_type})
    validate_mamba_state_copy_funcs({spec: [0]}, funcs)


def test_placement_follows_native_ladder():
    p = V4StateCheckpointPlacement(interval=8192, demand=True)
    # 12,288-token prompt: rung 8192, anchor one block short of the end.
    assert p.limit(12288) == 8192
    anchor = p.anchor(12288, 12288)
    assert anchor == 12160
    assert p.cut(0, 8192, 12288, 0, anchor) == 8192
    assert p.cut(8192, 12288, 12288, 0, anchor) == 12160
    assert p.cut(12160, 12288, 12288, 0, anchor) == 0
    # Earliest candidate wins: a demand below the rung is cut first.
    assert p.cut(0, 8192, 12288, 2944, anchor) == 2944
    # Keeps: rung, anchor, demand; never past the prompt (generation keeps none).
    assert p.keeps(8192, 12288, 0, anchor) and p.keeps(anchor, 12288, 0, anchor)
    assert p.keeps(2944, 12288, 2944, anchor)
    assert not p.keeps(4096, 12288, 0, anchor)
    assert not p.keeps(16384, 12288, 0, anchor)
    # Demand only when the proxy hit is longer than the state checkpoint hit, and room.
    assert p.demand_pos(0, 2944, True) == 2944
    assert p.demand_pos(2944, 2944, True) == 0
    assert p.demand_pos(0, 2944, False) == 0
    assert (
        V4StateCheckpointPlacement(interval=8192, demand=False).demand_pos(
            0, 2944, True
        )
        == 0
    )
    assert p.state_checkpoints_per_request() == 2
    # Generation: native's spacing rule, one interval past the last checkpoint.
    assert p.decode_keeps(9344, 0) and not p.decode_keeps(9344, 2944)
    assert not p.decode_keeps(9345, 0)
    assert not V4StateCheckpointPlacement(interval=-1, demand=True).decode_keeps(
        9344, 0
    )


def test_placement_interval_off_and_ladder_off():
    off = V4StateCheckpointPlacement(interval=0, demand=True)
    assert off.anchor(3000, 3000) == 0
    assert off.demand_pos(0, 2944, True) == 0
    assert not off.keeps(2944, 3000, 2944, 2944)
    assert off.state_checkpoints_per_request() == 0
    ladder_off = V4StateCheckpointPlacement(interval=-1, demand=True)
    assert ladder_off.limit(20000) == 0
    assert ladder_off.anchor(3000, 3000) == 2944
    assert not ladder_off.keeps(8192, 20000, 0, 19968)
    assert ladder_off.keeps(19968, 20000, 0, 19968)


def test_partial_eviction_restores_by_group():
    """vLLM evicts block by block (hybrid Mamba semantics): losing one group's
    block leaves the other groups hashed, the boundary is stored again, and a
    hit takes each group's first cached block -- the new store for the evicted
    group, the old one for the rest."""
    from vllm.v1.core.block_pool import BlockPool

    from atom.plugin.vllm.deepseek_v4_state_checkpoint import (
        V4SchedulerStateCheckpoints,
    )

    gids = [1, 2, 3]
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=128)
    st = object.__new__(V4SchedulerStateCheckpoints)
    st.block_pool = pool
    st.placement = V4StateCheckpointPlacement(interval=8192, demand=True)
    st.state_checkpoint_gids = gids
    st.k = len(gids)
    req = NS(block_hashes=[b"h0", b"h1"], request_id="r")

    def store():
        state_checkpoint = []
        for gid in gids:
            (blk,) = pool.get_new_blocks(1)
            pool.cache_full_blocks(
                request=req,
                blocks=[blk],
                num_cached_blocks=0,
                num_full_blocks=1,
                block_size=128,
                kv_cache_group_id=gid,
            )
            state_checkpoint.append(blk)
        pool.free_blocks(state_checkpoint)
        return state_checkpoint

    a = store()
    assert st._state_checkpoint_cached(req, 0)
    assert pool._maybe_evict_cached_block(a[1])
    assert a[0].block_hash is not None and a[2].block_hash is not None
    assert a[1].block_hash is None
    assert not st._state_checkpoint_cached(req, 0)
    b = store()
    assert st._state_checkpoint_cached(req, 0)
    hit = pool.get_cached_block(b"h0", gids)
    assert [blk.block_id for blk in hit] == [
        a[0].block_id,
        b[1].block_id,
        a[2].block_id,
    ]
