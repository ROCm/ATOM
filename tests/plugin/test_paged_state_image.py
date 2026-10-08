# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The model-agnostic image layer, driven by a fake (non-V4) adapter."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

pytest.importorskip("vllm", reason="the image layer patches vLLM")

from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.plugin.vllm import paged_state_image as G
from atom.plugin.vllm.paged_state_image_scheduler import ImagePlacement

PLUGIN = Path(__file__).parents[2] / "atom/plugin/vllm"
BLOCK = 256
UNIT = 1000
SLOT = 2600  # image = whole slot, 3 units (the last one partial)


class _Copier:
    """Slots and PAGE units as plain CPU byte tensors; records every call."""

    def __init__(self, num_slots=4, num_units=16):
        self.checkpoint_spec = PagedStateCheckpointSpec(
            UNIT, SLOT, "fake-layout", image_bytes=SLOT
        )
        self.slots = torch.zeros(num_slots, SLOT, dtype=torch.uint8)
        self.units = torch.zeros(num_units, UNIT, dtype=torch.uint8)
        self.calls = []

    def unit_segments(self, units):
        out, left = [], SLOT
        for u in units:
            take = min(UNIT, left)
            out.append(self.units[u, :take])
            left -= take
        return out

    def execute_paged_state_copies(self, stores, restores, descriptor_slot=0):
        self.calls.append(
            (
                "copies",
                [(o.src_slot, o.unit_ids) for o in stores],
                [(o.dst_slot, o.unit_ids) for o in restores],
                descriptor_slot,
            )
        )
        for op in stores:
            torch._foreach_copy_(
                self.unit_segments(op.unit_ids),
                list(torch.split(self.slots[op.src_slot], self._sizes())),
            )
        for op in restores:
            torch._foreach_copy_(
                list(torch.split(self.slots[op.dst_slot], self._sizes())),
                self.unit_segments(op.unit_ids),
            )

    @staticmethod
    def _sizes():
        return [UNIT, UNIT, SLOT - 2 * UNIT]


class _Backend(G.PagedStateImageBackend):
    block_size = BLOCK

    @staticmethod
    def get_name():
        return "FAKE_IMAGE"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [BLOCK]


class _FakeAdapter(G.PagedStateImageAdapter):
    name = "Fake"
    layer_suffix = "fake_image"
    block_size = BLOCK
    backend_cls = _Backend
    proxy_layer_name = "model.layers.0.fake_proxy"

    def matches(self, vllm_config):
        return "FakeForCausalLM" in vllm_config.model_config.architectures

    def sizing(self, vllm_config):
        return NS(
            spec=PagedStateCheckpointSpec(UNIT, SLOT, "fake-layout", image_bytes=SLOT),
            page_bytes=1024,
            num_slots=4,
            slot_bytes=SLOT,
            usable_blocks=lambda n: n - 11,
        )

    def checkpoint_defaults(self):
        return (4096, True)


@pytest.fixture
def fake():
    a = _FakeAdapter()
    G._adapters.append(a)
    yield a
    G._adapters.remove(a)


def _vcfg(arch="FakeForCausalLM", prefix=True):
    return NS(
        model_config=NS(architectures=[arch], max_model_len=8192),
        cache_config=NS(enable_prefix_caching=prefix, mamba_cache_mode="none"),
        scheduler_config=NS(long_prefill_token_threshold=0),
        speculative_config=None,
    )


def test_generic_layer_imports_no_model_code():
    for name in ("paged_state_image.py", "paged_state_image_scheduler.py"):
        tree = ast.parse((PLUGIN / name).read_text())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                mods.add(node.module or "")
        bad = [
            m
            for m in mods
            if "deepseek" in m or m.startswith(("atom.models", "atom.model_ops"))
        ]
        assert not bad, (name, bad)
        allowed = (
            "vllm",
            "torch",
            "numpy",
            "atom.model_engine.page_unit_checkpoint",
            "atom.plugin.vllm.paged_state_image",
            "atom.plugin.vllm.req_id_passthrough_patch",
        )
        atom_mods = [m for m in mods if m.startswith("atom")]
        assert all(m.startswith(allowed) for m in atom_mods), (name, atom_mods)


def test_registry_routes_by_config_and_by_kv_config(fake):
    assert G.adapter_for_config(_vcfg()) is fake
    assert G.adapter_for_config(_vcfg("LlamaForCausalLM")) is None
    cfg = NS(kv_cache_groups=[NS(layer_names=[fake.proxy_layer_name])])
    assert G.adapter_for_kv_cache_config(cfg) is fake
    assert (
        G.adapter_for_kv_cache_config(NS(kv_cache_groups=[NS(layer_names=["x"])]))
        is None
    )
    assert G.images_on(fake, _vcfg()) and not G.images_on(fake, _vcfg(prefix=False))
    assert G.register_image_adapter(_FakeAdapter()) is fake


def test_other_models_pass_through_the_kv_config_hook(fake):
    sentinel = object()
    called = []

    def original(vc, specs, mem):
        called.append(vc)
        return sentinel

    vc = _vcfg("LlamaForCausalLM")
    assert G._kv_cache_configs_hook(original, vc, {}, 0) is sentinel
    assert called == [vc] and vc.cache_config.mamba_cache_mode == "none"


def _groups(fake, k, block=BLOCK, page=1024):
    from vllm.v1.kv_cache_interface import MambaSpec

    names = fake.image_layer_names(k)
    proxy = NS(
        layer_names=[fake.proxy_layer_name],
        kv_cache_spec=NS(block_size=block, page_size_bytes=page),
    )
    imgs = [
        NS(
            layer_names=[n],
            kv_cache_spec=MambaSpec(
                block_size=block,
                shapes=((page,),),
                dtypes=(torch.uint8,),
                mamba_cache_mode="align",
            ),
        )
        for n in names
    ]
    shared = [fake.proxy_layer_name, *names]
    return NS(
        kv_cache_groups=[proxy, *imgs],
        kv_cache_tensors=[NS(shared_by=shared)],
        num_blocks=100,
    )


def test_group_check_takes_k_and_block_from_the_adapter(fake):
    sizing = fake.sizing(None)
    assert sizing.spec.units_per_checkpoint == 3
    assert fake.image_layer_names(3) == [
        f"model.layers.{i}.fake_image" for i in (1, 2, 3)
    ]
    G.check_image_groups(_groups(fake, 3), fake, sizing)
    with pytest.raises(RuntimeError, match="need k=3"):
        G.check_image_groups(_groups(fake, 2), fake, sizing)
    with pytest.raises(RuntimeError, match="block size 256"):
        G.check_image_groups(_groups(fake, 3, block=128), fake, sizing)


def test_kv_config_hook_aligns_and_withholds_slots(fake):
    vc = _vcfg()

    def original(vc_, specs, mem):
        return [_groups(fake, 3)]

    (cfg,) = G._kv_cache_configs_hook(original, vc, {}, 0)
    assert vc.cache_config.mamba_cache_mode == "align"
    assert cfg.num_blocks == 100 - 11
    vc.speculative_config = object()
    with pytest.raises(ValueError, match="Fake prefix caching"):
        G._kv_cache_configs_hook(original, vc, {}, 0)


def test_placement_at_another_block_size():
    p = ImagePlacement(interval=4096, demand=True, block=BLOCK)
    assert p.anchor(5000, 5000) == 4864
    assert p.cut(0, 5000, 5000, 0, 4864) == 4096
    assert p.cut(4096, 5000, 5000, 0, 4864) == 4864
    assert p.decode_keeps(4096 + 4096, 4096) and not p.decode_keeps(8192 + 128, 4096)
    with pytest.raises(TypeError):
        ImagePlacement(interval=4096, demand=True)  # no block-size default


def test_step_order_and_one_store_per_plan(fake, monkeypatch):
    copier = _Copier()
    worker = G.ImageWorker(fake, copier, [1, 2, 3])
    events = []
    md = NS(
        reset_slots={2},
        image_restore_slots=[2],
        image_restore_units=[(4, 5, 6)],
        image_store_slots=[2],
        image_store_units=[(7, 8, 9)],
    )
    worker.pending_stores = (
        [0],
        [(1, 2, 3)],
    )  # left by a step whose forward did not flush

    def plan():
        events.append(("plan", list(copier.calls)))

    def reset(slots):
        events.append(("reset", sorted(slots), len(copier.calls)))

    G.run_image_step(
        worker, md, capturing=False, plan=plan, reset_fn=reset, name="Fake"
    )
    # Left-over store first, then plan, reset, restore; this step's store waits.
    assert copier.calls[0] == ("copies", [(0, (1, 2, 3))], [], 1)
    assert events[0] == ("plan", [copier.calls[0]])
    assert events[1] == ("reset", [2], 1)
    assert copier.calls[1] == ("copies", [], [(2, (4, 5, 6))], 0)
    assert worker.pending_stores == ([2], [(7, 8, 9)])
    host = NS(_atom_image_worker=worker)
    G.flush_image_stores(host)
    G.flush_image_stores(host)
    assert copier.calls[2:] == [("copies", [(2, (7, 8, 9))], [], 1)]
    # Capture path: nothing planned, nothing restored or stored.
    md2 = NS(reset_slots=set())
    G.run_image_step(
        worker, md2, capturing=True, plan=lambda: 1 / 0, reset_fn=reset, name="Fake"
    )
    assert len(copier.calls) == 3


def test_stores_run_once_right_after_the_runner_forward(fake):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    G._apply_runner_patches()
    assert getattr(GPUModelRunner._model_forward, "_atom_state_image_patched", False)
    copier = _Copier()
    worker = G.ImageWorker(fake, copier, [1, 2, 3])
    calls = []

    class _Runner:
        def model(self, **kw):
            calls.append(("forward", len(copier.calls)))
            return "hidden"

    runner = _Runner()
    before = G.store_counts()
    G.register_image_worker(worker)
    try:
        md = NS(
            reset_slots=set(), image_store_slots=[3], image_store_units=[(10, 11, 12)]
        )
        G.run_image_step(
            worker, md, capturing=False, plan=lambda: None, reset_fn=None, name="Fake"
        )
        assert copier.calls == [] and worker.pending_stores is not None
        assert (
            GPUModelRunner._model_forward(runner, input_ids=None, positions=None)
            == "hidden"
        )
        # The store is issued after the forward (nothing was copied while it ran), once.
        assert calls == [("forward", 0)]
        assert copier.calls == [("copies", [(3, (10, 11, 12))], [], 1)]
        GPUModelRunner._model_forward(runner, input_ids=None, positions=None)
        assert len(copier.calls) == 1 and worker.pending_stores is None
        after = G.store_counts()
        assert after["planned"] - before["planned"] == 1
        assert after["after_forward"] - before["after_forward"] == 1
        assert after["at_next_build"] == before["at_next_build"]
    finally:
        G._workers.pop(fake.name, None)


def test_a_store_never_runs_inside_a_capture(fake, monkeypatch):
    worker = G.ImageWorker(fake, _Copier(), [1, 2, 3])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    G.flush_worker(worker)  # nothing pending: nothing to refuse
    worker.pending_stores = ([0], [(1, 2, 3)])
    with pytest.raises(RuntimeError, match="capture"):
        G.flush_worker(worker)
    assert worker.copier.calls == []


def test_a_left_over_store_runs_first_at_the_next_build_and_is_counted(fake):
    copier = _Copier()
    worker = G.ImageWorker(fake, copier, [1, 2, 3])
    worker.pending_stores = ([0], [(1, 2, 3)])
    before = G.store_counts()
    G.run_image_step(
        worker,
        NS(reset_slots=set()),
        capturing=False,
        plan=lambda: None,
        reset_fn=None,
        name="Fake",
    )
    assert copier.calls == [("copies", [(0, (1, 2, 3))], [], 1)]
    assert G.store_counts()["at_next_build"] - before["at_next_build"] == 1


def test_store_restore_round_trip(fake, monkeypatch):
    copier = _Copier()
    copier.slots[1] = torch.randint(0, 256, (SLOT,), dtype=torch.uint8)
    src = copier.slots[1].clone()
    G.execute_image_stores(copier, [1], [(9, 3, 12)])
    G.execute_image_restores(copier, [2], [(9, 3, 12)])
    assert torch.equal(copier.slots[2], src)


def test_plan_refuses_an_unrestored_resume_without_images(monkeypatch):
    md = NS()
    with pytest.raises(RuntimeError, match="Fake request"):
        G.plan_image_ops(
            None,
            md,
            rows=2,
            slots=np.array([0, 1]),
            chunk_start=np.array([0, 512]),
            lens=np.array([5, 7]),
            fresh_rows=[1],
            block=BLOCK,
            is_target=True,
            name="Fake",
        )
    G.plan_image_ops(
        None,
        md,
        rows=2,
        slots=np.array([0, 1]),
        chunk_start=np.array([0, 512]),
        lens=np.array([5, 7]),
        fresh_rows=[1],
        block=BLOCK,
        is_target=False,
        name="Fake",
    )
    assert md.image_restore_rows == [] and md.image_store_rows == []


def test_v4_adapter_spec_is_native_and_k_follows_it():
    pytest.importorskip("aiter", reason="the V4 bridge reaches AITER dtypes")
    from atom.plugin.vllm.deepseek_v4_image import (
        register_v4_image_adapter,
        v4_image_sizing,
    )

    ratios = [0, 0, 4, 128, 4, 128, 4, 0]
    hf = NS(
        compress_ratios=ratios,
        num_hidden_layers=len(ratios) - 1,
        head_dim=512,
        index_head_dim=128,
        qk_rope_head_dim=64,
        sliding_window=128,
        index_topk=512,
    )
    vc = NS(
        model_config=NS(
            hf_config=hf, max_model_len=4096, architectures=["DeepseekV4ForCausalLM"]
        ),
        scheduler_config=NS(max_num_seqs=4),
        cache_config=NS(cache_dtype="fp8", enable_prefix_caching=True),
        speculative_config=None,
    )
    a = register_v4_image_adapter()
    s = a.sizing(vc)
    assert isinstance(s.spec, PagedStateCheckpointSpec)
    assert (
        s.spec.page_unit_bytes,
        s.spec.slot_bytes,
        s.spec.image_bytes,
        s.spec.layout_id,
    ) == (s.page_unit_bytes, s.slot_bytes, s.image_bytes, s.layout_id)
    assert s.k == s.spec.units_per_checkpoint == -(-s.image_bytes // s.page_unit_bytes)
    assert a.block_size == 128 and a.matches(vc) and s == v4_image_sizing(vc)
