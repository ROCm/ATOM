# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The model-agnostic state checkpoint layer, driven by a fake (non-V4) adapter."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

pytest.importorskip("vllm", reason="the state checkpoint layer patches vLLM")

from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.plugin.vllm import paged_state_checkpoint as G
from atom.plugin.vllm.paged_state_checkpoint_scheduler import StateCheckpointPlacement

PLUGIN = Path(__file__).parents[2] / "atom/plugin/vllm"
BLOCK = 256
UNIT = 1000
SLOT = 2600  # state checkpoint = whole slot, 3 units (the last one partial)


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


class _Backend(G.PagedStateCheckpointBackend):
    block_size = BLOCK

    @staticmethod
    def get_name():
        return "FAKE_STATE_CHECKPOINT"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [BLOCK]


class _FakeAdapter(G.PagedStateCheckpointAdapter):
    name = "Fake"
    layer_suffix = "fake_state_checkpoint"
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
    for name in ("paged_state_checkpoint.py", "paged_state_checkpoint_scheduler.py"):
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
            "atom.plugin.vllm.paged_state_checkpoint",
            "atom.plugin.vllm.req_id_passthrough_patch",
            "atom.utils",
        )
        atom_mods = [m for m in mods if m.startswith("atom")]
        assert all(m.startswith(allowed) for m in atom_mods), (name, atom_mods)


def test_state_checkpoint_env_vars_are_registered_and_documented():
    from atom.utils import envs

    doc = (Path(__file__).parents[2] / "docs/environment_variables.md").read_text()
    names = (
        "ATOM_STATE_CHECKPOINT_INTERVAL_TOKENS",
        "ATOM_STATE_CHECKPOINT_DEMAND",
        "ATOM_STATE_CHECKPOINT_CHECK_BLOCKS",
        "ATOM_V4_CG_INSTR_DIR",
    )
    assert [n for n in names if n not in envs.environment_variables] == []
    assert [n for n in names if f"**{n}**" not in doc] == []
    # Every ATOM_* the state checkpoint layer and the V4 cudagraph patch read is
    # registered.
    for name in ("paged_state_checkpoint.py", "paged_state_checkpoint_scheduler.py"):
        src = (PLUGIN / name).read_text()
        assert "os.environ" not in src and "os.getenv" not in src, name


def test_registry_routes_by_config_and_by_kv_config(fake):
    assert G.adapter_for_config(_vcfg()) is fake
    assert G.adapter_for_config(_vcfg("LlamaForCausalLM")) is None
    cfg = NS(kv_cache_groups=[NS(layer_names=[fake.proxy_layer_name])])
    assert G.adapter_for_kv_cache_config(cfg) is fake
    assert (
        G.adapter_for_kv_cache_config(NS(kv_cache_groups=[NS(layer_names=["x"])]))
        is None
    )
    assert G.state_checkpoints_on(fake, _vcfg()) and not G.state_checkpoints_on(
        fake, _vcfg(prefix=False)
    )
    assert G.register_state_checkpoint_adapter(_FakeAdapter()) is fake


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

    names = fake.state_checkpoint_layer_names(k)
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
        kv_cache_tensors=[
            NS(layers=[n], offset=0, layer_stride=100 * page, block_stride=page)
            for n in shared
        ],
        num_blocks=100,
    )


def test_group_check_takes_k_and_block_from_the_adapter(fake):
    sizing = fake.sizing(None)
    assert sizing.spec.units_per_checkpoint == 3
    assert fake.state_checkpoint_layer_names(3) == [
        f"model.layers.{i}.fake_state_checkpoint" for i in (1, 2, 3)
    ]
    G.check_state_checkpoint_groups(_groups(fake, 3), fake, sizing)
    with pytest.raises(RuntimeError, match="need k=3"):
        G.check_state_checkpoint_groups(_groups(fake, 2), fake, sizing)
    with pytest.raises(RuntimeError, match="block size 256"):
        G.check_state_checkpoint_groups(_groups(fake, 3, block=128), fake, sizing)
    apart = _groups(fake, 3)
    apart.kv_cache_tensors[2].offset = 100 * 1024
    with pytest.raises(RuntimeError, match="not at byte 0"):
        G.check_state_checkpoint_groups(apart, fake, sizing)


def test_kv_config_hook_aligns_and_withholds_slots(fake):
    vc = _vcfg()

    def original(vc_, specs, mem):
        return [_groups(fake, 3)]

    specs = [{fake.proxy_layer_name: object()}]
    (cfg,) = G._kv_cache_configs_hook(original, vc, specs, 0)
    assert vc.cache_config.mamba_cache_mode == "align"
    assert cfg.num_blocks == 100 - 11
    vc.speculative_config = object()
    with pytest.raises(ValueError, match="Fake prefix caching .* drafter's state"):
        G._kv_cache_configs_hook(original, vc, specs, 0)


@pytest.mark.parametrize("eagle", [True, False])
def test_a_family_whose_draft_is_in_the_slot_runs_state_checkpoints_with_a_drafter(
    fake, monkeypatch, eagle
):
    monkeypatch.setattr(fake, "drafter_state_in_checkpoint", lambda vc: True)
    vc = _vcfg()
    vc.speculative_config = NS(use_eagle=lambda: eagle, disable_eagle_block_drop=False)
    specs = [{fake.proxy_layer_name: object()}]
    (cfg,) = G._kv_cache_configs_hook(lambda *a: [_groups(fake, 3)], vc, specs, 0)
    assert vc.cache_config.mamba_cache_mode == "align"
    assert cfg.num_blocks == 100 - 11
    # An EAGLE-class drafter: native's hit rule, through vLLM's own switch.
    assert vc.speculative_config.disable_eagle_block_drop is eagle


def test_state_checkpoints_refuse_a_drafter_the_family_does_not_keep_in_the_slot(
    fake, monkeypatch
):
    # The capability is per drafter, not per family: only the drafter the
    # family binds into the target's slot gets state checkpoints (and the hit rule).
    monkeypatch.setattr(
        fake,
        "drafter_state_in_checkpoint",
        lambda vc: vc.speculative_config.method == "mtp",
    )
    monkeypatch.setattr(fake, "state_checkpoint_drafters", "method 'mtp'")
    specs = [{fake.proxy_layer_name: object()}]
    for method in ("eagle3", "dspark", "ngram"):
        vc = _vcfg()
        vc.speculative_config = NS(
            method=method, use_eagle=lambda: True, disable_eagle_block_drop=False
        )
        with pytest.raises(ValueError) as e:
            G._kv_cache_configs_hook(lambda *a: [_groups(fake, 3)], vc, specs, 0)
        msg = str(e.value)
        assert f"speculative method '{method}'" in msg and "(method 'mtp')" in msg
        assert "--no-enable-prefix-caching" in msg
        assert vc.speculative_config.disable_eagle_block_drop is False
        assert vc.cache_config.mamba_cache_mode == "none"
        # Without prefix caching there are no state checkpoints, and nothing to refuse.
        vc.cache_config.enable_prefix_caching = False
        G._kv_cache_configs_hook(lambda *a: [_groups(fake, 3)], vc, specs, 0)
        assert vc.speculative_config.disable_eagle_block_drop is False
    vc = _vcfg()
    vc.speculative_config = NS(
        method="mtp", use_eagle=lambda: True, disable_eagle_block_drop=False
    )
    G._kv_cache_configs_hook(lambda *a: [_groups(fake, 3)], vc, specs, 0)
    assert vc.speculative_config.disable_eagle_block_drop is True


def test_a_matching_model_without_the_proxy_layer_passes_through(fake):
    sentinel = object()
    vc = _vcfg()
    vc.speculative_config = object()
    out = G._kv_cache_configs_hook(lambda *a: sentinel, vc, [{"x.attn": 0}], 0)
    assert out is sentinel and vc.cache_config.mamba_cache_mode == "none"


def test_placement_at_another_block_size():
    p = StateCheckpointPlacement(interval=4096, demand=True, block=BLOCK)
    assert p.anchor(5000, 5000) == 4864
    assert p.cut(0, 5000, 5000, 0, 4864) == 4096
    assert p.cut(4096, 5000, 5000, 0, 4864) == 4864
    assert p.decode_keeps(4096 + 4096, 4096) and not p.decode_keeps(8192 + 128, 4096)
    with pytest.raises(TypeError):
        StateCheckpointPlacement(interval=4096, demand=True)  # no block-size default


def test_step_order_and_one_store_per_plan(fake, monkeypatch):
    copier = _Copier()
    worker = G.StateCheckpointWorker(fake, copier, [1, 2, 3])
    events = []
    md = NS(
        reset_slots={2},
        state_checkpoint_restore_slots=[2],
        state_checkpoint_restore_units=[(4, 5, 6)],
        state_checkpoint_store_slots=[2],
        state_checkpoint_store_units=[(7, 8, 9)],
    )

    def plan():
        events.append(("plan", list(copier.calls)))

    def reset(slots):
        events.append(("reset", sorted(slots), len(copier.calls)))

    G.run_state_checkpoint_step(
        worker, md, capturing=False, plan=plan, reset_fn=reset, name="Fake"
    )
    # Plan, reset, restore; this step's store waits for the forward.
    assert events[0] == ("plan", [])
    assert events[1] == ("reset", [2], 0)
    assert copier.calls[0] == ("copies", [], [(2, (4, 5, 6))], 0)
    assert worker.pending_stores == ([2], [(7, 8, 9)])
    host = NS(_atom_state_checkpoint_worker=worker)
    G.flush_state_checkpoint_stores(host)
    G.flush_state_checkpoint_stores(host)
    assert copier.calls[1:] == [("copies", [(2, (7, 8, 9))], [], 1)]
    # Capture path: nothing planned, nothing restored or stored.
    md2 = NS(reset_slots=set())
    G.run_state_checkpoint_step(
        worker, md2, capturing=True, plan=lambda: 1 / 0, reset_fn=reset, name="Fake"
    )
    assert len(copier.calls) == 2


def test_stores_run_once_right_after_the_runner_forward(fake):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    G._apply_runner_patches()
    assert getattr(
        GPUModelRunner._model_forward, "_atom_state_checkpoint_patched", False
    )
    copier = _Copier()
    worker = G.StateCheckpointWorker(fake, copier, [1, 2, 3])
    calls = []

    class _Runner:
        def model(self, **kw):
            calls.append(("forward", len(copier.calls)))
            return "hidden"

    runner = _Runner()
    before = G.store_counts()
    G.register_state_checkpoint_worker(worker)
    try:
        md = NS(
            reset_slots=set(),
            state_checkpoint_store_slots=[3],
            state_checkpoint_store_units=[(10, 11, 12)],
        )
        G.run_state_checkpoint_step(
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


class _DraftRunner:
    """A runner with a drafter: the draft forward writes the slot after the
    target's (vLLM: ``sample_tokens`` -> ``propose_draft_token_ids``)."""

    speculative_config = object()

    def __init__(self, copier, slot):
        self.copier, self.slot = copier, slot

    def model(self, **kw):
        self.copier.slots[self.slot, :UNIT] = 1
        return "hidden"

    def draft(self):
        self.copier.slots[self.slot, UNIT:] = 2


def test_with_a_drafter_stores_run_at_the_next_build_after_both_forwards(fake):
    """Native drains a step's stores into the next batch's build, after the
    target and the draft forward; without a drafter they stay right after the
    target forward (``test_stores_run_once_right_after_the_runner_forward``)."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    G._apply_runner_patches()
    copier = _Copier()
    worker = G.StateCheckpointWorker(fake, copier, [1, 2, 3])
    runner = _DraftRunner(copier, 3)
    before = G.store_counts()
    G.register_state_checkpoint_worker(worker)
    try:
        md = NS(
            reset_slots=set(),
            state_checkpoint_store_slots=[3],
            state_checkpoint_store_units=[(10, 11, 12)],
        )
        G.run_state_checkpoint_step(
            worker, md, capturing=False, plan=lambda: None, reset_fn=None, name="Fake"
        )
        GPUModelRunner._model_forward(runner, input_ids=None, positions=None)
        assert copier.calls == [] and worker.forward_ran
        runner.draft()
        expected = copier.slots[3].clone()
        events = []
        G.run_state_checkpoint_step(
            worker,
            NS(reset_slots={3}),
            capturing=False,
            plan=lambda: events.append(("plan", len(copier.calls))),
            reset_fn=lambda s: events.append(("reset", len(copier.calls))),
            name="Fake",
        )
        # Issued first in the next build, before its plan, resets and restores,
        # with both forwards' rows.
        assert copier.calls == [("copies", [(3, (10, 11, 12))], [], 1)]
        assert events == [("plan", 1), ("reset", 1)]
        assert torch.equal(
            torch.cat([copier.units[u] for u in (10, 11)]), expected[: 2 * UNIT]
        )
        assert torch.equal(copier.units[12, : SLOT - 2 * UNIT], expected[2 * UNIT :])
        assert worker.pending_stores is None and not worker.forward_ran
        after = G.store_counts()
        assert after["next_build"] - before["next_build"] == 1
        assert after["after_forward"] == before["after_forward"]
        assert after["at_next_build"] == before["at_next_build"]
    finally:
        G._workers.pop(fake.name, None)


def test_with_a_drafter_a_store_whose_forward_never_ran_is_still_refused(fake):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    G._apply_runner_patches()
    copier = _Copier()
    worker = G.StateCheckpointWorker(fake, copier, [1, 2, 3])
    runner = _DraftRunner(copier, 2)
    G.register_state_checkpoint_worker(worker)
    try:
        md = NS(
            reset_slots=set(),
            state_checkpoint_store_slots=[2],
            state_checkpoint_store_units=[(7, 8, 9)],
        )
        G.run_state_checkpoint_step(
            worker, md, capturing=False, plan=lambda: None, reset_fn=None, name="Fake"
        )
        # A forward from before this build (or a dummy run) marks nothing.
        assert not worker.forward_ran
        before = G.store_counts()
        with pytest.raises(RuntimeError, match="never ran"):
            G.run_state_checkpoint_step(
                worker,
                NS(reset_slots=set()),
                capturing=False,
                plan=lambda: None,
                reset_fn=None,
                name="Fake",
            )
        assert copier.calls == []
        assert G.store_counts()["at_next_build"] - before["at_next_build"] == 1
        # A capture build never issues one, even after the forward ran.
        worker.pending_stores = ([2], [(7, 8, 9)])
        GPUModelRunner._model_forward(runner, input_ids=None, positions=None)
        G.run_state_checkpoint_step(
            worker,
            NS(reset_slots=set()),
            capturing=True,
            plan=lambda: 1 / 0,
            reset_fn=None,
            name="Fake",
        )
        assert copier.calls == [] and worker.pending_stores is not None
    finally:
        G._workers.pop(fake.name, None)


def test_vllm_runs_the_drafter_between_the_target_forward_and_the_next_build():
    """What the next-build site relies on in vLLM's V1 runner: the target
    forward runs in ``execute_model`` after the step's metadata build, and the
    drafter in ``sample_tokens``, which the engine calls before the next
    ``execute_model``."""
    import inspect

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    execute = inspect.getsource(GPUModelRunner.execute_model)
    assert execute.index("self._build_attention_metadata(") < execute.index(
        "self._model_forward("
    )
    assert "self.propose_draft_token_ids(" not in execute
    assert "self.propose_draft_token_ids(" in inspect.getsource(
        GPUModelRunner.sample_tokens
    )


def test_a_store_never_runs_inside_a_capture(fake, monkeypatch):
    worker = G.StateCheckpointWorker(fake, _Copier(), [1, 2, 3])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    G.flush_worker(worker)  # nothing pending: nothing to refuse
    worker.pending_stores = ([0], [(1, 2, 3)])
    with pytest.raises(RuntimeError, match="capture"):
        G.flush_worker(worker)
    assert worker.copier.calls == []


def test_a_store_whose_forward_never_ran_is_refused_not_copied(fake):
    """A build registers a store for its boundary (the scheduler has already
    hashed the state checkpoint blocks for it); the forward that would have advanced the
    slot there never runs. The next build must not copy the slot, which still
    holds an earlier position's state, into those blocks."""
    copier = _Copier()
    copier.slots[2] = torch.randint(0, 256, (SLOT,), dtype=torch.uint8)
    worker = G.StateCheckpointWorker(fake, copier, [1, 2, 3])
    md = NS(
        reset_slots=set(),
        state_checkpoint_store_slots=[2],
        state_checkpoint_store_units=[(7, 8, 9)],
    )
    G.run_state_checkpoint_step(
        worker, md, capturing=False, plan=lambda: None, reset_fn=None, name="Fake"
    )
    units_before = copier.units.clone()
    before = G.store_counts()
    # No GPUModelRunner._model_forward here.
    with pytest.raises(RuntimeError, match="never ran"):
        G.run_state_checkpoint_step(
            worker,
            NS(reset_slots=set()),
            capturing=False,
            plan=lambda: None,
            reset_fn=None,
            name="Fake",
        )
    assert copier.calls == [] and torch.equal(copier.units, units_before)
    assert G.store_counts()["at_next_build"] - before["at_next_build"] == 1


def test_store_restore_round_trip(fake, monkeypatch):
    copier = _Copier()
    copier.slots[1] = torch.randint(0, 256, (SLOT,), dtype=torch.uint8)
    src = copier.slots[1].clone()
    G.execute_state_checkpoint_stores(copier, [1], [(9, 3, 12)])
    G.execute_state_checkpoint_restores(copier, [2], [(9, 3, 12)])
    assert torch.equal(copier.slots[2], src)


def test_plan_refuses_an_unrestored_resume_without_state_checkpoints(monkeypatch):
    md = NS()
    with pytest.raises(RuntimeError, match="Fake request"):
        G.plan_state_checkpoint_ops(
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
    G.plan_state_checkpoint_ops(
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
    assert (
        md.state_checkpoint_restore_rows == [] and md.state_checkpoint_store_rows == []
    )


def test_v4_adapter_spec_is_native_and_k_follows_it(monkeypatch):
    pytest.importorskip("aiter", reason="the V4 bridge reaches AITER dtypes")
    from vllm.model_executor.models.registry import ModelRegistry

    monkeypatch.setitem(
        ModelRegistry.models,
        "DeepseekV4ForCausalLM",
        NS(module_name="atom.plugin.vllm.model_wrapper", class_name="X"),
    )
    from atom.plugin.vllm.deepseek_v4_state_checkpoint import (
        register_v4_state_checkpoint_adapter,
        v4_state_checkpoint_sizing,
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
    a = register_v4_state_checkpoint_adapter()
    s = a.sizing(vc)
    assert isinstance(s.spec, PagedStateCheckpointSpec)
    assert (
        s.spec.page_unit_bytes,
        s.spec.slot_bytes,
        s.spec.image_bytes,
        s.spec.layout_id,
    ) == (s.page_unit_bytes, s.slot_bytes, s.image_bytes, s.layout_id)
    assert s.k == s.spec.units_per_checkpoint == -(-s.image_bytes // s.page_unit_bytes)
    assert a.block_size == 128 and a.matches(vc) and s == v4_state_checkpoint_sizing(vc)
