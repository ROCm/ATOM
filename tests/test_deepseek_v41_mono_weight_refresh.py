# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""In-place weight updates and DeepSeek-V4.1's mono decode
(``deepseek_v41.mono.weight_refresh``), on stand-ins, no GPU:

- the registry: a derived weight is rebuilt in place (its address kept), and
  only when a source it reads was rewritten;
- the bind-time audit: every weight pointer in a parameter's storage or a
  registered derived buffer passes; any other is a refusal naming it, raised
  under a runner that takes weight updates;
- the refusals: V4.1 gfx950 answers the shared "rollout weight updates"
  refusal at bind instead (gfx1250 keeps it), and keeps refusing a runner whose
  sleep would move what mono binds;
- the rollout sync's model hooks: a routing wrapper's parameters keep the
  model's names (``weight_sync_target``), the sync's last bucket hands what it
  rewrote to ``after_weight_update``, and ``weight_update_refusal`` refuses the
  sync before anything is written.
"""

from types import SimpleNamespace

import pytest
import torch
from conftest import atom_config_double
from torch import nn

from atom.model_engine.collective_rpc import WeightReloadRefused
from atom.models.deepseek_v41.mono.weight_refresh import (
    DerivedWeights,
    ParameterStorage,
    rebuild_after_weight_update,
    registries_of,
    weight_update_refusal,
)
from atom.mono.runtime.consensus import MonoUnsupported
from atom.rollout.weight_updater import WeightUpdaterMixin
from atom.utils.selector import Family


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.Parameter(torch.arange(8, dtype=torch.float32), False)
        self.w = nn.Parameter(torch.arange(32, dtype=torch.float32).view(4, 8), False)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()


class _Wrapper(nn.Module):
    """A routing wrapper answering the update's model hooks with a registry, as
    V4.1's ``MonoDecodeModel`` does (``test_the_v41_wrapper_answers_the_hooks``
    in ``test_deepseek_v41_mono_weight_update.py`` holds the real one to it)."""

    def __init__(self, model, derived) -> None:
        super().__init__()
        self.model = model
        self.mono_derived = derived

    @property
    def weight_sync_target(self):
        return self.model

    def weight_update_refusal(self):
        return self.mono_derived.refusal()

    def after_weight_update(self, changed):
        self.mono_derived.rebuild(changed)


def _packed(block):
    """A derived copy of ``block.w``: re-laid out (transposed) and re-encoded."""
    return (block.w.t() * 2).contiguous()


def _register(derived, block, packed, counter=None):
    def rebuild():
        if counter is not None:
            counter.append("w")
        packed.copy_(_packed(block))

    return derived.register("packed w", (block.w,), {"w.p": packed}, rebuild)


# ---------------------------------------------------------------- the registry


def test_a_rebuild_rederives_in_place_and_keeps_the_address():
    model, derived = _Model(), DerivedWeights()
    packed = _packed(model.block)
    _register(derived, model.block, packed)
    address = packed.data_ptr()
    model.block.w.data.copy_(torch.full((4, 8), 3.0))

    report = derived.rebuild()

    assert report.rebuilt == ("packed w",)
    assert packed.data_ptr() == address
    assert torch.equal(packed, torch.full((8, 4), 6.0))


def test_only_the_entries_whose_sources_were_rewritten_are_rebuilt():
    model, derived = _Model(), DerivedWeights()
    calls = []
    _register(derived, model.block, _packed(model.block), calls)
    scaled = model.block.norm * 2
    derived.register(
        "norm x2",
        (model.block.norm,),
        {"n": scaled},
        lambda: (calls.append("norm"), scaled.copy_(model.block.norm * 2)),
    )

    assert derived.rebuild([model.block.norm]).rebuilt == ("norm x2",)
    assert calls == ["norm"]
    # a view of a source counts as the source (a fused parameter's rows)
    assert derived.rebuild([model.block.w[1:3]]).rebuilt == ("packed w",)
    assert derived.rebuild([torch.zeros(3)]).rebuilt == ()
    assert len(derived.rebuild().rebuilt) == 2


def test_a_rebuild_that_moves_its_destination_raises():
    model, derived = _Model(), DerivedWeights()
    packed = _packed(model.block)

    def rebuild():
        packed.data = _packed(model.block)  # a new allocation: the old address dangles

    derived.register("moving", (model.block.w,), {"w.p": packed}, rebuild)
    with pytest.raises(RuntimeError, match="moved"):
        derived.rebuild()


def test_a_destination_inside_a_source_is_refused():
    model, derived = _Model(), DerivedWeights()
    with pytest.raises(ValueError, match="overlaps a source"):
        derived.register("self", (model.block.w,), {"w": model.block.w[0]}, print)


def test_parameter_storage_holds_views_not_copies():
    model = _Model()
    storage = ParameterStorage(model, None)
    assert model.block.w in storage
    assert model.block.w.view(-1)[3:9] in storage
    assert model.block.w.t() in storage
    assert model.block.w.clone() not in storage


# ---------------------------------------------------------------- the audit


def test_the_audit_passes_in_place_and_registered_pointers():
    model, derived = _Model(), DerivedWeights(strict=True)
    packed = _packed(model.block)
    _register(derived, model.block, packed)
    weights = {"norm": model.block.norm, "w rows": model.block.w[1:], "w.p": packed}

    assert derived.audit("path", weights, model.parameters()) is None
    assert derived.refusal() is None


def test_the_audit_names_an_unregistered_copy():
    model, derived = _Model(), DerivedWeights()
    copy = model.block.norm.float().clone()
    weights = {"norm": model.block.norm, "norm f32 copy": copy}

    why = derived.audit("V4.1 mono layer kernel", weights, model.parameters())

    assert "V4.1 mono layer kernel" in why and "norm f32 copy" in why
    assert derived.refusal() == why
    assert weight_update_refusal(_Wrapper(model, derived)) == why


def test_a_strict_audit_raises_the_refusal():
    model, derived = _Model(), DerivedWeights(strict=True)
    with pytest.raises(MonoUnsupported, match="a copy"):
        derived.audit("path", {"a copy": model.block.w.clone()}, model.parameters())


def test_the_audit_refuses_a_derived_weight_rebuilt_from_a_copy():
    model, derived = _Model(), DerivedWeights()
    stale = model.block.w.clone()  # e.g. a cache derived once at load
    packed = (stale * 2).contiguous()
    derived.register("from a cache", (stale,), {"p": packed}, print)

    why = derived.audit("path", {"p": packed}, model.parameters())

    assert "from a cache" in why and "stale copy" in why


def test_a_refusal_no_pointer_shows_is_recorded():
    derived = DerivedWeights()
    derived.refuse("draft", "a cached row")
    assert derived.refusal() == "draft: a cached row"
    with pytest.raises(MonoUnsupported):
        DerivedWeights(strict=True).refuse("draft", "a cached row")


# ---------------------------------------------------------------- finding the registry


def test_the_registry_is_found_once():
    model, derived = _Model(), DerivedWeights()
    wrapper = _Wrapper(model, derived)

    # one registry, however many modules reach it
    outer = nn.ModuleDict({"a": wrapper, "b": _Wrapper(nn.Linear(1, 1), derived)})
    assert registries_of(outer) == [derived]
    assert registries_of(model) == []
    assert registries_of(None) == []


def test_rebuild_after_weight_update_walks_the_model():
    model, derived = _Model(), DerivedWeights()
    packed = _packed(model.block)
    _register(derived, model.block, packed)
    model.block.w.data.fill_(1.0)

    report = rebuild_after_weight_update(_Wrapper(model, derived), [model.block.w])

    assert report.rebuilt == ("packed w",)
    assert torch.equal(packed, torch.full((8, 4), 2.0))


# ---------------------------------------------------------------- the refusals


def _v41_atom_config(**overrides):
    mode = SimpleNamespace(uses_tr_hybrid=lambda: False)
    fields = {
        "tensor_parallel_size": 4,
        "enable_expert_parallel": False,
        "enable_dp_attention": False,
        "parallel_config": SimpleNamespace(data_parallel_size=1),
        "pipeline_parallel_size": 1,
        "is_multinode_tp": False,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "enable_tbo": False,
        "enable_tbo_decode": False,
        "eplb_enable": False,
        "compilation_config": SimpleNamespace(level=0, cudagraph_mode=mode),
        "kv_cache_dtype": "bf16",
        "index_cache_dtype": "fp8",
        "speculative_config": SimpleNamespace(
            method="dspark", num_speculative_tokens=5
        ),
        "dspark": SimpleNamespace(confidence_schedule=False, ragged=False),
        "hf_config": SimpleNamespace(
            sliding_window=128,
            index_topk=512,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
            index_n_heads=32,
            index_head_dim=128,
            fp4ptpc=False,
            compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
            candidate_source_layer_id=20,
            engram_layer_ids=[1, 14],
        ),
        "max_model_len": 1 << 20,
        "runner_qualname": "atom.rollout.model_runner_ext.RLHFModelRunner",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def native(monkeypatch):
    from atom.mono.runtime import deployment

    monkeypatch.setattr(deployment, "is_vllm", lambda: False)
    monkeypatch.setattr(deployment, "is_sglang", lambda: False)
    monkeypatch.setattr(deployment, "_takes_weight_updates", lambda c: True)
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    monkeypatch.setenv("ATOM_DSV41_SIDE_STREAMS", "0")
    return deployment


def test_v41_gfx950_audits_and_gfx1250_keeps_the_refusal(native, monkeypatch):
    from aiter_stub import atom_without_aiter

    monkeypatch.setattr(native, "_gfx", lambda: "gfx950")
    with atom_without_aiter() as load:
        v41 = load("atom.models.deepseek_v41.mono.config")
        gfx1250 = load("atom.models.deepseek_v41.mono.gfx1250.config")
        dims = load("atom.models.deepseek_v41.mono.kernels.dims").Dims
        # the imports bound their own references to the deployment's function
        monkeypatch.setattr(
            load("atom.mono.runtime.deployment"),
            "_takes_weight_updates",
            lambda c: True,
        )
        monkeypatch.setattr(v41, "_takes_weight_updates", lambda c: True)
        config = _v41_atom_config(sleep_keeps_memory_resident=True, enforce_eager=False)
        assert v41.config_refusal(config) is None
        # a runner whose sleep would move what mono binds keeps mono off
        assert "sleep_keeps_memory_resident" in v41.config_refusal(_v41_atom_config())
        # the shared refusal itself stands for a model that does not audit
        shared = native.first_refusal(native.common_refusals(config, dims, (4,)))
        assert shared.startswith("rollout weight updates")
        assert gfx1250.config_refusal(config, dims).startswith("rollout weight updates")


# ---------------------------------------------------------------- the rollout sync


def _updater(model):
    class _Updater(WeightUpdaterMixin):
        device = torch.device("cpu")
        label = "test"
        rank = 0
        world_size = 1
        config = atom_config_double()
        attn_family = Family.MHA

        def clear_kv_cache(self):
            pass

    updater = _Updater()
    updater._init_weight_updater()
    updater.model = model
    return updater


def test_the_sync_names_the_parameters_as_the_model_does():
    model, derived = _Model(), DerivedWeights()
    updater = _updater(_Wrapper(model, derived))

    assert updater._sync_target_model() is model
    assert "block.w" in updater._get_param_to_module_mapping()


def test_the_last_bucket_rebuilds_what_the_sync_rewrote_in_place():
    model, derived = _Model(), DerivedWeights()
    packed, calls = _packed(model.block), []
    _register(derived, model.block, packed, calls)
    scaled = model.block.norm * 2
    derived.register(
        "norm x2",
        (model.block.norm,),
        {"n": scaled},
        lambda: calls.append("norm"),
    )
    updater = _updater(_Wrapper(model, derived))
    address = packed.data_ptr()

    updater.update_weights([("block.w", torch.full((4, 8), 5.0))])

    assert calls == ["w"]
    assert packed.data_ptr() == address
    assert torch.equal(packed, torch.full((8, 4), 10.0))


def test_a_failed_audit_refuses_the_sync_before_any_write():
    model, derived = _Model(), DerivedWeights()
    derived.audit("V4.1 mono hi", {"stale": model.block.w.clone()}, model.parameters())
    updater = _updater(_Wrapper(model, derived))
    before = model.block.w.clone()

    with pytest.raises(WeightReloadRefused, match="cannot be reloaded: V4.1 mono hi"):
        updater.update_weights([("block.w", torch.zeros(4, 8))])

    assert torch.equal(model.block.w, before)


def test_a_model_without_mono_rebuilds_nothing():
    model = _Model()
    updater = _updater(model)
    updater.update_weights([("block.w", torch.zeros(4, 8))])
    assert torch.equal(model.block.w, torch.zeros(4, 8))
