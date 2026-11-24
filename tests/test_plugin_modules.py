# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Recipe v2 (`atom.plugin_recipe/2`) and whole-module takeover (`plugin_host`).

CPU only: plugins are stand-in entry points over a toy `nn.Module` tree; the
graph state a capture would report is patched in.
"""

import pickle
import re
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from atom import plugin_host as host
from atom import plugin_recipe as recipes

REPO = Path(__file__).resolve().parents[1]
ARCH = "DeepseekV41ForCausalLM"


class _EntryPoint:
    def __init__(self, register):
        self._register = register

    def load(self):
        return self._register


@pytest.fixture(autouse=True)
def fresh_host(monkeypatch):
    monkeypatch.setattr(host, "_entry_points", dict)
    monkeypatch.setattr(host, "_configure_aiter", lambda names: None)
    monkeypatch.setattr(host, "_graph_state", lambda: (False, None))
    monkeypatch.setattr(host, "_valid_rows", lambda rows: None)
    host._reset_for_testing()
    yield
    host._reset_for_testing()


def _plugins(monkeypatch, **registers):
    eps = {name: _EntryPoint(fn) for name, fn in registers.items()}
    monkeypatch.setattr(host, "_entry_points", lambda: eps)


def _built():
    """What a model build leaves: a bound seam (hot-plug needs a built model)."""
    host.bind("v41.attn.out", lambda *a, **k: (None, None))


def _v2(**fields):
    return {"schema": recipes.RECIPE_SCHEMA_V2, "arch": ARCH, **fields}


def _plan(**fields):
    return recipes.resolve([_v2(**fields)])


# -- a toy model: stack.N.{attn, ffn.shared} -------------------------------


class Scale(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.w = nn.Parameter(torch.full((1,), float(k)))

    def forward(self, x):
        return x * self.w


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = Scale(2)
        self.routed = Scale(3)

    def forward(self, x):
        return self.shared(x) + self.routed(x)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = FFN()
        self.attn = Scale(5)

    def forward(self, x):
        return self.ffn(self.attn(x))


class Toy(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.stack = nn.ModuleList(Layer() for _ in range(n))

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return x


def _module_owner(slot, owner, *, value=None, select=None, seen=None, **kw):
    """An entry point offering ``owner`` for ``slot``: its forward returns a
    constant (or reads the live module's weight when ``value`` is None)."""

    def factory(module, capability):
        if seen is not None:
            seen.append((capability.flags["path"], module))
        if value is None:
            return lambda x: x * module.w * 100
        return lambda *args: torch.full_like(args[0], float(value))

    def register(h):
        h.register_module(slot=slot, factory=factory, owner=owner, select=select, **kw)

    return register


# -- recipe v2 ---------------------------------------------------------------


def test_v2_options_parse_and_key_the_plan():
    plan = _plan(
        engine={"prefill_short_first_rows": 96, "prefill_cost_ceil": True},
        model_options={
            "ffn_mhc_fuse_quant": True,
            "fused_attn": True,
            "fused_attn_min_rows": 128,
            "expert_placement": {"3": [1, 0]},
            "mono_mode": "off",
        },
        dspark_options={"fused_prologue": True, "verify_argmax_once": True},
    )
    assert plan.schema_version == 2 and plan.has_options and not plan.empty
    assert plan.engine.prefill_short_first_rows == 96
    assert plan.engine.prefill_cost_ceil and plan.engine.prefill_long_cap_busy == 0
    assert plan.model_options.placement_spec() == {"3": [1, 0]}
    assert plan.dspark_options.fused_prologue and not plan.dspark_options.lean_head
    other = _plan(engine={"prefill_short_first_rows": 64})
    assert other.plan_id != plan.plan_id
    copy = pickle.loads(pickle.dumps(plan))
    assert copy == plan and copy.dump() == plan.dump()


@pytest.mark.parametrize(
    "fields",
    [
        {"engine": {"prefill_short_first_rows": -1}},
        {"engine": {"no_such_knob": 1}},
        {"engine": {"prefill_decode_interval_slope": -1}},
        {"engine": {"prefill_decode_interval_slope": "2"}},
        {"engine": {"prefill_decode_interval_slope": 2}},
        {"engine": {"prefill_decode_interval_slope_from": 1.5}},
        {"engine": {"prefill_admit_order": "shortest"}},
        {"engine": {"prefill_admit_max_age_ms": -1}},
        {"engine": {"thp_disable": 1}},
        {"engine": {"engram_pin_pages": "1"}},
        {"engine": {"prefill_short_first_rows": 96, "prefill_admit_order": "srpt"}},
        {"model_options": {"fused_attn": "yes"}},
        {"model_options": {"mono_mode": "sometimes"}},
        {"dspark_options": {"lean_head": 1}},
        {"dspark_options": {"dp_sharded_head": "on"}},
        {"dspark_options": {"dp_sharded_head_max_rows": 0}},
        {"dspark_options": {"sampler_skip_greedy_pick": 1}},
        {"dspark_options": {"verify_skip_unused_target_draw": None}},
        {"modules": {"layers..ffn": ["a"]}},
    ],
)
def test_v2_refuses_a_bad_option(fields):
    with pytest.raises(recipes.RecipeRefused):
        _plan(**fields)


def test_the_scheduler_options_of_a_load_dependent_interval_and_an_admit_order():
    plan = _plan(
        engine={
            "prefill_short_first_rows": 96,
            "prefill_decode_interval_busy": 110,
            "prefill_decode_interval_slope": 2,
            "prefill_decode_interval_slope_from": 30,
            "prefill_admit_order": "fifo",
        }
    )
    assert plan.engine.prefill_decode_interval_slope == 2.0
    assert plan.engine.prefill_decode_interval_slope_from == 30
    assert plan.engine.prefill_admit_order == "fifo"
    assert plan.engine.prefill_admit_max_age_ms == 3000
    assert recipes.EngineOptions().prefill_decode_interval_slope == 0
    srpt = _plan(engine={"prefill_admit_order": "srpt", "prefill_admit_max_age_ms": 0})
    assert srpt.engine.prefill_admit_order == "srpt" and srpt.plan_id != plan.plan_id


def test_the_dspark_options_of_a_sharded_head_and_the_samplers_unread_work():
    off = recipes.DSparkOptions()
    assert not off.dp_sharded_head and off.dp_sharded_head_max_rows == 1024
    assert not off.sampler_skip_greedy_pick
    assert not off.verify_skip_unused_target_draw
    plan = _plan(
        dspark_options={
            "dp_sharded_head": True,
            "dp_sharded_head_max_rows": 512,
            "sampler_skip_greedy_pick": True,
            "lean_head": True,
            "verify_argmax_once": True,
        }
    )
    options = plan.dspark_options
    assert options.dp_sharded_head and options.dp_sharded_head_max_rows == 512
    assert options.sampler_skip_greedy_pick and options.lean_head
    assert options.verify_argmax_once and not options.verify_skip_unused_target_draw
    assert plan.plan_id != _plan(dspark_options={"dp_sharded_head": True}).plan_id


def test_the_host_options_reach_code_without_a_config():
    assert not host.engine_options().thp_disable
    plan = _plan(engine={"thp_disable": True, "engram_pin_pages": True})
    assert plan.engine.thp_disable and plan.engine.engram_pin_pages
    host.configure(plan, kill_switch=False)
    assert host.engine_options().engram_pin_pages
    host._reset_for_testing()
    assert not host.engine_options().engram_pin_pages


def test_v1_keeps_its_schema_and_refuses_modules():
    doc = {
        "schema": recipes.RECIPE_SCHEMA,
        "model_arch": ARCH,
        "modules": {"layers.*.ffn": ["a"]},
    }
    with pytest.raises(recipes.RecipeRefused, match="modules requires schema"):
        recipes.resolve([doc])


def test_v2_infers_plugins_and_nests_overlapping_owners():
    plan = _plan(
        seams={"v41.block.ffn": ["outer"], "v41.moe.shared_act": ["inner"]},
        modules={"stack.*.attn": ["whole-attn"]},
    )
    assert set(plan.providers) == {"outer", "inner", "whole-attn"}
    # Parent and child owners coexist (v1 refuses this as recipe-shadow).
    assert plan.shadowed == (("v41.moe.shared_act", "inner", "outer"),)
    # A seam that spans exactly one slot is planned under both spellings.
    assert plan.modules["any"]["layers.*.ffn"] == ("outer",)


def test_a_recipe_set_must_agree_on_options():
    a = _v2(graph={"mode": "eager"}, model_options={"fused_attn": True})
    b = _v2(graph={"mode": "capture", "buckets": [8]})
    with pytest.raises(recipes.RecipeRefused, match="model_options differs"):
        recipes.resolve([a, b])


def test_engine_options_reach_the_engine_from_the_config(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(
        __import__("json").dumps(_v2(engine={"dp_session_prefix_owner_tokens": 512}))
    )
    source = recipes.from_cli(str(path), None)
    config = types.SimpleNamespace(plugin_source=source, plugin_arm="native")
    options = recipes.engine_options_for(config)
    assert options.dp_session_prefix_owner_tokens == 512
    assert (
        recipes.engine_options_for(types.SimpleNamespace()) == recipes.EngineOptions()
    )


def test_a_recipe_env_is_applied_once_and_refuses_a_conflict():
    plan = _plan(env={"ATOM_ENABLE_PREFILL_DELAYER": 1, "ATOM_DSV41_MONO_HI": "auto"})
    assert dict(plan.env) == {
        "ATOM_ENABLE_PREFILL_DELAYER": "1",
        "ATOM_DSV41_MONO_HI": "auto",
    }
    assert plan.plan_id != _plan().plan_id
    environ = {"ATOM_DSV41_MONO_HI": "auto"}
    assert recipes.apply_env(plan, environ) == {"ATOM_ENABLE_PREFILL_DELAYER": "1"}
    assert environ["ATOM_ENABLE_PREFILL_DELAYER"] == "1"
    with pytest.raises(recipes.RecipeRefused, match="recipe-env-conflict"):
        recipes.apply_env(plan, {"ATOM_DSV41_MONO_HI": "1"})
    for bad in ({"PATH": "/x"}, {"ATOM_PLUGIN_HOST": "0"}, {"atom_x": "1"}):
        with pytest.raises(recipes.RecipeRefused):
            _plan(env=bad)
    with pytest.raises(recipes.RecipeRefused, match="env requires schema"):
        recipes.resolve(
            [
                {
                    "schema": recipes.RECIPE_SCHEMA,
                    "model_arch": ARCH,
                    "env": {"ATOM_X": "1"},
                }
            ]
        )


def test_no_new_environment_variable_carries_a_recipe_option():
    envs = (REPO / "atom/utils/envs.py").read_text(encoding="utf-8")
    for name in (
        "ATOM_DSV41_FFN_SEAM_QUANT",
        "ATOM_DSV41_SHARED_ACT_QUANT",
        "ATOM_DSV41_FUSED_ATTN",
        "ATOM_DSPARK_FUSED_PROLOGUE",
        "ATOM_DSPARK_LEAN_HEAD",
        "ATOM_SPEC_VERIFY_ARGMAX_ONCE",
        "ATOM_PREFILL_SHORT_FIRST_ROWS",
        "ATOM_PREFILL_COST_CEIL",
        "ATOM_DP_SESSION_PREFIX_OWNER_TOKENS",
        "ATOM_MOE_EXPERT_PLACEMENT",
        "ATOM_PREFILL_DECODE_INTERVAL_SLOPE",
        "ATOM_PREFILL_ADMIT_ORDER",
        "ATOM_PREFILL_ADMIT_MAX_AGE_MS",
        "ATOM_DP_DSPARK_SHARDED_HEAD",
        "ATOM_SAMPLER_SKIP_GREEDY_PICK",
        "ATOM_SPEC_SKIP_UNUSED_TARGET_DRAW",
        "ATOM_THP_DISABLE",
        "ATOM_ENGRAM_PIN_PAGES",
    ):
        assert name not in envs, name


# -- the plugin model ------------------------------------------------------------


def test_atoms_own_v41_model_has_no_plugin_seam():
    native = REPO / "atom/models/deepseek_v41"
    for name in ("model.py", "attention.py", "moe.py", "dspark.py", "runtime.py"):
        text = (native / name).read_text(encoding="utf-8")
        assert "plugin_host" not in text and "plugin_transition" not in text, name


def test_the_plugin_model_is_chosen_only_under_a_plan(monkeypatch):
    native = "atom.models.deepseek_v41.runtime.DeepseekV41RuntimeModel"
    assert host.model_class_for(ARCH, native) == native
    host.configure(_plan(model_options={"fused_attn": True}), kill_switch=False)
    assert host.model_class_for(ARCH, native) == host.PLUGIN_MODEL_CLASSES[ARCH]
    assert host.model_class_for("Qwen3ForCausalLM", "q") == "q"
    assert host.model_options().fused_attn
    host._reset_for_testing()
    host.configure(
        _plan(
            model_options={"target_model_cls": "pkg.Mine", "draft_model_cls": "pkg.D"}
        ),
        kill_switch=False,
    )
    assert host.model_class_for(ARCH, native) == "pkg.Mine"
    assert host.draft_model_class_for("DeepseekV41DSparkModel", "x") == "pkg.D"
    host._reset_for_testing()
    host.configure(_plan(), kill_switch=True)
    assert host.model_class_for(ARCH, native) == native
    assert host.model_options() == recipes.ModelOptions()


def test_hotplug_refuses_a_plan_that_changes_the_built_model():
    host.configure(None, hotplug=True, kill_switch=False)
    _built()
    host.hotplug_plan(_plan(dspark_options={"lean_head": True}))
    assert host.dspark_options().lean_head
    with pytest.raises(recipes.RecipeRefused, match="recipe-boot-only"):
        host.hotplug_plan(_plan(model_options={"fused_attn": True}))
    assert host.dspark_options().lean_head  # the serving plan stays


# -- registration --------------------------------------------------------------


def test_owns_collectives_is_accepted_and_checked(monkeypatch):
    def offer(h):
        h.register(
            seam="v41.block.ffn",
            contract=1,
            impl=lambda x, **k: x,
            owner="mega",
            owns_collectives=("tp_all_reduce", "ep_all_to_all", "dp_gather"),
        )

    _plugins(monkeypatch, mega=offer)
    host.configure(_plan(seams={"v41.block.ffn": ["mega"]}), kill_switch=False)
    assert host.owners("v41.block.ffn") == ("mega",)
    assert host.SEAMS["v41.block.ffn"].collectives == ()
    with pytest.raises(host.RegistrationRefused, match="unknown-collective"):
        host.register(
            seam="v41.attn.out",
            contract=1,
            impl=lambda *a, **k: a,
            owner="x",
            owns_collectives=("teleport",),
        )


def test_a_module_owner_on_a_seam_slot_registers_as_the_seam(monkeypatch):
    _plugins(monkeypatch, whole=_module_owner("layers.*.ffn", "whole", value=9))
    host.configure(_plan(modules={"layers.*.ffn": ["whole"]}), kill_switch=False)
    assert host.owners("v41.block.ffn") == ("whole",)
    ffn = FFN()
    bound = host.bind(
        "v41.block.ffn",
        ffn.__call__,
        capability={"flags": {"layer_id": 0, "module": ffn}},
    )
    # The seam's call site (Block._ffn) reaches the module owner's callable.
    assert torch.equal(bound(torch.ones(2)), torch.full((2,), 9.0))


# -- module takeover -------------------------------------------------------------


def test_whole_module_takeover_routes_live_modules_and_keeps_names(monkeypatch):
    seen = []
    _plugins(
        monkeypatch, fast=_module_owner("stack.*.attn", "fast", value=None, seen=seen)
    )
    host.configure(_plan(modules={"stack.*.attn": ["fast"]}), kill_switch=False)
    model = Toy()
    names = [n for n, _ in model.named_parameters()]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    assert host.install_modules(model) == ["stack.0.attn", "stack.1.attn"]
    assert [p for p, _ in seen] == ["stack.0.attn", "stack.1.attn"]
    assert seen[0][1] is model.stack[0].attn
    assert [n for n, _ in model.named_parameters()] == names
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    x = torch.ones(1)
    # attn 5 -> 500 by the owner; ffn (2 + 3) native.
    assert torch.equal(model(x), torch.tensor([500.0 * 5 * 500 * 5]))
    host.set_arm("native")
    assert torch.equal(model(x), torch.tensor([5.0 * 5 * 5 * 5]))
    host.set_arm("plugin")
    assert host.status()["modules_installed"] == ["stack.0.attn", "stack.1.attn"]
    host._reset_for_testing()
    assert "forward" not in model.stack[0].attn.__dict__  # restored


def test_overlapping_owners_compose_outer_first_inner_on_decline(monkeypatch):
    """An outer owner serves the calls it selects; the rest run the module's
    own forward, which reaches the inner owner. Neither knows the other."""
    _plugins(
        monkeypatch,
        **{
            "outer-ffn": _module_owner(
                "stack.*.ffn",
                "outer-ffn",
                value=7,
                select=lambda cap, x: cap.rows <= 2,
            ),
            "inner-shared": _module_owner(
                "stack.*.ffn.shared", "inner-shared", value=40
            ),
        },
    )
    plan = _plan(
        modules={
            "stack.*.ffn": ["outer-ffn"],
            "stack.*.ffn.shared": ["inner-shared"],
        }
    )
    host.configure(plan, kill_switch=False)
    ffn = Toy(n=1).stack[0].ffn
    host.install_modules(ffn, prefix="stack.0.ffn")
    small, large = torch.ones(2), torch.ones(4)
    assert torch.equal(ffn(small), torch.full((2,), 7.0))  # outer serves
    # Outer declines: native FFN = inner owner (40) + routed native (3).
    assert torch.equal(ffn(large), torch.full((4,), 43.0))
    counts = host.decision_counts()
    assert counts[("module:stack.*.ffn", "plugin", "selected", "outer-ffn")] == 1
    assert counts[("module:stack.*.ffn", "native", "declined", None)] == 1


def test_a_hotplug_reinstalls_the_new_plans_owners(monkeypatch):
    _plugins(
        monkeypatch,
        a=_module_owner("stack.*.attn", "a", value=1),
        b=_module_owner("stack.*.attn", "b", value=2),
    )
    host.configure(
        _plan(modules={"stack.*.attn": ["a"]}), hotplug=True, kill_switch=False
    )
    model = Toy(n=1)
    _built()
    host.install_modules(model)
    attn = model.stack[0].attn
    assert torch.equal(attn(torch.ones(1)), torch.ones(1))
    host.hotplug_plan(_plan(modules={"stack.*.attn": ["b"]}))
    assert torch.equal(attn(torch.ones(1)), torch.full((1,), 2.0))
    host.hotplug_plan(None)
    assert torch.equal(attn(torch.ones(1)), torch.full((1,), 5.0))


def test_a_planned_slot_owner_that_did_not_register_refuses_the_start(monkeypatch):
    _plugins(monkeypatch, quiet=lambda h: None)
    with pytest.raises(recipes.RecipeRefused, match="recipe-coverage-miss"):
        host.configure(_plan(modules={"stack.*.attn": ["quiet"]}), kill_switch=False)


def test_a_factory_error_names_the_owner_and_the_module(monkeypatch):
    def register(h):
        h.register_module(
            slot="stack.*.attn",
            owner="broken",
            factory=lambda module, cap: 1 / 0,
        )

    _plugins(monkeypatch, broken=register)
    host.configure(_plan(modules={"stack.*.attn": ["broken"]}), kill_switch=False)
    with pytest.raises(RuntimeError, match=re.escape("'broken'") + ".*stack.0.attn"):
        host.install_modules(Toy(n=1))


# -- capture warmup ---------------------------------------------------------------


def test_a_cold_decline_in_bucket_warmup_is_asked_again_when_recording(monkeypatch):
    warm = []

    def select(cap, x_fp8, x_scale, w, ws, **kw):
        return bool(warm)

    def offer(h):
        h.register(
            seam="v41.attn.out",
            contract=1,
            impl=lambda *a, **k: ("plugin", None),
            owner="bmm",
            select=select,
        )

    _plugins(monkeypatch, bmm=offer)
    host.configure(["bmm"], kill_switch=False)
    native = lambda *a, **k: ("native", None)
    bound = host.bind("v41.attn.out", native)
    monkeypatch.setattr(host, "_graph_state", lambda: (True, 8))
    monkeypatch.setattr(host, "_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    x = torch.zeros(8, 2, 64)
    assert bound(x, x, x, x, dtype=torch.bfloat16)[0] == "native"  # warmup, cold
    warm.append(True)  # its kernels loaded on that warmup
    monkeypatch.setattr(host, "_stream_capturing", lambda: True)  # recording
    assert bound(x, x, x, x, dtype=torch.bfloat16)[0] == "plugin"


def test_a_taken_over_module_fakes_its_own_output_shape(monkeypatch):
    """Under compile a taken-over module is one opaque op; its default fake is
    the module's own forward on fake inputs (a projection changes the width)."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    _plugins(monkeypatch, proj=_module_owner("proj", "proj", value=3))
    host.configure(_plan(modules={"proj": ["proj"]}), kill_switch=False)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 8)

    net = Net()
    host.install_modules(net)
    router = net.proj.forward
    with FakeTensorMode() as mode:
        out = router._native_fake(mode.from_tensor(torch.ones(2, 4)))
    assert tuple(out.shape) == (2, 8)
