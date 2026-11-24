# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Recipe v2 (`atom.plugin_recipe/2`) and whole-module takeover (`plugin_host`).

CPU only: plugins are stand-in entry points over a toy `nn.Module` tree; the
graph state a capture would report is patched in.
"""

import pickle
import re
from pathlib import Path

import pytest
import torch
from plugin_plans import enable
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
    return {"schema": recipes.RECIPE_SCHEMA, "arch": ARCH, **fields}


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


def test_options_parse_and_key_the_plan():
    plan = _plan(
        model_options={"target_model_cls": "pkg.Model", "draft_model_cls": "pkg.D"},
        plugin_options={"pkg": {"lean_head": True}},
    )
    assert plan.has_options and not plan.empty
    assert plan.model_options.target_model_cls == "pkg.Model"
    other = _plan(plugin_options={"pkg": {"lean_head": False}})
    assert other.plan_id != plan.plan_id
    copy = pickle.loads(pickle.dumps(plan))
    assert copy == plan and copy.dump() == plan.dump()


@pytest.mark.parametrize(
    "fields",
    [
        {"engine": {"prefill_short_first_rows": 96}},
        {"dspark_options": {"lean_head": True}},
        {"model_options": {"fused_attn": True}},
        {"model_options": {"target_model_cls": ""}},
        {"modules": {"layers..ffn": ["a"]}},
        {"modules": {"@NotQualified": ["a"]}},
    ],
)
def test_a_bad_option_is_refused(fields):
    with pytest.raises(recipes.RecipeRefused):
        _plan(**fields)


def test_v2_infers_plugins_and_nests_overlapping_owners():
    plan = _plan(
        seams={"v41.block.ffn": ["outer"], "v41.moe.shared_act": ["inner"]},
        modules={"stack.*.attn": ["whole-attn"]},
    )
    assert set(plan.providers) == {"outer", "inner", "whole-attn"}
    # Parent and child owners coexist; the shadowing is reported against the
    # built model (plugin_host.check_plan).
    assert plan.shadow_policy == "nested"


def test_a_recipe_set_must_agree_on_options():
    a = _v2(graph={"mode": "eager"}, model_options={"target_model_cls": "pkg.M"})
    b = _v2(graph={"mode": "capture", "buckets": [8]})
    with pytest.raises(recipes.RecipeRefused, match="model_options differs"):
        recipes.resolve([a, b])


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
    with pytest.raises(recipes.RecipeRefused, match="expected schema"):
        recipes.resolve(
            [{"schema": "atom.plugin_recipe/1", "arch": ARCH, "env": {"ATOM_X": "1"}}]
        )


# -- the plugin model ------------------------------------------------------------


def test_atoms_own_v41_model_has_no_plugin_seam():
    native = REPO / "atom/models/deepseek_v41"
    for name in ("model.py", "attention.py", "moe.py", "dspark.py", "runtime.py"):
        text = (native / name).read_text(encoding="utf-8")
        assert "plugin_host" not in text and "plugin_transition" not in text, name


def test_the_plugin_model_is_chosen_only_under_a_plan(monkeypatch):
    native = "atom.models.deepseek_v41.runtime.DeepseekV41RuntimeModel"
    assert host.model_class_for(ARCH, native) == native
    host.configure(_plan(plugin_options={"pkg": {"x": 1}}), kill_switch=False)
    # ATOM has no plugin variant of its own: a plan without a target class
    # builds ATOM's model; a plugin package's model comes from the recipe.
    assert host.model_class_for(ARCH, native) == native
    assert host.model_class_for("Qwen3ForCausalLM", "q") == "q"
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
    host.configure(enable(["bmm"]), kill_switch=False)
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


# -- class slots: one plugin, every instance of a class, any path -------------


class Gate(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def pick(self, x):
        return x + self.k

    def forward(self, x):
        return self.pick(x) * 2


class Twin(nn.Module):
    """Two differently-pathed Gates: ``left`` and ``deep.right``."""

    def __init__(self):
        super().__init__()
        self.left = Gate(1)
        self.deep = nn.Module()
        self.deep.right = Gate(10)

    def forward(self, x):
        return self.left(x) + self.deep.right(x)


GATE = f"@{__name__}.Gate"


def test_a_class_slot_serves_every_instance_wherever_it_sits(monkeypatch):
    seen = []
    _plugins(monkeypatch, g=_module_owner(GATE, "g", value=None, seen=seen))
    # the owner reads `module.w`: give each Gate one
    model = Twin()
    model.left.w = 3.0
    model.deep.right.w = 4.0
    host.configure(_plan(modules={GATE: ["g"]}), kill_switch=False)
    assert sorted(host.install_modules(model)) == ["deep.right", "left"]
    assert sorted(p for p, _ in seen) == ["deep.right", "left"]
    x = torch.ones(1)
    assert torch.equal(model(x), torch.tensor([300.0 + 400.0]))
    host.set_arm("native")
    assert torch.equal(model(x), torch.tensor([(1 + 1) * 2.0 + (1 + 10) * 2.0]))


def test_a_method_is_taken_over_on_a_class_slot(monkeypatch):
    def offer(h):
        h.register_module(
            slot=GATE,
            method="pick",
            owner="p",
            factory=lambda module, cap: (lambda x: x * 0 + module.k * 100),
        )

    _plugins(monkeypatch, p=offer)
    host.configure(_plan(modules={f"{GATE}:pick": ["p"]}), kill_switch=False)
    model = Twin()
    assert sorted(host.install_modules(model)) == ["deep.right:pick", "left:pick"]
    # forward stays the module's own; only pick is the owner's
    assert torch.equal(model(torch.ones(1)), torch.tensor([100.0 * 2 + 1000.0 * 2]))
    host._reset_for_testing()
    assert "pick" not in model.left.__dict__  # restored


def test_a_class_slot_needs_its_inner_seams_planned_unless_it_takes_a_method(
    monkeypatch,
):
    saved = (dict(host._SEAM_SPECS), dict(host._SEAM_SOURCES))
    try:
        host.seam(
            {
                "name": "toy.gate.inner",
                "contract": 1,
                "kind": "op",
                "form": "call",
                "inputs": ("x",),
                "kwargs": (),
                "returns": ("y",),
                "key": ("x",),
                "graph": "capture_safe",
                "target": "toy.py",
                "scope": "deep.right",
            }
        )
        whole = _module_owner(GATE, "g", value=1)
        _plugins(monkeypatch, g=whole)
        host.configure(_plan(modules={GATE: ["g"]}), kill_switch=False)
        host.install_modules(Twin())
        with pytest.raises(recipes.RecipeRefused, match="recipe-inner-unplanned"):
            host.check_plan()
        host._reset_for_testing()

        def method_owner(h):
            h.register_module(
                slot=GATE, method="pick", owner="g", factory=lambda m, c: m.pick
            )

        _plugins(monkeypatch, g=method_owner)
        host.configure(_plan(modules={f"{GATE}:pick": ["g"]}), kill_switch=False)
        host.install_modules(Twin())
        host.check_plan()  # a method takeover has no inner seam to plan
    finally:
        for live, old in zip((host._SEAM_SPECS, host._SEAM_SOURCES), saved):
            live.clear()
            live.update(old)
