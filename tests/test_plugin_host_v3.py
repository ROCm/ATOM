# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Plugin host surface for the HOOK_API 3 migration (CPU only).

AITER follows the arm, AITER providers are prepared with the plan, a recipe pins
plugin code (never ATOM or AITER versions), and ``context()`` gives code a
seam's capability does not reach the capture state.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

from atom import plugin_host as host
from atom import plugin_recipe as recipes

ARCH = "DeepseekV41ForCausalLM"


class _EntryPoint:
    def __init__(self, obj):
        self._obj = obj

    def load(self):
        return self._obj


@pytest.fixture(autouse=True)
def fresh_host(monkeypatch):
    calls = []
    monkeypatch.setattr(host, "_entry_points", dict)
    monkeypatch.setattr(host, "_aiter_entry_points", dict)
    monkeypatch.setattr(
        host, "_configure_aiter", lambda names: calls.append(tuple(names))
    )
    monkeypatch.setattr(host, "_graph_state", lambda: (False, None))
    monkeypatch.setattr(host, "_valid_rows", lambda rows: None)
    host._reset_for_testing()
    calls.clear()
    yield calls
    host._reset_for_testing()


def _plan(**fields):
    return recipes.resolve([{"schema": recipes.RECIPE_SCHEMA, "arch": ARCH, **fields}])


# -- recipe: aiter by op, plugin pins -----------------------------------------


def test_aiter_providers_grouped_by_op_keep_one_global_order():
    plan = _plan(aiter={"topk_gating": ["b", "a"], "quant": ["a", "c"]})
    assert plan.aiter == ("b", "a", "c")
    with pytest.raises(recipes.RecipeRefused, match="aiter ops must be names"):
        _plan(aiter={"": ["a"]})


@pytest.mark.parametrize("pins", [{"p": "sha1:00"}, {"p": "sha256:" + "0" * 63}, ["p"]])
def test_plugin_pins_must_be_sha256_digests(pins):
    with pytest.raises(recipes.RecipeRefused, match="requires.plugins"):
        _plan(requires={"plugins": pins})


def test_a_recipe_carries_no_atom_or_aiter_version_pin():
    plan = _plan(requires={"plugins": {"p": "sha256:" + "a" * 64}})
    assert set(plan.requires) == {"plugins"}


# -- AITER follows the arm ----------------------------------------------------


def test_aiter_follows_the_arm_and_is_told_only_on_a_change(fresh_host):
    calls = fresh_host
    plan = _plan(aiter=["a", "b"], arms={"only-b": ["b"]})
    host.configure(plan, kill_switch=False)
    assert calls == [("a", "b")]
    host.set_arm("native")
    host.set_arm("native")
    assert calls[1:] == [()]
    host.set_arm("only-b")
    host.set_arm("plugin")
    host.set_arm("identity")  # AITER has no identity: its providers keep serving
    assert calls[2:] == [("b",), ("a", "b")]
    assert host.status()["aiter"]["enabled"] == ["a", "b"]


def test_the_kill_switch_tells_aiter_nothing_serves(fresh_host):
    host.configure(_plan(aiter=["a"]), kill_switch=True)
    assert fresh_host == [()]


def test_aiter_providers_are_prepared_with_the_plan(monkeypatch):
    seen = []

    class Provider:
        def prepare(self, device):
            seen.append(device)
            return {"warm": True}

    monkeypatch.setattr(
        host, "_aiter_entry_points", lambda: {"a": _EntryPoint(Provider())}
    )
    host.configure(_plan(aiter=["a", "missing"]), kill_switch=False)
    out = host.prepare_plan("cuda:0")
    assert seen == ["cuda:0"] and out["prepared"] == ["a"]
    assert out["native"] == {"a": {"warm": True}}


# -- plugin pins --------------------------------------------------------------


def _install_provider(tmp_path, monkeypatch, body="X = 1\n"):
    pkg = tmp_path / "pinpkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "hook.py").write_text("def register(host):\n    return None\n" + body)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m.startswith("pinpkg")]:
        del sys.modules[name]
    module = importlib.import_module("pinpkg.hook")
    monkeypatch.setattr(
        host, "_entry_points", lambda: {"pin-p": _EntryPoint(module.register)}
    )
    return pkg


def test_a_pinned_provider_must_hash_to_its_pin(tmp_path, monkeypatch):
    pkg = _install_provider(tmp_path, monkeypatch)
    digest = host.provider_digest("pin-p")
    assert digest.startswith("sha256:") and len(digest) == 71
    host.configure(_plan(requires={"plugins": {"pin-p": digest}}), kill_switch=False)
    host._reset_for_testing()
    (pkg / "hook.py").write_text("def register(host):\n    return None\nX = 2\n")
    with pytest.raises(recipes.RecipeRefused, match="recipe-plugin-drift"):
        host.configure(
            _plan(requires={"plugins": {"pin-p": digest}}), kill_switch=False
        )
    with pytest.raises(recipes.RecipeRefused, match="not installed"):
        host.configure(_plan(requires={"plugins": {"nope": digest}}), kill_switch=False)


def test_a_provider_digest_covers_its_declared_code_roots(tmp_path, monkeypatch):
    extra = tmp_path / "kernels"
    extra.mkdir()
    (extra / "k.h").write_text("int k;\n")
    _install_provider(
        tmp_path, monkeypatch, body=f"PLUGIN_CODE_ROOTS = ({str(extra)!r},)\n"
    )
    before = host.provider_digest("pin-p")
    (extra / "k.h").write_text("int k2;\n")
    assert host.provider_digest("pin-p") != before


# -- context() ----------------------------------------------------------------


def test_context_reports_capture_bucket_and_valid_rows(monkeypatch):
    monkeypatch.setattr(host, "_valid_rows", lambda rows: ("valid", rows))
    assert host.context() == host.Context(False, None, None)
    monkeypatch.setattr(host, "_graph_state", lambda: (True, 8))
    ctx = host.context()
    assert ctx.captured and ctx.bucket == 8 and ctx.valid_rows == ("valid", 8)
    assert host.context(rows=5).valid_rows == ("valid", 5)
    assert isinstance(host.context(), types.SimpleNamespace) is False


# -- seams written at their call sites; the plan checked against the model ---


@pytest.fixture
def seam_table():
    saved = (dict(host._SEAM_SPECS), dict(host._SEAM_SOURCES))
    yield
    for live, old in zip((host._SEAM_SPECS, host._SEAM_SOURCES), saved):
        live.clear()
        live.update(old)


_TOY = {
    "name": "toy.block.mix",
    "contract": 1,
    "kind": "region",
    "form": "bind",
    "inputs": ("x",),
    "kwargs": (),
    "returns": ("y",),
    "key": ("x",),
    "graph": "capture_safe",
    "target": "toy/model.py",
    "scope": "blocks.*",
}


def _mix_owner(name, value=1, contract=1):
    def register(h):
        h.register(
            seam="toy.block.mix",
            contract=contract,
            owner=name,
            impl=lambda x, **k: x + value,
        )

    return register


def test_the_v41_seams_are_written_at_their_call_sites():
    root = Path(__file__).resolve().parents[1]
    site = (root / "atom/model_ops/blockscale.py").read_text(encoding="utf-8")
    assert "plugin_host.seam(" in site and 'name="v41.op.quant_fp8_g32"' in site
    assert (
        "v41.mono.install" in host.boot_only() and "model.install" in host.boot_only()
    )


def test_a_model_declares_its_seam_where_it_binds_it(seam_table, monkeypatch):
    monkeypatch.setattr(
        host, "_entry_points", lambda: {"mix-p": _EntryPoint(_mix_owner("mix-p"))}
    )
    # the recipe names a seam the model has not been built yet to declare
    host.configure(_plan(seams={"toy.block.mix": ["mix-p"]}), kill_switch=False)
    router = host.bind("toy.block.mix", lambda x: x, spec=_TOY)
    assert router(1) == 2
    assert host.bind("toy.block.mix", lambda x: x, spec=_TOY)(1) == 2  # every layer
    host.check_plan()


def test_one_seam_name_has_one_contract(seam_table):
    host.seam(_TOY)
    with pytest.raises(host.SeamConflict, match="conflicts"):
        host.seam({**_TOY, "contract": 2})
    with pytest.raises(host.SeamConflict, match="dotted lower-case"):
        host.seam({**_TOY, "name": "Bad"})
    with pytest.raises(host.SeamConflict, match="key must name inputs"):
        host.seam({**_TOY, "name": "toy.other", "key": ("z",)})


def test_an_owner_is_checked_when_the_model_declares_its_seam(seam_table, monkeypatch):
    monkeypatch.setattr(
        host,
        "_entry_points",
        lambda: {"mix-p": _EntryPoint(_mix_owner("mix-p", contract=2))},
    )
    host.configure(_plan(seams={"toy.block.mix": ["mix-p"]}), kill_switch=False)
    with pytest.raises(host.PluginContractError, match="contract 2"):
        host.bind("toy.block.mix", lambda x: x, spec=_TOY)


def test_a_planned_seam_the_model_never_declares_refuses_the_start(
    seam_table, monkeypatch
):
    monkeypatch.setattr(
        host, "_entry_points", lambda: {"mix-p": _EntryPoint(_mix_owner("mix-p"))}
    )
    host.configure(_plan(seams={"toy.block.mix": ["mix-p"]}), kill_switch=False)
    with pytest.raises(recipes.RecipeRefused, match="recipe-seam-unbound"):
        host.check_plan()


def test_required_contracts_and_excludes_are_checked_on_the_built_model(
    seam_table, monkeypatch
):
    host.seam(_TOY)
    host.seam({**_TOY, "name": "toy.block.window", "excludes": ("toy.block.mix",)})
    eps = {
        "mix-p": _EntryPoint(_mix_owner("mix-p")),
        "win-p": _EntryPoint(
            lambda h: h.register(
                seam="toy.block.window",
                contract=1,
                owner="win-p",
                impl=lambda x, **k: x,
            )
        ),
    }
    monkeypatch.setattr(host, "_entry_points", lambda: eps)
    host.configure(
        _plan(
            seams={"toy.block.mix": ["mix-p"]}, requires={"seams": {"toy.block.mix": 2}}
        ),
        kill_switch=False,
    )
    with pytest.raises(recipes.RecipeRefused, match="recipe-contract-mismatch"):
        host.check_plan()
    host._reset_for_testing()
    host.configure(
        _plan(seams={"toy.block.mix": ["mix-p"], "toy.block.window": ["win-p"]}),
        kill_switch=False,
    )
    with pytest.raises(recipes.RecipeRefused, match="recipe-excludes"):
        host.check_plan()


def test_outer_owners_shadow_nested_and_covered_ones(seam_table, monkeypatch):
    host.seam(_TOY)
    host.seam({**_TOY, "name": "toy.block.inner", "parent": "toy.block.mix"})
    host.seam({**_TOY, "name": "toy.block.alt", "covers": ("toy.block.inner",)})

    def reg(seam, owner):
        return _EntryPoint(
            lambda h: h.register(
                seam=seam, contract=1, owner=owner, impl=lambda x, **k: x
            )
        )

    eps = {
        "o": reg("toy.block.mix", "o"),
        "i": reg("toy.block.inner", "i"),
        "a": reg("toy.block.alt", "a"),
    }
    monkeypatch.setattr(host, "_entry_points", lambda: eps)
    seams = {"toy.block.mix": ["o"], "toy.block.inner": ["i"], "toy.block.alt": ["a"]}
    host.configure(_plan(seams=seams), kill_switch=False)
    assert sorted(host.check_plan()["shadowed"]) == [
        ("toy.block.inner", "i", "toy.block.alt"),
        ("toy.block.inner", "i", "toy.block.mix"),
    ]
    host._reset_for_testing()
    host.configure(
        _plan(seams=seams, defaults={"shadow_policy": "error"}), kill_switch=False
    )
    with pytest.raises(recipes.RecipeRefused, match="recipe-shadow"):
        host.check_plan()


# -- M4: method takeover, plugin_options, model.install -----------------------


class _Mix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 2

    def forward(self, x):
        return self.combine(x) + 1

    def combine(self, x):
        return x * self.scale


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Mix(), _Mix()])


def test_a_method_is_taken_over_like_a_forward_and_restored():
    def register(h):
        h.register_module(
            slot="blocks.*",
            method="combine",
            owner="comb-p",
            factory=lambda module, cap: (lambda x: x * 10),
            select=lambda cap, x: cap.flags["method"] == "combine" and x.numel() < 4,
        )

    host._entry_points = lambda: {"comb-p": _EntryPoint(register)}
    host.configure(_plan(modules={"blocks.*:combine": ["comb-p"]}), kill_switch=False)
    net = _Net()
    assert host.install_modules(net) == ["blocks.0:combine", "blocks.1:combine"]
    x = torch.ones(2)
    assert torch.equal(net.blocks[0](x), x * 10 + 1)  # forward reaches the owner
    assert torch.equal(net.blocks[1](torch.ones(5)), torch.ones(5) * 2 + 1)  # declined
    host._uninstall_modules()
    assert "combine" not in net.blocks[0].__dict__
    assert torch.equal(net.blocks[0](x), x * 2 + 1)


def test_plugin_options_pass_through_unread():
    plan = _plan(plugin_options={"dsv41": {"fused_attn": True, "grid": [1, 2]}})
    assert dict(plan.plugin_options) == {"dsv41": {"fused_attn": True, "grid": [1, 2]}}
    other = _plan(plugin_options={"dsv41": {"fused_attn": False, "grid": [1, 2]}})
    assert plan.plan_id != other.plan_id
    host.configure(plan, kill_switch=False)
    assert host.plugin_options("dsv41")["grid"] == [1, 2]
    assert dict(host.plugin_options("other")) == {}
    with pytest.raises(recipes.RecipeRefused, match="plugin_options"):
        _plan(plugin_options={"dsv41": 3})


def test_model_install_is_a_core_boot_hook():
    spec = host.SEAMS[host.MODEL_INSTALL]
    assert spec.boot_only and spec.form == "call"
    assert host._SEAM_SOURCES[host.MODEL_INSTALL] == "atom.plugin_host"

    def register(h):
        h.register(
            seam=host.MODEL_INSTALL,
            contract=1,
            owner="wrap-p",
            graph="eager",
            impl=lambda model, cfg, drafter, **k: ("wrapped", model),
        )

    host._entry_points = lambda: {"wrap-p": _EntryPoint(register)}
    host.configure(_plan(seams={host.MODEL_INSTALL: ["wrap-p"]}), kill_switch=False)
    out = host.call(host.MODEL_INSTALL, lambda m, c, d, **k: m, "m", None, None)
    assert out == ("wrapped", "m")


# -- M5: route report, cross-rank agreement, inner seams of a taken-over module


def test_route_report_names_unreached_planned_owners(seam_table, monkeypatch):
    host.seam(_TOY)

    def reg_a(h):
        h.register(
            seam="toy.block.mix", contract=1, owner="mix-a", impl=lambda x, **k: x
        )

    def reg_b(h):
        h.register(
            seam="toy.block.mix",
            contract=1,
            owner="mix-b",
            impl=lambda x, **k: x,
            select=lambda cap, x: False,
        )

    monkeypatch.setattr(
        host,
        "_entry_points",
        lambda: {"mix-a": _EntryPoint(reg_a), "mix-b": _EntryPoint(reg_b)},
    )
    host.configure(
        _plan(seams={"toy.block.mix": ["mix-b", "mix-a"]}), kill_switch=False
    )
    monkeypatch.setattr(host, "_graph_state", lambda: (True, 16))
    host.bind("toy.block.mix", lambda x: x)(torch.ones(16, 4))
    report = host.route_report()
    assert report["routed"]["toy.block.mix"]["captured"] == {"mix-a": [16]}
    assert report["planned_but_unreached"] == ["toy.block.mix:mix-b"]
    assert report["unplanned"] == []


def test_owners_running_collectives_must_agree_across_ranks(monkeypatch):
    import torch.distributed as dist

    group = types.SimpleNamespace(world_size=2, cpu_group=object())
    fake_ps = types.ModuleType("aiter.dist.parallel_state")
    fake_ps.get_tp_group = lambda: group
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", fake_ps)
    monkeypatch.setitem(sys.modules, "aiter.dist", types.ModuleType("aiter.dist"))
    others = ["x"]

    def gather(out, mine, group=None):
        out[0], out[1] = mine, others[0]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    decision = host.Decision("s", "plugin", "selected", "x", 0, True, 8)
    assert host._agree(decision, "route") == (decision, "route")
    others[0] = "y"
    agreed, route = host._agree(decision, "route")
    assert (
        route is None and agreed.reason == "rank-disagree" and agreed.route == "native"
    )


def test_a_taken_over_module_needs_its_inner_seams_planned(seam_table, monkeypatch):
    host.seam(_TOY)  # scope blocks.*

    def reg(h):
        h.register_module(
            slot="blocks.*", owner="blk-p", factory=lambda m, c: m.forward
        )

    monkeypatch.setattr(host, "_entry_points", lambda: {"blk-p": _EntryPoint(reg)})
    host.configure(_plan(modules={"blocks.*": ["blk-p"]}), kill_switch=False)
    with pytest.raises(recipes.RecipeRefused, match="recipe-inner-unplanned"):
        host.check_plan()
    host._reset_for_testing()
    plan = _plan(modules={"blocks.*": ["blk-p"]}, seams={"toy.block.mix": []})
    host.configure(plan, kill_switch=False)
    host.check_plan()


def test_shared_weights_are_prepared_once_for_every_configuration():
    net = _Net()
    assert host.freeze_weights(net) == 0  # _Mix has no parameters
    lin = torch.nn.Linear(2, 2)
    model = torch.nn.Sequential(lin)
    assert host.freeze_weights(model, None, prefixes=("", "draft")) == 2
    assert host.weights_changed() == []
    with torch.no_grad():
        lin.weight.mul_(2)  # an owner re-preparing a shared weight in place
    lin.bias = torch.nn.Parameter(torch.zeros(2))  # or swapping it
    assert host.weights_changed() == ["0.bias", "0.weight"]
    report = host.log_route_report("test")
    assert "0.weight" in report["weights_changed"]


def test_one_inferred_owner_may_serve_several_seams():
    plan = _plan(seams={"toy.block.mix": ["p"], "toy.block.other": ["p"]})
    assert plan.providers == ("p",)


def test_a_recipes_metadata_fields_are_ignored():
    plan = _plan(seams={"toy.block.mix": []}, purpose="layer bench", baseline="x")
    assert plan.providers == ()


def test_a_process_without_a_plan_builds_the_recipes_model_class():
    """The weight-cache daemon builds the model without configuring the host: the class comes from the config's
    recipe, so the servers it serves can map its weights."""
    import json
    from types import SimpleNamespace

    doc = {
        "schema": recipes.RECIPE_SCHEMA,
        "arch": ARCH,
        "model_options": {
            "target_model_cls": "pkg.Model",
            "draft_model_cls": "pkg.Draft",
        },
    }
    plan = recipes.resolve([doc])
    source = recipes.RecipeSource(
        (json.dumps(doc, sort_keys=True),), plan.plan_id, False
    )
    config = SimpleNamespace(plugin_source=source)
    assert host.model_class_for("Arch", "atom.Model", config) == "pkg.Model"
    assert host.draft_model_class_for("Arch", "atom.Draft", config) == "pkg.Draft"
    assert host.model_class_for("Arch", "atom.Model") == "atom.Model"


def test_a_model_built_again_registers_its_spans_anew():
    (
        host.seam({**_TOY, "name": "toy.block.span", "graph": "opaque_op"})
        if "toy.block.span" not in host.SEAMS
        else None
    )
    host._owned = frozenset({"toy.block.span"})
    host.span_op("toy.block.span", "L0", lambda x, e: x, lambda x, e: x)
    with pytest.raises(ValueError, match="not unique"):
        host.span_op("toy.block.span", "L0", lambda x, e: x, lambda x, e: x)
    host.new_model_build()
    host.span_op("toy.block.span", "L0", lambda x, e: x, lambda x, e: x)


def test_the_runner_steps_are_the_host_calls_in_their_order(monkeypatch):
    calls = []
    for name in ("install_modules", "check_plan", "prepare_plan", "freeze_weights"):
        monkeypatch.setattr(
            host, name, lambda *a, _n=name, **k: calls.append((_n, a, k))
        )
    model, draft = object(), object()
    host.install_built(model, types.SimpleNamespace(model=draft))
    host.install_built(model)
    assert calls == [
        ("install_modules", (model,), {}),
        ("install_modules", (draft,), {"prefix": "draft"}),
        ("check_plan", (), {}),
        ("install_modules", (model,), {}),
        ("check_plan", (), {}),
    ]
    calls.clear()
    host.prepare_built("cuda:0", model, types.SimpleNamespace(model=draft))
    host.prepare_built("cuda:0", model)
    assert calls == [
        ("prepare_plan", ("cuda:0",), {}),
        ("freeze_weights", (model, draft), {"prefixes": ("", "draft")}),
        ("prepare_plan", ("cuda:0",), {}),
        ("freeze_weights", (model, None), {"prefixes": ("", "draft")}),
    ]


def test_the_runner_configures_from_the_config_and_a_stand_in_is_native(monkeypatch):
    seen = []
    monkeypatch.setattr(
        host, "configure", lambda plan=None, **kw: seen.append((plan, kw))
    )
    config = types.SimpleNamespace(plugin_source="src", plugin_arm="native")
    host.configure_runner(config)
    host.configure_runner(types.SimpleNamespace())
    assert seen == [
        ("src", {"arm": "native", "config": config}),
        (None, {"arm": "plugin", "config": seen[1][1]["config"]}),
    ]
