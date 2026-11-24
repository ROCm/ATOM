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

import pytest

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
    return recipes.resolve(
        [{"schema": recipes.RECIPE_SCHEMA_V2, "arch": ARCH, **fields}]
    )


# -- recipe: aiter by op, plugin pins -----------------------------------------


def test_aiter_providers_grouped_by_op_keep_one_global_order():
    plan = _plan(aiter={"topk_gating": ["b", "a"], "quant": ["a", "c"]})
    assert plan.aiter == ("b", "a", "c")
    with pytest.raises(recipes.RecipeRefused, match="aiter by op requires schema 2"):
        recipes.resolve(
            [
                {
                    "schema": recipes.RECIPE_SCHEMA,
                    "model_arch": ARCH,
                    "aiter": {"x": ["a"]},
                }
            ]
        )


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


# -- seam spec registry (M2) --------------------------------------------------


@pytest.fixture
def seam_table():
    saved = (
        dict(host._SEAM_SPECS),
        dict(host._SEAM_SOURCES),
        dict(host._SEAM_TO_SLOT),
        dict(host._SLOT_TO_SEAM),
        host._seam_sources_loaded,
    )
    yield
    for live, old in zip(
        (host._SEAM_SPECS, host._SEAM_SOURCES, host._SEAM_TO_SLOT, host._SLOT_TO_SEAM),
        saved,
    ):
        live.clear()
        live.update(old)
    host._seam_sources_loaded = saved[4]


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
    "module_slot": "blocks.*.mix",
}


def test_the_builtin_v41_seams_are_declared_through_the_registry():
    assert host._SEAM_SOURCES["v41.block.ffn"] == "atom.plugin_seams_v41"
    assert host.SEAM_TO_MODULE_SLOT["v41.block.ffn"] == "layers.*.ffn"
    assert host.MODULE_SLOT_TO_SEAM["mono"] == "v41.mono.install"
    assert "v41.mono.install" in host.boot_only()


def test_a_package_declares_a_seam_a_recipe_plans_and_a_plugin_serves(
    seam_table, monkeypatch
):
    assert host.declare_seams([_TOY], source="toy") == ["toy.block.mix"]
    assert host.declare_seams([_TOY], source="again") == []  # identical: a no-op
    assert host.MODULE_SLOT_TO_SEAM["blocks.*.mix"] == "toy.block.mix"

    def register(h):
        h.register(
            seam="toy.block.mix", contract=1, owner="mix-p", impl=lambda x, **k: x + 1
        )

    monkeypatch.setattr(host, "_entry_points", lambda: {"mix-p": _EntryPoint(register)})
    host.configure(_plan(seams={"toy.block.mix": ["mix-p"]}), kill_switch=False)
    router = host.bind("toy.block.mix", lambda x: x)
    assert router(1) == 2


@pytest.mark.parametrize(
    "change, error",
    [
        ({"contract": 2}, "conflicts"),
        ({"name": "Bad"}, "dotted lower-case"),
        ({"key": ("z",)}, "key must name inputs"),
        ({"parent": "toy.none"}, "parent"),
        ({"graph": "sometimes"}, "graph"),
    ],
)
def test_a_seam_spec_is_checked_and_one_name_has_one_contract(
    seam_table, change, error
):
    host.declare_seams([_TOY], source="toy")
    with pytest.raises(host.SeamConflict, match=error):
        host.declare_seams([{**_TOY, **change}], source="other")


def test_installed_packages_declare_seams_through_entry_points(seam_table, monkeypatch):
    class _EP:
        name = "toy"

        def load(self):
            return lambda: [_TOY]

    host._seam_sources_loaded = False
    monkeypatch.setattr(
        host.metadata,
        "entry_points",
        lambda group=None: [_EP()] if group == host.SEAM_SPEC_GROUP else [],
    )
    plan = _plan(seams={"toy.block.mix": []})  # the recipe parse loads them
    assert "toy.block.mix" in host.SEAMS and plan is not None
    assert host._SEAM_SOURCES["toy.block.mix"] == "atom.seam_specs:toy"
