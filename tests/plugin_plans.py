# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Test helper: the recipe plan that gives the named stand-in plugins every
seam and module slot they register (what a recipe lists for them), owners in
the order named."""

from types import SimpleNamespace

from atom import plugin_host as host
from atom import plugin_recipe as recipes


def _registrations(name):
    seams, slots = [], []

    def register(**kwargs):
        seams.append(kwargs["seam"])
        return SimpleNamespace(accepted=True)

    def register_module(*, slot, method=None, **kwargs):
        slots.append(slot if method is None else f"{slot}:{method}")
        return SimpleNamespace(accepted=True)

    recorder = SimpleNamespace(
        **{k: getattr(host, k) for k in dir(host) if not k.startswith("__")}
    )
    recorder.register = register
    recorder.register_module = register_module
    points = host._entry_points()
    if name in points:
        try:
            points[name].load()(recorder)
        except Exception:  # noqa: BLE001, S110 -- a plugin broken on purpose
            pass
    return seams, slots


def enable(names, **recipe):
    """``names``: a list (or a comma string) of entry point names."""
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    seams: dict[str, list[str]] = {}
    modules: dict[str, list[str]] = {}
    for name in names:
        got_seams, got_slots = _registrations(name)
        for seam in got_seams:
            seams.setdefault(seam, []).append(name)
        for slot in got_slots:
            modules.setdefault(slot, []).append(name)
    document = {
        "schema": recipes.RECIPE_SCHEMA,
        "arch": "DeepseekV41ForCausalLM",
        "seams": seams,
        "modules": modules,
        **recipe,
    }
    plan = recipes.resolve([document])
    # A name that registers nothing is still enabled (a broken plugin).
    missing = [n for n in names if n not in plan.providers]
    if missing:
        arms = dict(plan.arms)
        from dataclasses import replace
        from types import MappingProxyType

        providers = (*plan.providers, *missing)
        arms["plugin"] = arms["identity"] = providers
        plan = replace(plan, providers=providers, arms=MappingProxyType(arms))
    return plan
