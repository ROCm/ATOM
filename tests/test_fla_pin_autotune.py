# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_FLA_PIN_AUTOTUNE: every autotuned FLA Triton kernel keeps only the
first config of its list, so no run's timing picks its bits."""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")

_UTILS = (
    pathlib.Path(__file__).parents[1] / "atom" / "model_ops" / "fla_ops" / "utils.py"
)

# The FLA modules that declare autotuned kernels with more than one config.
MODULES = [
    "chunk_delta_h",
    "chunk_delta_h_vk",
    "chunk_fused",
    "chunk_o",
    "chunk_o_vk",
    "chunk_scaled_dot_kkt",
    "cumsum",
    "l2norm",
    "solve_tril",
    "wy_fast",
]

# Imports the modules in a fresh interpreter (the decorators read the switch at
# import) and prints each autotuned kernel's config list.
_PROBE = """
import importlib, json, sys
from triton.runtime.autotuner import Autotuner
out = {}
for name in sys.argv[1:]:
    mod = importlib.import_module("atom.model_ops.fla_ops." + name)
    for attr, obj in vars(mod).items():
        tuner = obj
        while tuner is not None and not isinstance(tuner, Autotuner):
            tuner = getattr(tuner, "fn", None)  # through @triton.heuristics
        if isinstance(tuner, Autotuner) and tuner.base_fn.__module__ == mod.__name__:
            out[name + "." + attr] = [str(c) for c in tuner.configs]
print(json.dumps(out))
"""


def _fla_utils():
    """`fla_ops/utils.py`: the module itself where its package imports (it
    pulls in aiter, which needs a GPU), else the file alone."""
    try:
        return importlib.import_module("atom.model_ops.fla_ops.utils")
    except Exception:  # noqa: BLE001 - no GPU for aiter to probe
        spec = importlib.util.spec_from_file_location("_fla_utils_alone", _UTILS)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


@triton.jit
def _kernel(x, BLOCK: tl.constexpr):
    pass


@pytest.mark.parametrize(("value", "kept"), [("1", 1), ("true", 1), ("0", 3), ("", 3)])
def test_the_helper_keeps_the_first_config_only_when_pinned(monkeypatch, value, kept):
    monkeypatch.setenv("ATOM_FLA_PIN_AUTOTUNE", value)
    configs = [
        triton.Config({"BLOCK": block}, num_warps=warps)
        for block, warps in ((64, 2), (128, 4), (32, 8))
    ]
    tuner = _fla_utils().autotune(configs=configs, key=["x"])(_kernel)
    assert [str(c) for c in tuner.configs] == [str(c) for c in configs[:kept]]


def _configs(pin: str | None) -> dict[str, list[str]]:
    env = dict(os.environ)
    env.pop("ATOM_FLA_PIN_AUTOTUNE", None)
    if pin is not None:
        env["ATOM_FLA_PIN_AUTOTUNE"] = pin
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, *MODULES],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def unpinned():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("the FLA package imports aiter, which probes the GPU arch")
    return _configs(None)


def test_unpinned_kernels_keep_their_config_lists(unpinned):
    # The pin below is only meaningful while these kernels autotune: each has
    # configs that give different bits (BK for the output, num_warps for the
    # 16x16 triangular solve).
    assert len(unpinned["chunk_o.chunk_fwd_kernel_o"]) > 1
    assert len(unpinned["solve_tril.solve_tril_16x16_kernel"]) > 1
    assert len(unpinned) >= 10


@pytest.mark.parametrize("value", ["1", "true"])
def test_every_fla_kernel_keeps_its_first_config_when_pinned(unpinned, value):
    pinned = _configs(value)
    assert pinned.keys() == unpinned.keys()
    assert {name: configs[:1] for name, configs in unpinned.items()} == pinned


@pytest.mark.parametrize("value", ["0", ""])
def test_an_unset_pin_changes_nothing(unpinned, value):
    assert _configs(value) == unpinned
