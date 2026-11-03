# SPDX-License-Identifier: MIT
"""AITER / Triton stand-ins, so a CPU runner imports the real ATOM modules.

Only the unavailable boundary is replaced: an attribute not listed below
resolves to a MagicMock. ATOM modules first imported against the stand-ins are
dropped again afterwards, so no later test sees them.
"""

import importlib
import importlib.abc
import importlib.util
import sys
from contextlib import contextmanager
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch


class ActivationType(Enum):
    Silu = 0
    Swiglu = 1
    Situv2 = 2


class QuantType(Enum):
    No = 0
    per_Tensor = 1
    per_Token = 2
    per_1x128 = 3
    per_1x32 = 4


class GateMode(Enum):
    INTERLEAVE = "interleave"
    SEPARATED = "separated"


def _identity_decorator(*_args, **_kwargs):
    return lambda function: function


def _triton_jit(function=None, **_options):
    """`triton.jit`, bare or with options (`@triton.jit(do_not_specialize=...)`)."""
    return function if function is not None else (lambda function: function)


def _external_module_attributes(name: str) -> dict[str, object]:
    dtypes = SimpleNamespace(
        bf16=torch.bfloat16,
        fp32=torch.float32,
        fp4x2=torch.uint8,
        fp8=torch.float8_e4m3fnuz,
        fp8_e8m0=torch.uint8,
        i4x2=torch.uint8,
        i8=torch.int8,
        d_dtypes={},
    )
    return {
        "aiter": {
            "ActivationType": ActivationType,
            "QuantType": QuantType,
            "dtypes": dtypes,
        },
        "aiter.jit.utils.torch_guard": {
            "torch_compile_guard": _identity_decorator,
        },
        "aiter.ops.flydsl.moe_common": {"GateMode": GateMode},
        "triton": {
            "jit": _triton_jit,
            "heuristics": _identity_decorator,
        },
        # An annotation, and `tl.constexpr(value)` at module scope.
        "triton.language": {"constexpr": lambda value=None: value},
    }.get(name, {})


class _ExternalModuleLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        module.__file__ = f"<test stub for {module.__name__}>"
        module.__path__ = []
        module.__dict__.update(_external_module_attributes(module.__name__))

        def resolve(attribute: str):
            value = MagicMock(name=f"{module.__name__}.{attribute}")
            setattr(module, attribute, value)
            return value

        module.__getattr__ = resolve


class _ExternalModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in {"aiter", "triton"} or fullname.startswith(
            ("aiter.", "triton.")
        ):
            return importlib.util.spec_from_loader(
                fullname, _ExternalModuleLoader(), is_package=True
            )
        return None


@contextmanager
def stubbed_external_kernel_modules():
    # Torch imports its own optional Triton integration lazily. Resolve that
    # first so this test's deliberately small Triton stub cannot affect it.
    importlib.import_module("torch._dynamo.config")

    prefixes = ("aiter", "triton")
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name in prefixes or name.startswith(("aiter.", "triton."))
    }
    for name in saved:
        sys.modules.pop(name, None)

    finder = _ExternalModuleFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name in prefixes or name.startswith(("aiter.", "triton.")):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


@contextmanager
def atom_under_stubs():
    """`importlib.import_module` against the stand-ins."""
    existing_atom_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "atom" or name.startswith("atom.")
    }
    # A package imported before keeps what it picks up meanwhile (a lazily
    # cached export, a submodule attribute) unless its namespace is put back.
    namespaces = {name: dict(vars(m)) for name, m in existing_atom_modules.items()}
    with stubbed_external_kernel_modules():
        try:
            yield importlib.import_module
        finally:
            for name in list(sys.modules):
                if (name == "atom" or name.startswith("atom.")) and (
                    name not in existing_atom_modules
                ):
                    sys.modules.pop(name, None)
            sys.modules.update(existing_atom_modules)
            for name, namespace in namespaces.items():
                vars(existing_atom_modules[name]).clear()
                vars(existing_atom_modules[name]).update(namespace)


@contextmanager
def atom_without_aiter():
    """`importlib.import_module`, against the stand-ins only where AITER is
    not installed (the CPU job); a GPU host keeps the real kernels."""
    try:
        import aiter  # noqa: F401
    except ImportError:
        with atom_under_stubs() as load:
            yield load
    else:
        yield importlib.import_module
