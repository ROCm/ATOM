# SPDX-License-Identifier: MIT
"""Import ATOM modules without a GPU AITER build: `stubbed_aiter` for engine
modules that bind a few `aiter.dist` names, `atom_under_stubs` /
`atom_without_aiter` for model code that needs AITER's enums and Triton."""

import importlib
import importlib.abc
import importlib.util
import sys
import types
from contextlib import contextmanager
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch


@contextmanager
def stubbed_aiter(*extra: str):
    """Import engine modules against a fake ``aiter``, then drop the fake.

    ``extra`` names more modules to fake the same way, for import chains that
    reach past the ones below.

    ``async_proc`` needs a real AITER build at import time, which a CPU runner
    does not have. Leaving the stub in ``sys.modules`` would also make
    ``pytest.importorskip("aiter")`` succeed in every module collected
    afterwards, so it is removed as soon as the engine classes are bound.

    Dropping the stub is not enough on its own. Any ``atom`` module imported
    while it was installed has already bound ``MagicMock``s for whatever it
    took from ``aiter``, and it stays in ``sys.modules`` for every later test
    to import -- so a test that stubs here silently hands mocks to an
    unrelated test file that imports the same module against a real AITER.
    That is a cross-file failure with no connection to the test that caused
    it, so the ``atom`` modules first imported under the stub are dropped too
    and re-imported for real by whoever needs them next.

    Only the ones that actually took something, though. Dropping every
    ``atom`` module imported under the stub also drops the ones that never
    touched ``aiter``, and a caller still holding such a module gets the
    two-copies problem described in ``_drop`` -- from the other side. It bit
    ``test_scheduler_metrics``: ``engine_core_protocol`` has no AITER
    dependency, but it was dropped anyway, so pickling an enum member
    re-imported the module, got a different class, and raised
    ``PicklingError``.
    """
    before = set(sys.modules)
    installed = []
    # every object the stubs handed out, by identity: a module that bound one
    # is poisoned, a module that bound none is not.
    handed_out = []
    for name in (
        "aiter",
        "aiter.dist",
        "aiter.dist.communication_op",
        "aiter.dist.shm_broadcast",
        # Bound at import by the P/D connectors; the real module probes the
        # GPU arch through `rocminfo` on the way in.
        "aiter.dist.parallel_state",
        *extra,
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)

        def _stub_attr(_attr, _sink=handed_out):
            mock = MagicMock()
            _sink.append(mock)
            return mock

        module.__getattr__ = _stub_attr
        sys.modules[name] = module
        installed.append(name)
        handed_out.append(module)  # bound by a plain ``import aiter``
    try:
        yield
    finally:
        for name in installed:
            _drop(name)
        # Only when this context actually stubbed something: with a real AITER
        # present nothing was poisoned, and dropping modules the caller still
        # holds references to is how two copies of one class appear.
        if installed:
            handed = {id(obj) for obj in handed_out}
            fresh = {
                name
                for name in set(sys.modules) - before
                if name == "atom" or name.startswith("atom.")
            }
            # Seed: modules that bound something a stub handed out.
            tainted = {
                name for name in fresh if _binds_any(sys.modules.get(name), handed)
            }
            # Closure: a module holding a class from a tainted module is just
            # as stale once that module is dropped and re-imported -- the
            # second copy lands in the holder, not in `sys.modules`, which is
            # what `conftest.atom_modules_are_imported_once` catches.
            # `engine_core` binds `async_proc.AsyncIOProcManager` and takes
            # nothing from AITER itself, so only the closure reaches it.
            while True:
                more = {
                    name
                    for name in fresh - tainted
                    if _binds_from(sys.modules.get(name), tainted)
                }
                if not more:
                    break
                tainted |= more
            for name in tainted:
                _drop(name)


def _binds_any(module, ids: set[int]) -> bool:
    """True if the module holds, at module level, one of these objects."""
    if module is None:
        return False
    return any(id(value) in ids for value in vars(module).values())


def _binds_from(module, modules: set[str]) -> bool:
    """True if the module holds something defined in one of these modules."""
    if module is None:
        return False
    return any(
        getattr(value, "__module__", None) in modules for value in vars(module).values()
    )


def _drop(name: str) -> None:
    """Remove a module from ``sys.modules`` *and* from its parent package.

    Popping ``sys.modules`` alone is not enough: the parent package keeps an
    attribute bound to the same module object, so a later
    ``import a.b.c`` rebuilds ``sys.modules["a.b.c"]`` while ``a.b.c`` as an
    attribute still resolves to the stubbed one. The process then holds two
    copies of every class in it, and a dotted-path monkeypatch can patch the
    copy the test is not holding.
    """
    sys.modules.pop(name, None)
    parent, _, child = name.rpartition(".")
    if parent:
        module = sys.modules.get(parent)
        if module is not None and hasattr(module, child):
            try:
                delattr(module, child)
            except AttributeError:
                pass


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
    """Serve every `aiter` / `triton` import from a stand-in: the names in
    `_external_module_attributes` are real, any other attribute is a MagicMock."""
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
    """`importlib.import_module` against the stand-ins. ATOM modules first
    imported here are dropped again afterwards, so no later test sees them."""
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
