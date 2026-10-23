# SPDX-License-Identifier: MIT
# Test helper: import engine modules without a GPU AITER build.

import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock


@contextmanager
def stubbed_aiter():
    """Import engine modules against a fake ``aiter``, then drop the fake.

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
