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
    """
    before = set(sys.modules)
    installed = []
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
        module.__getattr__ = lambda _attr: MagicMock()
        sys.modules[name] = module
        installed.append(name)
    try:
        yield
    finally:
        for name in installed:
            sys.modules.pop(name, None)
        for name in set(sys.modules) - before:
            if name == "atom" or name.startswith("atom."):
                sys.modules.pop(name, None)
