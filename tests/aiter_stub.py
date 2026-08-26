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
    """
    installed = []
    for name in (
        "aiter",
        "aiter.dist",
        "aiter.dist.shm_broadcast",
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
