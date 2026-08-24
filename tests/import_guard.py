# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Skip when a dependency is absent. Fail when our own code is broken.

Several test modules import something heavy — an `aiter`-backed kernel, the
API server's transformers/uvicorn/fastapi stack — and guard it so a bare
non-GPU runner can still collect the rest of the suite. Those guards caught
`Exception`, which also catches a `SyntaxError`, a `NameError` or a bad
relative import *inside the module under test*, and turned it into a skip.

Measured: a `yield from` inside an `async def` generator in `api_server.py`
made nineteen tests go quiet. The run reported `279 passed, 40 skipped` — no
failure anywhere, and the count looked plausible. A syntax error in a shipped
module read as green.

So the question is asked once, here: an import error naming a third-party
module is an environment that cannot run this test, and an import error
naming one of ours — or anything that is not an import error at all — is a
bug that has to be seen.
"""

from __future__ import annotations

import pytest


def skip_if_dependency_missing(exc: BaseException, what: str) -> None:
    """Skip the module for an absent dependency; re-raise anything else.

    `what` names the import that failed, for the skip line.

    Only `ImportError` can be environmental, and only when the module it names
    is not ours. `ModuleNotFoundError` is a subclass, so "no module named
    aiter" and "cannot import name 'dtypes' from 'aiter'" are both covered —
    the second is what a namespace package left on the path produces, and it
    is how the non-GPU CI actually fails.
    """
    name = getattr(exc, "name", None) or ""
    if isinstance(exc, ImportError) and not name.startswith("atom"):
        pytest.skip(f"{what}: {exc}", allow_module_level=True)
    raise exc
