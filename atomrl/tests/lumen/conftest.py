# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Root test hooks.

``tests/ops/conftest.py`` is the helper module other tests import as
``conftest``. This file is also named ``conftest``, so once it exists those
imports land here. Re-export the ops helpers, and restore ``sys.modules``
after every test.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_OPS_CONFTEST = Path(__file__).with_name("ops") / "conftest.py"
_spec = importlib.util.spec_from_file_location(
    "_lumen_ops_conftest_helpers", _OPS_CONFTEST
)
_ops_helpers = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ops_helpers)
for _name, _value in vars(_ops_helpers).items():
    if not _name.startswith("_"):
        globals()[_name] = _value
del _name, _value


@pytest.fixture(autouse=True)
def _restore_sys_modules_after_test():
    before = dict(sys.modules)
    yield
    for key, module in before.items():
        if sys.modules.get(key) is not module:
            sys.modules[key] = module
    for key in list(sys.modules):
        if key in before:
            continue
        module = sys.modules[key]
        if (
            getattr(module, "__file__", None) is None
            and getattr(module, "__path__", None) is None
        ):
            del sys.modules[key]
