# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiter_stub import stubbed_aiter


def test_a_stand_in_another_module_left_serves_what_it_lacks(monkeypatch):
    """test_pd_pp leaves aiter's parallel_state with the two functions its
    connectors need. The stub kept it as it was, so importing any other name
    from it under the stub raised ImportError at collection."""
    stand_in = types.ModuleType("aiter.dist.parallel_state")
    stand_in.get_tp_group = lambda: "tp"
    monkeypatch.setitem(sys.modules, stand_in.__name__, stand_in)

    with stubbed_aiter():
        from aiter.dist.parallel_state import get_tp_group, get_world_group

    assert get_tp_group() == "tp" and callable(get_world_group)
    assert sys.modules[stand_in.__name__] is stand_in
    assert not hasattr(stand_in, "get_world_group")
