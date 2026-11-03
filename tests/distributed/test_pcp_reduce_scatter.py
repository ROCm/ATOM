# SPDX-License-Identifier: MIT
"""Only SP takes AITER's lossy INT4 reduce-scatter."""

from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import pytest
import torch
from aiter_stubs import atom_without_aiter


@pytest.fixture(scope="module")
def pcp_utils():
    with atom_without_aiter() as load:
        yield load("atom.distributed.pcp_utils")


@pytest.mark.parametrize("sp", [False, True])
def test_pcp_merge_alone_keeps_the_exact_reduce_scatter(monkeypatch, pcp_utils, sp):
    qr_comm = Mock(**{"should_quick_reduce_scatter.return_value": True})
    qr_comm.quick_reduce_scatter.return_value = sentinel.int4
    group = SimpleNamespace(
        device_communicator=SimpleNamespace(qr_comm=qr_comm),
        reduce_scatter=Mock(return_value=sentinel.exact),
    )
    monkeypatch.setattr(pcp_utils, "get_pcp_group", lambda: group)
    monkeypatch.setattr(pcp_utils, "sp_is_enabled", lambda: sp)
    out = pcp_utils.pcp_reduce_scatter(torch.zeros(4, 8), pcp_size=4)
    assert out is (sentinel.int4 if sp else sentinel.exact)
    assert qr_comm.quick_reduce_scatter.call_count == int(sp)
