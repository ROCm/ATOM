# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``dcp.comm_backend="auto"`` (the default): which backend it resolves to, and
the post-capture check that every rank captured the same mailbox merges."""

import sys
from types import SimpleNamespace

import pytest
import torch

from atom import config as atom_config
from atom.config import Config

try:
    from atom.model_ops import dcp_ops
except ImportError as _e:
    dcp_ops, _DCP_OPS_ERR = None, _e
else:
    _DCP_OPS_ERR = None
needs_dcp_ops = pytest.mark.skipif(
    _DCP_OPS_ERR is not None, reason=f"requires full atom import env: {_DCP_OPS_ERR}"
)


def _resolve(monkeypatch, backend, dcp=4, plugin=False, tbo=False, gfx="gfx950"):
    monkeypatch.setattr(atom_config, "is_plugin_mode", lambda: plugin)

    def get_gfx():
        return gfx

    monkeypatch.setitem(
        sys.modules, "aiter.jit.utils.chip_info", SimpleNamespace(get_gfx=get_gfx)
    )
    cfg = SimpleNamespace(
        dcp_config=SimpleNamespace(comm_backend=backend),
        decode_context_parallel_size=dcp,
        enable_tbo_decode=tbo,
    )
    Config._resolve_dcp_comm_backend(cfg)
    return cfg.dcp_config.comm_backend


@pytest.mark.parametrize(
    "kw, expected",
    [
        ({}, "mailbox"),
        ({"dcp": 1}, "a2a"),
        ({"plugin": True}, "a2a"),
        ({"tbo": True}, "a2a"),
        ({"gfx": "gfx1250"}, "a2a"),
        ({"gfx": "gfx942"}, "a2a"),
    ],
)
def test_auto_takes_the_mailbox_only_where_it_was_validated(monkeypatch, kw, expected):
    assert _resolve(monkeypatch, "auto", **kw) == expected


@pytest.mark.parametrize("backend", ["a2a", "ag_rs", "mailbox"])
def test_an_explicit_backend_is_kept(monkeypatch, backend):
    assert _resolve(monkeypatch, backend, gfx="gfx942") == backend


def test_an_explicit_mailbox_takes_all_to_all_in_plugin_mode(monkeypatch):
    """Plugin mode rebinds its DCP group after the model is built."""
    assert _resolve(monkeypatch, "mailbox", plugin=True) == "a2a"
    assert _resolve(monkeypatch, "mailbox", plugin=True, dcp=1) == "mailbox"
    assert _resolve(monkeypatch, "ag_rs", plugin=True) == "ag_rs"


def test_an_explicit_mailbox_is_refused_next_to_decode_tbo(monkeypatch):
    """Not quietly swapped like `auto`. Checked after `_resolve_tbo`, so a TBO
    it turned off (tbo=False here) no longer refuses the mailbox."""
    with pytest.raises(AssertionError, match="decode TBO"):
        _resolve(monkeypatch, "mailbox", tbo=True)
    assert _resolve(monkeypatch, "mailbox", tbo=False) == "mailbox"
    assert _resolve(monkeypatch, "a2a", tbo=True) == "a2a"


def _peer_vote(peer_count):
    """all_reduce stub: one peer that captured `peer_count` merges."""

    def all_reduce(t, op, group):
        assert op == torch.distributed.ReduceOp.MIN
        assert group == "gloo", "the vote must cross the workspace's CPU group"
        t.copy_(torch.minimum(t, torch.tensor([peer_count, -peer_count])))

    return all_reduce


@needs_dcp_ops
@pytest.mark.parametrize("peer, ok", [(5, True), (6, False), (4, False)])
def test_ranks_that_captured_different_mailbox_merges_refuse_to_serve(
    monkeypatch, peer, ok
):
    key = ("g", 4, 128)
    other = ("g", 4, 512)  # a layout whose mapping fell back: not voted on
    monkeypatch.setattr(
        dcp_ops,
        "_MAILBOX_WORKSPACES",
        {key: SimpleNamespace(group="gloo"), other: None},
    )
    monkeypatch.setattr(dcp_ops, "_MAILBOX_LAUNCHES", {key: 5})
    monkeypatch.setattr(torch.distributed, "all_reduce", _peer_vote(peer))
    if ok:
        dcp_ops.check_dcp_mailbox_launches()
    else:
        with pytest.raises(RuntimeError, match="captured between"):
            dcp_ops.check_dcp_mailbox_launches()


@needs_dcp_ops
def test_only_a_merge_that_takes_the_mailbox_is_counted(monkeypatch):
    group = SimpleNamespace(world_size=4, rank_in_group=0)
    key = dcp_ops._mailbox_key(group, 4, 128)
    monkeypatch.setitem(dcp_ops._MAILBOX_WORKSPACES, key, SimpleNamespace(max_tokens=8))
    monkeypatch.setattr(dcp_ops, "_MAILBOX_LAUNCHES", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(dcp_ops, "cp_lse_a2a", lambda *a, **k: "a2a")
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.dcp_mailbox",
        SimpleNamespace(dcp_lse_combine=lambda ws, o, l, base_e, row_counts=None: o),
    )
    lse = torch.zeros(16, 16)
    dcp_ops.dcp_lse_merge(
        torch.zeros(2, 16, 128, dtype=torch.bfloat16), lse[:2], group, backend="mailbox"
    )
    # Over capacity: a2a, not counted.
    dcp_ops.dcp_lse_merge(
        torch.zeros(9, 16, 128, dtype=torch.bfloat16), lse[:9], group, backend="mailbox"
    )
    assert dcp_ops._MAILBOX_LAUNCHES == {key: 1}
