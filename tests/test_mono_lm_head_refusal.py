# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Whether a deployment takes the shared mono LM head, decided before its
kernels load (``serves.deployment_refusal``): what any peer-buffer launch
refuses -- a TP group across nodes, which its HIP IPC peer buffers cannot
reach -- refused before every rank agrees to bind and opens them."""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter")

from atom.mono.lm_head import serves


@pytest.fixture
def refusal(monkeypatch):
    """``deployment_refusal`` on a served arch and TP, the TP group on one
    node or across (``multinode``)."""
    monkeypatch.setattr(serves, "get_gfx", lambda: serves.ARCH)
    monkeypatch.setattr(serves.envs, "ATOM_MONO_ENABLE", True)

    def ask(multinode: bool, tp: int = serves.SERVED_TPS[0]):
        monkeypatch.setattr(
            serves,
            "get_current_atom_config",
            lambda: SimpleNamespace(is_multinode_tp=multinode),
        )
        return serves.deployment_refusal(tp)

    return ask


def test_a_tp_group_across_nodes_is_refused(refusal):
    assert "spans nodes" in refusal(multinode=True)


def test_a_tp_group_on_one_node_is_served(refusal):
    assert refusal(multinode=False) is None


def test_mono_off_is_refused_before_any_kernel_loads(refusal, monkeypatch):
    monkeypatch.setattr(serves.envs, "ATOM_MONO_ENABLE", False)
    assert refusal(multinode=False) == "ATOM_MONO_ENABLE is off"
