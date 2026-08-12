# SPDX-License-Identifier: MIT
# state_entry_views must cover a group's whole state, contiguously.
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_ops.attentions.gdn_attn import GDNStateMixin

LAYERS, GROUPS = 3, 4
SHAPE_K, SHAPE_V = (2, 5), (2, 3, 4)


def build(num_spec: int):
    span = 1 + num_spec
    slots = GROUPS * span
    k = torch.arange(LAYERS * slots * 10, dtype=torch.float32)[
        : LAYERS * slots * SHAPE_K[0] * SHAPE_K[1]
    ].reshape((LAYERS, slots) + SHAPE_K)
    v = torch.zeros((LAYERS, slots) + SHAPE_V)
    stub = SimpleNamespace(
        num_spec=num_spec,
        model_runner=SimpleNamespace(mamba_k_cache=k, mamba_v_cache=v),
    )
    return stub, k, v, span


@pytest.mark.parametrize("num_spec", [0, 2])
def test_every_view_is_contiguous(num_spec):
    """_build_meta rejects a strided segment (triton_kv_staging.py:135)."""
    stub, _, _, _ = build(num_spec)
    views = GDNStateMixin.state_entry_views(stub, 1)
    assert views
    assert all(v.is_contiguous() for v in views)


def test_views_cover_the_whole_group_and_nothing_else():
    stub, k, v, span = build(num_spec=2)
    views = GDNStateMixin.state_entry_views(stub, 1)
    total = sum(int(x.numel()) for x in views)
    expected = LAYERS * span * (k[0, 0].numel() + v[0, 0].numel())
    assert total == expected


def test_writing_through_the_views_writes_the_cache():
    """Views must alias, not copy — the packer reads them in place."""
    stub, k, _, span = build(num_spec=1)
    for view in GDNStateMixin.state_entry_views(stub, 2):
        view.fill_(7.0)
    assert torch.all(k[:, 2 * span : 3 * span] == 7.0)
    assert not torch.all(k[:, 0:span] == 7.0)
