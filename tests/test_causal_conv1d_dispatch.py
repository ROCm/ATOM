# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which kernel `causal_conv1d_fn` launches for a channel-last prefill.

The tiled / sliding-window kernels are opt-in (``token_tiles=True``, passed by
Kimi-K3's KDA only): GatedDeltaNet and the plugin GDN layers call it without
the argument and must keep the row kernel at every shape, including one the
tiles would accept. The kernels are replaced by recorders, so this runs on the
CPU job.
"""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter

with atom_without_aiter() as load:
    cc = load("atom.model_ops.mamba_ops.causal_conv1d")


@pytest.fixture
def launched(monkeypatch):
    """Records which kernel `causal_conv1d_fn` ran: "row", "slide" or "tiled"."""
    calls = []

    def tiles(name):
        def run(x, weight, bias, conv_states, query_start_loc, k_dim, v_dim, *rest):
            calls.append(name)
            tokens = x.shape[1]
            return (
                x.new_empty(tokens, k_dim),
                x.new_empty(tokens, k_dim),
                x.new_empty(tokens, v_dim),
            )

        return run

    class _RowKernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append("row")

    monkeypatch.setattr(cc, "_causal_conv1d_fwd_slide", tiles("slide"))
    monkeypatch.setattr(cc, "_causal_conv1d_fwd_tiled", tiles("tiled"))
    monkeypatch.setattr(cc, "_causal_conv1d_fwd_kernel", _RowKernel())
    monkeypatch.setattr(
        cc,
        "compute_causal_conv1d_metadata",
        lambda qsl, num_tokens, block_m=8: (
            {8: {"tot": 1, "num_tokens": num_tokens}},
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        ),
    )
    return calls


def _gdn_prefill(width, token_tiles=None, k_dim=512, v_dim=1024, tokens=37):
    """`causal_conv1d_fn` called as GatedDeltaNet's prefill calls it
    (attention_gdn.py), at a GDN shape: 16 key / 32 value heads of 128 at TP4.
    The arguments meet every other condition of the tiled path."""
    dim = 2 * k_dim + v_dim
    x = torch.randn(tokens, dim).transpose(0, 1)  # channel-last view
    kwargs = {} if token_tiles is None else {"token_tiles": token_tiles}
    return cc.causal_conv1d_fn(
        x,
        torch.randn(dim, width),
        None,
        activation="silu",
        conv_states=torch.zeros(4, dim, width - 1),
        has_initial_state=torch.tensor([True, False]),
        cache_indices=torch.tensor([0, 1], dtype=torch.int32),
        cache_indices_in=None,
        query_start_loc=torch.tensor([0, 20, tokens], dtype=torch.int32),
        k_dim_size=k_dim,
        v_dim_size=v_dim,
        metadata=SimpleNamespace(nums_dict=None),
        **kwargs,
    )


@pytest.mark.parametrize("width", [2, 3, 4])
def test_the_gdn_call_keeps_the_row_kernel(launched, width):
    _gdn_prefill(width)
    assert launched == ["row"]


@pytest.mark.parametrize("width, kernel", [(4, "slide"), (3, "tiled")])
def test_the_kda_opt_in_takes_the_tiles(launched, width, kernel):
    q, k, v = _gdn_prefill(width, token_tiles=True)
    assert launched == [kernel]
    assert (q.shape, k.shape, v.shape) == ((37, 512), (37, 512), (37, 1024))
