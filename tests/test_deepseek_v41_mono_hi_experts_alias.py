# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The hi packs read in place from ATOM's tensors: the routed experts' weights (``hi_weights.expert_packs``: the hi
runtime's unit order, ``[rg, st, j, r, b]`` with gate / up interleaved a 16-row group, is the order of aiter's gfx950
lane-level GUGU shuffle, written out here from both sides' definitions) and the plain tensors (``hi_weights.alias_plain``).
"""

import torch

E, INTER, HIDDEN = (
    3,
    128,
    256,
)  # (a 128-wide padded slice: 8 row groups a half; K: 2 / 1 steps of 128)


def _atom_gugu(w):
    """ATOM's gfx950 gate / up shuffle (aiter ``shuffle_weight(is_guinterleave=True, gate_up=True)``, layout (16,
    16)): ``[E, 2 N, K/2]`` GGUU -> ``[E, n0, gate | up, k0, j, r, b]``."""
    e, n2, kb = w.shape
    x = w.view(e, 2, n2 // 2 // 16, 16, kb // 64, 4, 16)
    return x.permute(0, 2, 1, 4, 5, 3, 6).contiguous().view(e, n2, kb)


def _atom_down(w):
    """... and its down shuffle (``gate_up=False``): ``[E, N, K/2]`` -> ``[E, n0, k0, j, r, b]``."""
    e, n, kb = w.shape
    return (
        w.view(e, n // 16, 16, kb // 64, 4, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view(e, n, kb)
    )


def _units(x):
    """the hi runtime's ``gemv.pack_weight`` MXFP4 unit order: ``[rg, r, st, j, b] -> [rg, st, j, r, b]``."""
    n, kb = x.shape
    return x.reshape(n // 16, 16, kb // 64, 4, 16).permute(0, 2, 3, 1, 4).reshape(-1)


def _hi_gate_up(w):
    """the hi runtime's ``pack_experts``: gate / up interleaved a 16-row group, then the units, every expert concatenated."""
    _, n2, kb = w.shape
    n = n2 // 2
    return torch.cat([_units(torch.stack((x[:n].reshape(n // 16, 16, kb), x[n:].reshape(n // 16, 16, kb)),
                                         1).reshape(n2, kb)) for x in w])  # fmt: skip


def test_unit_order_is_atoms_shuffle():
    w13 = torch.randint(0, 256, (E, 2 * INTER, HIDDEN // 2), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, HIDDEN, INTER // 2), dtype=torch.uint8)
    assert torch.equal(_atom_gugu(w13).reshape(-1), _hi_gate_up(w13))
    assert torch.equal(_atom_down(w2).reshape(-1), torch.cat([_units(x) for x in w2]))


def test_plain_tensors_alias_only_where_the_bytes_match():
    """``hi_weights.alias_plain``: a packed plain tensor (``PLAIN``) takes the block's own where its bytes match (the compressor's
    wkv / wgate: rows of ATOM's fused ``wkv_gate``), keeps its pack where they differ or the block has none.
    """
    from atom.models.deepseek_v41.mono import hi_weights

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = torch.nn.Module()
            self.ffn.gate = torch.nn.Module()
            self.ffn.gate.weight = torch.nn.Parameter(
                torch.randn(8, 4).bfloat16(), requires_grad=False
            )
            self.attn_norm = torch.nn.Module()
            self.attn_norm.weight = torch.nn.Parameter(
                torch.randn(4).bfloat16(), requires_grad=False
            )
            self.attn = torch.nn.Module()
            self.attn.compressor = torch.nn.Module()
            self.attn.compressor.wkv_gate = torch.nn.Module()
            self.attn.compressor.wkv_gate.weight = torch.nn.Parameter(
                torch.randn(1024, 2).bfloat16(), requires_grad=False
            )

    b = Block()
    wkv_gate = b.attn.compressor.wkv_gate.weight
    packed = {
        "router.w": b.ffn.gate.weight.clone(),  # the same bytes: aliased
        "attn_norm.g": b.attn_norm.weight.clone() + 1,  # other bytes: kept
        "ffn_norm.g": torch.zeros(4, dtype=torch.bfloat16),  # no ATOM tensor: kept
        "comp.wkv": wkv_gate[:512].clone(),
        "comp.wgate": wkv_gate[512:].clone(),
    }
    kept = packed["attn_norm.g"]
    assert hi_weights.alias_plain(b, packed) == 3
    assert packed["router.w"].data_ptr() == b.ffn.gate.weight.data_ptr()
    assert packed["comp.wkv"].data_ptr() == wkv_gate.data_ptr()
    assert packed["comp.wgate"].data_ptr() == wkv_gate[512:].data_ptr()
    assert packed["attn_norm.g"] is kept
    assert all(
        getattr(packed[k], "_atom_alias", False)
        for k in ("router.w", "comp.wkv", "comp.wgate")
    )
