# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``hi_weights``: the hi packs from the tensors ATOM loaded equal the packs from the checkpoint, byte for byte.

A synthetic layer in the checkpoint's form is packed for TP2, TP4 and TP8 rank 1 the checkpoint's way (``layer_impl.pack``:
sliced there); the same tensors put through what ATOM's load does to them (this rank's slices, the fused
projections, the FP8 16 x 16 preshuffle, the routed experts' gfx950 gate / up shuffle and scale shuffle: aiter's
own functions) are packed from ATOM's form (``block_tensors`` / ``expert_packs``, ``pack(local=True)``).
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl")
shuffle = pytest.importorskip("aiter.ops.shuffle")

RANK = 1  # (a rank past the first, at either TP size)


def _fp8(*shape, g=None):
    return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g).view(
        torch.float8_e4m3fn
    )


def _codes(*shape, g=None):
    return torch.randint(110, 135, shape, dtype=torch.uint8, generator=g)


def _shuffled(w):
    """ATOM's FP8 linear weight as it holds it: aiter's 16 x 16 preshuffle, flagged."""
    p = torch.nn.Parameter(
        shuffle.shuffle_weight(w.contiguous(), layout=(16, 16)), requires_grad=False
    )
    p.is_shuffled = True
    return p


def _linear(w, s, **kw):
    return SimpleNamespace(weight=_shuffled(w), weight_scale=torch.nn.Parameter(s.contiguous(), requires_grad=False),
                           **kw)  # fmt: skip


def _part(t, dim, tp):
    size = t.shape[dim] // tp
    return t.narrow(dim, RANK * size, size)


def _experts(full: dict, n: int, inter: int, hidden: int, g, tp: int):
    """ATOM's ``FusedMoE`` (TP, no EP: every expert's slice of the rank, padded to the 128 step) of the checkpoint
    experts in ``full``: the load's slices, then aiter's gfx950 shuffles (``moe.py``'s FlyDSL branch).
    """
    live = inter // tp
    pad = -(-live // 128) * 128
    w13 = torch.zeros(n, 2 * pad, hidden // 2, dtype=torch.uint8)
    s13 = _codes(
        n, 2 * pad, hidden // 32, g=g
    )  # (the padding's codes arbitrary: the packs' are 0)
    w2 = torch.zeros(n, hidden, pad // 2, dtype=torch.uint8)
    s2 = _codes(n, hidden, pad // 32, g=g)
    for e in range(n):
        for h, name in enumerate(("w1", "w3")):
            w13[e, h * pad : h * pad + live] = _part(
                full[f"ffn.experts.{e}.{name}.weight"].view(torch.uint8), 0, tp
            )
            s13[e, h * pad : h * pad + live] = _part(
                full[f"ffn.experts.{e}.{name}.scale"], 0, tp
            )
        w2[e, :, : live // 2] = _part(
            full[f"ffn.experts.{e}.w2.weight"].view(torch.uint8), 1, tp
        )
        s2[e, :, : live // 32] = _part(full[f"ffn.experts.{e}.w2.scale"], 1, tp)
    fp4 = torch.float4_e2m1fn_x2
    out = SimpleNamespace(
        w13_weight=shuffle.shuffle_weight(
            w13.view(fp4), is_guinterleave=True, gate_up=True
        ),
        w2_weight=shuffle.shuffle_weight(
            w2.view(fp4), is_guinterleave=True, gate_up=False
        ),
        w13_weight_scale=shuffle.shuffle_scale(
            s13.reshape(-1, s13.shape[-1]), n, is_guinterleave=True, gate_up=True
        ),
        w2_weight_scale=shuffle.shuffle_scale(
            s2.reshape(-1, s2.shape[-1]), n, is_guinterleave=True, gate_up=False
        ),
    )
    out.w13_weight.is_shuffled = out.w2_weight.is_shuffled = True
    return out


def hi_order_scales(
    experts, inter: int, tp: int, name: str = "moe"
) -> dict[str, torch.Tensor]:
    """ATOM's routed experts' scale codes (``hi_weights``' view of them) reordered into the hi runtime's own unit
    order (``pack_experts``': ``[E, rg, st, j, r]``, ``rg`` = 2 n0 + g / n0; the padding's codes 0): what the
    checkpoint's pack holds."""
    from atom.models.deepseek_v41.mono.hi_weights import (
        _GROUP,
        _expert_layout,
        _unshuffled_codes,
    )

    e, _n2, _kb, hidden, pad, live, groups, _s13, _s2, _g2 = _expert_layout(
        experts, inter, tp
    )
    c13, c2 = _unshuffled_codes(experts, inter, tp)
    c13 = c13.clone()
    c13[:, :, live:] = 0
    ug = (
        c13.reshape(e, 2, pad // 16, 16, groups // 4, 4)
        .permute(0, 2, 1, 4, 5, 3)
        .reshape(-1)
    )
    c2 = c2[:, :, : pad // _GROUP].clone()
    c2[:, :, live // _GROUP :] = 0
    down = (
        c2.reshape(e, hidden // 16, 16, pad // 128, 4)
        .permute(0, 1, 3, 4, 2)
        .reshape(-1)
    )
    return {f"{name}.ug.ws": ug.contiguous(), f"{name}.down.ws": down.contiguous()}


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_expert_packs_are_the_checkpoints(tp):
    """``expert_packs`` of ATOM's shuffled experts == ``pack_experts(.., "intra")`` of the checkpoint's (a small
    shape: 384-wide experts, a rank's 96-wide (TP4) or 192-wide (TP2) slice padded to the 128 step; down's 4 or 6
    scale groups padded to 8 by aiter; at TP8 the model's 2304-wide experts, a 288-wide slice padded to 384, down's 9
    scale groups padded to 16).
    """
    from atom.models.deepseek_v41.mono import hi_weights
    from atom.models.deepseek_v41.mono.hi.blocks.gfx950.moe_experts import pack_experts

    g = torch.Generator().manual_seed(7)
    n, inter, hidden = 3, 2304 if tp == 8 else 384, 256
    full = {}
    for e in range(n):
        for name, shape in (
            ("w1", (inter, hidden)),
            ("w3", (inter, hidden)),
            ("w2", (hidden, inter)),
        ):
            full[f"ffn.experts.{e}.{name}.weight"] = torch.randint(0, 256, (shape[0], shape[1] // 2), dtype=torch.uint8,
                                                                   generator=g)  # fmt: skip
            full[f"ffn.experts.{e}.{name}.scale"] = _codes(
                shape[0], shape[1] // 32, g=g
            )
    experts = _experts(full, n, inter, hidden, g, tp)
    ours = hi_weights.expert_packs(experts, inter, tp)
    ref = [SimpleNamespace(**{k: SimpleNamespace(packed=full[f"ffn.experts.{e}.{k}.weight"],
                                                 codes=full[f"ffn.experts.{e}.{k}.scale"]) for k in ("w1", "w2", "w3")})
           for e in range(n)]  # fmt: skip
    theirs = pack_experts(ref, "moe", tp, RANK, "intra")
    assert set(ours) == set(theirs)
    for k in ("moe.ug.w", "moe.down.w"):  # the weights: ATOM's bytes are the packs'
        assert torch.equal(ours[k].view(torch.uint8), theirs[k].view(torch.uint8)), k
    for k, v in hi_order_scales(
        experts, inter, tp
    ).items():  # the codes, reordered: the packs'
        assert torch.equal(v, theirs[k]), k
    # the kernels' index over ATOM's codes in place (``moe_experts._gemv(cols=)``: the staged task's 32-row blocks,
    # byte (rg // 2) cols 32 + (st // 2) 256 + lane 4 + (st % 2) 2 + rg % 2) == the pack's byte (rg steps + st) 64 +
    # lane, for every expert, row group, K step and lane of the live rows and K groups
    live = inter // tp
    pad = -(-live // 128) * 128
    for key, n_rows, kk, live_rows, live_k in (("moe.ug", 2 * pad, hidden, live, hidden), ("moe.down", hidden, pad,
                                                                                         hidden, live)):  # fmt: skip
        steps, cols = kk // 128, -(-kk // 32 // 8) * 8
        e, rg, st, lane = torch.meshgrid(torch.arange(n), torch.arange(n_rows // 16), torch.arange(steps),
                                         torch.arange(64), indexing="ij")  # fmt: skip
        at = (
            e * (n_rows * cols)
            + (rg // 2) * (cols * 32)
            + (st // 2) * 256
            + lane * 4
            + (st % 2) * 2
            + rg % 2
        )
        want = e * (n_rows * kk // 32) + (rg * steps + st) * 64 + lane
        row = (
            (rg % 2) * pad + (rg // 2) * 16 + lane % 16
            if key == "moe.ug"
            else rg * 16 + lane % 16
        )
        ok = ((row % pad if key == "moe.ug" else row) < live_rows) & (
            4 * st + lane // 16 < live_k // 32
        )
        assert torch.equal(
            ours[key + ".ws"][at[ok]], theirs[key + ".ws"][want[ok]]
        ), key
    # a padding code 255 (E8M0 NaN) is refused, before anything is packed (``check_experts``, the install's check):
    # the kernels apply the padding's codes to its zero weights
    hi_weights.check_experts(experts, inter, tp)
    experts.w13_weight_scale.view(torch.uint8).reshape(-1)[
        -1
    ] = 255  # (the last byte: the up half's last row: padding)
    with pytest.raises(hi_weights.MonoUnsupported, match="255"):
        hi_weights.check_experts(experts, inter, tp)


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_a_layer_packed_from_atoms_tensors_is_the_checkpoints(tp):
    """A window layer with Engram (4 routed experts: the packs do not depend on the count) packed both ways."""
    from atom.models.deepseek_v41.mono import hi_weights
    from atom.models.deepseek_v41.mono.hi.v41 import layer as L
    from atom.models.deepseek_v41.mono.hi.v41.recipe import (
        HD,
        HEADS,
        OGROUPS,
        ORANK,
    )

    g = torch.Generator().manual_seed(11)
    H, QL, INTER, NE = L.HIDDEN, 1280, 2304, 4
    OUT = OGROUPS * ORANK
    bf = lambda *s: torch.randn(*s, generator=g).bfloat16()
    f32 = lambda *s: torch.randn(*s, generator=g)
    w = {}

    def lin(name, n, k):
        w[f"{name}.weight"], w[f"{name}.scale"] = _fp8(n, k, g=g), _codes(
            n // 32, k // 32, g=g
        )

    lin("attn.wq_a", QL, H)
    lin("attn.wkv", HD, H)
    lin("attn.wq_b", HEADS * HD, QL)
    lin("attn.wo_b", H, OUT)
    lin("ffn.shared_experts.w1", INTER, H)
    lin("ffn.shared_experts.w3", INTER, H)
    lin("ffn.shared_experts.w2", H, INTER)
    raw = {"layers.3.attn.wo_a.weight": _fp8(OUT, HEADS * HD // OGROUPS, g=g),
           "layers.3.attn.wo_a.scale": _codes(OUT // 32, HEADS * HD // OGROUPS // 32, g=g)}  # fmt: skip
    for e in range(NE):
        for name, (n, k) in (
            ("w1", (INTER, H)),
            ("w3", (INTER, H)),
            ("w2", (H, INTER)),
        ):
            w[f"ffn.experts.{e}.{name}.weight"] = torch.randint(
                0, 256, (n, k // 2), dtype=torch.uint8, generator=g
            )
            w[f"ffn.experts.{e}.{name}.scale"] = _codes(n, k // 32, g=g)
    w.update({
        "attn.q_norm.weight": bf(QL), "attn.kv_norm.weight": bf(HD), "attn.attn_sink": f32(HEADS),
        "attn_norm.weight": bf(H), "ffn_norm.weight": bf(H), "ffn.gate.weight": bf(384, H), "ffn.gate.bias": f32(384),
        "engram.q_weight": bf(4, H), "engram.k_weight": bf(4, H),
    })  # fmt: skip
    for hc in ("attn", "ffn"):
        w[f"hc_{hc}_fn"], w[f"hc_{hc}_scale"], w[f"hc_{hc}_base"] = (
            f32(24, 4 * H),
            f32(3),
            f32(24),
        )
    theirs = L.pack(3, w, raw, device="cpu", tp=tp, rank=RANK, engram=True, experts=NE)

    # ATOM's tensors: this rank's slices, the fused projections, the FP8 preshuffle
    sc = lambda n: w[f"{n}.scale"]
    p = lambda t: torch.nn.Parameter(t.contiguous(), requires_grad=False)
    attn = SimpleNamespace(
        wqkv_a=_linear(torch.cat((w["attn.wq_a.weight"].view(torch.uint8), w["attn.wkv.weight"].view(torch.uint8)))
                       .view(torch.float8_e4m3fn), torch.cat((sc("attn.wq_a"), sc("attn.wkv"))), output_sizes=[QL, HD]),
        wq_b=_linear(_part(w["attn.wq_b.weight"], 0, tp), _part(sc("attn.wq_b"), 0, tp)),
        wo_a=_linear(_part(raw["layers.3.attn.wo_a.weight"], 0, tp), _part(raw["layers.3.attn.wo_a.scale"], 0, tp)),
        wo_b=_linear(_part(w["attn.wo_b.weight"], 1, tp), _part(sc("attn.wo_b"), 1, tp)),
        q_norm=SimpleNamespace(weight=p(w["attn.q_norm.weight"])), kv_norm=SimpleNamespace(weight=p(w["attn.kv_norm.weight"])),
        attn_sink=p(_part(w["attn.attn_sink"], 0, tp)), compressor=None, indexer=None,
    )  # fmt: skip
    se = "ffn.shared_experts."
    gate_up = torch.cat(
        [_part(w[se + n + ".weight"].view(torch.uint8), 0, tp) for n in ("w1", "w3")]
    )
    ffn = SimpleNamespace(
        shared_experts=SimpleNamespace(
            gate_up_proj=_linear(gate_up.view(torch.float8_e4m3fn), torch.cat([_part(sc(se + n), 0, tp) for n in ("w1", "w3")])),
            w2=_linear(_part(w[se + "w2.weight"], 1, tp), _part(sc(se + "w2"), 1, tp)),
        ),
        gate=SimpleNamespace(weight=p(w["ffn.gate.weight"]), e_score_correction_bias=p(w["ffn.gate.bias"])),
        experts=_experts(w, NE, INTER, H, g, tp),
    )  # fmt: skip
    block = SimpleNamespace(attn=attn, ffn=ffn, attn_norm=SimpleNamespace(weight=p(w["attn_norm.weight"])),
                            ffn_norm=SimpleNamespace(weight=p(w["ffn_norm.weight"])),
                            engram=SimpleNamespace(q_weight=p(w["engram.q_weight"]), k_weight=p(w["engram.k_weight"])),
                            **{k: p(w[k]) for k in w if k.startswith("hc_")})  # fmt: skip
    lw, lraw = hi_weights.block_tensors(block, "layers.3.")
    moe = hi_weights.expert_packs(block.ffn.experts, INTER, tp)
    ours = L.pack(
        3,
        lw,
        lraw,
        device="cpu",
        tp=tp,
        rank=RANK,
        local=True,
        moe=moe,
        engram=True,
        experts=NE,
    )

    # (the routed experts' codes compared in the pack's order: ATOM's in place, ``hi_order_scales``)
    ours = {**ours, **hi_order_scales(block.ffn.experts, INTER, tp)}
    assert set(ours) == set(theirs) and len(theirs) > 20
    bad = [k for k in sorted(theirs) if ours[k].dtype != theirs[k].dtype or not torch.equal(
        ours[k].contiguous().reshape(-1).view(torch.uint8), theirs[k].contiguous().reshape(-1).view(torch.uint8))]  # fmt: skip
    assert not bad, bad

    # the install's check (``check_block``) refuses what ``block_tensors`` / ``expert_packs`` would, before anything
    # is packed: each FP8 linear they read holding another dtype
    hi_weights.check_block(block, INTER, tp)
    lins = {"attn." + n: getattr(attn, n) for n in ("wqkv_a", "wq_b", "wo_a", "wo_b")}
    lins.update(
        {se + n: getattr(ffn.shared_experts, n) for n in ("gate_up_proj", "w2")}
    )
    for name, lin in lins.items():
        fp8 = lin.weight
        lin.weight = torch.nn.Parameter(fp8.data.view(torch.uint8), requires_grad=False)
        with pytest.raises(hi_weights.MonoUnsupported, match="torch.uint8"):
            hi_weights.check_block(block, INTER, tp)
        with pytest.raises(hi_weights.MonoUnsupported, match="torch.uint8"):
            hi_weights.block_tensors(block, "layers.3.")
        lin.weight = fp8
