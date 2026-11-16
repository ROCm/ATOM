# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports TileRT 0.1.7 (MIT, Copyright (c) TileRT-team) ``tilert/models/ds_v4_1_flash/ops/{mhc,q_proj,compressor,
# indexer,attn_projo,moe,moe_rx,head,main_proj,dspark_head,engram}.py``: the weight packers.
"""The packers of DeepSeek-V4.1-Flash's TP8 decode weights: a rank's logical tensors (``ckpt``) in, the byte
layouts the TileRT-derived gfx950 kernels read out. Permutations, dtype views and exact float32 / bf16
conversions on the CPU, so a port check is a byte compare.

The twelve of TileRT's ``PACKER_SPECS`` (``params.PACKERS``): ``pack_mhc``, ``pack_final_norm``, ``pack_engram``,
``pack_q_proj``, ``pack_window_kv``, ``pack_compressor``, ``pack_indexer``, ``pack_attn_projo``, ``pack_moe``,
``pack_head``, ``pack_main_proj``, ``pack_dspark_head``; plus ``rx_from_main`` (``moe_rx``: the DSpark draft
layers' 128-expert MoE, repacked from ``pack_moe``'s layout) and ``engram_hash_params`` (the hash constants the
kernels take).
"""

from __future__ import annotations

import torch

HIDDEN = 5120
_U8 = torch.uint8


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(_U8)


# ---------------------------------------------------------------------------------------------------------------
# mHC (ops/mhc.py)

HC = 4
MHC_MIX = 24
HC_DIM = HC * HIDDEN


def _gamma(norm: torch.Tensor) -> torch.Tensor:
    """An RMSNorm gamma: bf16 when it is bf16 or bf16-exact, else float32."""
    if norm.dtype == torch.bfloat16:
        return norm.contiguous()
    g = norm.float().contiguous()
    b = g.bfloat16()
    return b.contiguous() if torch.equal(b.float(), g) else g


def _pack_half(fn, base, scale, norm) -> dict[str, torch.Tensor]:
    assert fn.shape == (MHC_MIX, HC_DIM), fn.shape
    assert base.shape == (MHC_MIX,) and scale.shape == (3,), (base.shape, scale.shape)
    assert norm.shape == (HIDDEN,), norm.shape
    return {
        "fn": fn.float().contiguous(),
        "base": base.float().contiguous(),
        "scale": scale.float().contiguous(),
        "gamma": _gamma(norm),
    }


def pack_mhc(
    hc_attn_fn: torch.Tensor,
    hc_attn_base: torch.Tensor,
    hc_attn_scale: torch.Tensor,
    hc_ffn_fn: torch.Tensor,
    hc_ffn_base: torch.Tensor,
    hc_ffn_scale: torch.Tensor,
    attn_norm: torch.Tensor,
    ffn_norm: torch.Tensor,
) -> dict[str, torch.Tensor]:
    out = {}
    for half, t in (
        ("attn", (hc_attn_fn, hc_attn_base, hc_attn_scale, attn_norm)),
        ("ffn", (hc_ffn_fn, hc_ffn_base, hc_ffn_scale, ffn_norm)),
    ):
        for k, v in _pack_half(*t).items():
            out[f"{half}_{k}"] = v
    return out


def pack_final_norm(norm: torch.Tensor) -> dict[str, torch.Tensor]:
    assert norm.shape == (HIDDEN,), norm.shape
    return {"gamma": _gamma(norm)}


# ---------------------------------------------------------------------------------------------------------------
# q / window kv projections (ops/q_proj.py)


def pack_pair_tiles_fp8(w: torch.Tensor) -> torch.Tensor:
    """[n, k] fp8 -> 16-row tiles, k in 64-byte pairs of 4 x 8-byte lanes."""
    n, k = w.shape
    assert n % 16 == 0 and k % 64 == 0
    u = _u8(w).view(n // 16, 16, k // 64, 2, 4, 8)
    return u.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1)


def pack_q_proj(wq_a, wq_a_s, q_norm, wq_b, wq_b_s) -> dict[str, torch.Tensor]:
    assert wq_a.shape == (1280, 5120) and wq_a_s.shape == (40, 160)
    assert wq_b.shape == (4096, 1280) and wq_b_s.shape == (128, 40)
    return {
        "qa_w": pack_pair_tiles_fp8(wq_a),
        "qa_s": _u8(wq_a_s).contiguous(),
        "q_norm": q_norm.float().contiguous(),
        "qb_w": pack_pair_tiles_fp8(wq_b),
        "qb_s": _u8(wq_b_s).contiguous(),
    }


def pack_window_kv(wkv, wkv_s, kv_norm) -> dict[str, torch.Tensor]:
    assert wkv.shape == (512, 5120) and wkv_s.shape == (16, 160)
    return {
        "kv_w": pack_pair_tiles_fp8(wkv),
        "kv_s": _u8(wkv_s).contiguous(),
        "kv_norm": kv_norm.float().contiguous(),
    }


# ---------------------------------------------------------------------------------------------------------------
# the KV compressor (ops/compressor.py)


def pack_tiles_bf16(w: torch.Tensor) -> torch.Tensor:
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and n % 16 == 0 and k % 32 == 0
    u = w.contiguous().view(n // 16, 16, k // 32, 4, 8)
    return u.permute(0, 2, 3, 1, 4).contiguous().view(-1)


def pack_compressor(c_wkv, c_norm, c_wgate=None) -> dict[str, torch.Tensor]:
    assert c_wkv.shape == (512, 5120) and c_wkv.dtype == torch.bfloat16
    w = c_wkv if c_wgate is None else torch.cat([c_wkv, c_wgate], dim=0)
    return {"c_w": pack_tiles_bf16(w), "c_norm": c_norm.float().contiguous()}


# ---------------------------------------------------------------------------------------------------------------
# the indexer (ops/indexer.py)


def pack_indexer(
    i_wq_b: torch.Tensor,
    i_wq_b_s: torch.Tensor,
    i_wproj: torch.Tensor,
    i_wk: torch.Tensor | None = None,
    i_k_norm: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    out = {
        "wq_b": i_wq_b.contiguous().view(_U8),
        "wq_b_s": i_wq_b_s.contiguous().view(_U8),
        "weights_proj": i_wproj.to(torch.bfloat16).contiguous(),
    }
    assert out["wq_b"].shape == (4096, 1280) and out["wq_b_s"].shape == (128, 40)
    assert out["weights_proj"].shape == (32, 5120)
    if i_wk is not None:
        out["wk"] = i_wk.to(torch.bfloat16).contiguous()
        out["k_norm"] = i_k_norm.float().contiguous()
        assert out["wk"].shape == (128, 512) and out["k_norm"].shape == (128,)
    return out


# ---------------------------------------------------------------------------------------------------------------
# the attention output projection (ops/attn_projo.py)

PROJO_HEADS = 8
PROJO_HEAD_DIM = 512
O_LORA = 1024
PROJO_K_A = PROJO_HEADS * PROJO_HEAD_DIM


def pack_projo(
    wo_a: torch.Tensor, wo_a_s: torch.Tensor, wo_b: torch.Tensor, wo_b_s: torch.Tensor
) -> dict[str, torch.Tensor]:
    assert wo_a.shape == (O_LORA, PROJO_K_A), wo_a.shape
    assert wo_a_s.shape == (O_LORA // 32, PROJO_K_A // 32), wo_a_s.shape
    assert wo_b.shape == (HIDDEN, O_LORA), wo_b.shape
    assert wo_b_s.shape == (HIDDEN // 32, O_LORA // 32), wo_b_s.shape
    a = _u8(wo_a).view(32, 2, 16, 8, 8, 2, 4, 8)
    pa = a.permute(0, 3, 4, 1, 6, 2, 5, 7).contiguous().reshape(-1)
    b = _u8(wo_b).view(160, 2, 16, 8, 2, 2, 4, 8)
    pb = b.permute(0, 3, 1, 4, 6, 2, 5, 7).contiguous().reshape(-1)
    return {
        "wo_a": pa,
        "wo_a_s": _u8(wo_a_s).reshape(-1).contiguous(),
        "wo_b": pb,
        "wo_b_s": _u8(wo_b_s).reshape(-1).contiguous(),
    }


def pack_attn_projo(
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_a_s: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_s: torch.Tensor,
) -> dict[str, torch.Tensor]:
    assert attn_sink.shape == (PROJO_HEADS,), attn_sink.shape
    out = {"attn_sink": attn_sink.float().contiguous()}
    out.update(pack_projo(wo_a, wo_a_s, wo_b, wo_b_s))
    return out


# ---------------------------------------------------------------------------------------------------------------
# the MoE (ops/moe.py): the routed experts' fp4 up/gate (``ug``) and down (``dn``), the shared expert's fp8 ``sug`` /
# ``sdn``; INTER is a rank's 288 of the 2304 intermediate columns

INTER = 288
_TILES = INTER // 8
_CHUNKS = HIDDEN // 128
_QUADS = HIDDEN // 4
_DTILES = HIDDEN // 16


def pack_dn(w2: torch.Tensor, w2_s: torch.Tensor):
    n = w2.shape[0]
    d = w2.reshape(n, _DTILES, 16, INTER // 2)
    frag = [
        d[..., 64 * c : 64 * (c + 1)]
        .reshape(n, _DTILES, 16, 4, 16)
        .transpose(2, 3)
        .reshape(n, _DTILES, 1024)
        for c in (0, 1)
    ]
    dn = torch.cat(frag + [d[..., 128:144].reshape(n, _DTILES, 256)], -1)
    s = w2_s.reshape(n, _DTILES, 16, 9)
    lo = torch.stack((s[..., 0:4], s[..., 4:8]), -1).transpose(2, 3)
    dn_s = torch.cat((lo.reshape(n, _DTILES, 128), s[..., 8]), -1)
    return dn.reshape(n, -1), dn_s.reshape(n, -1)


def unpack_dn(dn: torch.Tensor, dn_s: torch.Tensor):
    """``pack_dn``'s inverse."""
    n = dn.shape[0]
    d = dn.reshape(n, _DTILES, 2304)
    frag = [
        d[..., 1024 * c : 1024 * (c + 1)]
        .reshape(n, _DTILES, 4, 16, 16)
        .transpose(2, 3)
        .reshape(n, _DTILES, 16, 64)
        for c in (0, 1)
    ]
    w2 = torch.cat(frag + [d[..., 2048:].reshape(n, _DTILES, 16, 16)], -1)
    s = dn_s.reshape(n, _DTILES, 144)
    lo = s[..., :128].reshape(n, _DTILES, 4, 16, 2).transpose(2, 3)
    w2_s = torch.cat((lo[..., 0], lo[..., 1], s[..., 128:].unsqueeze(-1)), -1)
    return w2.reshape(n, HIDDEN, 144), w2_s.reshape(n, HIDDEN, 9)


def pack_moe_experts(w1, w1_s, w3, w3_s, w2, w2_s, chunk: int = 32):
    w1, w1_s, w3, w3_s, w2, w2_s = (_u8(t) for t in (w1, w1_s, w3, w3_s, w2, w2_s))
    e = w1.shape[0]
    assert w1.shape == (e, INTER, HIDDEN // 2) and w3.shape == w1.shape
    assert w1_s.shape == (e, INTER, HIDDEN // 32) and w3_s.shape == w1_s.shape
    assert w2.shape == (e, HIDDEN, INTER // 2) and w2_s.shape == (
        e,
        HIDDEN,
        INTER // 32,
    )
    ug = torch.empty(e, _TILES * _CHUNKS * 1024, dtype=_U8)
    ug_s = torch.empty(e, _TILES * _CHUNKS * 64, dtype=_U8)
    dn = torch.empty(e, _DTILES * 2304, dtype=_U8)
    dn_s = torch.empty(e, _DTILES * 144, dtype=_U8)
    for e0 in range(0, e, chunk):
        e1 = min(e, e0 + chunk)
        n = e1 - e0
        a = w1[e0:e1].view(n, _TILES, 8, HIDDEN // 2)
        b = w3[e0:e1].view(n, _TILES, 8, HIDDEN // 2)
        rows = torch.cat((a, b), dim=2)
        v = rows.view(n, _TILES, 16, _CHUNKS, 4, 16).permute(0, 1, 3, 4, 2, 5)
        ug[e0:e1] = v.reshape(n, -1)
        a = w1_s[e0:e1].view(n, _TILES, 8, HIDDEN // 32)
        b = w3_s[e0:e1].view(n, _TILES, 8, HIDDEN // 32)
        s = torch.cat((a, b), dim=2).view(n, _TILES, 16, _CHUNKS, 4)
        ug_s[e0:e1] = s.permute(0, 1, 3, 2, 4).reshape(n, -1)
        dn[e0:e1], dn_s[e0:e1] = pack_dn(w2[e0:e1], w2_s[e0:e1])
    return {"ug": ug, "ug_s": ug_s, "dn": dn, "dn_s": dn_s}


def pack_moe_shared(sw1, sw1_s, sw3, sw3_s, sw2, sw2_s):
    sw1, sw1_s, sw3, sw3_s, sw2, sw2_s = (
        _u8(t) for t in (sw1, sw1_s, sw3, sw3_s, sw2, sw2_s)
    )
    assert sw1.shape == (INTER, HIDDEN) and sw3.shape == sw1.shape
    assert sw1_s.shape == (INTER // 32, HIDDEN // 32) and sw3_s.shape == sw1_s.shape
    assert sw2.shape == (HIDDEN, INTER) and sw2_s.shape == (HIDDEN // 32, INTER // 32)
    a = sw1.view(_TILES, 8, HIDDEN)
    b = sw3.view(_TILES, 8, HIDDEN)
    rows = torch.cat((a, b), dim=1).view(_TILES, 16, _CHUNKS, 2, 4, 16)
    sug = rows.permute(0, 2, 3, 4, 1, 5).contiguous().view(-1)
    sug_s = torch.stack((sw1_s, sw3_s)).contiguous().view(-1)
    d = sw2.view(_QUADS, 4, INTER)
    sdn = (
        torch.cat(
            [d[..., 64 * i : 64 * (i + 1)].reshape(_QUADS, 256) for i in range(4)]
            + [d[..., 256:288].reshape(_QUADS, 128)],
            dim=1,
        )
        .contiguous()
        .view(-1)
    )
    return {
        "sug": sug,
        "sug_s": sug_s,
        "sdn": sdn,
        "sdn_s": sw2_s.contiguous().view(-1),
    }


def pack_moe(
    gate_w,
    gate_b,
    w1,
    w1_s,
    w3,
    w3_s,
    w2,
    w2_s,
    sw1,
    sw1_s,
    sw3,
    sw3_s,
    sw2,
    sw2_s,
    chunk: int = 32,
) -> dict:
    out = {
        "gate_w": gate_w.to(torch.bfloat16).contiguous(),
        "gate_b": gate_b.to(torch.float32).contiguous(),
    }
    out.update(pack_moe_experts(w1, w1_s, w3, w3_s, w2, w2_s, chunk=chunk))
    out.update(pack_moe_shared(sw1, sw1_s, sw3, sw3_s, sw2, sw2_s))
    return out


# ---------------------------------------------------------------------------------------------------------------
# the DSpark draft layers' MoE (ops/moe_rx.py): 128 experts, gate bf16-swizzled, scales / down in the "rx" order


def swizzle_bf16_m4(w: torch.Tensor) -> torch.Tensor:
    rows, k = w.shape
    assert rows % 8 == 0 and k % 64 == 0
    v = (
        w.to(torch.bfloat16)
        .view(torch.uint16)
        .reshape(rows // 8, 2, 4, k // 64, 2, 8, 4)
    )
    return v.permute(0, 3, 1, 5, 2, 4, 6).reshape(-1).contiguous().view(_U8)


def pack_moe_rx(lg: dict, chunk: int = 32) -> dict:
    w1, w3 = _u8(lg["w1"]), _u8(lg["w3"])
    e, inter, _ = w1.shape
    t = inter // 8
    out = {
        "gate_w": swizzle_bf16_m4(lg["gate_w"]).view(e, -1),
        "gate_b": lg["gate_b"].to(torch.float32).contiguous(),
    }
    ug = torch.empty(e, t * _CHUNKS * 1024, dtype=_U8)
    # 16 bytes of E8M0 1.0 past the scales
    ug_s = torch.empty(e * t * 64 * _CHUNKS + 16, dtype=_U8)
    ug_s[-16:] = 127
    ugsv = ug_s[:-16].view(e, -1)
    w1s, w3s = _u8(lg["w1_s"]), _u8(lg["w3_s"])
    for e0 in range(0, e, chunk):
        e1 = min(e, e0 + chunk)
        n = e1 - e0
        a = w1[e0:e1].view(n, t, 8, HIDDEN // 2)
        b = w3[e0:e1].view(n, t, 8, HIDDEN // 2)
        rows = torch.cat((a, b), dim=2)
        v = rows.view(n, t, 16, _CHUNKS, 4, 16).permute(0, 1, 3, 4, 2, 5)
        ug[e0:e1] = v.reshape(n, -1)
        a = w1s[e0:e1].view(n, t, 8, HIDDEN // 32)
        b = w3s[e0:e1].view(n, t, 8, HIDDEN // 32)
        s = torch.cat((a, b), dim=2).view(n, t, 16, _CHUNKS, 4)
        ugsv[e0:e1] = s.permute(0, 1, 4, 2, 3).reshape(n, -1)
    out["ug"], out["ug_s"] = ug, ug_s
    kb = inter // 32
    w2 = _u8(lg["w2"])
    out["dn"] = (
        w2.view(e, _QUADS, 4, kb, 16).permute(0, 1, 3, 2, 4).contiguous().view(e, -1)
    )
    out["dn_s"] = _u8(lg["w2_s"]).contiguous()
    a = _u8(lg["sw1"]).view(t, 8, HIDDEN)
    b = _u8(lg["sw3"]).view(t, 8, HIDDEN)
    rows = torch.cat((a, b), dim=1).view(t, 16, _CHUNKS, 2, 4, 16)
    out["sug"] = rows.permute(0, 2, 3, 4, 1, 5).contiguous().view(-1)
    out["sug_s"] = (
        torch.stack((_u8(lg["sw1_s"]), _u8(lg["sw3_s"]))).contiguous().view(-1)
    )
    sw2 = _u8(lg["sw2"])
    out["sdn"] = sw2.view(_QUADS, 4, kb, 32).permute(0, 2, 1, 3).contiguous().view(-1)
    out["sdn_s"] = _u8(lg["sw2_s"]).contiguous().view(-1)
    return out


def _logical_from_main(p: dict) -> dict:
    """``pack_moe``'s output back to the logical tensors (its inverse)."""
    e = p["gate_b"].shape[0]

    def ex(k):
        return p[k].view(e, -1)

    def sh(k):
        return p[k].reshape(-1)

    x = ex("ug").view(e, 36, 40, 4, 16, 16).permute(0, 1, 4, 2, 3, 5)
    x = x.reshape(e, 36, 16, HIDDEN // 2)
    s = ex("ug_s").view(e, 36, 40, 16, 4).permute(0, 1, 3, 2, 4)
    s = s.reshape(e, 36, 16, HIDDEN // 32)
    w2, w2_s = unpack_dn(ex("dn"), ex("dn_s"))
    g = sh("sug").view(36, 40, 2, 4, 16, 16).permute(0, 4, 1, 2, 3, 5)
    g = g.reshape(36, 16, HIDDEN)
    gs = sh("sug_s").view(2, 9, HIDDEN // 32)
    q = sh("sdn").view(_QUADS, 1152)
    sw2 = torch.cat(
        [q[:, 256 * i : 256 * (i + 1)].reshape(_QUADS, 4, 64) for i in range(4)]
        + [q[:, 1024:].reshape(_QUADS, 4, 32)],
        -1,
    ).reshape(HIDDEN, 288)
    return {
        "gate_w": p["gate_w"],
        "gate_b": p["gate_b"],
        "w1": x[:, :, :8].reshape(e, 288, -1),
        "w3": x[:, :, 8:].reshape(e, 288, -1),
        "w1_s": s[:, :, :8].reshape(e, 288, -1),
        "w3_s": s[:, :, 8:].reshape(e, 288, -1),
        "w2": w2,
        "w2_s": w2_s,
        "sw1": g[:, :8].reshape(288, -1),
        "sw3": g[:, 8:].reshape(288, -1),
        "sw1_s": gs[0],
        "sw3_s": gs[1],
        "sw2": sw2,
        "sw2_s": sh("sdn_s").view(HIDDEN // 32, 9),
    }


def rx_from_main(p: dict) -> dict:
    """A DSpark draft layer's MoE: ``pack_moe``'s output repacked in the rx layout (TileRT's HF loader route)."""
    return pack_moe_rx(_logical_from_main(p))


# ---------------------------------------------------------------------------------------------------------------
# the LM head (ops/head.py)

HEAD_VOCAB_LOCAL = 16160


def swizzle_head_bf16(w: torch.Tensor) -> torch.Tensor:
    assert w.shape == (HEAD_VOCAB_LOCAL, HIDDEN), w.shape
    w16 = w.contiguous().to(torch.bfloat16).view(torch.uint16)
    w16 = w16.reshape(HEAD_VOCAB_LOCAL // 16, 16, HIDDEN // 32, 4, 8)
    packed = w16.permute(0, 2, 3, 1, 4).contiguous()
    return packed.reshape(-1).view(_U8)


def pack_head(head: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"head_w": swizzle_head_bf16(head)}


# ---------------------------------------------------------------------------------------------------------------
# DSpark's main_proj (ops/main_proj.py): draft layer 0's [5120, 3 * 5120] fp8 projection, a rank's 640 output rows

MAIN_K_IN = 3 * HIDDEN
MAIN_N_LOCAL = HIDDEN // 8


def pack_main_proj(
    main_proj: torch.Tensor,
    main_proj_s: torch.Tensor,
    main_norm: torch.Tensor,
    rank: int,
) -> dict[str, torch.Tensor]:
    assert main_proj.shape == (HIDDEN, MAIN_K_IN), main_proj.shape
    assert main_proj_s.shape == (HIDDEN // 32, MAIN_K_IN // 32), main_proj_s.shape
    tiles, kchunks = MAIN_N_LOCAL // 16, MAIN_K_IN // 128
    w = _u8(main_proj)[MAIN_N_LOCAL * rank : MAIN_N_LOCAL * (rank + 1)]
    w = w.reshape(tiles, 16, kchunks, 2, 2, 4, 8)
    w = w.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
    s = _u8(main_proj_s)[MAIN_N_LOCAL // 32 * rank : MAIN_N_LOCAL // 32 * (rank + 1)]
    s = s.repeat_interleave(2, dim=0)
    s = s.reshape(tiles, kchunks, 4).contiguous()
    return {
        "w": w.reshape(-1),
        "w_s": s.reshape(-1),
        "main_norm": main_norm.float().contiguous(),
    }


# ---------------------------------------------------------------------------------------------------------------
# DSpark's markov head (ops/dspark_head.py)

MARKOV_RANK = 256


def pack_dspark_head(
    mtp_norm: torch.Tensor, markov_embed: torch.Tensor, markov_head: torch.Tensor
) -> dict[str, torch.Tensor]:
    assert markov_embed.shape == (HEAD_VOCAB_LOCAL * 8, MARKOV_RANK), markov_embed.shape
    assert markov_head.shape == (HEAD_VOCAB_LOCAL, MARKOV_RANK), markov_head.shape
    return {
        "mtp_norm": mtp_norm.float().contiguous(),
        "markov_embed": markov_embed.to(torch.bfloat16).contiguous(),
        "markov_head": markov_head.to(torch.bfloat16).contiguous(),
    }


# ---------------------------------------------------------------------------------------------------------------
# Engram (ops/engram.py), layers 1 and 14

ENGRAM_LAYERS = 2
ENGRAM_COLS = 24
ENGRAM_HEAD_DIM = 256
ENGRAM_K = ENGRAM_COLS * ENGRAM_HEAD_DIM
ENGRAM_OUT = HIDDEN * (HC + 1)


def swizzle_wkv(w: torch.Tensor) -> torch.Tensor:
    n = w.shape[0]
    assert w.shape[1] == ENGRAM_K and n % 32 == 0, w.shape
    u = w.contiguous().view(_U8).view(n // 16, 16, ENGRAM_K // 64, 2, 4, 8)
    return u.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1)


def pack_engram_proj(
    e_wkv: torch.Tensor, e_wkv_s: torch.Tensor, e_qk: torch.Tensor
) -> dict[str, torch.Tensor]:
    """``pack_engram`` without the table: ``wkv`` / ``wkv_s`` / ``qk`` (TileRT's HF loader packs these so, and reads
    the table rows apart)."""
    return {
        "wkv": swizzle_wkv(e_wkv),
        "wkv_s": e_wkv_s.contiguous().view(_U8),
        "qk": e_qk.float().contiguous(),
    }


def pack_engram(
    e_table: torch.Tensor,
    e_table_s: torch.Tensor,
    e_row_lo: torch.Tensor,
    e_wkv: torch.Tensor,
    e_wkv_s: torch.Tensor,
    e_qk: torch.Tensor,
) -> dict[str, torch.Tensor]:
    assert e_table.dim() == 2 and e_table.shape[1] == ENGRAM_HEAD_DIM, e_table.shape
    assert e_table_s.shape == (e_table.shape[0], ENGRAM_HEAD_DIM // 32), e_table_s.shape
    lo, hi = (int(v) for v in e_row_lo.flatten()[:2].tolist())
    assert hi - lo == e_table.shape[0], (lo, hi, e_table.shape)
    n = e_wkv.shape[0]
    assert e_wkv.shape == (n, ENGRAM_K) and n in (
        ENGRAM_OUT,
        ENGRAM_OUT // 2,
        ENGRAM_OUT // 4,
        ENGRAM_OUT // 8,
    ), f"e_wkv must be the rank's output-row slice [25600/npes, 6144], got {tuple(e_wkv.shape)}"
    assert e_wkv_s.shape == (n // 32, ENGRAM_K // 32), e_wkv_s.shape
    assert e_qk.shape == (HC, HIDDEN), e_qk.shape
    return {
        "table": e_table.contiguous().view(_U8),
        "table_s": e_table_s.contiguous().view(_U8),
        "row_range": torch.tensor([lo, hi], dtype=torch.int64),
        **pack_engram_proj(e_wkv, e_wkv_s, e_qk),
    }


def engram_hash_params(consts: dict) -> dict:
    """The hash constants as the kernels take them (TileRT's ``hash_consts_to``, on the CPU)."""
    return {
        "token_map": consts["token_map"].to(torch.int32).contiguous(),
        "mult": consts["multipliers"].to(torch.int64).contiguous(),
        "primes": consts["primes"]
        .to(torch.int64)
        .reshape(ENGRAM_LAYERS, ENGRAM_COLS)
        .contiguous(),
        "offsets": consts["offsets"].to(torch.int64).contiguous(),
        "pad_id": int(consts["pad_id"]),
    }
