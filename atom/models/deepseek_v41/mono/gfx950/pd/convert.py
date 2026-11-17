# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports TileRT 0.1.7 (MIT, Copyright (c) TileRT-team) ``tilert/pd_vllm/profiles/dsv41_rocm_convert.py`` and the
# receive half of ``Dsv41RocmProfile.convert`` (``tilert/pd_vllm/profiles/dsv41_rocm.py``).
"""A prefill's DeepSeek-V4.1-Flash KV, landed in the TileRT wire image (``wire``), turned into the decode caches the
gfx950 kernels read. Pure torch, on whatever device the inputs are on.

The wire records (the prefill's, vLLM's ``fp8_ds_mla`` and MXFP4 index caches):

- main / window: 584-byte V4 records, a page being ``cb`` records' data (448 E4M3 nope bytes + 64 bf16 rope dims
  each, 576 B) then their scales (7 UE8M0, one a 64-dim nope group, + 1 pad byte, 8 B);
- index: 68-byte MXFP4 records (64 B of E2M1 pairs, even element in the low nibble, + 4 UE8M0 scales, one a 32-dim
  group), a page shuffled in 32-row tiles (``_idx_offsets``);
- compressor ring: rows of 2 x 512 float32 (kv state then score state).

The decode planes (``build_planes``), per request:

- ``main_kv[s]`` for the KV-source layers 2 / 8 / 14 (ratio 2) and 20 (ratio 1), ``n // ratio`` rows: FP4 E2M1
  (uint8 [rows, 256]) and an E4M3 scale per 16 dims (uint8 [rows, 32]), from the dequantized records (``to_main``);
- ``index_k[s]``: the index records unshuffled, ``n // ratio`` rows (uint8 [rows, 64], [rows, 4]);
- ``win[i]`` for the 40 layers and 3 DSpark layers: a 256-row ring of positions ``[n - 128, n)`` at ``pos % 256``,
  E4M3 with a UE8M0 scale per 32 dims (``to_window``), zeros elsewhere and for a layer the prefill did not send;
- ``state[s]`` for the ratio-2 sources: 16 slots of (kv, score) float32, score -inf, slot ``(n - 1) % 16`` the ring
  row of position ``n - 1`` when ``n`` is odd;
- ``hist``: the last 16 prompt tokens' compressed (Engram) ids at ``pos % 16``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from . import wire as W

logger = logging.getLogger(__name__)

U8, BF16, F32, I32, I64 = (
    torch.uint8,
    torch.bfloat16,
    torch.float32,
    torch.int32,
    torch.int64,
)
E4M3 = torch.float8_e4m3fn
HEAD_DIM = 512
V4_REC, V4_DATA, V4_SC = (584, 576, 8)
NOPE = 448
IDX_REC, IDX_W, IDX_S = (68, 64, 4)
NPT, DPT, LANES = (32, 16, 2)
RING_ROW = 2 * HEAD_DIM
WINDOW = 128
WIN_RING = 256
STATE_SLOTS = 16
HIST = 16
KV_SOURCES = (2, 8, 14, 20)
RATIO = {2: 2, 8: 2, 14: 2, 20: 1}
NUM_BACKBONE, NUM_DRAFT = (40, 3)
NUM_WIN = NUM_BACKBONE + NUM_DRAFT
_F32_INV_448 = torch.tensor(1.0 / 448.0, dtype=F32).item()


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """The smallest power of two >= ``x`` (positive normal float32), by its bits."""
    bits = x.view(I32)
    e = (bits >> 23 & 255) - 127 + (bits & 8388607 != 0).to(I32)
    return (e + 127 << 23).view(F32)


def _a8_ue8m0(x: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 with a power-of-two (UE8M0) scale per ``block``: amax / 448 rounded up, at least 1e-4 / 448."""
    xf = x.float().unflatten(-1, (-1, block))
    s = _pow2_ceil(xf.abs().amax(-1).clamp_min(0.0001) * _F32_INV_448)
    y = (xf / s.unsqueeze(-1)).clamp(-448.0, 448.0).to(E4M3)
    return y.flatten(-2), s.to(torch.float8_e8m0fnu)


def _fp4_code(v: torch.Tensor) -> torch.Tensor:
    """E2M1 codes of ``v`` (|v| <= 6), round to nearest, ties to even; sign in bit 3."""
    a = v.abs()
    code = (
        (a > 0.25).to(U8)
        + (a >= 0.75)
        + (a > 1.25)
        + (a >= 1.75)
        + (a > 2.5)
        + (a >= 3.5)
        + (a > 5.0)
    ).to(U8)
    return code | torch.signbit(v).to(U8) << 3


def _fp4_e4m3(x: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """E2M1 pairs (even element low) with an E4M3 scale per ``block``: amax / 6, at least 2^-9."""
    xf = x.float().unflatten(-1, (-1, block))
    s = (xf.abs().amax(-1).clamp_min(6.0 * 2.0 ** (-9)) / 6.0).to(E4M3).float()
    code = _fp4_code((xf / s.unsqueeze(-1)).clamp(-6.0, 6.0)).flatten(-2)
    return code[..., 0::2] | code[..., 1::2] << 4, s.to(E4M3)


def v4_split(pages: torch.Tensor, cb: int):
    """V4 pages of ``cb`` records -> (nope E4M3 bytes [rows, 448], rope bf16 [rows, 64], scales [rows, 7])."""
    p = pages.reshape(pages.shape[0], -1)
    assert p.shape[1] >= cb * V4_REC, (tuple(pages.shape), cb)
    data = p[:, : cb * V4_DATA].reshape(-1, V4_DATA)
    sc = p[:, cb * V4_DATA : cb * V4_REC].reshape(-1, V4_SC)
    return data[:, :NOPE], data[:, NOPE:].contiguous().view(BF16), sc[:, :7]


def v4_dequant(
    nope: torch.Tensor, rope: torch.Tensor, sc: torch.Tensor
) -> torch.Tensor:
    """float32 [rows, 512]: nope * 2^(scale - 127) per 64 dims, then rope."""
    s = torch.exp2(sc.float() - 127.0).repeat_interleave(64, -1)
    return torch.cat([nope.view(E4M3).float() * s, rope.float()], -1)


def _idx_offsets(cb: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Byte offsets in an index page of each record's 64 value bytes and 4 scale bytes (row-major)."""
    assert cb % NPT == 0, f"index K page of {cb} rows is not a multiple of {NPT}"
    pos = torch.arange(cb, device=device)[:, None]
    b = torch.arange(IDX_W, device=device)[None, :]
    voff = (
        pos // NPT * (NPT * IDX_W) + b // DPT * (NPT * DPT) + pos % NPT * DPT + b % DPT
    )
    s = torch.arange(IDX_S, device=device)[None, :]
    shi = IDX_S // LANES
    soff = (
        cb * IDX_W
        + pos // NPT * (NPT * IDX_S)
        + s % LANES * (NPT * shi)
        + pos % NPT * shi
        + s // LANES
    )
    return voff.reshape(-1), soff.reshape(-1)


def idx_unshuffle(pages: torch.Tensor, cb: int) -> tuple[torch.Tensor, torch.Tensor]:
    p = pages.reshape(pages.shape[0], -1)
    assert p.shape[1] >= cb * IDX_REC, (tuple(pages.shape), cb)
    voff, soff = _idx_offsets(cb, p.device)
    vals = p[:, voff].reshape(-1, IDX_W)
    scs = p[:, soff].reshape(-1, IDX_S)
    return vals, scs


def to_window(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A window row: E4M3 (uint8 [rows, 512]) and a UE8M0 scale per 32 (uint8 [rows, 16])."""
    if x.shape[0] == 0:
        return (
            torch.zeros(0, HEAD_DIM, dtype=U8, device=x.device),
            torch.zeros(0, HEAD_DIM // 32, dtype=U8, device=x.device),
        )
    q, s = _a8_ue8m0(x.to(BF16), 32)
    return q.view(U8), s.view(U8)


def to_main(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A main KV row: E2M1 pairs (uint8 [rows, 256]) and an E4M3 scale per 16 (uint8 [rows, 32])."""
    if x.shape[0] == 0:
        return (
            torch.zeros(0, HEAD_DIM // 2, dtype=U8, device=x.device),
            torch.zeros(0, HEAD_DIM // 16, dtype=U8, device=x.device),
        )
    p, s = _fp4_e4m3(x, 16)
    return p.view(U8), s.view(U8)


MAIN_CHUNK_ROWS = 16384


def main_plane(
    pages: torch.Tensor, cb: int, ne: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The first ``ne`` records of V4 ``pages`` as main KV rows (``to_main``), ``MAIN_CHUNK_ROWS`` at a time."""
    dev = pages.device
    if ne == 0:
        return to_main(torch.zeros(0, HEAD_DIM, dtype=F32, device=dev))
    assert pages.shape[0] * cb >= ne, (tuple(pages.shape), cb, ne)
    vals = torch.empty(ne, HEAD_DIM // 2, dtype=U8, device=dev)
    scs = torch.empty(ne, HEAD_DIM // 16, dtype=U8, device=dev)
    step = max(1, MAIN_CHUNK_ROWS // cb)
    for k0 in range(0, -(-ne // cb), step):
        r0 = k0 * cb
        x = v4_dequant(*v4_split(pages[k0 : k0 + step], cb))
        r1 = min(ne, r0 + x.shape[0])
        vals[r0:r1], scs[r0:r1] = to_main(x[: r1 - r0])
    return vals, scs


def index_plane(
    pages: torch.Tensor, cb: int, ne: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The first ``ne`` index records of ``pages``, unshuffled."""
    if ne == 0:
        return (
            torch.zeros(0, IDX_W, dtype=U8, device=pages.device),
            torch.zeros(0, IDX_S, dtype=U8, device=pages.device),
        )
    vals, scs = idx_unshuffle(pages, cb)
    assert vals.shape[0] >= ne, (vals.shape, ne)
    return vals[:ne].contiguous(), scs[:ne].contiguous()


def window_first_block(n: int, cb: int = 32) -> int:
    """The first window page a prefill of ``n`` positions sends: the one holding position ``n - 128``."""
    return max(0, n - WINDOW) // cb


def window_plane(
    pages: torch.Tensor | None, first_block: int, n: int, cb: int = 32, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """A layer's window ring (E4M3 [256, 512], uint8 [256, 16]): positions ``[n - 128, n)`` of V4 ``pages`` (the
    pages from ``first_block`` on) at ``pos % 256``."""
    dev = device if device is not None else pages.device if pages is not None else "cpu"
    kv = torch.zeros(WIN_RING, HEAD_DIM, dtype=U8, device=dev)
    sc = torch.zeros(WIN_RING, HEAD_DIM // 32, dtype=U8, device=dev)
    if pages is None or n <= 0:
        return kv.view(E4M3), sc
    lo = max(0, n - WINDOW)
    x = v4_dequant(*v4_split(pages, cb))
    off = lo - first_block * cb
    assert 0 <= off and off + (n - lo) <= x.shape[0], (first_block, n, x.shape)
    q, s = to_window(x[off : off + (n - lo)])
    slots = torch.arange(lo, n, device=dev) & WIN_RING - 1
    kv[slots] = q
    sc[slots] = s
    return kv.view(E4M3), sc


def state_plane(ring: torch.Tensor | None, n: int, device=None) -> torch.Tensor:
    """A ratio-2 source's compressor state (float32 [16, 2, 512]): position ``n - 1``'s ring row when ``n`` is odd
    (the group it opens is incomplete), else empty (kv 0, score -inf)."""
    dev = device if device is not None else ring.device if ring is not None else "cpu"
    st = torch.zeros(STATE_SLOTS, 2, HEAD_DIM, dtype=F32, device=dev)
    st[:, 1] = -torch.inf
    if n % 2:
        assert ring is not None, "odd prompt length needs the compressor ring"
        rows = ring.reshape(-1, RING_ROW)
        st[n - 1 & STATE_SLOTS - 1] = rows[(n - 1) % rows.shape[0]].view(2, HEAD_DIM)
    return st


def hist_plane(tail_ids, n: int, token_map: torch.Tensor, device=None) -> torch.Tensor:
    """int32 [16]: the compressed ids (``token_map``) of the last ``min(n, 16)`` prompt tokens at ``pos % 16``."""
    hist = torch.zeros(HIST, dtype=I32, device=device or "cpu")
    k = min(n, HIST)
    if k == 0:
        return hist
    ids = torch.as_tensor(list(tail_ids)[-k:], dtype=I64).to(hist.device)
    assert (
        ids.numel() == k
    ), f"{ids.numel()} prompt ids for the history of {n} positions"
    tm = token_map.to(hist.device)
    ids = tm[ids.clamp(0, tm.numel() - 1)]
    pos = torch.arange(n - k, n, device=hist.device)
    hist[pos & HIST - 1] = ids.to(I32)
    return hist


def build_planes(
    n: int,
    main: dict,
    index: dict,
    win: list,
    win_first_block: int,
    rings: dict,
    tail_ids,
    token_map,
    device=None,
    win_cb: int = 32,
) -> dict:
    """A request's decode planes from its wire pages: ``main`` / ``index`` ``{source: (pages, cb)}``, ``win`` the
    window pages of each layer sent, ``rings`` ``{ratio-2 source: ring rows}``."""
    planes: dict = {"main_kv": {}, "index_k": {}, "win": [], "state": {}}
    for s in KV_SOURCES:
        ne = n // RATIO[s]
        pg, cb = main[s]
        planes["main_kv"][s] = main_plane(pg, cb, ne)
        pg, cb = index[s]
        planes["index_k"][s] = index_plane(pg, cb, ne)
    for i in range(NUM_WIN):
        pg = win[i] if i < len(win) else None
        planes["win"].append(window_plane(pg, win_first_block, n, win_cb, device))
    for s in KV_SOURCES:
        if RATIO[s] == 2:
            planes["state"][s] = state_plane(rings.get(s), n, device)
    planes["hist"] = hist_plane(tail_ids, n, token_map, device)
    return planes


# ---------------------------------------------------------------------------------------------------------------
# the landed buffer (Dsv41RocmProfile.convert)


@dataclass
class ConvertedRequest:
    rid: str
    seq_len: int
    last_prompt_token: int
    first_token_id: int | None
    sampling: dict | None
    planes: dict


_WARNED: set = set()


@torch.inference_mode()
def convert(
    buffer: torch.Tensor,
    max_seq_len: int,
    received: W.ReceivedRequest,
    token_map: torch.Tensor,
    device=None,
) -> ConvertedRequest:
    """``received``'s planes from the landed wire image ``buffer`` (uint8, ``wire.wire_bytes(max_seq_len)``), on
    ``device`` (default the buffer's). ``token_map``: the Engram compressed-token map.
    """
    dev = torch.device(device) if device is not None else buffer.device
    rm = W.regions(max_seq_len)
    h = W.parse_header(buffer[rm["hdr"].offset :])
    n = h["n"]
    if n != int(received.seq_len):
        raise RuntimeError(
            f"dsv41: header n={n} != handshake seq_len={received.seq_len}"
        )

    def take(r: W.Region, layer: int, nbytes: int) -> torch.Tensor:
        o = r.offset + layer * r.layer_bytes
        assert nbytes <= r.layer_bytes, (r.name, nbytes)
        t = buffer[o : o + nbytes]
        return t.to(dev, non_blocking=True) if t.device != dev else t

    main, index = {}, {}
    for pl in W.PAGE_PLANES:
        cb, npg = h["cb"][pl], h["pages"][pl]
        page = cb * W.PLANE_REC[pl]
        dst = main if pl.startswith("main") else index
        for i, s in enumerate(W.PLANE_LAYERS[pl]):
            dst[s] = (take(rm[pl], i, npg * page).view(npg, page), cb)
    nwin, nb, wcb = h["nwin"], h["win_blocks"], h["win_cb"]
    if nwin < NUM_WIN and "nwin" not in _WARNED:
        _WARNED.add("nwin")
        logger.warning(
            "dsv41: the prefill sent %d window layers (no DSpark caches); DSpark windows start empty",
            nwin,
        )
    wpage = wcb * V4_REC
    win = [take(rm["swa"], i, nb * wpage).view(nb, wpage) for i in range(nwin)]
    rows = h["ring_rows"]
    rings = {
        s: take(rm["ring"], i, rows * W.RING_ROW_BYTES).view(F32).view(rows, RING_ROW)
        for i, s in enumerate(W.R2)
    }
    tm = torch.as_tensor(token_map).to(I64).to(dev)
    planes = build_planes(
        n,
        main,
        index,
        win,
        h["win_first"],
        rings,
        h["tail"],
        tm,
        device=dev,
        win_cb=wcb,
    )
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return ConvertedRequest(
        rid=received.rid,
        seq_len=n,
        last_prompt_token=received.last_prompt_token,
        first_token_id=received.first_token_id,
        sampling=received.sampling,
        planes=planes,
    )
