# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports TileRT 0.1.7 (MIT, Copyright (c) TileRT-team) ``tilert/pd_vllm/wire.py``, the request record of
# ``tilert/pd_vllm/receive_server.py`` (``ReceivedRequest``) and the wire layout of
# ``tilert/pd_vllm/profiles/dsv41_rocm.py`` (regions, header, hello).
"""The TileRT PD wire a DeepSeek-V4.1-Flash prefill ("P") sends a decode ("D") host, as the D side reads it.

Control plane (TCP, each message a 4-byte big-endian length then JSON, ``send_msg`` / ``recv_msg``):

1. D, on accept: ``hello_msg`` -- ``magic`` ``"tilert-pd"``, ``protocol_version`` 2, ``layout_version`` 2041,
   ``transport``, ``max_seq_len``, ``busy``, the transport's own fields and ``hello_layout``'s (``senders`` 1,
   ``dsv41_base`` the receive buffer's address, ``dsv41_wire`` 1);
2. P: ``{rid, rank, seq_len, last_prompt_token, sampling, admission_window_s[, prompt_token_ids][, kv_scales]}``;
3. D: ``accept_msg`` (``{accepted: true, rid, rank, generation}``) or ``reject_msg``;
4. P RDMA-writes its staging image into D's buffer (rank 0 only: ``SENDER_RANKS``), then sends ``done_msg``.

Data plane: one buffer of ``wire_bytes(max_seq_len)`` bytes, its ``regions`` 4 KiB aligned -- the 256-byte header
(``parse_header``), then per layer: the V4 main pages of the ratio-2 sources (2, 8, 14) and their MXFP4 index
pages, the same of the ratio-1 source (20), 5 window pages of 32 V4 records for each of 43 window layers, and 16
compressor ring rows (2 x 512 float32) for each ratio-2 source. A region's layer slot is sized for
``max_seq_len``; the header says how many pages of it a request used.
"""

from __future__ import annotations

import json
import os
import socket
import struct
from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------------------------------------------
# control plane (wire.py)

MAGIC = "tilert-pd"
PROTOCOL_VERSION = 2
NUM_RANKS = 8
EXPECTED_RANKS = tuple(range(NUM_RANKS))


def local_ip(probe_addr: str | None = None) -> str:
    probe = probe_addr or os.environ.get("TILERT_PD_PROBE_ADDR", "8.8.8.8")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((probe, 1))
        return s.getsockname()[0]
    finally:
        s.close()


def wants_prompt_token_ids(sampling: dict | None) -> bool:
    """Whether a request's sampling needs the prompt's ids (a repetition penalty other than 1)."""
    if not sampling:
        return False
    rep = sampling.get("repetition_penalty")
    if rep is None:
        return False
    try:
        return float(rep) != 1.0
    except (TypeError, ValueError):
        return False


def derive_rid(request_id: str) -> str:
    """The request id both sides key a transfer by: an OpenAI id without its ``chatcmpl-`` / ``cmpl-`` prefix, a
    short hex suffix and a short numeric suffix."""
    rid = request_id
    for prefix in ("chatcmpl-", "cmpl-"):
        if rid.startswith(prefix):
            rid = rid[len(prefix) :]
            break
    parts = rid.rsplit("-", 1)
    if (
        len(parts) == 2
        and len(parts[1]) <= 8
        and all(c in "0123456789abcdef" for c in parts[1])
    ):
        rid = parts[0]
    parts = rid.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 3:
        rid = parts[0]
    return rid


def send_msg(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode()
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_msg(sock: socket.socket) -> dict:
    hdr = _recv_exact(sock, 4)
    (n,) = struct.unpack("!I", hdr)
    if n > 16 << 20:
        raise ValueError(f"control message too large: {n}")
    return json.loads(_recv_exact(sock, n).decode())


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed mid-message")
        buf += chunk
    return buf


def hello_msg(
    transport: str,
    transport_meta: dict,
    max_seq_len: int,
    layout_version: int,
    layout: dict,
    busy: bool,
) -> dict:
    return {
        "magic": MAGIC,
        "protocol_version": PROTOCOL_VERSION,
        "layout_version": layout_version,
        "transport": transport,
        "max_seq_len": max_seq_len,
        "busy": busy,
        **transport_meta,
        **layout,
    }


def accept_msg(rid: str, rank: int, generation: int) -> dict:
    return {"accepted": True, "rid": rid, "rank": rank, "generation": generation}


def reject_msg(reason: str, **extra) -> dict:
    return {"accepted": False, "error": reason, **extra}


def done_msg(rid: str, rank: int, generation: int) -> dict:
    return {"done": True, "rid": rid, "rank": rank, "generation": generation}


# a receive slot's states (receive_server.py)
FREE = "free"
RESERVED = "reserved"
TRANSFERRING = "transferring"
COMPLETE = "complete"
CANCELLING = "cancelling"


@dataclass
class ReceivedRequest:
    """An admitted request, from P's message 2 (``seq_len``, ``last_prompt_token``, ``sampling``,
    ``prompt_token_ids``; ``first_token_id`` reaches D apart, with the decode call)."""

    rid: str
    seq_len: int
    last_prompt_token: int
    first_token_id: int | None
    sampling: dict | None
    prompt_token_ids: list = field(default_factory=list)
    done_ranks: set = field(default_factory=set)
    t_first_conn: float = 0.0
    t_complete: float = 0.0
    generation: int = 0
    state: str = RESERVED
    active_writers: int = 0
    claimed: bool = (
        False  # a decode call is reading its KV (``receive.ReceiveServer.claim``)
    )

    @property
    def has_live_writer(self) -> bool:
        return self.active_writers > 0


# ---------------------------------------------------------------------------------------------------------------
# the DeepSeek-V4.1 data plane (profiles/dsv41_rocm.py)

NAME = "dsv41"
LAYOUT_VERSION = 2041
WIRE_MAGIC = 825512772
WIRE_VERSION = 1
SENDER_RANKS = frozenset({0})
SUPPORTED_NUM_MTP = (5,)
BLOCK = 128
R2, R1 = (2, 8, 14), (20,)
V4_REC = 584
IDX_REC = 68
WINDOW = 128
SWA_CB = 32
WIN_BLOCKS = WINDOW // SWA_CB + 1
RING_ROW = 1024
RING_ROW_BYTES = RING_ROW * 4
RING_ROWS_MAX = 16
HIST = 16
NUM_WIN = 43
HDR_WORDS = 64
HDR_BYTES = HDR_WORDS * 4
_ALIGN = 4096
# header int32 words: magic, version, n, ring rows, window layers, first window page, window pages; cb of each
# PAGE_PLANES then of a window page (H_CB + 4); pages of each PAGE_PLANES; the tail ids' count and the ids
H_MAGIC, H_VER, H_N, H_RING_ROWS, H_NWIN, H_WIN_FIRST, H_WIN_BLOCKS = range(7)
H_CB = 7
H_PAGES = 12
H_NTAIL = 16
H_TAIL = 32
PAGE_PLANES = ("main_r2", "index_r2", "main_r1", "index_r1")
PLANE_RATIO = {"main_r2": 2, "index_r2": 2, "main_r1": 1, "index_r1": 1}
PLANE_REC = {
    "main_r2": V4_REC,
    "index_r2": IDX_REC,
    "main_r1": V4_REC,
    "index_r1": IDX_REC,
}
PLANE_LAYERS = {"main_r2": R2, "index_r2": R2, "main_r1": R1, "index_r1": R1}


@dataclass(frozen=True)
class Region:
    name: str
    layers: int
    layer_bytes: int
    offset: int

    @property
    def nbytes(self) -> int:
        return self.layers * self.layer_bytes


def _align(x: int) -> int:
    return -(-x // _ALIGN) * _ALIGN


def regions(max_seq_len: int) -> dict[str, Region]:
    """The wire image's regions for ``max_seq_len`` positions, in order: ``hdr``, ``main_r2``, ``index_r2``,
    ``main_r1``, ``index_r1``, ``swa``, ``ring``."""
    p = -(-int(max_seq_len) // BLOCK)
    spec = [
        ("hdr", 1, HDR_BYTES),
        ("main_r2", len(R2), p * (BLOCK // 2) * V4_REC),
        ("index_r2", len(R2), p * (BLOCK // 2) * IDX_REC),
        ("main_r1", len(R1), p * BLOCK * V4_REC),
        ("index_r1", len(R1), p * BLOCK * IDX_REC),
        ("swa", NUM_WIN, WIN_BLOCKS * SWA_CB * V4_REC),
        ("ring", len(R2), RING_ROWS_MAX * RING_ROW_BYTES),
    ]
    out, off = {}, 0
    for name, layers, lb in spec:
        out[name] = Region(name, layers, lb, off)
        off = _align(off + layers * lb)
    return out


def wire_bytes(max_seq_len: int) -> int:
    """The receive buffer's size for ``max_seq_len`` positions."""
    last = list(regions(max_seq_len).values())[-1]
    return _align(last.offset + last.nbytes)


def hello_layout(base_ptr: int, max_seq_len: int) -> dict:
    """D's layout fields of the hello (``base_ptr``: the receive buffer's address)."""
    del max_seq_len
    return {
        "senders": len(SENDER_RANKS),
        "dsv41_base": int(base_ptr),
        "dsv41_wire": WIRE_VERSION,
    }


def parse_header(buf: torch.Tensor) -> dict:
    """The header at the start of ``buf`` (uint8): ``n``, ``ring_rows``, ``nwin``, ``win_first``, ``win_blocks``,
    ``win_cb``, ``cb`` / ``pages`` per ``PAGE_PLANES`` and ``tail`` (the last ``min(n, 16)`` prompt ids).
    """
    h = buf[:HDR_BYTES].cpu().clone().view(torch.int32).tolist()
    if h[H_MAGIC] != WIRE_MAGIC or h[H_VER] != WIRE_VERSION:
        raise RuntimeError(
            f"dsv41: bad wire header (magic {h[H_MAGIC]:#x}, version {h[H_VER]})"
        )
    ntail = h[H_NTAIL]
    return {
        "n": h[H_N],
        "ring_rows": h[H_RING_ROWS],
        "nwin": h[H_NWIN],
        "win_first": h[H_WIN_FIRST],
        "win_blocks": h[H_WIN_BLOCKS],
        "win_cb": h[H_CB + 4],
        "cb": {p: h[H_CB + i] for i, p in enumerate(PAGE_PLANES)},
        "pages": {p: h[H_PAGES + i] for i, p in enumerate(PAGE_PLANES)},
        "tail": h[H_TAIL : H_TAIL + ntail],
    }
