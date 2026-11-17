# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# The buffer shapes and initial values port TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert/models/ds_v4_1_flash/end2end.py`` (``_alloc_caches`` / ``_alloc_temps`` / ``sym_table``) and
# ``e2e_graph.py`` (``DsV41Graph._temps``).
"""Every buffer one TP8 rank's DSpark decode step (``decode_launches``) reads or writes, besides the packed
parameters: its specs (``specs``, CPU-only) and its allocation on the device (``Buffers``).

- ``temp:`` (57, ``e2e_graph._temps`` in ``TEMP_NAMES`` order) and ``cache:`` (105, ``end2end._alloc_caches``): torch
  tensors, the shapes TileRT's formulas give at ``max_seq_len``; the workspace sizes TileRT's .so computes
  (``dsv41_*_ws_bytes``) are the recorded values at TP8 and the 1M-position context (``WS_BYTES``).
- ``internal:`` (what the .so allocates itself, ``decode_launches.INTERNAL``): torch tensors, and the 4 KiB uncached
  mailbox from ``hi.runtime.uncached`` (``hipExtMallocWithFlags`` 0x3, zeroed: TileRT's ``dsv41_mega_tail_alloc``).
- ``toktab:`` (the DSpark token tables): allocated here, filled by the caller (``Buffers.toktab``).
- ``sym:`` (8 families, ``dsv41_*_sym_bytes(8)``): one peer arena a rank, every family at a 2 MiB-aligned offset,
  exchanged over the TP group's IPC handles (``hi.runtime.peer.PeerGroup``, uncached; or ``cached=True``: ordinary
  device memory, as TileRT's ``torch.zeros``, through the same ``hip_ipc`` exchange); then the 7 pointer tables
  (``temp:tab_*``, int64 [8]: entry v the family's buffer on rank v, as this process maps it; ``sym_table``).

``Buffers.address_map()`` is what ``decode_launches.kernargs`` takes: this rank's ``{name: address}`` and, for each
peer, its ``sym:`` buffers as mapped here (the only peer memory a kernarg names: ``sym:indexer``).
"""

from __future__ import annotations

from dataclasses import dataclass

NPES = 8
MAX_SEQ_LEN = (
    1 << 20
)  # the context the plan was recorded at (its kernargs and WS_BYTES are this context's)
DIM, HC, HEADS_LOCAL, HEAD_DIM, Q_LORA = 5120, 4, 8, 512, 1280
ROWS = 8  # max_forward_seq
INDEX_TOPK, INDEX_HEAD_DIM, HASH_COLS = 512, 128, 24
VOCAB = 129280
VOCAB_LOCAL = VOCAB // NPES
N_LAYERS, N_MTP = 40, 3
N_WIN = N_LAYERS + N_MTP
WIN_RING = 256
KV_SOURCES = {2: 2, 8: 2, 14: 2, 20: 1}  # layer: compression ratio
DSPARK_TAPS = 3
ALIGN_SYM = 2 << 20

# TileRT's .so size functions at TP8 (``ws_ix``: at MAX_SEQ_LEN), as recorded
WS_BYTES = {
    "ws_mhc": 458752,  # dsv41_mhc_ws_bytes(1)
    "ws_qkv": 917504,  # max(dsv41_q_kv_proj_ws_bytes(8), (6))
    "ws_at": 4476928,  # dsv41_attn_projo_ws_bytes(8, 8)
    "ws_moe": 153600,  # dsv41_moe_ws_bytes(8)
    "ws_main_proj": 420864,  # dsv41_main_proj_ws_bytes(8)
    "ws_dspark": 163840,  # dsv41_dspark_head_ws_bytes(8)
    "ws_ix": 61924352,  # dsv41_indexer_ws_bytes(8, 1 << 20)
    "ws_sample": 103936,  # dsv41_sample_ws_bytes()
}
SYM_BYTES = {
    "projo": 4587520,
    "moe": 3997696,
    "engram": 2043904,
    "top1": 1792,
    "main_proj": 327680,
    "dspark": 131072,
    "indexer": 7373312,
    "sample": 4177920,
}
TABLES = (
    "projo",
    "moe",
    "engram",
    "top1",
    "main_proj",
    "dspark",
    "indexer",
    "sample",
)  # temp:tab_<family>
VIRTUAL_FAMILIES = frozenset(("projo", "moe"))
INTERNAL = {  # bytes, uncached
    "internal:tail_workspace": (0x334300, False),
    "internal:next_mhc": (0x23000, False),
    "internal:mhcf": (0x69000, False),
    "internal:tail_4096": (0x1000, True),
}
TOKTAB = {  # per-token rows of the scope's first (attention-side) block: q (8 heads x 512), kv, mHC mix
    f"toktab:{scope}.{t}": (shape, dtype)
    for scope in ("mtp.0", "layers.0")
    for t, shape, dtype in (
        ("q", (VOCAB, HEADS_LOCAL * HEAD_DIM), "bfloat16"),
        ("kv", (VOCAB, HEAD_DIM), "bfloat16"),
        ("hc", (VOCAB, 24), "float32"),
    )
}
_SIZE = {"bfloat16": 2, "float32": 4, "int32": 4, "int64": 8, "uint8": 1, "bool": 1}


@dataclass(frozen=True)
class Spec:
    name: str
    shape: tuple[int, ...]
    dtype: str  # torch dtype name
    init: object = (
        0  # a fill value, or (index, value) pairs: ``t[index] = value`` after the fill
    )
    uncached: bool = False

    @property
    def nbytes(self) -> int:
        n = _SIZE[self.dtype]
        for d in self.shape:
            n *= d
        return n


def _stream_len(cap: int) -> int:
    return cap + 64


def temp_specs(cap: int = MAX_SEQ_LEN) -> list[Spec]:
    """The 57 temps in ``TEMP_NAMES`` order (``e2e_graph._temps`` over ``end2end._alloc_temps(8)``)."""
    if cap != MAX_SEQ_LEN:
        raise ValueError(
            f"the plan and WS_BYTES are {MAX_SEQ_LEN}-position values, not {cap}"
        )
    S, D = ROWS, DIM
    i32, i64, f32, bf16, u8 = "int32", "int64", "float32", "bfloat16", "uint8"
    t = [
        ("seq", (32,), i32),
        ("X", (S, HC, D), bf16),
        ("X2", (S, HC, D), bf16),
        ("hcA", (S, 24), f32),
        ("hcF", (S, 24), f32),
        ("P", (S, HC), f32, (((slice(None), 0), 1.0),)),
        ("Xn", (S, D), bf16),
        ("Xq", (S, D), u8),
        ("Xs", (S, D // 32), u8),
        ("Qr", (S, Q_LORA), bf16),
        ("Q", (S, HEADS_LOCAL, HEAD_DIM), bf16),
        ("Y", (S, D), bf16),
        ("latent", (S, HEAD_DIM), bf16),
        ("latent_idx", (S,), i32, -1),
        ("topk", (S, INDEX_TOPK), i32, -1),
        ("hash", (S, 2, HASH_COLS), i64),
        ("act", (S, 2, 6144), u8),
        ("act_s", (S, 2, 192), u8),
        ("hist", (16,), i32),
        ("cap", (DSPARK_TAPS, S, D), bf16),
        ("bkv", (S, HEAD_DIM), u8),
        ("bks", (S, HEAD_DIM // 32), u8),
        ("logits", (S, VOCAB_LOCAL), f32),
        ("tok_out", (S,), i32),
        *(
            (k, (WS_BYTES[k],), u8)
            for k in (
                "ws_mhc",
                "ws_qkv",
                "ws_at",
                "ws_moe",
                "ws_main_proj",
                "ws_dspark",
            )
        ),
        ("ws_top1", (16,), i64),
        ("ws_ix", (WS_BYTES["ws_ix"],), u8),
        ("cand", (S, 2050), i32),
        *((f"tab_{f}", (NPES,), i64) for f in TABLES[:7]),
        ("aux0", (N_WIN, S, HC, D), bf16),
        ("aux1", (S, S, 512), i32, -1),
        ("plain_acc", (1, _stream_len(cap) + cap // 8), i32),
        ("ctl", (16,), i32),
        ("blk_tok", (S,), i32),
        ("blk_pos", (S,), i32, -1),
        ("vtok", (S,), i32),
        ("vpos", (S,), i32, -1),
        ("dsp_pos", (S,), i32, -1),
        ("desc", (4,), i32),
        ("ar_acc", (1, _stream_len(cap)), i32),
        ("ar_num", (1, _stream_len(cap)), i32),
        # pack_params(SamplingParams()): seed 0, temperature 1.0 | top_p 1.0 << 32, top_k 0 | min_p 0.0 << 32, 0
        ("samp_params", (4,), i64, (((1,), 0x3F8000003F800000),)),
        ("gmax", (S,), f32),
        ("draft_z", (5, VOCAB_LOCAL), f32),
        ("ws_sample", (WS_BYTES["ws_sample"],), u8),
        ("tab_sample", (NPES,), i64),
    ]
    return [Spec(f"temp:{x[0]}", x[1], x[2], x[3] if len(x) > 3 else 0) for x in t]


def cache_specs(cap: int = MAX_SEQ_LEN) -> list[Spec]:
    """The 105 caches in ``end2end._alloc_caches`` order: the 43 window rings, then main KV, index K and the
    ratio-2 sources' compressor state."""
    out = []
    for i in range(N_WIN):
        out += [
            Spec(f"cache:win.{i}.kv", (WIN_RING, HEAD_DIM), "uint8"),
            Spec(f"cache:win.{i}.s", (WIN_RING, HEAD_DIM // 32), "uint8"),
        ]
    for s, r in KV_SOURCES.items():
        n = cap // r
        out += [
            Spec(f"cache:main_kv.{s}.0", (n, HEAD_DIM // 2), "uint8"),
            Spec(f"cache:main_kv.{s}.1", (n, HEAD_DIM // 16), "uint8"),
        ]
    for s, r in KV_SOURCES.items():
        n = cap // r
        out += [
            Spec(f"cache:index_k.{s}.0", (n, INDEX_HEAD_DIM // 2), "uint8"),
            Spec(f"cache:index_k.{s}.1", (n, INDEX_HEAD_DIM // 32), "uint8", 127),
        ]
    for s, r in KV_SOURCES.items():
        if r == 2:
            out.append(
                Spec(
                    f"cache:state.{s}",
                    (16, 2, HEAD_DIM),
                    "float32",
                    (((slice(None), 1), float("-inf")),),
                )
            )
    return out


def internal_specs() -> list[Spec]:
    return [Spec(n, (b,), "uint8", 0, u) for n, (b, u) in INTERNAL.items()]


def toktab_specs() -> list[Spec]:
    return [Spec(n, shape, dt, None) for n, (shape, dt) in TOKTAB.items()]


def sym_layout() -> tuple[dict[str, int], int]:
    """(``{family: offset}`` in a rank's peer arena, the arena's bytes): each family at a 2 MiB boundary."""
    off, out = 0, {}
    for f in TABLES:
        out[f] = off
        off += -(-SYM_BYTES[f] // ALIGN_SYM) * ALIGN_SYM
    return out, off


def sym_table(fam: str, ptrs: list[int]) -> list[int]:
    """TileRT's ``end2end.sym_table`` (the int64 values): ``projo`` / ``moe`` are virtual families (entry v is the PE
    ``v * n // 8``, n the PEs), the others padded with 0 past the n PEs."""
    n = len(ptrs)
    if fam in VIRTUAL_FAMILIES:
        return [ptrs[v * n // NPES] for v in range(NPES)]
    return list(ptrs) + [0] * (NPES - n)


def specs(cap: int = MAX_SEQ_LEN) -> list[Spec]:
    """Every non-parameter buffer but the ``sym:`` arena."""
    return temp_specs(cap) + cache_specs(cap) + internal_specs() + toktab_specs()


# ---------------------------------------------------------------------------------------------------------------
# on the device


def _tensor(spec: Spec, device):
    import torch

    if spec.uncached:
        from atom.models.deepseek_v41.mono.hi.runtime import uncached

        t = uncached.zeros(spec.nbytes, device)  # (zeroed, pooled, never freed)
        return t.view(getattr(torch, spec.dtype)).view(spec.shape)
    dtype = getattr(torch, spec.dtype)
    if spec.init is None:
        return torch.empty(spec.shape, dtype=dtype, device=device)
    fill = spec.init if isinstance(spec.init, (int, float)) else 0
    t = torch.full(spec.shape, fill, dtype=dtype, device=device)
    if not isinstance(spec.init, (int, float)):
        for index, value in spec.init:
            t[index] = value
    return t


class Buffers:
    """One rank's buffers on its device (module docstring). Collective over ``group`` (the TP group's
    ``torch.distributed`` group, 8 ranks): the peer arena's exchange."""

    def __init__(
        self,
        rank: int,
        group=None,
        device=None,
        cap: int = MAX_SEQ_LEN,
        cached: bool = False,
    ):
        import torch

        self.rank = rank
        self.device = torch.device(
            device if device is not None else f"cuda:{torch.cuda.current_device()}"
        )
        self.tensors = {s.name: _tensor(s, self.device) for s in specs(cap)}
        self.sym_offsets, nbytes = sym_layout()
        with torch.cuda.device(self.device):
            self.peer_bases, self._peers = _exchange(nbytes, group, cached)
        for f in TABLES:
            ptrs = [b + self.sym_offsets[f] for b in self.peer_bases]
            self.tensors[f"temp:tab_{f}"].copy_(
                torch.tensor(sym_table(f, ptrs), dtype=torch.int64)
            )
        torch.cuda.synchronize(self.device)
        import torch.distributed as dist

        # every rank initialized before any rank's first launch polls it
        dist.barrier(group=group)

    def toktab(self, scope: str, q, kv, hc) -> None:
        """Fill ``scope``'s token tables (``mtp.0`` / ``layers.0``; tensors of ``TOKTAB``'s shapes)."""
        for t, src in (("q", q), ("kv", kv), ("hc", hc)):
            self.tensors[f"toktab:{scope}.{t}"].copy_(src)

    def address_map(self, params: dict | None = None) -> list[dict]:
        """``addr[r]``: rank r's buffers as this process addresses them (all of them for this rank, the ``sym:``
        families for a peer), ``params`` (name -> tensor) included."""
        addr: list[dict] = [{} for _ in range(NPES)]
        for r, base in enumerate(self.peer_bases):
            for f, off in self.sym_offsets.items():
                addr[r][f"sym:{f}"] = base + off
        own = addr[self.rank]
        own.update({n: t.data_ptr() for n, t in self.tensors.items()})
        own.update({n: t.data_ptr() for n, t in (params or {}).items()})
        return addr


def _exchange(nbytes: int, group, cached: bool) -> tuple[list[int], object]:
    """(every rank's arena base as this process maps it, the owner object): ``PeerGroup`` (uncached), or a
    hipExtMallocWithFlags(0) arena exchanged the same way (``PeerGroup.__init__``'s handshake).
    """
    if not cached:
        from atom.models.deepseek_v41.mono.hi.runtime.peer import PeerGroup

        g = PeerGroup(nbytes, group)
        return [int(x) for x in g.addresses[: g.tp].tolist()], g
    import torch.distributed as dist

    from atom.mono.runtime import hip_ipc

    rank, tp = dist.get_rank(group), dist.get_world_size(group)
    local = hip_ipc._DEFAULT_RUNTIME.allocate_uncached(
        nbytes, 0
    )  # (flags 0: default device memory, zeroed)
    handles = [None] * tp
    dist.all_gather_object(handles, hip_ipc.get_ipc_handle(local), group=group)
    bases, error = [], None
    for peer, h in enumerate(handles):
        if peer == rank:
            bases.append(local)
            continue
        try:
            bases.append(hip_ipc.open_ipc_handle(h))
        except RuntimeError as e:  # every rank must still reach the agreement below
            error = f"rank {rank} mapping rank {peer}: {e}"
            break
    errors = [None] * tp
    dist.all_gather_object(errors, error, group=group)
    if any(errors):
        raise RuntimeError(
            "sym arena exchange failed: " + "; ".join(e for e in errors if e)
        )
    return bases, local
