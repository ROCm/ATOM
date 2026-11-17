# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports TileRT 0.1.7 (MIT, Copyright (c) TileRT-team) ``tilert/models/ds_v4_1_flash/toktab.py`` (``build_table``,
# ``build_rank_tables``): its buffers, launch sequence and chunking.
"""The DSpark token tables built on the GPU by the kernels TileRT's ``toktab.build_table`` runs -- our FlyDSL ports
of them, from source (``flybuild``) -- so every rounding is the device's own: bit-exact by construction.

A scope (``layers.0`` / ``mtp.0``) is built ``ROWS`` tokens at a time (one chunk), as ``build_table`` drives it:

- ``X`` [rows, 4, 5120] bf16 <- the 4 mHC copies of each token's embedding (``embed_kernel``: a row copy);
- ``mhc_v3_kernel<6, 2>`` (PRE, ``kernels/mhc_pre.py``) over ``X`` with the mix (1, 0, 0, 0): ``Xn`` / its
  1 x 32 E4M3 + ue8m0 and the mix GEMV's partials in ``ws_mhc``;
- ``q_kv_xf_kg_kernel<6, true, 8, 0>`` (``kernels/qkv.py``) with the mHC finalize bundle (``ws_qkv | 1``,
  ``ws_mhc``, ``attn_scale``, ``attn_base``, ``hcA``, rows): ``hcA`` (the hc table), q_a / kv, q-norm, q_b + RoPE at
  position 0 of the identity rope (cos 1, sin 0) -> ``Q`` (the q table);
- ``toktab::kv_extract_kernel`` (``kernels/kv_extract.py``): the kv GEMM's packs in ``ws_qkv`` RMS-normed ->
  the kv table.

The workspaces are this builder's own, zeroed once (the kernels' epochs and hand-off tags live in them, as in
``build_table``). TileRT builds ``mtp.0`` 5 rows a chunk with the <5, ..> instances of the same kernels; every row's
arithmetic is the same at 6, so both scopes run the 6-row kernels here.

``Builder(table, device).build(params, scope)`` -> ``[q, kv, hc]`` on the device; ``build_rank_tables`` -> a rank's 6
tensors in TileRT's order. ``python -m ...tk.toktab_device --rank R [--ref rank<R>.pt]``: build one rank (parameters
packed from the checkpoint), compare byte for byte.
"""

from __future__ import annotations

import argparse
import ctypes
import struct
import time

import torch

from . import alloc, codeobj, flybuild

ROWS = 6
SCOPES = ("layers.0", "mtp.0")
MHC = flybuild.BY_KEY["mhc_pre6"].symbol
QKV = flybuild.BY_KEY["qkv_kg6_idx0"].symbol
KVX = flybuild.BY_KEY["kv_extract"].symbol
RECIPES = tuple(flybuild.BY_KEY[k] for k in ("mhc_pre6", "qkv_kg6_idx0", "kv_extract"))
MHC_GRID, QKV_GRID = 6 * 10 + 160, 481 + ROWS
# the parameters of a scope's first block a table needs (slot names after ``param:<scope>.``)
SLOTS = ("mhc.attn_fn", "mhc.attn_scale", "mhc.attn_base", "mhc.attn_gamma", "q_proj.qa_w", "q_proj.qa_s",
         "q_proj.q_norm", "q_proj.qb_w", "q_proj.qb_s", "window_kv.kv_w", "window_kv.kv_s", "window_kv.kv_norm")  # fmt: skip


def _blob(words: list[int]) -> bytes:
    return struct.pack(f"<{len(words)}Q", *(int(w) & (2**64 - 1) for w in words))


class Builder:
    """One device's token table builder: its kernels (from ``table``: ``RECIPES``' symbols), buffers and workspaces."""

    def __init__(self, table: codeobj.CodeTable, device):
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            self.launcher = codeobj.Launcher(table, [MHC, QKV, KVX])
        self._alloc()

    def _alloc(self) -> None:
        """The buffers and workspaces on ``self.device``, every dtype explicit (a ModelRunner process's defaults are
        bf16 on its GPU: a bf16 rope read as f32 broke the q table there)."""

        def z(*shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=self.device)

        S, D, bf16, u8 = ROWS, alloc.DIM, torch.bfloat16, torch.uint8
        self.X = z(S, 4, D, dtype=bf16)
        self.P = z(S, 4, dtype=torch.float32)
        self.P[:, 0] = 1.0
        self.hcA = z(S, 24, dtype=torch.float32)
        self.xn, self.xq, self.xs = (
            z(S, D, dtype=bf16),
            z(S, D, dtype=u8),
            z(S, D // 32, dtype=u8),
        )
        self.ws_m = z(alloc.WS_BYTES["ws_mhc"], dtype=u8)
        self.ws_q = z(alloc.WS_BYTES["ws_qkv"], dtype=u8)
        # TileRT's rope_id: 16 positions of 32 cos (1) | 32 sin (0), f32
        self.rope = torch.ones(16, 64, dtype=torch.float32, device=self.device)
        self.rope[:, 32:] = 0.0
        self.pos = z(S, dtype=torch.int32)
        self.qr, self.q = z(S, alloc.Q_LORA, dtype=bf16), z(
            S, alloc.HEADS_LOCAL, alloc.HEAD_DIM, dtype=bf16
        )
        self.wk, self.wsc = z(alloc.WIN_RING, alloc.HEAD_DIM, dtype=u8), z(
            alloc.WIN_RING, alloc.HEAD_DIM // 32, dtype=u8
        )
        self.kv = z(S, alloc.HEAD_DIM, dtype=bf16)

    def _launches(self, p: dict) -> list:
        """(symbol, grid, block, kernelParams, buffer) of the three launches, ``p`` the scope's ``SLOTS`` tensors."""
        a = {k: t.data_ptr() for k, t in p.items()}
        S = ROWS
        mhc = [0] * 48
        mhc[1], mhc[3], mhc[4] = self.X.data_ptr(), self.P.data_ptr(), self.P.stride(0)
        mhc[5], mhc[6], mhc[7], mhc[8] = (
            a["mhc.attn_fn"],
            a["mhc.attn_scale"],
            a["mhc.attn_base"],
            a["mhc.attn_gamma"],
        )
        mhc[9] = int(p["mhc.attn_gamma"].dtype == torch.bfloat16)
        mhc[10], mhc[11], mhc[12], mhc[13] = (
            self.hcA.data_ptr(),
            self.xn.data_ptr(),
            self.xq.data_ptr(),
            self.xs.data_ptr(),
        )
        mhc[16] = self.ws_m.data_ptr()
        qkv = [0] * 47
        qkv[0:6] = [
            self.xq.data_ptr(),
            self.xs.data_ptr(),
            a["q_proj.qa_w"],
            a["q_proj.qa_s"],
            a["window_kv.kv_w"],
            a["window_kv.kv_s"],
        ]
        # the mHC finalize bundle (MhcFinalizeArgs): the flag in ws's bit 0, mHC ws, scale, base, hcA, rows
        qkv[6:12] = [
            self.ws_q.data_ptr() | 1,
            self.ws_m.data_ptr(),
            a["mhc.attn_scale"],
            a["mhc.attn_base"],
            self.hcA.data_ptr(),
            S,
        ]
        qkv[31:42] = [
            self.ws_q.data_ptr(),
            a["q_proj.q_norm"],
            a["q_proj.qb_w"],
            a["q_proj.qb_s"],
            a["window_kv.kv_norm"],
            self.rope.data_ptr(),
            self.pos.data_ptr(),
            self.qr.data_ptr(),
            self.q.data_ptr(),
            self.wk.data_ptr(),
            self.wsc.data_ptr(),
        ]
        kvx = [self.ws_q.data_ptr(), a["window_kv.kv_norm"], self.kv.data_ptr()]
        out = []
        for sym, words, grid, block in (
            (MHC, mhc, MHC_GRID, 256),
            (QKV, qkv, QKV_GRID, 512),
            (KVX, kvx, S, 256),
        ):
            blob = _blob(words)
            buf = ctypes.create_string_buffer(
                blob, max(len(blob), self.launcher.arg_bytes(sym))
            )
            out.append(
                (sym, (grid, 1, 1), (block, 1, 1), self.launcher.params(sym, buf), buf)
            )
        return out

    def build(
        self,
        p: dict,
        embed: torch.Tensor,
        tokens: torch.Tensor | None = None,
        out=None,
        log=None,
    ) -> list:
        """One scope's ``[q, kv, hc]`` (device tensors; ``out`` to fill given ones) for ``tokens`` (default all), ``p``:
        the scope's ``SLOTS`` -> device tensor, ``embed`` the device embedding [129280, 5120] bf16.
        """
        dev = self.device
        tokens = (
            torch.arange(alloc.VOCAB, device="cpu") if tokens is None else tokens.cpu()
        )
        n = tokens.numel()
        if out is None:
            out = [torch.empty(n, alloc.HEADS_LOCAL * alloc.HEAD_DIM, dtype=torch.bfloat16, device=dev),
                   torch.empty(n, alloc.HEAD_DIM, dtype=torch.bfloat16, device=dev),
                   torch.empty(n, 24, dtype=torch.float32, device=dev)]  # fmt: skip
        tq, tkv, thc = out
        # the last chunk padded with token 0
        ids = torch.zeros(-(-n // ROWS) * ROWS, dtype=torch.long, device="cpu")
        ids[:n] = tokens
        ids = ids.to(dev)
        launches = self._launches({k: v for k, v in p.items()})
        t0 = time.time()
        with torch.cuda.device(dev):
            stream = torch.cuda.current_stream(dev).cuda_stream
            for c in range(0, n, ROWS):
                t = ids[c : c + ROWS]
                self.X.copy_(
                    embed.index_select(0, t)[:, None, :].expand(ROWS, 4, alloc.DIM)
                )
                for sym, grid, block, prm, _ in launches:
                    self.launcher.launch(sym, grid, block, 0, stream, prm)
                k = min(ROWS, n - c)
                tq[c : c + k].copy_(self.q.view(ROWS, -1)[:k])
                tkv[c : c + k].copy_(self.kv[:k])
                thc[c : c + k].copy_(self.hcA[:k])
                if log and c % (ROWS * 4096) == 0:
                    log(f"toktab (device): {c}/{n} rows ({time.time() - t0:.0f}s)")
            torch.cuda.synchronize(dev)
        return out


def first_block_params(ckpt, scope: str, rank: int) -> dict:
    """``SLOTS`` of ``scope`` for ``rank``, packed from the checkpoint on the CPU (as ``weights.params`` packs them)."""
    from .weights import ckpt as CK
    from .weights import pack as PK
    from .weights import params as PP

    idx = CK.get_index(ckpt)
    pre = f"{scope}."
    d: list[dict] = [{}]
    CK._hc(idx, pre, d)
    CK._attn(idx, pre, [rank], d)
    lg = d[0]
    lg["ffn_norm"] = CK._gamma(idx, pre + "ffn_norm.weight")
    for (
        k,
        name,
    ) in (
        PP._GAMMA_SRC.items()
    ):  # the checkpoint's own bf16 gammas, as params._logical keeps them
        src = pre + name
        if k in lg and src in idx and idx.info(src).dtype == torch.bfloat16:
            lg[k] = idx.read(src)
    groups = {
        "mhc": PK.pack_mhc(*(lg[k] for k in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn",
                                             "hc_ffn_base", "hc_ffn_scale", "attn_norm", "ffn_norm"))),
        "q_proj": PK.pack_q_proj(lg["wq_a"], lg["wq_a_s"], lg["q_norm"], lg["wq_b"], lg["wq_b_s"]),
        "window_kv": PK.pack_window_kv(lg["wkv"], lg["wkv_s"], lg["kv_norm"]),
    }  # fmt: skip
    flat = PP.scope_params(scope, groups)
    return {s: flat[f"{scope}.{s}"] for s in SLOTS}


def build_rank_tables(
    builder: Builder, params: dict, embed: torch.Tensor, tokens=None, log=None
) -> list:
    """A rank's 6 tables in TileRT's order (q, kv, hc of ``layers.0``, then of ``mtp.0``), device tensors; ``params``:
    ``param:<scope>.<slot>`` -> device tensor (``TkRank.params``)."""
    out = []
    for scope in SCOPES:
        p = {s: params[f"param:{scope}.{s}"] for s in SLOTS}
        out += builder.build(p, embed, tokens, log=log)
    return out


def compare(
    got: list[torch.Tensor], ref: list[torch.Tensor], tokens: torch.Tensor
) -> list[dict]:
    """Per table: rows, bytes, differing bytes, differing rows (``ref`` the full tables, ``got`` rows ``tokens``)."""
    out = []
    for g, r in zip(got, ref):
        r = r[tokens]
        gb = g.contiguous().view(torch.uint8).view(g.shape[0], -1)
        rb = r.contiguous().view(torch.uint8).view(r.shape[0], -1)
        diff = gb != rb
        out.append({"rows": g.shape[0], "bytes": gb.numel(), "diff_bytes": int(diff.sum()),
                    "diff_rows": int(diff.any(1).sum()), "dtype": str(g.dtype).rsplit(".", 1)[-1]})  # fmt: skip
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="build a rank's DSpark token tables on the GPU, compare to a reference"
    )
    ap.add_argument("--ckpt", default="/shared_nfs/lingzha/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--device", type=int, help="default: the rank")
    ap.add_argument("--ref", help="rank<r>.pt: TileRT's g.toktab[r] (6 tensors)")
    ap.add_argument("--tokens", default="all", help="all | a:b")
    ap.add_argument("--fly-cache")
    ap.add_argument("--out", help="torch.save the 6 tables here")
    a = ap.parse_args(argv)
    from .weights.ckpt import get_index

    dev = torch.device("cuda", a.rank if a.device is None else a.device)
    tokens = torch.arange(alloc.VOCAB)
    if a.tokens != "all":
        lo, hi = (int(x) for x in a.tokens.split(":"))
        tokens = torch.arange(lo, hi)
    table = flybuild.code_table(a.fly_cache, RECIPES, log=print)
    builder = Builder(table, dev)
    embed = get_index(a.ckpt).read("embed.weight").to(dev)
    ref = torch.load(a.ref, map_location="cpu", mmap=True) if a.ref else None
    got, total = [], 0
    for k, scope in enumerate(SCOPES):
        p = {s: t.to(dev) for s, t in first_block_params(a.ckpt, scope, a.rank).items()}
        t0 = time.time()
        tabs = [t.cpu() for t in builder.build(p, embed, tokens, log=print)]
        print(
            f"rank {a.rank} {scope}: {tokens.numel()} rows in {time.time() - t0:.1f}s",
            flush=True,
        )
        got += tabs
        if ref is not None:
            for name, res in zip(
                ("q", "kv", "hc"), compare(tabs, ref[3 * k : 3 * k + 3], tokens)
            ):
                total += res["diff_bytes"]
                print(f"rank {a.rank} {scope}.{name}: {res}", flush=True)
    if ref is not None:
        print(
            f"rank {a.rank}: RESULT {'exact' if total == 0 else f'{total} differing bytes'}",
            flush=True,
        )
    if a.out:
        torch.save(got, a.out)


if __name__ == "__main__":
    main()
