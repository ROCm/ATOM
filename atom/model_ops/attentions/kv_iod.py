# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""IOD-local A8K4V8 KV cache on gfx1250 (ATOM_KV_IOD): the pool side.

aiter.ops.kv_iod re-lays one attention layer's K slice and V slice (the pool's layer-major `k` / `v` fields) so
that every 16 KiB granule of block `c` is homed on IOD `c & 1`, the block id's parity. The scheduler's block ids
are these codes: one id names one placed block in every layer. Three things follow here:

* capacity: a layer's codes are what its whole 2 MiB pages hold (28 blocks a domain in a K page, 16 in a V page),
  a little under the pool's entries; `capacity` is a bound that holds for any alignment of the slices.
* the kernels' x bytes: each layer has a [codes] uint8 table of its pages' home phases. Every attention call reads
  a block table whose entries carry the layer's x (code | x << 29), resolved for all layers in one launch a forward.
* every other reader or writer of pool bytes by block id (released checkpoint units' zeroing, GDN state
  checkpoints stored in PAGE units) goes through the placed offsets too.

Correctness never depends on a block's parity: a kernel reads a block at its own placed address. Parity only
decides which IOD the bytes are local to, so the block manager's placement (chunks of ATOM_KV_IOD_CHUNK blocks
alternating parity) is a preference with any block as the fallback.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import triton
import triton.language as tl

from atom.utils import envs

logger = logging.getLogger("atom")

PAGE = 2 << 20
GRAN = 16384
K_ROW = 2 * (
    16384 + 2048
)  # one block's NVFP4 K of both heads (data then scales, per head)
V_ROW = 2 * 32768  # one block's fp8 V of both heads


def enabled() -> bool:
    return envs.ATOM_KV_IOD


def chunk() -> int:
    return envs.ATOM_KV_IOD_CHUNK


def capacity(entries: int) -> int:
    """Codes every layer holds at `entries` pool entries, whatever the slices' alignment (each slice may lose one
    page to it)."""
    from aiter.ops import kv_iod as A

    k_pages = (entries * K_ROW) // PAGE - 1
    v_pages = (entries * V_ROW) // PAGE - 1
    return 2 * max(0, min(k_pages * A.K_PER_PAGE, v_pages * A.V_PER_PAGE))


@triton.jit
def _resolve_kernel(
    bt_ptr, x_ptr, out_ptr, rows, cols, bt_stride, codes, n_layers, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n = rows * cols
    m = offs < n
    r = offs // cols
    c = offs % cols
    code = tl.load(bt_ptr + r * bt_stride + c, mask=m, other=0)
    ok = m & (code >= 0) & (code < codes)
    for layer in range(n_layers):
        x = tl.load(x_ptr + layer * codes + code, mask=ok, other=0).to(tl.int32)
        tl.store(out_ptr + layer * n + offs, code | (x << 29), mask=m)


def resolve(block_table: torch.Tensor, x_all: torch.Tensor) -> torch.Tensor:
    """[layers, rows, cols] int32: code | x_all[layer, code] << 29 (entries outside the codes kept as they are)."""
    assert (
        block_table.dim() == 2
        and block_table.dtype == torch.int32
        and block_table.stride(1) == 1
    )
    layers, codes = x_all.shape
    rows, cols = block_table.shape
    out = torch.empty(
        (layers, rows, cols), dtype=torch.int32, device=block_table.device
    )
    n = rows * cols
    if n:
        BLOCK = 1024
        _resolve_kernel[(triton.cdiv(n, BLOCK),)](
            block_table,
            x_all,
            out,
            rows,
            cols,
            block_table.stride(0),
            codes,
            layers,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out


class KvIodPools:
    """The placed layers of one builder's MHA pools, one per (pool, row), and their x tables."""

    def __init__(self, pools, codes: int, device):
        from aiter.ops import kv_iod as A

        self.codes = codes
        self.layers: dict[tuple[int, int], A.KvIodLayer] = {}
        self.index: dict[tuple[int, int], int] = {}
        self.pools = list(pools)
        xs = []
        for p, pool in enumerate(self.pools):
            assert (
                pool.k_nvfp4
            ), "ATOM_KV_IOD places an A8K4V8 cache (--kv_cache_dtype fp8_k4v8)"
            assert (
                pool.num_kv_heads == 2
                and pool.head_dim == 256
                and pool.block_size == 128
            ), "ATOM_KV_IOD's layout is for 2 KV heads of 256 in 128-token blocks"
            kf, vf = pool._views["k"], pool._views["v"]
            assert kf.shape[-1] == K_ROW and vf.shape[-1] == V_ROW, (kf.shape, vf.shape)
            for row in range(kf.shape[0]):
                layer = A.KvIodLayer(kf[row], vf[row])
                if layer.codes < codes:
                    raise RuntimeError(
                        f"IOD layout of pool {p} row {row} holds {layer.codes} codes < {codes}"
                    )
                self.index[(p, row)] = len(xs)
                self.layers[(p, row)] = layer
                xs.append(layer.x_table(codes, device))
        # A layer whose probe failed is still exact, its homes unknown (aiter.ops.kv_iod.KvIodLayer): then every
        # layer runs the stock workgroup order (mode 1), which needs no label.
        placed = [layer for layer in self.layers.values() if layer.placed]
        label = {layer.label_xor for layer in placed}
        self.mode = envs.ATOM_KV_IOD_MODE
        if len(placed) < len(self.layers) or len(label) != 1:
            logger.warning(
                "ATOM_KV_IOD: %d of %d layers placed (labels %s): the decode keeps the stock workgroup order",
                len(placed),
                len(self.layers),
                sorted(label),
            )
            self.mode = 1
        self.label_xor = label.pop() if len(label) == 1 else 0
        self.x_all = (
            torch.stack(xs)
            if xs
            else torch.empty((0, codes), dtype=torch.uint8, device=device)
        )
        self.group_rows: dict[str, list[int]] = {}
        self.group_x: dict[str, torch.Tensor] = {}
        logger.info(
            "ATOM_KV_IOD: %d placed attention layers, %d block codes, label_xor %d, chunk %d blocks, mode %d",
            len(xs),
            codes,
            self.label_xor,
            chunk(),
            self.mode,
        )

    def handle(self, pool, row: int, group: str = "target") -> KvIodLayerHandle:
        """The placed layer of `pool` row `row`, bound for a module of model `group` (target / draft): a pass
        resolves the block tables of its group's layers only. The group's x stack is built here, eagerly.
        """
        p = next(i for i, q in enumerate(self.pools) if q is pool)
        index = self.index[(p, row)]
        rows = self.group_rows.setdefault(group, [])
        if index not in rows:
            rows.append(index)
        self.group_x[group] = self.x_all[rows].contiguous()
        return KvIodLayerHandle(
            self, index, self.layers[(p, row)], group, rows.index(index)
        )

    # ---- per forward -------------------------------------------------------------------------------------------
    def resolved(
        self, fwd_ctx, block_table: torch.Tensor, index: int, group: str = "target"
    ) -> torch.Tensor:
        """The resolved block tables of `group`'s layers for this pass's `block_table`, computed once a pass
        (cached on the forward context). A pass starts again when a layer index is not above the last one seen: a
        CUDA graph capture after an eager warm-up under the same context, or the next draft pass, must record
        its own resolve rather than reuse a tensor the graph would never refresh."""
        cache = getattr(fwd_ctx, "_kv_iod_resolved", None)
        last = getattr(fwd_ctx, "_kv_iod_last", None)
        capturing = torch.cuda.is_current_stream_capturing()
        if (
            cache is None
            or (last is not None and index <= last)
            or getattr(fwd_ctx, "_kv_iod_capturing", None) != capturing
        ):
            cache = {}
            fwd_ctx._kv_iod_resolved = cache
        fwd_ctx._kv_iod_last = index
        fwd_ctx._kv_iod_capturing = capturing
        key = (
            id(self),
            group,
            block_table.data_ptr(),
            tuple(block_table.shape),
            block_table.stride(0),
        )
        r = cache.get(key)
        if r is None:
            r = cache[key] = resolve(block_table, self.group_x[group])
        return r

    # ---- pool maintenance by block id ---------------------------------------------------------------------------
    def zero_blocks(self, blocks) -> None:
        """Zero every byte of `blocks` in every placed layer (released checkpoint units)."""
        from aiter.ops import kv_iod as A

        cn = np.asarray(list(blocks), dtype=np.int64)
        if not cn.size:
            return
        for key, L in self.layers.items():
            xn = self._x_host(key)[cn]
            dev = L.k.device
            g_k, s_k, g_v = [], [], []
            for h in range(A.HEADS):
                kd, ks, v0, v1 = A.placed_offsets(cn, xn, h)
                g_k.append(kd // GRAN)
                s_k.append(ks // 2048)
                g_v += [v0 // GRAN, v1 // GRAN]
            kg = torch.from_numpy(np.concatenate(g_k)).to(dev)
            ks = torch.from_numpy(np.concatenate(s_k)).to(dev)
            vg = torch.from_numpy(np.concatenate(g_v)).to(dev)
            L.k.view(-1, GRAN).index_fill_(0, kg, 0)
            L.k.view(-1, 2048).index_fill_(0, ks, 0)
            L.v.view(-1, GRAN).index_fill_(0, vg, 0)

    def _x_host(self, key):
        cache = getattr(self, "_x_np", None)
        if cache is None:
            cache = self._x_np = {
                k: self.x_all[i].cpu().numpy() for k, i in self.index.items()
            }
        return cache[key]

    def unit_segments(self, region_keys, unit_ids):
        """Placed (address, size) segments of PAGE units, in the order `mha_page_unit_regions` walks a unit (the
        pool's regions: every layer's K, then every layer's V, of each pool), a unit's K region as (data, scales)
        of head 0 then head 1, its V region as the four 16 KiB halves. Returns (bases [images, segments] int64,
        sizes [segments of one unit] int64)."""
        from aiter.ops import kv_iod as A

        ids = np.asarray(unit_ids, dtype=np.int64)
        flat = ids.reshape(-1)
        cols, sizes = [], []
        for field, key in region_keys:
            L = self.layers[key]
            xn = self._x_host(key)[flat]
            for h in range(A.HEADS):
                kd, ks, v0, v1 = A.placed_offsets(flat, xn, h)
                if field == "k":
                    cols += [L.k.data_ptr() + kd, L.k.data_ptr() + ks]
                    sizes += [GRAN, 2048]
                else:
                    cols += [L.v.data_ptr() + v0, L.v.data_ptr() + v1]
                    sizes += [GRAN, GRAN]
        per_unit = np.stack(cols, axis=-1)  # [units, segments of a unit]
        bases = (
            per_unit.reshape(ids.shape[0], -1)
            if ids.ndim == 2
            else per_unit.reshape(1, -1)
        )
        return bases.astype(np.int64), np.asarray(sizes, dtype=np.int64)

    def region_keys(self):
        """(field, (pool, row)) in `mha_page_unit_regions` order."""
        keys = []
        for p, pool in enumerate(self.pools):
            for name in pool._views:
                if name not in ("k", "v"):
                    raise RuntimeError(f"ATOM_KV_IOD cannot place pool field {name!r}")
                for row in range(pool._views[name].shape[0]):
                    keys.append((name, (p, row)))
        return keys


class KvIodLayerHandle:
    """One attention module's view of its placed layer."""

    def __init__(
        self,
        pools: KvIodPools,
        index: int,
        layer,
        group: str = "target",
        pos: int | None = None,
    ):
        self.pools = pools
        self.index = index
        self.layer = layer
        self.group = group
        self.pos = index if pos is None else pos  # row in the group's resolved tables
        self.x = pools.x_all[index]

    def write_loc(self, v_dtype):
        """`kv_loc` of aiter's fused_qk_norm_mrope_cache: (k_base, v_base, x_table)."""
        return (self.layer.k, self.layer.v.view(v_dtype), self.x)

    def attn(self, fwd_ctx, block_table: torch.Tensor, v_dtype):
        """(resolved block table, kv_loc) for aiter unified_attention."""
        bt = self.pools.resolved(fwd_ctx, block_table, self.index, self.group)[self.pos]
        kv_loc = (
            self.x,
            self.layer.k,
            self.layer.v.view(v_dtype),
            self.pools.label_xor,
            chunk(),
            self.pools.mode,
            True,
        )
        return bt, kv_loc


def preferred_parity(table, pos: int) -> int | None:
    """The parity block `pos` of a sequence whose blocks are `table` should have: chunk k of a sequence is on
    domain (k + salt) & 1, salt the parity of its first block. None for a sequence's first block (the caller
    balances); a first block replaced in place keeps its parity."""
    if not table:
        return None
    if pos == 0:
        return table[0] & 1
    return ((pos // chunk()) + (table[0] & 1)) & 1
