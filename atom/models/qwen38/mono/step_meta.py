# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A Mono step's host decisions and contracts, without a launch (CPU-testable).

- ``verify_step``: whether a target forward is a served MTP verify step;
- ``StepMeta`` / ``meta_buffers`` / ``target_puts``: the kernels' int32 metadata args and how a step fills them
  (one fused launch, ``prep.fused_puts``);
- ``GDNPools`` / ``QSAPools``: ATOM's pools a layer reads and writes in place, checked against the geometry;
- ``draft_rows_error``: the addressing contract of a served draft step;
- ``draft_attention_fused``: which path the fallback draft's QSA attention takes.

A step has ``rows = batch * spec`` rows; row ``r`` is request ``r // spec`` at offset ``r % spec``. Classic verify
(the GDN state snapshotted per verify row): with slots ``idx = spec_state_indices [batch, SPEC]`` and ``a`` accepted
tokens, the conv window is read and rewritten in place in slot ``idx[:, 0]`` from row ``a - 1``, the state is read from
``spec_state_indices_in[b, a - 1]`` and the state after row ``t`` written to ``idx[b, t]``. A padded request
(``valid`` false) gets slots -1 and a context of 0 over a block-table row of -1: the kernels touch no pool for it.
"""

from dataclasses import dataclass

import torch

from atom.models.qwen38.mono.config import SPEC

FP8_TABLE = (torch.float8_e4m3fn, torch.uint8)


def verify_step(fwd, rows: int, batches, replayssm: bool = False) -> int | None:
    """The requests of a served verify step (``SPEC`` rows each, nothing else in the step, on the verify the kernels
    were built for: ReplaySSM or classic) under forward context ``fwd``, else None (Native).
    """
    if rows % SPEC or rows // SPEC not in batches:
        return None
    batch = rows // SPEC
    if fwd.context is None or fwd.context.is_prefill or fwd.ubatch_slices is not None:
        return None
    md = getattr(fwd, "attn_metadata", None)
    if md is None or getattr(md, "image_mask", None) is not None:
        return None
    gdn = getattr(md, "gdn_metadata", None)
    if gdn is None or bool(getattr(gdn, "replayssm", False)) != replayssm:
        return None
    if (
        gdn.num_prefills != 0
        or gdn.num_decodes != 0
        or not 0 < gdn.num_spec_decodes <= batch
    ):
        return None
    idx = gdn.spec_state_indices_tensor
    if idx is None or idx.dim() != 2 or idx.shape[1] != SPEC or idx.shape[0] < batch:
        return None
    return batch


@dataclass
class StepMeta:
    """A served target step's metadata (device tensors)."""

    batch: int
    valid: torch.Tensor  # [batch] bool
    block_tables: torch.Tensor  # [batch, width]
    context_lens: torch.Tensor  # [batch], including the step's rows
    conv_slot: torch.Tensor  # [batch]: the conv window, read and rewritten in place
    conv_off: torch.Tensor  # [batch]: its first row (accepted - 1)
    ssm_read: torch.Tensor  # [batch]: the state slot read
    ssm_write: torch.Tensor  # [batch, SPEC]: the state slot after each row
    spec: int = SPEC
    # [batch]: ATOM's RoPE position of the request's rows minus their cache rows (an image request's MRoPE
    # offset; 0 for text)
    rope_off: torch.Tensor | None = None


@dataclass
class Put:
    """``dst[i] = valid[i // group] ? src[...] - sub : fill`` for ``i < dst.numel()`` (``masked``), else
    ``src[...] - sub``. ``src`` is read in runs of ``inner`` elements ``stride`` apart (a ``[batch, width]`` slice of a
    wider table: ``inner = width``, ``stride = src.stride(0)``); by default it is contiguous.
    """

    dst: torch.Tensor
    src: torch.Tensor
    masked: bool = True
    fill: int = -1
    sub: int = 0
    group: int = 1
    inner: int | None = None
    stride: int | None = None


def meta_buffers(batch: int, spec: int, width: int, io_width: int, device) -> dict:
    """The kernels' metadata args under the recipe's arg names (int32), the ``h1_mem`` scratch and the staging buffer
    of an input that is not a registered buffer."""
    rows, i32 = batch * spec, torch.int32
    return {
        "gdn.conv_slot": torch.zeros(batch, dtype=i32, device=device),
        "gdn.conv_off": torch.zeros(batch, dtype=i32, device=device),
        "gdn.ssm_read": torch.zeros(batch, dtype=i32, device=device),
        "gdn.ssm_write": torch.zeros(rows, dtype=i32, device=device),
        "seq_lens": torch.zeros(batch, dtype=i32, device=device),
        "rope_off": torch.zeros(batch, dtype=i32, device=device),
        "block_table": torch.zeros(batch, width, dtype=i32, device=device),
        "sel": torch.zeros(batch, dtype=i32, device=device),
        "h1_mem": torch.zeros(rows, io_width, dtype=torch.bfloat16, device=device),
        "stage": torch.zeros(rows, io_width, dtype=torch.bfloat16, device=device),
    }


def target_puts(m: dict, meta: StepMeta) -> list[Put]:
    """A target step's metadata args: the kernels place the step's K / V and index keys at ``seq_lens[b] + s``
    through the block table (ATOM's ``slot_mapping``) and rotate them at ``seq_lens[b] + s + rope_off[b]``
    (ATOM's positions)."""
    bt = meta.block_tables
    puts = [
        Put(m["gdn.conv_slot"], meta.conv_slot),
        Put(m["gdn.conv_off"], meta.conv_off, masked=False),
        Put(m["gdn.ssm_read"], meta.ssm_read),
        Put(m["gdn.ssm_write"], meta.ssm_write.reshape(-1), group=meta.spec),
        Put(m["seq_lens"], meta.context_lens, fill=0, sub=meta.spec),
        Put(
            m["block_table"],
            bt,
            group=bt.shape[1],
            inner=bt.shape[1],
            stride=bt.stride(0),
        ),
    ]
    if meta.rope_off is not None:
        puts.append(Put(m["rope_off"], meta.rope_off, fill=0))
    return puts


@dataclass
class GDNPools:
    """One GDN layer's conv window pool ``[slots, W, conv_dim]`` (token-major, dense) and recurrent state pool ``[slots, v_heads, K, V]`` (``geo.state_dtype``). Under
    ReplaySSM the state pool holds each request's checkpoint, and ``rec`` the layer's records ``(k, u, g)``:
    bf16 ``[slots, v_heads, L, K]``, bf16 ``[slots, v_heads, L, V]``, f32 ``[slots, v_heads, L]``, with the record
    cursor ``write_pos`` int32 ``[slots]`` (one for every layer).
    """

    geo: object  # GDNGeometry
    conv: torch.Tensor
    ssm: torch.Tensor
    rec: tuple | None = None
    write_pos: torch.Tensor | None = None

    def __post_init__(self):
        g = self.geo
        # ATOM's pool (``gdn_attn``'s ``mamba_k_cache``) is token-major, as the kernels read it
        if self.conv.shape[-1] != g.conv_dim or self.conv.shape[1] < g.conv_kernel - 1:
            raise ValueError(
                f"unsupported conv pool {tuple(self.conv.shape)}: mono reads [slots, rows, {g.conv_dim}]"
            )
        if tuple(self.ssm.shape[1:]) != (g.v_heads, g.k_dim, g.v_dim):
            raise ValueError(f"unsupported state pool {tuple(self.ssm.shape)}")
        # the kernels address both pools by slot with dense strides
        if not self.conv.is_contiguous() or not self.ssm.is_contiguous():
            raise ValueError(
                f"GDN pools must be dense (token-major conv, slot-major state): strides {self.conv.stride()}, "
                f"{self.ssm.stride()}"
            )
        if self.ssm.dtype != g.state_dtype:
            raise ValueError(
                f"GDN state pool is {self.ssm.dtype}, mono stores the state as {g.state_dtype}: "
                "run ATOM with a state pool dtype matching it (the checkpoint's mamba_ssm_dtype)"
            )
        if self.rec is not None:
            rk, ru, rg = self.rec
            slots, cap = self.ssm.shape[0], rk.shape[2]
            want = (
                (rk, (slots, g.v_heads, cap, g.k_dim), torch.bfloat16),
                (ru, (slots, g.v_heads, cap, g.v_dim), torch.bfloat16),
                (rg, (slots, g.v_heads, cap), torch.float32),
            )
            for t, shape, dtype in want:
                if tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous():
                    raise ValueError(
                        f"ReplaySSM records {tuple(t.shape)} {t.dtype}, mono reads {shape} {dtype}"
                    )
            wp = self.write_pos
            if wp is None or wp.dtype != torch.int32 or wp.numel() < slots:
                raise ValueError(
                    "ReplaySSM: the record cursor (int32, one a slot) is missing"
                )

    @property
    def replay(self) -> int:
        """The record capacity under ReplaySSM, 0 for the classic verify."""
        return 0 if self.rec is None else int(self.rec[0].shape[2])

    def check_window(self, spec: int) -> None:
        """The kernels address a request's conv window with the step's own width (a compile-time stride): ATOM's
        pool must hold ``2 + spec`` rows, as its MTP verify pool does."""
        if self.conv.shape[1] != 2 + spec:
            raise ValueError(
                f"mono kernels at spec {spec}: GDN conv rows {self.conv.shape[1]} != {2 + spec}"
            )


@dataclass
class QSAPools:
    """One QSA layer's paged caches (one block table): ``k``, ``v`` [blocks, bs, kv_heads, head_dim] (bf16, or fp8
    with fp32 ``k_scale`` / ``v_scale`` [blocks, bs, kv_heads]), ``index_raw`` [blocks, bs, 1, index_dim],
    ``index_cmp`` [blocks, bs / compress, 1, index_dim] (bf16)."""

    geo: object  # Geometry
    k: torch.Tensor
    v: torch.Tensor
    index_raw: torch.Tensor
    index_cmp: torch.Tensor
    k_scale: torch.Tensor | None = None
    v_scale: torch.Tensor | None = None

    def __post_init__(self):
        g = self.geo
        n, bs = self.k.shape[:2]
        want = {
            "k": (n, bs, g.kv_heads, g.head_dim),
            "v": (n, bs, g.kv_heads, g.head_dim),
            "index_raw": (n, bs, 1, g.index_dim),
            "index_cmp": (n, bs // g.index_compress, 1, g.index_dim),
        }
        fp8 = self.k.dtype in FP8_TABLE
        for name, shape in want.items():
            t = getattr(self, name)
            dtype_ok = (
                (t.dtype in FP8_TABLE)
                if (fp8 and name in ("k", "v"))
                else t.dtype == torch.bfloat16
            )
            if tuple(t.shape) != shape or not dtype_ok or not t.is_contiguous():
                raise ValueError(
                    f"unsupported QSA {name} cache {tuple(t.shape)} {t.dtype}, want {shape}"
                )
        if fp8:
            for name in ("k_scale", "v_scale"):
                t = getattr(self, name)
                if (
                    t is None
                    or tuple(t.shape[:3]) != (n, bs, g.kv_heads)
                    or t.dtype != torch.float32
                ):
                    raise ValueError(
                        f"fp8 QSA caches need fp32 {name} [{n}, {bs}, {g.kv_heads}]"
                    )
        if bs % g.index_compress:
            raise ValueError(
                f"QSA block size {bs} is not a multiple of {g.index_compress}"
            )


def draft_attention_fused(
    meta, is_prefill: bool, rows: int, rope_position_cache
) -> bool:
    """Whether the fallback draft's QSA attention takes its fused path (``draft_fallback.attention``): a decode step
    of one or ``SPEC`` rows a request, with the three-axis RoPE position cache the fused index preparation reads.
    Otherwise ATOM's module: a prefill, no QSA metadata, another row count, or a text-only pool (no position cache;
    ATOM's module rebuilds text positions arithmetically)."""
    if meta is None or is_prefill or rope_position_cache is None:
        return False
    requests = meta.block_tables.shape[0]
    return rows in (requests, SPEC * requests)


def draft_rows_error(
    positions: list, slots: list, table: list, block: int, batch: int, spec: int
) -> str | None:
    """Why a draft step's rows break the kernels' addressing, or None. Mono places request ``b``'s row ``s`` at
    ``seq_lens[b] + s`` through its block-table row: ATOM's rows must be consecutive positions a request and its
    ``slot_mapping`` the block table's slots."""
    want = [
        table[r // spec][p // block] * block + p % block
        for r, p in enumerate(positions)
    ]
    runs = all(
        positions[b * spec : (b + 1) * spec]
        == list(range(positions[b * spec], positions[b * spec] + spec))
        for b in range(batch)
    )
    if want != slots or not runs:
        return f"positions {positions}, block-table slots {want}, slot_mapping {slots}"
    return None
