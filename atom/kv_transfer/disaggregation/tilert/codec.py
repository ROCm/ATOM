# SPDX-License-Identifier: MIT
"""ATOM V4.1 BF16 PAGE/STATE to TileRT 0.1.7 request wire image.

This correctness-first codec deliberately uses the published TileRT profile
for header and RDMA geometry. No ATOM pool bytes are treated as wire bytes.
"""

import torch


def pack_mla(rows, cb):
    """584-byte fp8_ds_mla records, data then scales within each page."""
    from tilert.pd_vllm.profiles import dsv41_rocm_convert as c

    rows = rows.reshape(-1, cb, 512)
    x = rows[..., :448].float().reshape(-1, cb, 7, 64)
    scale = c._pow2_ceil(x.abs().amax(-1).clamp_min(0.0001) / 448)
    q = (x / scale[..., None]).clamp(-448, 448).to(c.E4M3)
    data = torch.cat(
        (
            q.reshape(-1, cb, 448).view(torch.uint8),
            rows[..., 448:].contiguous().to(torch.bfloat16).view(torch.uint8),
        ),
        -1,
    )
    scales = torch.zeros(*scale.shape[:-1], 8, dtype=torch.uint8, device=rows.device)
    scales[..., :7] = scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    return torch.cat((data.flatten(1), scales.flatten(1)), -1)


def unpack_atom_index(pages, cb, tile=8):
    """Reverse ATOM's scorer-block FP8 shuffle and planar FP32 scales."""
    # Each scorer block carries its own data then scales. A PAGE
    # contains several such blocks; its scales are not all at PAGE's end.
    pages = pages.flatten(1).contiguous().reshape(-1, cb // tile, tile * 132)
    data = pages[..., : tile * 128].reshape(-1, cb // tile, 8, tile, 16)
    values = data.permute(0, 1, 3, 2, 4).contiguous().reshape(-1, cb, 128)
    scales = pages[..., tile * 128 :].contiguous().view(torch.float32).reshape(-1, cb)
    return values.view(torch.float8_e4m3fn).float() * scales[..., None]


def pack_index(rows, cb):
    """Quantize to MXFP4 and apply TileRT's 32-row wire shuffle."""
    from tilert.pd_vllm.profiles import dsv41_rocm_convert as c

    x = rows.float().reshape(-1, cb, 4, 32)
    s = c._pow2_ceil(x.abs().amax(-1).clamp_min(0.0001) / 6)
    codes = c._fp4_code((x / s[..., None]).clamp(-6, 6)).reshape(-1, cb, 128)
    vals = codes[..., 0::2] | (codes[..., 1::2] << 4)
    scales = s.to(torch.float8_e8m0fnu).view(torch.uint8)
    out = torch.zeros(vals.shape[0], cb * 68, dtype=torch.uint8, device=rows.device)
    vi, si = c._idx_offsets(cb, rows.device)
    out[:, vi] = vals.flatten(1)
    out[:, si] = scales.flatten(1)
    return out


class AtomWireEncoder:
    def __init__(self, cache, max_seq_len):
        from tilert.pd_vllm.profiles.dsv41_rocm import Dsv41RocmProfile

        g = cache.geometry
        if (
            g.kv_dtype != "bf16"
            or g.index_fp4
            or g.index_block_rows not in (8, 16)
            or g.head_dim != 512
            or g.index_dim != 128
            or g.owners != ((2, 2), (8, 2), (14, 2), (20, 1))
            or g.layers not in (40, 43)
            or g.window_size != 128
            or g.block_size != 128
        ):
            raise ValueError(f"Unsupported TileRT source geometry: {g}")
        self.cache, self.max_seq_len = cache, max_seq_len
        self.profile = Dsv41RocmProfile()
        self.staging = torch.zeros(
            self.profile.buffer_bytes(max_seq_len),
            dtype=torch.uint8,
            device=cache.backing.device,
        )

    @torch.inference_mode()
    def encode(self, request):
        from tilert.pd_vllm.profiles import dsv41_rocm as w

        c, n = self.cache, len(request.prompt_ids)
        if not 0 < n <= self.max_seq_len:
            raise ValueError("Prompt exceeds TileRT staging capacity")
        slot = request.slot
        if not 0 <= slot < c.state_bytes.shape[0]:
            raise ValueError("Invalid retained STATE slot")
        if int(c.cursor[slot, 0].item()) != n:
            raise ValueError("TileRT handoff must contain exactly the committed prompt")
        c.require_committed()
        regs, stg = w.regions(self.max_seq_len), self.staging
        hdr = torch.zeros(w.HDR_WORDS, dtype=torch.int32)
        pages, page_bytes = {}, {}
        ids = torch.tensor(
            request.blocks[: (n + 127) // 128], dtype=torch.long, device=stg.device
        )
        if ids.numel() != (n + 127) // 128:
            raise ValueError("Insufficient retained PAGE blocks")
        for pi, plane in enumerate(w.PAGE_PLANES):
            ratio = w._PLANE_RATIO[plane]
            cb = 128 // ratio
            pages[plane], page_bytes[plane] = ids.numel(), cb * w._PLANE_REC[plane]
            hdr[w.H_CB + pi], hdr[w.H_PAGES + pi] = cb, ids.numel()
            for li, owner in enumerate(w._PLANE_LAYERS[plane]):
                if plane.startswith("main"):
                    rows = c.pages.view(f"main_{owner}")[0].index_select(0, ids)
                    wire = pack_mla(rows, cb)
                else:
                    raw = c.index_planes[owner][0].index_select(0, ids)
                    wire = pack_index(
                        unpack_atom_index(raw, cb, c.geometry.index_block_rows), cb
                    )
                off = regs[plane].offset + li * regs[plane].layer_bytes
                stg[off : off + wire.numel()].copy_(wire.flatten())
        first, last = w.C.window_first_block(n), (n + 31) // 32
        nb = last - first
        absolute = torch.arange(first * 32, last * 32, device=stg.device)
        # Rows before n-128 and after n are padding and never consumed by D.
        valid = (absolute >= max(0, n - 128)) & (absolute < n)
        windows = c.state.view("window")[:, slot]
        for layer in range(c.geometry.layers):
            rows = (
                windows[layer].index_select(0, absolute % c.geometry.ring_slots).clone()
            )
            rows[~valid] = 0
            wire = pack_mla(rows, 32)
            off = regs["swa"].offset + layer * regs["swa"].layer_bytes
            stg[off : off + wire.numel()].copy_(wire.flatten())
        ring_rows = 16
        for li, owner in enumerate(w.R2):
            kv, score = c.compress_state(owner)
            ring = torch.zeros(16, 1024, device=stg.device, dtype=torch.float32)
            if n % 2:
                pos = (n - 1) % c.geometry.compress_ring_slots
                ring[(n - 1) % 16] = torch.cat((kv[slot, pos], score[slot, pos]))
            off = regs["ring"].offset + li * regs["ring"].layer_bytes
            stg[off : off + ring.numel() * 4].copy_(ring.flatten().view(torch.uint8))
        tail = request.prompt_ids[-16:]
        hdr[w.H_MAGIC], hdr[w.H_VER], hdr[w.H_N] = w.WIRE_MAGIC, w.WIRE_VERSION, n
        hdr[w.H_RING_ROWS], hdr[w.H_NWIN] = ring_rows, c.geometry.layers
        hdr[w.H_WIN_FIRST], hdr[w.H_WIN_BLOCKS], hdr[w.H_CB + 4] = first, nb, 32
        hdr[w.H_NTAIL] = len(tail)
        hdr[w.H_TAIL : w.H_TAIL + len(tail)] = torch.tensor(tail, dtype=torch.int32)
        stg[: w.HDR_BYTES].copy_(hdr.view(torch.uint8).to(stg.device))
        # Source compute and packing must be visible before the CPU RDMA call.
        torch.cuda.current_stream().synchronize()
        return {
            "seq": n,
            "stage_max": self.max_seq_len,
            "pages": pages,
            "page_bytes": page_bytes,
            "nwin": c.geometry.layers,
            "win_blocks": nb,
            "ring_rows": ring_rows,
        }
