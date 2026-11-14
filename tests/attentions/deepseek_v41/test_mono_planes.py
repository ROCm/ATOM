# SPDX-License-Identifier: MIT
"""Execute mono's cache I/O helpers against independently gathered plane bytes.

These small launches cover K1 stores (including the peer-row receiver), K2
QK's global-to-LDS staging and PV's loads/dequantization. They do not execute
the full mono layer or its inter-rank synchronization.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)
pytest.importorskip("aiter")
pytest.importorskip("flydsl")
if "gfx950" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""):
    pytest.skip("gfx950 mono kernels required", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from atom.model_ops.attentions.deepseek_v41.planes import split_source
from atom.models.deepseek_v41.mono.kernels import attention, attn_pre
from atom.mono.device.ops import ld_f32, ld_i32, rsrc


def _writer(peer):
    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel(
        data: Int64,
        scales: Int64,
        rows: Int64,
        offset: Int32,
        values: Int64,
        codes: Int64,
        payload: Int64,
    ):
        tid, t = fx.thread_idx.x, fx.block_idx.x
        c = {
            "tid": tid,
            "rank": 99,
            "kv_fp8": True,
            "tok_par": False,
            "args": {
                "ring": data,
                "ring_scale": scales,
                "ring_rows": rows,
                "ring_off": offset,
            },
        }
        if const_expr(peer):

            def poll(requests):
                return [
                    fx.Vector.from_elements([ld_i32(ptr, at)], fx.Int32)
                    for ptr, at, _ in requests
                ]

            c["poll"] = poll
            attn_pre.ring_row_from(c, payload, t, 1)
        else:
            tt = fx.min(tid, 255)
            rel = ld_i32(rows, t)
            pair = [ld_f32(values, t * 512 + 2 * tt + j) for j in range(2)]
            code = ld_i32(codes, t * 16 + tt // 16)
            attn_pre._ring_row_fp8(c, t, tt, tid < 256, rel, rel + offset, pair, code)

    @flyc.jit
    def launch(
        data: Int64,
        scales: Int64,
        rows: Int64,
        offset: Int32,
        values: Int64,
        codes: Int64,
        payload: Int64,
        stream: fx.Stream,
    ):
        kernel(data, scales, rows, offset, values, codes, payload).launch(
            grid=(5,), block=(512,), stream=stream
        )

    return launch


@pytest.mark.parametrize("peer", [False, True])
def test_mono_window_stores_preserve_plane_bytes_and_padding(peer):
    torch.manual_seed(418)
    # Nonzero base, first/last rows, a padding token and inactive rows.
    rows = torch.tensor([0, 7, -1, 32, 65], dtype=torch.int32, device="cuda")
    offset, capacity = 3, 72
    values = torch.randint(-8, 9, (5, 512), device="cuda").float()
    codes = torch.randint(118, 131, (5, 16), device="cuda", dtype=torch.int32)
    packed = values.to(torch.float8_e4m3fn).view(torch.uint8)
    backing = torch.full((capacity * 528,), 0xA5, dtype=torch.uint8, device="cuda")
    data, scales = split_source(backing, 512)
    expected = backing.clone()
    expected_data, expected_scales = split_source(expected, 512)
    for t, row in enumerate(rows.cpu().tolist()):
        if row >= 0:
            expected_data[offset + row] = packed[t]
            expected_scales[offset + row] = codes[t].to(torch.uint8)
    # KG mailbox payload: data as 128 dwords, 16 codes as dwords, padded to
    # the production 256-dword token stride. Only readiness polling is stubbed.
    payload = torch.full((5, 256), -1, dtype=torch.int32, device="cuda")
    payload[:, :128] = packed.view(torch.int32)
    payload[:, 128:144] = codes
    _run_compiled(
        _writer(peer),
        data.data_ptr(),
        scales.data_ptr(),
        rows.data_ptr(),
        offset,
        values.data_ptr(),
        codes.data_ptr(),
        payload.data_ptr(),
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    assert torch.equal(backing, expected)


def _reader():
    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel(
        main: Int64,
        main_scale: Int64,
        window: Int64,
        window_scale: Int64,
        boundary: Int32,
        ids: Int64,
        raw: Int64,
        decoded: Int64,
    ):
        lane = fx.thread_idx.x
        c = {
            "lane": lane,
            "kv_fp8": True,
            "args": {
                "pool": main,
                "pool_scale": main_scale,
                "window": window,
                "window_scale": window_scale,
                "ring_off": boundary,
            },
        }
        alloc = fx.SharedAllocator()
        kv = (
            alloc.allocate(fx.Array[fx.Int32, attention.SCORE_LDS_WORDS, 16]).peek().ptr
        )
        slot = ld_i32(ids, lane % 16)
        attention._rows_to_lds(c, main, slot, kv)
        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
        gpu.barrier()
        for row in range_constexpr(16):
            for chunk in range_constexpr(3):
                col = lane + 64 * chunk
                if col < 132:
                    word = fx.Int32(
                        fx.ptr_load(
                            kv + row * attention.SCORE_LDS_ROW + col, result_type=T.i32
                        )
                    )
                    bo.buffer_store(word, rsrc(raw), row * 132 + col)
            ident = ld_i32(ids, row)
            # One lane's eight columns; real PV loads and E8M0 dequantization.
            tile = attention._ld_row4(c, main, ident, lane * 8)
            code_word = attention._ld_codes(c, ident, lane // 4)
            code = (code_word >> ((lane // 4 % 4) * 8)) & 255
            words = attention._tile_words(c, tile, code)
            bo.buffer_store(words, rsrc(decoded), row * 256 + lane * 4)

    @flyc.jit
    def launch(
        main: Int64,
        main_scale: Int64,
        window: Int64,
        window_scale: Int64,
        boundary: Int32,
        ids: Int64,
        raw: Int64,
        decoded: Int64,
        stream: fx.Stream,
    ):
        kernel(
            main, main_scale, window, window_scale, boundary, ids, raw, decoded
        ).launch(grid=(1,), block=(64,), stream=stream)

    return launch


@pytest.mark.parametrize("main_rows", [0, 11])
def test_mono_qk_and_pv_read_the_same_csa_swa_planes(main_rows):
    torch.manual_seed(419)
    sources = []
    for rows in (main_rows, 73):
        backing = torch.empty(rows * 528, dtype=torch.uint8, device="cuda")
        data, scale = split_source(backing, 512)
        data.copy_(
            torch.randint(-8, 9, (rows, 512), device="cuda")
            .float()
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        scale.copy_(
            torch.randint(118, 131, (rows, 16), device="cuda", dtype=torch.uint8)
        )
        sources.append((data, scale))
    # Probe both sides of the CSA/SWA boundary, plus interior and final rows.
    chosen = ([0, main_rows - 1] if main_rows else []) + [
        main_rows,
        main_rows + 17,
        main_rows + 72,
    ]
    ids = torch.tensor((chosen * 16)[:16], device="cuda", dtype=torch.int32)
    expected_rows = []
    for ident in ids.cpu().tolist():
        source, row = (
            (sources[0], ident)
            if ident < main_rows
            else (sources[1], ident - main_rows)
        )
        expected_rows.append(torch.cat([source[0][row], source[1][row]]))
    expected = torch.stack(expected_rows)
    raw = torch.empty((16, 528), device="cuda", dtype=torch.uint8)
    decoded = torch.empty((16, 512), device="cuda", dtype=torch.bfloat16)
    _run_compiled(
        _reader(),
        *(p.data_ptr() for source in sources for p in source),
        main_rows,
        ids.data_ptr(),
        raw.data_ptr(),
        decoded.data_ptr(),
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    assert torch.equal(raw, expected)
    values = expected[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = torch.exp2(expected[:, 512:].float() - 127).repeat_interleave(32, dim=1)
    assert torch.equal(decoded, (values * scales).bfloat16())
