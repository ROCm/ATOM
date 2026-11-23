"""GPU writer vs the unchanged CPU wire conversion and import reference."""

from types import SimpleNamespace

import pytest
import torch

from atom.models.deepseek_v41.mono.gfx950.pd import convert as C
from atom.models.deepseek_v41.mono.gfx950.pd import wire as W
from atom.models.deepseek_v41.mono.gfx950.tk import alloc
from atom.models.deepseek_v41.mono.gfx950.tk.host import TkRank


def test_fused_configuration_refuses_incompatible_sources():
    from atom.models.deepseek_v41.mono.gfx950.pd.tkworker import TkRankWorker

    with pytest.raises(ValueError, match="needs IPC"):
        TkRankWorker(0, 8, "/unused", kv_share="gloo", kv_convert="fused")
    with pytest.raises(ValueError, match="needs IPC"):
        TkRankWorker(0, 8, "/unused", planes="/unused.pt", kv_convert="fused")
    with pytest.raises(ValueError, match="needs IPC"):
        TkRankWorker(0, 8, "/unused", receive=False, kv_convert="fused")


@pytest.mark.parametrize(
    "options,expected",
    [
        ([], "fused"),
        (["--kv-convert", "torch"], "torch"),
        (["--kv-convert", "fused"], "fused"),
        (["--kv-share", "gloo"], "torch"),
    ],
)
def test_server_dispatches_default_conversion(monkeypatch, options, expected):
    from atom.models.deepseek_v41.mono.gfx950.pd import server

    class Dispatched(Exception):
        pass

    def capture(nranks, worker, kwargs):
        assert nranks == 8 and worker.endswith(":TkRankWorker")
        assert kwargs["kv_convert"] == expected
        raise Dispatched

    monkeypatch.setattr(server.F, "RankGroup", capture)
    args = server.build_parser().parse_args(
        ["--model-weights-dir", "/unused", *options]
    )
    with pytest.raises(Dispatched):
        server.build_engine(args)


def test_saved_planes_keep_torch_default():
    from atom.models.deepseek_v41.mono.gfx950.pd.tkworker import resolve_kv_convert

    assert resolve_kv_convert(None, planes="/unused.pt") == "torch"
    assert resolve_kv_convert(None, receive=False) == "torch"


def wire_image(n, capacity=512, nwin=40, kind="random", alternate=False):
    gen = torch.Generator().manual_seed(591402 + n)
    buf = torch.zeros(W.wire_bytes(capacity), dtype=torch.uint8)
    regions = W.regions(capacity)
    hdr = torch.zeros(W.HDR_WORDS, dtype=torch.int32)
    hdr[W.H_MAGIC], hdr[W.H_VER], hdr[W.H_N] = W.WIRE_MAGIC, W.WIRE_VERSION, n
    hdr[W.H_RING_ROWS], hdr[W.H_NWIN] = 16, nwin
    wcb = 16 if alternate else 32
    first = C.window_first_block(n, wcb)
    nb = (n - first * wcb + wcb - 1) // wcb
    hdr[W.H_WIN_FIRST], hdr[W.H_WIN_BLOCKS], hdr[W.H_CB + 4] = first, nb, wcb
    k = min(n, C.HIST)
    hdr[W.H_NTAIL] = k
    tail = torch.arange(k, dtype=torch.int32) - 3
    if k:
        tail[-1] = 1000
    hdr[W.H_TAIL : W.H_TAIL + k] = tail

    def records(pages, cb):
        values = torch.randn(pages, cb, 512, generator=gen)
        exponents = torch.randint(
            116, 133, (pages, cb, 8), generator=gen, dtype=torch.uint8
        )
        if kind == "zero":
            values.zero_()
            exponents.fill_(127)
        elif kind == "tiny":
            values.fill_(2.0**-40)
            exponents.fill_(127)
        elif kind == "ties":
            vals = torch.tensor(
                [
                    -0.0,
                    0.25,
                    -0.25,
                    0.75,
                    -0.75,
                    1.25,
                    -1.25,
                    1.75,
                    -1.75,
                    2.5,
                    -2.5,
                    3.5,
                    -3.5,
                    5.0,
                    -5.0,
                    6.0,
                ]
            )
            values = vals.repeat(pages, cb, 32)
            exponents.fill_(127)
        raw = torch.cat(
            (
                values[..., :448].to(torch.float8_e4m3fn).view(torch.uint8),
                values[..., 448:].to(torch.bfloat16).view(torch.uint8),
            ),
            -1,
        )
        return torch.cat(
            (raw.reshape(pages, cb * 576), exponents.reshape(pages, cb * 8)), -1
        )

    for i, plane in enumerate(W.PAGE_PLANES):
        cb = 32 if alternate else W.BLOCK // W.PLANE_RATIO[plane]
        rows = n // W.PLANE_RATIO[plane]
        pages = (rows + cb - 1) // cb
        hdr[W.H_CB + i], hdr[W.H_PAGES + i] = cb, pages
        region = regions[plane]
        for layer in range(region.layers):
            value = (
                torch.randint(
                    0, 256, (pages, cb * W.IDX_REC), generator=gen, dtype=torch.uint8
                )
                if plane.startswith("index")
                else records(pages, cb)
            )
            start = region.offset + layer * region.layer_bytes
            buf[start : start + value.numel()] = value.reshape(-1)
    region = regions["swa"]
    for layer in range(nwin):
        value = records(nb, wcb)
        start = region.offset + layer * region.layer_bytes
        buf[start : start + value.numel()] = value.reshape(-1)
    region = regions["ring"]
    for layer in range(region.layers):
        value = torch.randn(16, 1024, generator=gen)
        value[0, :4] = torch.tensor([-0.0, float("inf"), float("-inf"), 1.0])
        start = region.offset + layer * region.layer_bytes
        buf[start : start + value.numel() * 4] = value.view(torch.uint8).reshape(-1)
    buf[: W.HDR_BYTES] = hdr.view(torch.uint8)
    return buf


def target(capacity=512, device="cpu"):
    tensors = {
        s.name: torch.full(
            s.shape,
            0xA5 if s.dtype == "uint8" else 2.5,
            dtype=getattr(torch, s.dtype),
            device=device,
        )
        for s in alloc.cache_specs(capacity)
    }
    tensors["temp:hist"] = torch.full((16,), 99, dtype=torch.int32, device=device)
    return tensors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU conversion test")
@pytest.mark.parametrize(
    "n,nwin,alternate,kind",
    [
        (0, 0, False, "random"),
        (1, 1, False, "random"),
        (31, 40, False, "random"),
        (32, 43, False, "random"),
        (33, 0, True, "random"),
        (127, 40, False, "random"),
        (128, 43, True, "random"),
        (129, 40, False, "random"),
        (511, 40, False, "random"),
        (512, 43, False, "random"),
        (129, 43, False, "zero"),
        (129, 43, False, "tiny"),
        (129, 43, False, "ties"),
    ],
)
def test_fused_import_matches_cpu_reference(n, nwin, alternate, kind):
    from atom.models.deepseek_v41.mono.gfx950.pd.convert_fused import Converter

    buf = wire_image(n, nwin=nwin, alternate=alternate, kind=kind)
    token_map = torch.arange(128, dtype=torch.int64) * 31 + 2**31
    received = W.ReceivedRequest("test", n, 0, None, None)
    planes = C.convert(buf, 512, received, token_map, device="cpu").planes
    reference = target()
    TkRank.import_planes(
        SimpleNamespace(buffers=SimpleNamespace(tensors=reference)), planes, n
    )
    actual = target(device="cuda:0")
    converter = Converter(actual, token_map, 512)
    converter.import_wire(buf.cuda(), n)
    torch.cuda.synchronize()
    for name, expected in reference.items():
        got = actual[name].cpu()
        assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8)), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU conversion test")
def test_fused_import_clears_previous_window_and_state():
    from atom.models.deepseek_v41.mono.gfx950.pd.convert_fused import Converter

    tm = torch.arange(128, dtype=torch.int64)
    actual = target(device="cuda:0")
    reference = target()
    converter = Converter(actual, tm, 512)
    for n, nwin in ((511, 43), (32, 0), (1, 1), (0, 0)):
        buf = wire_image(n, nwin=nwin)
        rec = W.ReceivedRequest("test", n, 0, None, None)
        planes = C.convert(buf, 512, rec, tm, device="cpu").planes
        TkRank.import_planes(
            SimpleNamespace(buffers=SimpleNamespace(tensors=reference)), planes, n
        )
        converter.import_wire(buf.cuda(), n)
        torch.cuda.synchronize()
        for name, expected in reference.items():
            assert torch.equal(
                actual[name].cpu().view(torch.uint8), expected.view(torch.uint8)
            ), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU conversion test")
def test_bad_header_is_rejected_before_cache_writes():
    from atom.models.deepseek_v41.mono.gfx950.pd.convert_fused import Converter

    actual = target(device="cuda:0")
    converter = Converter(actual, torch.arange(128, dtype=torch.int64), 512)
    buf = wire_image(129)
    buf.view(torch.int32)[W.H_WIN_FIRST] = 1000
    with pytest.raises(ValueError, match="window page range"):
        converter.import_wire(buf.cuda(), 129)
    for name, expected in target().items():
        assert torch.equal(
            actual[name].cpu().view(torch.uint8), expected.view(torch.uint8)
        ), name
