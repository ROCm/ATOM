# SPDX-License-Identifier: MIT
"""V4 Mono shares native FP4 pages across both scale layouts."""

import importlib.util
import inspect
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from atom.model_ops.attentions.pool_layout.v4_pool_fields import (
    FP4_GFX950_OPUS,
    FP4_GFX950_PRESHUFFLE,
    FP4_GFX1250_NATURAL,
)


@pytest.fixture(scope="module")
def config_module():
    # Load the real, host-only config without importing the V4 model's kernels.
    path = (
        Path(__file__).resolve().parents[2] / "atom/models/deepseek_v4/mono/config.py"
    )
    name = "_v4_mono_layout_test_config"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def test_layout_selects_a_distinct_compiled_variant(config_module):
    m = config_module
    old = m.V4Config(compress_ratio=4, max_seq=8192)
    new = replace(old, indexer_layout=FP4_GFX950_OPUS)
    old.validate()
    new.validate()

    def key(cfg):
        return m._variant_key(cfg, 4, 8, m.MoeMode.A8W4, False, None, 4, "two")

    assert old.indexer_layout == FP4_GFX950_PRESHUFFLE
    assert key(old) != key(new)
    # Storage sizes stay the same; only the byte interpretation differs.
    before, after = m.shape_dims(old), m.shape_dims(new)
    assert not before.pop("indexer_mfma_scales")
    assert after.pop("indexer_mfma_scales")
    assert before == after
    assert (
        m.shape_dims(replace(new, compress_ratio=128))["indexer_mfma_scales"] is False
    )


def test_other_arch_layout_is_refused(config_module):
    cfg = config_module.V4Config(indexer_layout=FP4_GFX1250_NATURAL)
    with pytest.raises(ValueError, match="unsupported Mono FP4 index layout"):
        cfg.validate()


def _gfx950():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    pytest.importorskip("aiter")
    pytest.importorskip("flydsl")
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx950":
        pytest.skip("MI355 scale layouts")


def _scale_probe(mfma, write):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import const_expr
    from flydsl.expr.typing import T

    from atom.models.deepseek_v4.mono.kernels import buffer_ops as bo
    from atom.models.deepseek_v4.mono.kernels.indexer import index_scale_address
    from atom.models.deepseek_v4.mono.kernels.primitives import rsrc

    @flyc.kernel
    def probe(src: fx.Int64, dst: fx.Int64, pages: fx.Int64):
        p = fx.block_idx.x
        t = fx.thread_idx.x
        page = fx.Int32(bo.buffer_load(rsrc(pages), p, vec_width=1, dtype=T.i32))
        word, byte = index_scale_address(t // 4, t % 4, mfma)
        # Exercise a nonzero owner base, noncontiguous page IDs and guard pages.
        word = 4 + page * 64 + word
        if const_expr(write):
            value = bo.buffer_load(rsrc(src), p * 256 + t, vec_width=1, dtype=T.i8)
            bo.buffer_store(value, rsrc(dst), word * 4 + byte)
        else:
            packed = fx.Int32(bo.buffer_load(rsrc(src), word, vec_width=1, dtype=T.i32))
            value = fx.Int8((packed >> (byte * 8)) & 255)
            bo.buffer_store(value, rsrc(dst), p * 256 + t)

    @flyc.jit
    def launch(src: fx.Int64, dst: fx.Int64, pages: fx.Int64, stream: fx.Stream):
        probe(src, dst, pages).launch(grid=(4, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("mfma", [False, True], ids=["flydsl", "opus"])
def test_scale_reads_and_writes_match_native_packing_and_graph(mfma):
    _gfx950()
    from flydsl.expr import Stream

    torch.manual_seed(1337)
    dev = "cuda"
    natural = torch.randint(0, 256, (4, 64, 4), dtype=torch.uint8, device=dev)
    pages = torch.tensor([2, 0, 4, 1], dtype=torch.int32, device=dev)
    cache = torch.full((16 + 5 * 256 + 16,), 0xA5, dtype=torch.uint8, device=dev)
    decoded = torch.empty_like(natural)
    store, load = _scale_probe(mfma, True), _scale_probe(mfma, False)

    def expected():
        # Native layouts expressed as independent tensor permutations.
        if mfma:
            packed = natural.reshape(4, 2, 32, 2, 2).permute(0, 4, 2, 3, 1)
        else:
            packed = natural.reshape(4, 4, 16, 4).permute(0, 3, 2, 1)
        out = torch.full_like(cache, 0xA5)
        out[16:-16].view(5, 256)[pages.long()] = packed.reshape(4, 256)
        return out

    def run():
        stream = Stream(torch.cuda.current_stream())
        store(natural.data_ptr(), cache.data_ptr(), pages.data_ptr(), stream)
        load(cache.data_ptr(), decoded.data_ptr(), pages.data_ptr(), stream)

    run()
    torch.testing.assert_close(cache, expected(), rtol=0, atol=0)
    torch.testing.assert_close(decoded, natural, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    natural.copy_(torch.randint_like(natural, 0, 256))
    cache.fill_(0xA5)
    graph.replay()
    torch.testing.assert_close(cache, expected(), rtol=0, atol=0)
    torch.testing.assert_close(decoded, natural, rtol=0, atol=0)


@pytest.mark.parametrize("mfma", [False, True], ids=["flydsl", "opus"])
@pytest.mark.parametrize("rows,qlen,phase", [(1, 1, "one"), (4, 4, "attention")])
def test_real_indexer_stages_compile(mfma, rows, qlen, phase):
    _gfx950()
    from atom.models.deepseek_v4.mono.build import kernel_kwargs
    from atom.models.deepseek_v4.mono.config import MoeMode, V4Config
    from atom.models.deepseek_v4.mono.kernels.layer import build_dsv4_kernel
    from atom.mono.runtime.compile import compile_only, launcher_resources

    cfg = V4Config(
        compress_ratio=4,
        max_seq=8192,
        kv_fp8=True,
        indexer_hadamard=False,
        indexer_layout=FP4_GFX950_OPUS if mfma else FP4_GFX950_PRESHUFFLE,
    )
    launch = build_dsv4_kernel(
        rows,
        cfg.heads,
        8,
        cfg.window,
        phase=phase,
        **kernel_kwargs(
            cfg,
            tokens_per_seq=qlen,
            timeline=False,
            moe_mode=MoeMode.A8W4,
            index_topk_refresh=True,
        ),
    )
    n = sum(
        name != "stream" for name in inspect.signature(launch._original_func).parameters
    )
    with compile_only():
        launch(*([0] * n))
    assert launcher_resources(launch)
