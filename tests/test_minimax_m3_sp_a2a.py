# SPDX-License-Identifier: MIT
"""The SP prefill o_proj hand-off through aiter's general all-to-all: how it picks
the registered-buffer path and when it steps aside for the RCCL hand-off."""

import types

import pytest

torch = pytest.importorskip("torch")
sp_prefill = pytest.importorskip("atom.models.minimax_m3.sp_prefill")


class _Tp:
    """A TP group with the all-to-all of aiter's `GroupCoordinator`."""

    def __init__(self, pool_bytes):
        self.pool = torch.zeros(pool_bytes, dtype=torch.uint8)
        self.calls = []

    def all_to_all_buffer(self, shape, dtype, byte_offset=0):
        n = dtype.itemsize
        for s in shape:
            n *= s
        if byte_offset + n > self.pool.numel():
            raise ValueError("outside the pool")
        return self.pool[byte_offset : byte_offset + n].view(dtype).view(*shape)

    def all_to_all(self, x, scatter_dim, gather_dim, scales=None, registered=False):
        self.calls.append((tuple(x.shape), tuple(scales.shape), registered))
        return x, scales


def _setup(monkeypatch, tp, quant_calls):
    monkeypatch.setattr(sp_prefill, "_tp", lambda: tp)
    blockscale = pytest.importorskip("atom.model_ops.blockscale")
    monkeypatch.setattr(
        blockscale, "quantize_fp8", lambda x, out: quant_calls.append(out)
    )
    return types.SimpleNamespace(padded=64, pad=lambda x: x)


def test_quant_writes_into_the_registered_buffer(monkeypatch):
    tp, quant = _Tp(1 << 20), []
    sp = _setup(monkeypatch, tp, quant)
    got = sp_prefill._a2a_mxfp8_registered(torch.zeros(64, 1024), sp, 1024)
    assert got is not None
    ((codes, scales),) = quant
    assert codes.untyped_storage().data_ptr() == tp.pool.untyped_storage().data_ptr()
    assert scales.data_ptr() % 16 == 0 and scales.shape == (64, 32)
    assert tp.calls == [((64, 1024), (64, 32), True)]


@pytest.mark.parametrize("d,pool", [(1000, 1 << 20), (1024, 1024)])
def test_steps_aside(monkeypatch, d, pool):
    tp, quant = _Tp(pool), []
    sp = _setup(monkeypatch, tp, quant)
    assert sp_prefill._a2a_mxfp8_registered(torch.zeros(64, d), sp, d) is None
    assert not quant and not tp.calls


def test_older_aiter_without_the_buffer(monkeypatch):
    tp, quant = types.SimpleNamespace(), []
    sp = _setup(monkeypatch, tp, quant)
    assert sp_prefill._a2a_mxfp8_registered(torch.zeros(64, 1024), sp, 1024) is None
