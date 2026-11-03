# SPDX-License-Identifier: MIT
"""Dispatch rules of the opus flat-row MLA decode (``ATOM_OPUS_MLA``).
Meta tensors and a recording decoder: no GPU and no aiter kernel needed."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_ops import opus_mla

FP8 = torch.float8_e4m3fn


class _Decoder:
    """Records the calls ``opus_mla`` makes on ``OpusMlaFlatDecode``."""

    def __init__(self, max_batch):
        self.max_batch = max_batch
        self.calls = []

    def plan(self, kv_indptr, q_len, num_heads, cp_world_size=1):
        self.calls.append(
            ("plan", np.asarray(kv_indptr).copy(), q_len, num_heads, cp_world_size)
        )

    def clear(self):
        self.calls.append(("clear",))

    def __call__(self, q, kv, out, *args):
        self.calls.append(("decode", q, kv, out, args))
        return "lse"


@pytest.fixture
def decoder(monkeypatch):
    monkeypatch.setenv("ATOM_OPUS_MLA", "1")
    monkeypatch.delenv("ATOM_OPUS_MLA_MAX_BS", raising=False)
    monkeypatch.setattr(opus_mla, "OpusMlaFlatDecode", _Decoder)
    monkeypatch.setattr(opus_mla, "_decoder", None)
    opus_mla.make_marker(1, 1)
    return opus_mla._decoder


def _args(
    q_len=6,
    *,
    heads=96,
    q_dtype=FP8,
    kv_dtype=FP8,
    o_dtype=torch.bfloat16,
    kv_rows=1 << 20,
    marker_q_len=None,
    **over,
):
    bs = 16
    q = torch.empty((bs * q_len, heads, 576), dtype=q_dtype, device="meta")
    args = {
        "q": q,
        "kv_buffer": torch.empty((kv_rows, 1, 1, 576), dtype=kv_dtype, device="meta"),
        "o": torch.empty((bs * q_len, heads, 512), dtype=o_dtype, device="meta"),
        "q_scale": torch.ones(1),
        "kv_scale": torch.ones(1),
        "causal": True,
        "cp_world_size": 8,
        "max_q_len": q_len,
        "page_size": 1,
    }
    args.update(over)
    marker = opus_mla.PlanMarker(bs, q_len if marker_q_len is None else marker_q_len)
    return SimpleNamespace(opus_plan=marker), args


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("ATOM_OPUS_MLA", raising=False)
    monkeypatch.setattr(opus_mla, "OpusMlaFlatDecode", _Decoder)
    assert not opus_mla.enabled()


def test_aiter_without_the_kernel_keeps_mla_decode_fwd(monkeypatch):
    monkeypatch.setenv("ATOM_OPUS_MLA", "1")
    monkeypatch.setattr(opus_mla, "OpusMlaFlatDecode", None)
    assert not opus_mla.enabled()


def test_defaults(decoder):
    assert opus_mla.enabled()
    assert decoder.max_batch == 128
    assert [b for b in range(140) if opus_mla.uses_opus_for_batch(b)] == list(
        range(1, 129)
    )


def test_max_batch_follows_env(monkeypatch, decoder):
    monkeypatch.setenv("ATOM_OPUS_MLA_MAX_BS", "64")
    assert [b for b in (1, 64, 65) if opus_mla.uses_opus_for_batch(b)] == [1, 64]


@pytest.mark.parametrize("q_len", [2, 4, 6, 8, 10])
def test_tested_shape_goes_opus(decoder, q_len):
    meta, args = _args(q_len)
    assert opus_mla.wanted(meta, **args)


@pytest.mark.parametrize(
    "case",
    [
        {"causal": False},
        {"cp_world_size": 1},
        {"q_len": 1},
        {"marker_q_len": 4},
        {"page_size": 64},
        {"heads": 128},
        {"heads": 16},
        {"q_dtype": torch.bfloat16},
        {"q_dtype": torch.float8_e4m3fnuz, "kv_dtype": torch.float8_e4m3fnuz},
        {"kv_dtype": torch.bfloat16},
        {"o_dtype": torch.float32},
        {"q_scale": None},
        {"kv_scale": None},
        {"kv_rows": (1 << 32) // 576 + 1},
    ],
    ids=lambda c: ",".join(f"{k}={v}" for k, v in c.items()),
)
def test_unsupported_calls_stay_on_mla_decode_fwd(decoder, case):
    meta, args = _args(**case)
    assert not opus_mla.wanted(meta, **args)


def test_unmarked_metadata_stays_on_mla_decode_fwd(decoder):
    _, args = _args()
    assert not opus_mla.wanted(SimpleNamespace(), **args)
    assert not opus_mla.wanted(SimpleNamespace(opus_plan=None), **args)


def test_switch_off_after_marking_stays_on_mla_decode_fwd(monkeypatch, decoder):
    meta, args = _args()
    monkeypatch.setenv("ATOM_OPUS_MLA", "0")
    assert not opus_mla.wanted(meta, **args)


def test_non_contiguous_q_stays_on_mla_decode_fwd(decoder):
    meta, args = _args()
    args["q"] = args["q"].transpose(0, 1)
    assert not opus_mla.wanted(meta, **args)


def test_update_plan_covers_the_scheduled_requests(decoder):
    kv_indptr = np.arange(0, 130 * 100, 100, dtype=np.int32)
    opus_mla.update_plan(kv_indptr, 12, 6, 8)
    name, kv, q_len, heads, cp = decoder.calls[-1]
    assert (name, q_len, heads, cp) == ("plan", 6, 96, 8)
    np.testing.assert_array_equal(kv, kv_indptr[:13])


def test_capture_marker_publishes_an_empty_plan(decoder):
    marker = opus_mla.capture_marker(32, 6)
    assert decoder.calls[-1] == ("clear",)
    assert (marker.bs, marker.q_len) == (32, 6)


def test_decode_passes_the_mla_decode_fwd_arguments(decoder):
    q, kv, o = object(), object(), object()
    out = opus_mla.decode(
        q,
        kv,
        o,
        "qo_indptr",
        "kv_indptr",
        "kv_indices",
        "kv_last_page_lens",
        6,
        page_size=1,
        num_kv_splits=16,
        sm_scale=0.1,
        q_scale="qs",
        kv_scale="ks",
        return_lse=True,
        g_kv_indptr="g",
        cp_world_size=8,
        cp_rank=3,
        causal=True,
    )
    assert out == (o, "lse")
    name, got_q, got_kv, got_o, rest = decoder.calls[-1]
    assert (name, got_q, got_kv, got_o) == ("decode", q, kv, o)
    assert rest == ("qo_indptr", "kv_indptr", "kv_indices", 0.1, "qs", "ks", "g", 8, 3)


def _builder(*, is_sparse=False, tbo_decode=False):
    return SimpleNamespace(
        is_sparse=is_sparse,
        model_runner=SimpleNamespace(
            config=SimpleNamespace(enable_tbo_decode=tbo_decode)
        ),
    )


def test_builder_gate(decoder):
    try:
        from atom.model_ops.attentions.aiter_mla import AiterMLAMetadataBuilder
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"aiter MLA backend unavailable: {exc}")

    uses = AiterMLAMetadataBuilder._uses_opus_mla
    assert uses(_builder(), 32, True)
    assert not uses(_builder(), 32, False)
    assert uses(_builder(), 1, True)
    assert not uses(_builder(), 129, True)
    assert not uses(_builder(is_sparse=True), 32, True)
    assert not uses(_builder(tbo_decode=True), 32, True)
