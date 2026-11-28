# SPDX-License-Identifier: MIT
"""The KDA prefill's initial-state gathers and its
final-state index. The kernels run on the CPU through Triton's interpreter, in
a child process (the flag must precede the triton import)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="the gathers are @triton.jit kernels")

ROOT = Path(__file__).resolve().parents[1]


def _state(lines=6, heads=3, dim=64, dtype=torch.float32):
    generator = torch.Generator().manual_seed(3)
    return torch.randn(lines, heads, dim, dim, generator=generator).to(dtype)


def _reference(state, slots, has, rows):
    """`gather_kda_initial_state`'s semantics, then `_prefill_kda`'s
    ``index_select``."""
    initial = state[slots].clone()
    if has is not None:
        initial[~has] = 0
    return initial.index_select(0, rows)


def _kda_state_module():
    """`kda_state` alone: the package's other modules import AITER."""
    import importlib.util

    path = ROOT / "atom/model_ops/kimi_k3/kda_state.py"
    spec = importlib.util.spec_from_file_location("spg_kda_state", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_kernels() -> None:
    kda_state = _kda_state_module()

    for dtype in (torch.float32, torch.bfloat16):
        state = _state(dtype=dtype)
        # Padded slot stride: each [H, V, K] plane is still dense.
        padded = torch.zeros(6, 4, 64, 64, dtype=dtype)[:, :3]
        padded.copy_(state)
        for src in (state, padded):
            slots = torch.tensor([4, 1, 5], dtype=torch.int32)
            rows = torch.tensor([0, 1, 2, 1, 1, 0], dtype=torch.int32)
            for has in (None, torch.tensor([True, False, True])):
                want = _reference(src, slots, has, rows)
                for k_first in (False, True):
                    got = kda_state._gather_rows(src, slots, has, rows, k_first)
                    assert got.shape == want.shape and got.dtype == want.dtype
                    assert torch.equal(got, want), (dtype, k_first, has)
                    if k_first:
                        assert got.transpose(-1, -2).is_contiguous()
                        assert got.transpose(-1, -2).contiguous().data_ptr() == (
                            got.data_ptr()
                        )
                    else:
                        assert got.is_contiguous()
                    # The int8 copy and the bool's own bytes read alike.
                    if has is not None:
                        again = kda_state._gather_rows(
                            src, slots, has.to(torch.int8), rows, k_first
                        )
                        assert torch.equal(again, want)


def test_gathers_match_the_reference_on_the_cpu():
    code = (
        "import importlib.util;"
        f"s = importlib.util.spec_from_file_location('probe', {__file__!r});"
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m);"
        "m.run_kernels()"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env={**os.environ, "TRITON_INTERPRET": "1", "HIP_VISIBLE_DEVICES": ""},
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert out.returncode == 0, out.stderr[-3000:]


def test_mask_bytes_views_a_dense_bool(monkeypatch):
    kda_state = _kda_state_module()

    has = torch.tensor([True, False, True])
    view = kda_state._mask_bytes(has)
    assert view.dtype == torch.uint8 and view.data_ptr() == has.data_ptr()
    assert view.tolist() == [1, 0, 1]
    # Anything else is cast to one byte per sequence.
    assert kda_state._mask_bytes(has.int()).dtype == torch.int8
    assert kda_state._mask_bytes(has[::2]).dtype == torch.int8


@pytest.mark.parametrize("k_first", [False, True])
def test_cpu_rows_gather_is_the_reference(k_first):
    gather_kda_initial_rows = _kda_state_module().gather_kda_initial_rows

    state = _state()
    slots, has = torch.tensor([2, 0]), torch.tensor([False, True])
    rows = torch.tensor([0, 1, 1])
    got = gather_kda_initial_rows(state, slots, has, rows, k_first)
    assert torch.equal(got, _reference(state, slots, has, rows))
    assert got.transpose(-1, -2).is_contiguous() is k_first


@pytest.fixture(scope="module")
def k3():
    from aiter_stub import atom_without_aiter

    with atom_without_aiter() as load:
        yield SimpleNamespace(
            model=load("atom.models.kimi_k3.model"),
            ops=load("atom.model_ops.kimi_k3"),
        )


def test_prefill_initial_is_one_state_per_sequence_with_heads(monkeypatch, k3):
    model = k3.model
    # The production gather is a kernel; its CPU stand-in is the reference.
    monkeypatch.setattr(
        k3.ops,
        "gather_kda_initial_state",
        lambda state, slots, has: _reference(state, slots, has, torch.arange(2)),
    )
    layer = SimpleNamespace(_paged_prefill=lambda *a: False)
    state = _state()
    ckpt = {"kda_tokens": torch.arange(4), "kda_rows": torch.tensor([0, 1, 1])}
    metadata = SimpleNamespace(
        has_initial_state=torch.tensor([True, False]), ssm_checkpoints=ckpt
    )
    q = v = torch.zeros(1, 4, 3, 64, dtype=torch.bfloat16)
    slots = torch.tensor([3, 5])
    initial, expanded = model.KimiKDAAttention._prefill_initial(
        layer, q, v, metadata, state, slots
    )
    assert expanded
    want = _reference(state, slots, metadata.has_initial_state, ckpt["kda_rows"])
    assert torch.equal(initial, want)


def test_prefill_initial_without_heads_keeps_the_gather(monkeypatch, k3):
    model = k3.model
    seen = []
    monkeypatch.setattr(
        k3.ops,
        "gather_kda_initial_state",
        lambda *a: seen.append(a) or "gathered",
    )
    layer = SimpleNamespace(_paged_prefill=lambda *a: False)
    metadata = SimpleNamespace(has_initial_state=None, ssm_checkpoints=None)
    q = torch.zeros(1, 4, 3, 64)
    assert model.KimiKDAAttention._prefill_initial(
        layer, q, q, metadata, "pool", "slots"
    ) == ("gathered", False)
    assert seen == [("pool", "slots", None)]
    layer._paged_prefill = lambda *a: True
    assert model.KimiKDAAttention._prefill_initial(
        layer, q, q, metadata, "pool", "slots"
    ) == (None, False)


def test_state_write_index_takes_the_int64_copy(k3):
    model = k3.model
    narrow = torch.tensor([4, 2], dtype=torch.int32)
    wide = narrow.long()
    meta = SimpleNamespace(non_spec_state_indices_i64=wide)
    assert model._state_write_index(meta, narrow) is wide
    assert model._state_write_index(SimpleNamespace(), narrow) is narrow
    other = SimpleNamespace(non_spec_state_indices_i64=torch.tensor([1, 2, 3]))
    assert model._state_write_index(other, narrow) is narrow


def test_direct_flash_kda_needs_several_sequences(k3):
    model = k3.model
    q = torch.zeros(1, 8, 2, 128, dtype=torch.bfloat16)
    args = (True, -5.0, torch.zeros(2), dict.fromkeys(model._FLASH_KDA_FLAGS, True))
    assert model._direct_flash_kda_cps(q, q, torch.tensor([0, 8]), *args) is None
    assert model._direct_flash_kda_cps(q, q, None, *args) is None
