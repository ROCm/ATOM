# SPDX-License-Identifier: MIT
"""Dispatch rules of the persistent HCA decode kernel (``ATOM_V4_HCA_PERSIST``).
Mocked: no GPU and no aiter kernel needed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="paged_decode defines Triton kernels")
pytest.importorskip("aiter", reason="paged_decode imports the AITER runtime")

from atom.model_ops.v4_kernels import hca_persist, paged_decode


@pytest.fixture
def routes(monkeypatch):
    """Replace both backends by recorders; return the call log."""
    calls = []
    monkeypatch.setattr(paged_decode.envs, "ATOM_USE_V4_PREFILL_ASM_FOR_DECODE", False)
    monkeypatch.setattr(
        paged_decode,
        "_sparse_attn_v4_paged_decode_asm",
        lambda *a, **k: calls.append(("asm", a, k)) or "asm",
    )
    monkeypatch.setattr(
        hca_persist,
        "hca_persist_decode",
        lambda *a, **k: calls.append(("persist", a, k)) or "persist",
    )
    monkeypatch.setattr(hca_persist, "mla_decode_fwd_v4_nm_ps", object())
    monkeypatch.setattr(hca_persist, "workspace_ready", lambda device: True)
    return calls


def _env(monkeypatch, enabled=True, min_rows=15, gfx="gfx950"):
    monkeypatch.setenv("ATOM_V4_HCA_PERSIST", "1" if enabled else "0")
    monkeypatch.setenv("ATOM_V4_HCA_PERSIST_MIN_ROWS", str(min_rows))
    monkeypatch.setattr(paged_decode, "get_gfx", lambda: gfx)


def _call(n, *, heads=128, ratio=128, kv_rows=8, kv=None, kv_rope=None, plan=None):
    kv = torch.empty((kv_rows, 512), dtype=torch.uint8) if kv is None else kv
    kv_rope = (
        torch.empty((kv_rows, 64), dtype=torch.bfloat16) if kv_rope is None else kv_rope
    )
    return paged_decode.sparse_attn_v4_paged_decode(
        q=None,
        unified_kv=kv,
        kv_indices=torch.zeros(0, dtype=torch.int32),
        kv_indptr=torch.zeros(n + 1, dtype=torch.int32),
        attn_sink=torch.empty(heads),
        softmax_scale=512**-0.5,
        unified_kv_rope=kv_rope,
        q_packed_in=torch.empty((n, heads, 512), dtype=torch.uint8),
        q_rope_in=torch.empty((n, heads, 64), dtype=torch.bfloat16),
        qo_indptr=torch.arange(n + 1, dtype=torch.int32),
        split_plan=plan,
        compress_ratio=ratio,
    )


@pytest.mark.parametrize("n", [15, 21, 112, 448, 896, 32768])
def test_hca_rows_at_or_above_min_go_persistent(monkeypatch, routes, n):
    _env(monkeypatch)
    assert _call(n) == "persist"


@pytest.mark.parametrize("n", [1, 7, 14])
def test_small_batches_stay_on_asm(monkeypatch, routes, n):
    _env(monkeypatch)
    plan = (2, torch.arange(0, 2 * (n + 1), 2, dtype=torch.int32))
    assert _call(n, plan=plan) == "asm"
    # the ASM path keeps its split plan unchanged
    assert routes[-1][2]["num_kv_splits"] == 2
    assert routes[-1][2]["split_indptr"] is plan[1]


def test_min_rows_is_configurable(monkeypatch, routes):
    _env(monkeypatch, min_rows=8)
    assert _call(7) == "asm"
    assert _call(8) == "persist"


def test_rows_above_limit_fall_back(monkeypatch, routes):
    _env(monkeypatch)
    assert _call(32769) == "asm"


@pytest.mark.parametrize(
    "kw",
    [
        {"ratio": 0},
        {"ratio": 4},
        {"ratio": None},
        {"heads": 64},
        {"heads": 16},
    ],
)
def test_non_hca_or_other_heads_stay_on_asm(monkeypatch, routes, kw):
    _env(monkeypatch)
    assert _call(112, **kw) == "asm"


@pytest.mark.parametrize("gfx", ["gfx942", "gfx1250"])
def test_other_arch_stays_on_asm(monkeypatch, routes, gfx):
    _env(monkeypatch, gfx=gfx)
    # gfx1250 H=128 could take the prefill-ASM route; it is off in `routes`
    assert _call(112) == "asm"


def test_switch_on_by_default(monkeypatch, routes):
    monkeypatch.delenv("ATOM_V4_HCA_PERSIST", raising=False)
    monkeypatch.setattr(paged_decode, "get_gfx", lambda: "gfx950")
    assert _call(896) == "persist"


def test_switch_off(monkeypatch, routes):
    _env(monkeypatch, enabled=False)
    assert _call(896) == "asm"


def test_aiter_without_kernel_stays_on_asm(monkeypatch, routes):
    _env(monkeypatch)
    monkeypatch.setattr(hca_persist, "mla_decode_fwd_v4_nm_ps", None)
    assert _call(896) == "asm"
    monkeypatch.setattr(hca_persist, "get_mla_v4_nm_ps_workspace", None)
    monkeypatch.setattr(hca_persist, "_workspaces", {})
    hca_persist.prepare("cuda:0")  # quietly nothing to prepare
    assert hca_persist._workspaces == {}


def test_strided_pool_falls_back(monkeypatch, routes):
    _env(monkeypatch)
    wide = torch.empty((8, 1024), dtype=torch.uint8)[:, :512]
    assert _call(112, kv=wide) == "asm"
    wide_rope = torch.empty((8, 128), dtype=torch.bfloat16)[:, :64]
    assert _call(112, kv_rope=wide_rope) == "asm"


def test_persistent_gets_the_asm_path_tensors(monkeypatch, routes):
    _env(monkeypatch)
    _call(21)
    kind, args, _ = routes[-1]
    assert kind == "persist"
    kv, _kv_indices, kv_indptr, sink, kv_rope, q_packed, q_rope = args
    assert kv.shape == (8, 512) and kv_rope.shape == (8, 64)
    assert kv_indptr.numel() == 22 and q_packed.shape == (21, 128, 512)
    assert sink.numel() == 128 and q_rope.shape == (21, 128, 64)


def test_no_workspace_under_capture_stays_on_asm(monkeypatch, routes):
    _env(monkeypatch)
    monkeypatch.setattr(hca_persist, "workspace_ready", lambda device: False)
    assert _call(896) == "asm"


# ------------------------------------------------------------ the workspace


def _fake_alloc(monkeypatch, capturing):
    made = []
    monkeypatch.setattr(hca_persist, "_workspaces", {})
    monkeypatch.setattr(hca_persist, "mla_decode_fwd_v4_nm_ps", object())
    monkeypatch.setattr(
        hca_persist,
        "get_mla_v4_nm_ps_workspace",
        lambda dev, num_partitions: made.append(dev) or dev,
    )
    monkeypatch.setattr(
        hca_persist.torch.cuda, "is_current_stream_capturing", lambda: capturing
    )
    return made


def test_workspace_ready_allocates_when_eager(monkeypatch):
    made = _fake_alloc(monkeypatch, capturing=False)
    assert hca_persist.workspace_ready("cuda:2")
    assert hca_persist.workspace_ready("cuda:2")
    assert made == [torch.device("cuda", 2)]


def test_workspace_ready_never_allocates_under_capture(monkeypatch):
    made = _fake_alloc(monkeypatch, capturing=True)
    assert not hca_persist.workspace_ready("cuda:2")
    assert made == []
    hca_persist._workspaces[2] = object()  # prepared before capture
    assert hca_persist.workspace_ready("cuda:2")


@pytest.mark.parametrize(
    "enabled,kv_fp8,heads,gfx,expect",
    [
        (True, True, 128, "gfx950", True),
        (False, True, 128, "gfx950", False),
        (True, False, 128, "gfx950", False),
        (True, True, 64, "gfx950", False),
        (True, True, 128, "gfx942", False),
    ],
)
def test_layer_init_prepares_only_when_usable(
    monkeypatch, enabled, kv_fp8, heads, gfx, expect
):
    """Model-construction hook shared by native ATOM and the vLLM / SGLang
    plugins (their builders never call prepare())."""
    monkeypatch.setenv("ATOM_V4_HCA_PERSIST", "1" if enabled else "0")
    made = _fake_alloc(monkeypatch, capturing=False)
    got = hca_persist.prepare_if_usable(
        kv_fp8=kv_fp8, heads=heads, gfx=gfx, device="cuda:1"
    )
    assert got is expect
    assert made == ([torch.device("cuda", 1)] if expect else [])


def test_attention_layer_calls_prepare_for_hca_only():
    """DeepseekV4Attention.__init__ (every serving path) gates on the ratio."""
    import inspect

    from atom.models import deepseek_v4

    src = inspect.getsource(deepseek_v4.DeepseekV4Attention.__init__)
    assert "hca_persist.prepare_if_usable(" in src
    assert "self.compress_ratio == hca_persist.HCA_RATIO" in src


# ---------------------------------------------------------------- the entry


def test_decode_trims_indptr_to_real_rows(monkeypatch):
    """Eager forward with N < T_pad: pass exactly the N rows the ASM would."""
    calls = []
    ws = object()
    monkeypatch.setitem(hca_persist._workspaces, 0, ws)
    monkeypatch.setattr(hca_persist, "_dev_index", lambda d: 0)
    monkeypatch.setattr(
        hca_persist,
        "mla_decode_fwd_v4_nm_ps",
        lambda *a, **k: calls.append((a, k)),
    )
    n, t_pad = 5, 16
    indptr = torch.arange(3, 3 + t_pad + 1, dtype=torch.int32)  # offset CSR
    out = hca_persist.hca_persist_decode(
        torch.empty((4, 512), dtype=torch.uint8),
        torch.zeros(t_pad, dtype=torch.int32),
        indptr,
        torch.zeros(128),
        torch.empty((4, 64), dtype=torch.bfloat16),
        torch.empty((n, 128, 512), dtype=torch.uint8),
        torch.empty((n, 128, 64), dtype=torch.bfloat16),
    )
    assert out.shape == (n, 128, 512) and out.dtype == torch.bfloat16
    ((args, kwargs),) = calls
    assert args[4].tolist() == list(range(3, 3 + n + 1))
    assert args[7] is ws and kwargs["out"] is out


@pytest.mark.parametrize("n", [32769])
def test_decode_enforces_row_limit(n):
    with pytest.raises(ValueError, match="32768"):
        hca_persist.hca_persist_decode(
            None,
            None,
            torch.zeros(n + 1, dtype=torch.int32),
            None,
            None,
            torch.empty((n, 128, 1), dtype=torch.uint8),
            None,
        )


def test_prepare_allocates_once_per_device(monkeypatch):
    made = []
    monkeypatch.setattr(hca_persist, "_workspaces", {})
    monkeypatch.setattr(hca_persist, "mla_decode_fwd_v4_nm_ps", object())
    monkeypatch.setattr(
        hca_persist,
        "get_mla_v4_nm_ps_workspace",
        lambda dev, num_partitions: made.append((dev, num_partitions)) or dev,
    )
    hca_persist.prepare("cuda:3")
    hca_persist.prepare("cuda:3")
    assert made == [(torch.device("cuda", 3), hca_persist.PARTITIONS)]


# ---------------------------------------------------------------- the builder


@pytest.mark.parametrize(
    "kv_fp8,hca_layers,heads,gfx,expect",
    [
        (True, [2], 128, "gfx950", True),
        (False, [2], 128, "gfx950", False),
        (True, [], 128, "gfx950", False),
        (True, [2], 16, "gfx950", False),
        (True, [2], 128, "gfx1250", False),
    ],
)
def test_builder_prepares_only_when_usable(
    monkeypatch, kv_fp8, hca_layers, heads, gfx, expect
):
    from atom.model_ops.attentions import deepseek_v4_attn

    monkeypatch.setenv("ATOM_V4_HCA_PERSIST", "1")
    monkeypatch.setattr(deepseek_v4_attn, "get_gfx", lambda: gfx)
    prepared = []
    monkeypatch.setattr(hca_persist, "prepare", prepared.append)
    builder = SimpleNamespace(
        _kv_fp8=kv_fp8, hca_layers=hca_layers, _local_heads=heads, device="cuda:3"
    )
    deepseek_v4_attn.DeepseekV4AttentionMetadataBuilder._prepare_hca_persist(builder)
    assert prepared == (["cuda:3"] if expect else [])


def test_builder_does_nothing_when_off(monkeypatch):
    from atom.model_ops.attentions import deepseek_v4_attn

    monkeypatch.setenv("ATOM_V4_HCA_PERSIST", "0")
    monkeypatch.setattr(
        hca_persist, "prepare", lambda d: pytest.fail("prepared while off")
    )
    deepseek_v4_attn.DeepseekV4AttentionMetadataBuilder._prepare_hca_persist(
        SimpleNamespace()
    )
