# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MiniMax-M3.1 TP1 q/k norm (atom/model_ops/minimax_m3/tp1_qknorm.py): which steps
aiter's qknorm_v3 takes, what it is handed, and (gfx950) the bytes it writes."""

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from atom.model_ops.minimax_m3 import tp1_qknorm as T

NQ, NKV, NIQ, D = 64, 4, 4, 128
WIDTH = (NQ + 2 * NKV + NIQ + 1) * D


@pytest.fixture
def aiter_op(monkeypatch):
    """Installs a fake aiter whose m3_tp1_qknorm is supported (True), not (False) or
    absent (None); returns the qknorm_v3 calls it records."""

    def install(supported):
        aiter, ops = ModuleType("aiter"), ModuleType("aiter.ops")
        aiter.ops = ops
        monkeypatch.setitem(sys.modules, "aiter", aiter)
        monkeypatch.setitem(sys.modules, "aiter.ops", ops)
        calls = []
        op = None
        if supported is not None:
            op = ModuleType("aiter.ops.m3_tp1_qknorm")
            op.is_supported = lambda: supported
            op.qknorm_v3 = lambda **kw: calls.append(kw)
            ops.m3_tp1_qknorm = op
        monkeypatch.setitem(sys.modules, "aiter.ops.m3_tp1_qknorm", op)
        T._qknorm_v3.cache_clear()
        return calls

    yield install
    T._qknorm_v3.cache_clear()


def tp1_impl(**changes):
    attrs = {
        "num_heads": NQ,
        "num_kv_heads": NKV,
        "num_idx_heads": NIQ,
        "head_dim": D,
        "index_head_dim": D,
        "rotary_emb": SimpleNamespace(rotary_dim=64),
        "kv_cache_dtype": "fp8",
        "index_cache_dtype": "fp8",
        "skip_index_topk": False,
    }
    attrs.update(changes)
    return SimpleNamespace(**attrs)


def qkv_of(tokens, dtype=torch.bfloat16):
    return torch.empty(tokens, WIDTH, dtype=dtype, device="meta")


@pytest.mark.parametrize("tokens", [640, 8192])
def test_the_tp1_layout_takes_qknorm_v3_from_640_tokens(aiter_op, tokens):
    aiter_op(True)
    assert T.serves(tp1_impl(), qkv_of(tokens))
    assert not T.serves(tp1_impl(), qkv_of(639))  # shorter steps keep aiter's kernel


@pytest.mark.parametrize(
    "changes",
    [
        {"num_heads": 32, "num_kv_heads": 2, "num_idx_heads": 2},  # TP2
        {"num_idx_heads": 1},
        {"head_dim": 64},
        {"index_head_dim": 64},
        {"rotary_emb": SimpleNamespace(rotary_dim=128)},
        {"kv_cache_dtype": "bf16"},
        {"kv_cache_dtype": "fp4"},
        {"index_cache_dtype": "bf16"},
        {"index_cache_dtype": "fp4"},
        {"skip_index_topk": True},
    ],
)
def test_any_other_layout_keeps_fused_qknorm_idxrqknorm(aiter_op, changes):
    aiter_op(True)
    assert not T.serves(tp1_impl(**changes), qkv_of(8192))


def test_an_fp16_qkv_keeps_fused_qknorm_idxrqknorm(aiter_op):
    aiter_op(True)
    assert not T.serves(tp1_impl(), qkv_of(8192, torch.float16))


@pytest.mark.parametrize("supported", [None, False])  # no op; not gfx950
def test_without_the_op_it_falls_back_and_says_so_once(aiter_op, caplog, supported):
    aiter_op(supported)
    with caplog.at_level(logging.INFO, logger="atom"):
        assert not T.serves(tp1_impl(), qkv_of(8192))
        assert not T.serves(tp1_impl(), qkv_of(8192))
    assert sum("qknorm_v3 is unavailable" in r.message for r in caplog.records) == 1


def test_run_hands_qknorm_v3_the_layer_and_the_step(aiter_op):
    calls = aiter_op(True)
    norm = [SimpleNamespace(weight=object(), variance_epsilon=1e-6) for _ in range(4)]
    impl = tp1_impl(q_norm=norm[0], k_norm=norm[1], index_cache=object())
    impl.index_q_norm, impl.index_k_norm = norm[2], norm[3]
    k_cache = torch.empty(64, NKV, D // 16, 16, 16, device="meta")
    step = {
        n: object() for n in ("qkv", "q", "iq", "cs", "pos", "slots", "v", "ks", "vs")
    }
    kv = (k_cache, step["v"], step["ks"], step["vs"])
    args = ("qkv", "q", "iq", "cs", "pos", "slots")
    T.run(impl, *(step[n] for n in args), kv)

    assert calls == [
        {
            "qkv": step["qkv"],
            "q_out": step["q"],
            "index_q_out": step["iq"],
            "q_norm_weight": norm[0].weight,
            "k_norm_weight": norm[1].weight,
            "index_q_norm_weight": norm[2].weight,
            "index_k_norm_weight": norm[3].weight,
            "cos_sin_cache": step["cs"],
            "positions": step["pos"],
            "slot_mapping": step["slots"],
            "index_slot_mapping": step["slots"],
            "k_cache": k_cache,
            "v_cache": step["v"],
            "index_cache": impl.index_cache,
            "k_scale": step["ks"],
            "v_scale": step["vs"],
            "eps": 1e-6,
            "block_size": 16,
        }
    ]


def _gfx950_op() -> bool:
    if not torch.cuda.is_available():
        return False
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return False
    try:
        from aiter.ops import m3_tp1_qknorm
    except ImportError:
        return False
    return m3_tp1_qknorm.is_supported()


DEV = "cuda"
BLOCKS = 160  # page-128 blocks; a step's sequences take scattered ones
MAX_POS = 1 << 17


def make_impl():
    from atom.model_ops.attention_mha import SparseMHAPagedAttentionImpl

    impl = object.__new__(SparseMHAPagedAttentionImpl)
    torch.nn.Module.__init__(impl)
    g = torch.Generator(device=DEV).manual_seed(0)

    def norm():
        w = torch.randn(D, device=DEV, generator=g) * 0.2
        return SimpleNamespace(weight=w.to(torch.bfloat16), variance_epsilon=1e-6)

    inv = 1e7 ** (-torch.arange(0, 64, 2, device=DEV, dtype=torch.float32) / 64)
    ang = torch.arange(MAX_POS, device=DEV, dtype=torch.float32)[:, None] * inv[None]
    rope = torch.nn.Module()
    rope.register_buffer("cos_cache", ang.cos().to(torch.bfloat16)[:, None, None, :])
    rope.register_buffer("sin_cache", ang.sin().to(torch.bfloat16)[:, None, None, :])
    rope.rotary_dim = 64
    impl.rotary_emb = rope
    impl.q_norm, impl.k_norm = norm(), norm()
    impl.index_q_norm, impl.index_k_norm = norm(), norm()
    impl.num_heads, impl.num_kv_heads, impl.head_dim = NQ, NKV, D
    impl.num_idx_heads, impl.index_q_size, impl.index_head_dim = NIQ, NIQ * D, D
    impl.skip_index_topk = False
    impl.kv_cache_dtype = impl.index_cache_dtype = "fp8"
    impl.layer_num = 0
    impl._index_q = impl._index_q_cache_key_info = None
    return impl


def ragged_step(tokens: int, seed: int):
    """Four sequences of a prefill chunk at uneven contexts (page-aligned and not),
    in scattered blocks, then three padding rows (slot -1)."""
    g = torch.Generator().manual_seed(seed)
    real = tokens - 3
    cuts = sorted(torch.randperm(real - 1, generator=g)[:3].add(1).tolist())
    lens = [b - a for a, b in zip([0, *cuts], [*cuts, real])]
    starts = [0, 16 * 37, 1001, 65536 + 77]
    blocks = iter(torch.randperm(BLOCKS, generator=g).tolist())
    pos, slots = [], []
    for start, n in zip(starts, lens):
        p = torch.arange(start, start + n)
        table = {b: next(blocks) for b in (p // 128).unique().tolist()}
        pos.append(p)
        slots.append(
            torch.tensor([table[b] for b in (p // 128).tolist()]) * 128 + p % 128
        )
    pos.append(torch.zeros(3, dtype=torch.int64))
    slots.append(torch.full((3,), -1))
    qkv = torch.randn(tokens, WIDTH, generator=g) * 2
    qkv.view(-1)[torch.randperm(qkv.numel(), generator=g)[:4096]] *= 64  # outliers
    to = {"device": DEV}
    return (
        qkv.to(torch.bfloat16).to(**to),
        torch.cat(pos).to(**to),
        torch.cat(slots).to(**to),
    )


def fresh_caches(seed: int):
    """Page-128 SHUFFLE FP8 K / V with per-token scales and the FP8 index cache, as
    the runner binds them, filled with noise that the writes must leave alone."""
    from atom.config import kv_cache_torch_dtype

    g = torch.Generator(device=DEV).manual_seed(seed)

    def noise(*shape, dtype=None):
        raw = torch.randint(0, 256, shape, generator=g, device=DEV, dtype=torch.uint8)
        return raw.view(dtype or kv_cache_torch_dtype("fp8"))

    k = noise(BLOCKS, NKV, D // 16, 128, 16)
    v = noise(BLOCKS, NKV, 128 // 16, D, 16)
    scales = torch.rand(2, BLOCKS, NKV, 128, generator=g, device=DEV)
    return k, v, scales[0], scales[1], noise(BLOCKS, 128, D)


def rope_cache(impl, qkv, pos, slots, caches):
    k, v, ks, vs, index = caches
    impl.index_cache = index
    layer = SimpleNamespace(k_cache=k, v_cache=v, k_scale=ks, v_scale=vs)
    ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(slot_mapping=slots),
        kv_cache_data={"layer_0": layer},
    )
    q = impl.rope_cache(None, None, None, qkv, pos, ctx)[0]
    return [q, impl._index_q, *caches]


@pytest.mark.skipif(not _gfx950_op(), reason="needs gfx950 and aiter m3_tp1_qknorm")
@pytest.mark.parametrize("tokens", [640, 1000, 3001, 8192])
def test_rope_cache_writes_the_same_bytes_through_qknorm_v3(monkeypatch, tokens):
    T._qknorm_v3.cache_clear()
    impl = make_impl()
    qkv, pos, slots = ragged_step(tokens, seed=tokens)
    with monkeypatch.context() as m:
        m.setattr(T, "serves", lambda impl, qkv: False)
        want = rope_cache(impl, qkv, pos, slots, fresh_caches(7))
    ran = []
    real_run = T.run
    monkeypatch.setattr(T, "run", lambda *a: ran.append(1) or real_run(*a))
    got = rope_cache(impl, qkv, pos, slots, fresh_caches(7))
    torch.cuda.synchronize()

    assert ran == [1]
    names = ["q", "index_q", "k_cache", "v_cache", "k_scale", "v_scale", "index_cache"]
    for name, a, b in zip(names, got, want):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name
