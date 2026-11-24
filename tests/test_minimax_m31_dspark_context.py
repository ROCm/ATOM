# SPDX-License-Identifier: MIT
"""MiniMax-M3.1 DSpark draft context rows: the two opt-in reductions.

* ``ATOM_DSPARK_FC_TP``: ``fc`` column-parallel, all-gathered before
  ``hidden_norm``, gives the replicated ``project_context`` bit for bit.
* ``ATOM_DSPARK_CTX_KV_ONLY``: the context write projects only the K/V rows of
  ``qkv_proj`` and hands the cache write the same K/V as the full projection.

Every TP rank is simulated in one process: each rank's layer is built and
loaded through the real loader under that rank's TP coordinates, and the
all-gather is the concatenation of the ranks' own outputs. The GEMM is a torch
stand-in, so what these show is the sharding, loading, row selection and
gather order; whether the GPU kernel for a narrower N accumulates in the same
order is the GPU A/B's to show.
"""

import contextlib
import importlib
import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.aiter_stub import (
    _ExternalModuleLoader,
    atom_under_stubs,
)

HIDDEN, TAPS, HEADS, KV_HEADS, HEAD_DIM = 256, 3, 8, 4, 32


class _FlydslFinder:
    """`flydsl`, which the M3 model module reaches through its mono kernels,
    from the same stand-in loader as `aiter` / `triton`."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "flydsl" or fullname.startswith("flydsl."):
            return importlib.util.spec_from_loader(
                fullname, _ExternalModuleLoader(), is_package=True
            )
        return None


@contextlib.contextmanager
def _m3_modules():
    try:
        import aiter  # noqa: F401
    except ImportError:
        finder = _FlydslFinder()
        saved = {
            n: m
            for n, m in sys.modules.items()
            if n == "flydsl" or n.startswith("flydsl.")
        }
        sys.meta_path.insert(0, finder)
        try:
            with atom_under_stubs() as load:
                yield load
        finally:
            sys.meta_path.remove(finder)
            for n in list(sys.modules):
                if n == "flydsl" or n.startswith("flydsl."):
                    sys.modules.pop(n, None)
            sys.modules.update(saved)
    else:
        yield importlib.import_module


@pytest.fixture(scope="module")
def mods():
    with _m3_modules() as load:
        yield SimpleNamespace(
            m3=load("atom.models.minimax_m3.dspark"),
            lin=load("atom.model_ops.linear"),
        )


def _mm(x, w, b=None, otype=None, **_):
    return F.linear(x, w, b)


@pytest.fixture
def rank(mods, monkeypatch):
    """Set the simulated TP coordinates every layer built next reads."""
    monkeypatch.setattr(mods.lin, "tgemm", SimpleNamespace(mm=_mm))

    def set_rank(r, tp):
        group = SimpleNamespace(rank_in_group=r, world_size=tp)
        monkeypatch.setattr(mods.lin, "get_tp_group", lambda: group)
        monkeypatch.setattr(mods.m3, "get_tensor_model_parallel_world_size", lambda: tp)

    return set_rank


class _RefGemmaNorm(nn.Module):
    """Gemma RMSNorm over the whole row, ``x * (1 + w)``."""

    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (xf * (1.0 + self.w.float())).to(x.dtype)


def _draft(mods, norm_w):
    hf = SimpleNamespace(
        hidden_size=HIDDEN,
        dspark_noise_token_id=1,
        dspark_target_layer_ids=list(range(TAPS)),
        dspark_markov_rank=8,
        vocab_size=64,
        rms_norm_eps=1e-6,
        num_hidden_layers=0,  # the context projection alone
        sliding_window=0,
    )
    cfg = SimpleNamespace(
        hf_config=hf,
        decode_context_parallel_size=1,
        max_num_seqs=4,
        kv_cache_block_size=16,
    )
    model = mods.m3.MiniMaxM3DSpark(cfg)
    model.hidden_norm = _RefGemmaNorm(norm_w)
    return model


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 7, 33])
def test_fc_tp_matches_replicated(mods, rank, monkeypatch, tp, dtype, rows):
    torch.manual_seed(tp * 100 + rows)
    weight = (torch.randn(HIDDEN, TAPS * HIDDEN) * 0.05).to(dtype)
    norm_w = torch.randn(HIDDEN).to(dtype)
    aux = torch.randn(rows, TAPS * HIDDEN).to(dtype)

    monkeypatch.delenv("ATOM_DSPARK_FC_TP", raising=False)
    rank(0, tp)
    replicated = _draft(mods, norm_w)
    assert not replicated.fc_column_parallel
    replicated.fc.weight.data = replicated.fc.weight.data.to(dtype)
    replicated.fc.weight_loader(replicated.fc.weight, weight)
    expected = replicated.project_context(aux)

    monkeypatch.setenv("ATOM_DSPARK_FC_TP", "1")
    shards = []
    for r in range(tp):
        rank(r, tp)
        model = _draft(mods, norm_w)
        assert model.fc_column_parallel
        assert tuple(model.fc.weight.shape) == (HIDDEN // tp, TAPS * HIDDEN)
        model.fc.weight.data = model.fc.weight.data.to(dtype)
        model.fc.weight_loader(model.fc.weight, weight)
        shards.append(model)
    local = [mods.m3.linear_out(m.fc(aux)) for m in shards]

    for r, model in enumerate(shards):

        def all_gather(x, use_custom=False, dim=-1, r=r):
            assert torch.equal(x, local[r]), "gathers this rank's own columns"
            assert not use_custom and dim == -1
            return torch.cat(local, dim=-1)

        monkeypatch.setattr(mods.m3, "tensor_model_parallel_all_gather", all_gather)
        got = model.project_context(aux)
        assert got.shape == expected.shape
        assert torch.equal(got, expected), f"rank {r}: differs from replicated"


def _attention(mods, weights):
    hf = SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        rotary_dim=HEAD_DIM // 2,
    )
    att = mods.m3.M3DSparkAttention(
        SimpleNamespace(kv_cache_dtype="fp8"), hf, None, layer_num=0
    )
    for shard, w in weights.items():
        att.qkv_proj.weight_loader(att.qkv_proj.weight, w, shard)
    att.kv.k_cache = torch.zeros(1)  # bound: the context write runs
    return att


def _record_context_write(mods, monkeypatch):
    calls = []
    monkeypatch.setattr(mods.m3, "to_page16_shuffle", lambda *a: a)

    def write(q, k, v, positions, **kw):
        calls.append(dict(q=q, k=k, v=v, positions=positions, **kw))
        return q, k

    monkeypatch.setattr(mods.m3, "triton_fused_norm_rope_cache", write)
    return calls


@pytest.mark.parametrize("tp", [4, 8])
def test_ctx_kv_only_writes_the_same_kv(mods, rank, monkeypatch, tp):
    torch.manual_seed(tp)
    weights = {
        "q": torch.randn(HEADS * HEAD_DIM, HIDDEN).bfloat16(),
        "k": torch.randn(KV_HEADS * HEAD_DIM, HIDDEN).bfloat16(),
        "v": torch.randn(KV_HEADS * HEAD_DIM, HIDDEN).bfloat16(),
    }
    ctx = torch.randn(9, HIDDEN).bfloat16()
    positions = torch.arange(9)
    slots = torch.arange(100, 109)
    calls = _record_context_write(mods, monkeypatch)

    for r in range(tp):
        rank(r, tp)
        written = []
        for flag in ("0", "1"):
            monkeypatch.setenv("ATOM_DSPARK_CTX_KV_ONLY", flag)
            att = _attention(mods, weights)
            assert att.context_kv_only == (flag == "1")
            calls.clear()
            att.write_context_kv(ctx, positions, slots)
            (call,) = calls
            assert call["num_heads"] == 0 and call["q"] is call["k"]
            assert call["slot_mapping"] is slots
            written.append((call["k"], call["v"]))
        (k0, v0), (k1, v1) = written
        assert torch.equal(k0, k1) and torch.equal(v0, v1), f"rank {r}"
        # And it is this rank's KV head, not a query head.
        head = r * KV_HEADS // tp
        rows = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
        assert torch.equal(k1, F.linear(ctx, weights["k"][rows]))
        assert torch.equal(v1, F.linear(ctx, weights["v"][rows]))


def test_ctx_kv_only_native_mxfp8_takes_the_kv_scale_rows(mods, rank, monkeypatch):
    """A native MXFP8 [1, 32] ``qkv_proj``: the K/V view must carry the K/V
    rows of the per-row 1x32 scales along with the weight rows."""
    blockscale = importlib.import_module("atom.model_ops.blockscale")

    def native_ref(x, weight, scale, *, weight_group_rows, **_):
        assert weight_group_rows == 1
        w = weight.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, 1)
        return F.linear(x.float(), w).to(x.dtype)

    monkeypatch.setattr(blockscale, "native_quant_linear", native_ref)
    calls = _record_context_write(mods, monkeypatch)
    torch.manual_seed(0)
    weights = {
        "q": torch.randn(HEADS * HEAD_DIM, HIDDEN).bfloat16(),
        "k": torch.randn(KV_HEADS * HEAD_DIM, HIDDEN).bfloat16(),
        "v": torch.randn(KV_HEADS * HEAD_DIM, HIDDEN).bfloat16(),
    }
    ctx = torch.randn(5, HIDDEN).bfloat16()
    rank(1, 4)  # 2 q heads (64 rows), then K and V (32 each): 32-aligned
    written = []
    for flag in ("0", "1"):
        monkeypatch.setenv("ATOM_DSPARK_CTX_KV_ONLY", flag)
        att = _attention(mods, weights)
        proj = att.qkv_proj
        n, k = proj.weight.shape
        # Stand in for the native layout: fp8 codes plus one e8m0 exponent per
        # row and 32 columns, distinct per row so a wrong scale row shows.
        proj.weight = nn.Parameter(
            (proj.weight.data.float() * 4).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        exps = 120 + (torch.arange(n).view(n, 1) + torch.arange(k // 32)) % 11
        proj.weight_scale = nn.Parameter(exps.to(torch.uint8), requires_grad=False)
        proj.native_a8_group_rows = 1
        calls.clear()
        att.write_context_kv(ctx, torch.arange(5), torch.arange(5))
        (call,) = calls
        written.append((call["k"], call["v"]))
    (k0, v0), (k1, v1) = written
    assert torch.equal(k0, k1) and torch.equal(v0, v1)


def test_ctx_kv_only_view_follows_a_rebound_weight(mods, rank, monkeypatch):
    """The cached K/V view is rebuilt when the weight's storage moves (a load
    that rebinds instead of copying), never left reading the old bytes."""
    _record_context_write(mods, monkeypatch)
    monkeypatch.setenv("ATOM_DSPARK_CTX_KV_ONLY", "1")
    rank(0, 4)
    weights = {
        "q": torch.randn(HEADS * HEAD_DIM, HIDDEN).bfloat16(),
        "k": torch.randn(KV_HEADS * HEAD_DIM, HIDDEN).bfloat16(),
        "v": torch.randn(KV_HEADS * HEAD_DIM, HIDDEN).bfloat16(),
    }
    att = _attention(mods, weights)
    first = att._context_kv_proj()
    assert att._context_kv_proj() is first
    assert "_kv_rows" not in dict(att.named_modules())
    att.qkv_proj.weight.data = att.qkv_proj.weight.data.clone() * 2
    second = att._context_kv_proj()
    assert second is not first
    assert torch.equal(second.weight, att.qkv_proj.weight.data[att.q_size :])
