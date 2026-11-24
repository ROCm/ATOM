# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 indexer CP: the lean shard selection (ATOM_M3_CP_SELECT_LEAN)
returns the full top-k's keys bit for bit, through the shard pass and through
the one-launch mailbox exchange.

Runs natively on a GPU, or on the CPU under ``TRITON_INTERPRET=1`` (the four
ranks of the exchange then run one after another against host mailboxes)."""

import os

import numpy as np
import pytest
import torch

triton = pytest.importorskip("triton", reason="requires Triton")

DEV = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(
    DEV == "cpu" and os.environ.get("TRITON_INTERPRET") != "1",
    reason="requires a GPU or TRITON_INTERPRET=1",
)

WORLD, QLEN, TOPK = 4, 8, 16


@pytest.fixture(scope="module")
def ex():
    import importlib

    from aiter_stub import stubbed_aiter

    mp = pytest.MonkeyPatch()
    if DEV == "cpu":
        # The interpreter makes a loop bound from a 1-element array, which
        # NumPy >= 2.4 no longer converts with int(): read its one element.
        from triton.runtime import interpreter

        patch_lang = interpreter._patch_lang

        def _patch_lang(fn):
            scope = patch_lang(fn)
            scope.set_attr(
                triton.language.core.tensor,
                "__index__",
                lambda self: int(np.asarray(self.handle.data).reshape(-1)[0]),
            )
            return scope

        mp.setattr(interpreter, "_patch_lang", _patch_lang)
        # ... and evaluates the exchange's `1 << _TAG_SHIFT` in Python.
        constexpr = triton.language.core.constexpr
        mp.setattr(
            constexpr,
            "__rlshift__",
            lambda self, other: constexpr(other << self.value),
            raising=False,
        )
    # Real Triton (native or interpreted); only aiter's arch probe is stubbed.
    with stubbed_aiter("aiter.jit", "aiter.jit.utils", "aiter.jit.utils.chip_info"):
        yield importlib.import_module(
            "atom.model_ops.minimax_m3.indexer_candidate_exchange"
        )
    mp.undo()


def _scores(kind, heads, tokens, local, seed):
    g = np.random.default_rng(seed)
    if kind == "iid":
        s = g.standard_normal((heads, tokens, local))
    elif kind == "ties":  # coarse values: many equal scores, the id decides
        s = np.round(g.standard_normal((heads, tokens, local)) * 2) / 2
    elif kind == "const":
        s = np.zeros((heads, tokens, local))
    elif kind == "crowded":  # every winner in one prefilter group: the fallback
        s = g.standard_normal((heads, tokens, local))
        s[..., ::32] += 10
    elif kind == "special":  # NaN, +-inf, -0.0
        s = g.standard_normal((heads, tokens, local))
        for v in (np.nan, np.inf, -np.inf, -0.0, 0.0):
            s[g.random(s.shape) < 0.03] = v
    return torch.tensor(s, dtype=torch.float32, device=DEV)


def _key(score, block):
    """`_pack_score_key` in numpy (score fp32 array, 1-based ids)."""
    bits = score.astype(np.float32).view(np.uint32).astype(np.uint64)
    bits[bits == 0x80000000] = 0
    ordered = np.where(bits >> 31 != 0, bits ^ 0xFFFFFFFF, bits ^ 0x80000000)
    ordered[(bits & 0x7FFFFFFF) > 0x7F800000] = 0
    return ((1 << 48) | (ordered << 16) | block.astype(np.uint64)).astype(np.int64)


def _reference_keys(scores, lengths, rank, global_blocks, init, keep):
    """Each (head, row)'s shard top-k keys, sorted, as `_local_topk` stores them."""
    heads, tokens, local = scores.shape
    out = np.zeros((heads, tokens, TOPK), dtype=np.int64)
    s = scores.cpu().numpy()
    for row in range(tokens):
        length = int(lengths[row // QLEN])
        causal = (length - QLEN + row % QLEN + 128) // 128
        block = np.arange(local) * WORLD + rank
        valid = (block < min(global_blocks, causal)) & (np.arange(local) < local)
        for h in range(heads):
            v = s[h, row].astype(np.float32).copy()
            v[valid & (block < init)] = 1e30
            v[valid & (block >= max(0, causal - keep))] = 1e29
            k = np.where(valid, _key(v, block + 1), 0)
            out[h, row] = np.sort(k)[::-1][:TOPK]
    return out


@pytest.mark.parametrize("kind", ["iid", "ties", "const", "crowded", "special"])
@pytest.mark.parametrize(
    "rank,context,init,keep",
    [(0, 150_000, 1, 4), (3, 150_000, 0, 0), (1, 300_000, 1, 4), (2, 9_000, 1, 4)],
)
def test_lean_shard_keys_match_full_topk(ex, kind, rank, context, init, keep):
    # A 1M-context graph: 2048 local blocks, 512-wide tiles; 300K spans 2 tiles.
    global_blocks = triton.cdiv(1 << 20, 128)
    local = triton.cdiv(global_blocks, WORLD)
    batch = 2
    lengths = torch.tensor([context, context - 4321], dtype=torch.int32, device=DEV)
    scores = _scores(kind, WORLD, batch * QLEN, local, seed=rank + context)
    got = {
        lean: ex.local_candidate_keys(
            scores,
            lengths,
            TOPK,
            rank,
            WORLD,
            QLEN,
            global_blocks,
            init,
            keep,
            lean=lean,
        )
        .cpu()
        .numpy()
        for lean in (False, True)
    }
    want = _reference_keys(scores, lengths.cpu(), rank, global_blocks, init, keep)
    np.testing.assert_array_equal(got[False], want)
    np.testing.assert_array_equal(got[True], want)


class _Mailbox:
    """`indexer_cp.CandidateMailbox` on one device: rank r's mailbox is
    ``boxes[r]``, every rank addresses all four."""

    MAX_TOPK = 64
    max_rows = 64

    def __init__(self, boxes, max_polls):
        self.MAX_POLLS = max_polls
        self.peers = torch.tensor(
            [b.data_ptr() for b in boxes], dtype=torch.int64, device=DEV
        )
        self.send_calls = torch.zeros(
            WORLD * self.max_rows, dtype=torch.int32, device=DEV
        )
        self.recv_calls = torch.zeros(self.max_rows, dtype=torch.int32, device=DEV)
        self.timeouts = torch.zeros(1, dtype=torch.int32, device=DEV)

    def fits(self, tokens, topk):
        return tokens <= self.max_rows and topk <= self.MAX_TOPK


def _exchange_rounds(ex, monkeypatch, lean, shard_scores, lengths, table, init, keep):
    """Round j runs every rank once, rank j last, so rank j's merge finds all
    four shards' keys (the others time out on the later ranks' and are not
    read). Returns rank j's (indices, sparse_bt, sparse_ctx) of round j."""
    monkeypatch.setattr(ex, "_prewarm_exchange", lambda *a, **k: None)
    if DEV == "cpu":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    boxes = [
        torch.zeros(2 * WORLD * _Mailbox.max_rows * 64, dtype=torch.int64, device=DEV)
        for _ in range(WORLD)
    ]
    # Ranks wait on peers only in the interpreter, where they run in turn.
    polls = 1 if DEV == "cpu" else 20_000_000
    mailboxes = [_Mailbox(boxes, polls) for _ in range(WORLD)]
    results = []
    for j in range(WORLD):
        for r in [r for r in range(WORLD) if r != j] + [j]:
            out = ex.select_exchanged_candidates(
                shard_scores[r],
                lengths,
                table,
                TOPK,
                r,
                WORLD,
                QLEN,
                init,
                keep,
                mailboxes[r],
                lean=lean,
            )
        results.append([t.cpu() for t in out])
    return results


@pytest.mark.skipif(DEV != "cpu", reason="sequential ranks need the interpreter")
@pytest.mark.parametrize("kind", ["iid", "ties", "crowded"])
def test_lean_exchange_is_bit_identical(ex, monkeypatch, kind):
    batch, context = 2, 150_000
    global_blocks = triton.cdiv(1 << 20, 128)
    local = triton.cdiv(global_blocks, WORLD)
    lengths = torch.tensor([context, context - 777], dtype=torch.int32, device=DEV)
    table = torch.randperm(
        batch * global_blocks, generator=torch.Generator().manual_seed(0)
    )
    table = table.to(torch.int32).view(batch, global_blocks).to(DEV)
    shard_scores = [
        _scores(kind, WORLD, batch * QLEN, local, seed=10 + r) for r in range(WORLD)
    ]
    base = _exchange_rounds(ex, monkeypatch, False, shard_scores, lengths, table, 1, 4)
    lean = _exchange_rounds(ex, monkeypatch, True, shard_scores, lengths, table, 1, 4)
    for j in range(WORLD):
        for a, b in zip(base[j], lean[j]):
            assert torch.equal(a, b)
        # and the global top-k of head j: all shards' blocks, forced pins
        full = torch.zeros((batch * QLEN, local * WORLD))
        for r in range(WORLD):
            full[:, r::WORLD] = shard_scores[r][j].cpu()
        idx = base[j][0][0]
        for row in range(batch * QLEN):
            length = int(lengths[row // QLEN])
            causal = (length - QLEN + row % QLEN + 128) // 128
            v = full[row, :causal].numpy().astype(np.float32).copy()
            v[:1] = 1e30
            v[max(0, causal - 4) :] = 1e29
            k = _key(v, np.arange(causal) + 1)
            top = (np.sort(k)[::-1][:TOPK] & 0xFFFF) - 1
            np.testing.assert_array_equal(idx[row].numpy(), top)
