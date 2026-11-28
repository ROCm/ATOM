# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""`shard_gemm_pad` on the real GEMM: a pad it measures on its random probe
rows keeps every rank's vocab slice bit-identical on other rows too.

One GPU stands in for every DP rank: a rank's sharded GEMM is its slice of a
replicated head, so the ranks differ only in which slice they compute. Shapes
are DeepSeek-V4.1's DSpark draft head (dim 5120, V 129280, T 5) over DP 8.
"""

import functools
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("requires a real ROCm GPU", allow_module_level=True)

pytest.importorskip("aiter")

from atom.model_ops import dp_vocab_shard_pad as pad_mod
from atom.model_ops.embed_head import ParallelLMHead

DIM, V, T, DP = 5120, 129280, 5, 8


@pytest.fixture(scope="module")
def head():
    gen = torch.Generator(device="cuda").manual_seed(1)
    weight = (
        torch.randn(V, DIM, generator=gen, device="cuda", dtype=torch.float32) * 0.02
    ).bfloat16()
    head = SimpleNamespace(weight=weight, bias=None, num_embeddings=V)
    head.local_logits = functools.partial(ParallelLMHead.local_logits, head)
    head._vocab_shard_logits = functools.partial(
        ParallelLMHead._vocab_shard_logits, head
    )
    return head


def _agreed_pad(head, rows):
    """`shard_gemm_pad`'s DP verdict, every rank measured on this one GPU."""
    exact = [pad_mod._exact_pads(head, rows, rank, DP) for rank in range(DP)]
    agreed = [all(col) for col in zip(*exact)]
    return next((p for p, ok in zip(pad_mod.PAD_CANDIDATES, agreed) if ok), None)


@pytest.mark.parametrize("batch", [1, 2, 4, 9, 12, 18, 23, 32, 45, 51])
def test_a_measured_pad_is_exact_on_other_rows(head, batch):
    rows = batch * T
    pad = _agreed_pad(head, rows)
    if pad is None:
        pytest.skip(f"no candidate pad is exact at rows={rows} on this GEMM build")
    gen = torch.Generator(device="cuda").manual_seed(1000 + batch)
    x = torch.randn(DP * rows, DIM, generator=gen, device="cuda").bfloat16()
    xp = torch.cat([x, x.new_zeros(pad, DIM)])
    vs = V // DP
    for rank in range(DP):
        got = head._vocab_shard_logits(xp, rank, DP)[: DP * rows]
        ref = torch.cat(
            [
                head.local_logits(x[s * rows : (s + 1) * rows])[
                    :, rank * vs : (rank + 1) * vs
                ]
                for s in range(DP)
            ]
        )
        assert torch.equal(got, ref), f"rank {rank} differs at pad {pad}"


class _ThreadGroup:
    """DP ranks as threads on this one GPU: an all-gather through a barrier.
    Every thread launches on the same stream, so a rank's copy is ordered
    before any peer's read of it."""

    def __init__(self, world):
        import threading

        self.world = world
        self.slots = [None] * world
        self.barrier = threading.Barrier(world)

    def rank(self, rank):
        shared = self

        class Group:
            world_size = shared.world
            rank_in_group = rank

            def all_gather(self, t, dim=0, use_custom=True):
                shared.barrier.wait()
                shared.slots[rank] = t.clone()
                shared.barrier.wait()
                out = torch.cat(shared.slots, dim)
                shared.barrier.wait()
                return out

        return Group()


@pytest.mark.parametrize("batch", [1, 4, 9, 23, 45, 51])
def test_the_dp_block_drafts_the_replicated_heads_ids(head, monkeypatch, batch):
    """Every DP rank's sharded block (real GEMM at the measured pad, real
    fused Markov pick over the ranks) gives the ids and W1 rows its own
    replicated head and `dspark_markov_argmax` give."""
    import threading

    from atom.models.deepseek_v4_dspark import DSparkMarkovHead
    from atom.models.deepseek_v41 import dspark

    cls = dspark.DeepseekV41DSpark
    rows, r = batch * T, 64
    pad = _agreed_pad(head, rows)
    if pad is None:
        pytest.skip(f"no candidate pad is exact at rows={rows} on this GEMM build")
    gen = torch.Generator(device="cuda").manual_seed(7 + batch)
    markov = DSparkMarkovHead(V, r).cuda().bfloat16()
    markov.fused_sample = True
    for w in (markov.markov_w1.weight, markov.markov_w2.weight):
        w.data.normal_(0, 0.05, generator=gen)
    seen = {}

    def confidence(hidden, embeds):
        seen.setdefault(threading.get_ident(), []).append(embeds.clone())
        return hidden.float().sum(-1)

    # The replicated reference reads the whole head through `get_logits`.
    head.get_logits = head.local_logits
    model = SimpleNamespace(
        head=head, mtp=[SimpleNamespace(markov_head=markov, confidence_head=confidence)]
    )
    model.forward_head = functools.partial(cls.forward_head, model)
    inputs = [
        (
            torch.randn(rows, DIM, generator=gen, device="cuda").bfloat16(),
            torch.randn(batch, T, DIM, generator=gen, device="cuda").bfloat16(),
            torch.randint(0, V, (batch,), generator=gen, device="cuda").int(),
        )
        for _ in range(DP)
    ]
    group, local = _ThreadGroup(DP), threading.local()
    monkeypatch.setattr(dspark, "get_dp_group", lambda: local.group)
    got, errors = [None] * DP, []

    def rank_main(rank):
        local.group = group.rank(rank)
        try:
            ids, _ = cls._head_and_sample_dp(model, *inputs[rank], pad)
            got[rank] = (ids, seen[threading.get_ident()][-1])
        except Exception as e:  # noqa: BLE001 -- surfaced below
            errors.append(e)
            group.barrier.abort()

    threads = [threading.Thread(target=rank_main, args=(i,)) for i in range(DP)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for rank in range(DP):
        ref_ids, _ = cls._head_and_sample(model, *inputs[rank])
        ref_embeds = seen[threading.get_ident()][-1]
        ids, embeds = got[rank]
        assert torch.equal(ids, ref_ids.to(ids.dtype)), f"rank {rank}"
        assert torch.equal(embeds, ref_embeds), f"rank {rank}"
