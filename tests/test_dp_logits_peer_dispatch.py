# SPDX-License-Identifier: MIT
"""CPU integration coverage for head dispatch; GPU transport has its own test."""

from types import SimpleNamespace as NS

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield NS(
            head=load("atom.model_ops.embed_head"),
            exchange=load("atom.distributed.dp_logits_exchange"),
            runner=load("atom.model_engine.model_runner"),
            indexer_cp=load("atom.distributed.indexer_cp"),
        )


def head(modules, *, vocab=256, tp=1, dtype=torch.bfloat16):
    h = modules.head.ParallelLMHead.__new__(modules.head.ParallelLMHead)
    torch.nn.Module.__init__(h)
    h.tp_size, h.num_embeddings = tp, vocab
    h.weight = torch.nn.Parameter(
        ((torch.arange(vocab * 8).reshape(vocab, 8) % 13 - 6) / 16).to(dtype),
        requires_grad=False,
    )
    h.bias = torch.nn.Parameter(
        (torch.arange(vocab) % 5).to(dtype), requires_grad=False
    )
    return h


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("local_rows", [0, 1, 3, 4])
@pytest.mark.parametrize("path", ["peer", "capacity_fallback", "native", "all2all"])
def test_target_head_keeps_only_own_rows(modules, monkeypatch, rank, local_rows, path):
    mod = modules.head
    h = head(modules)
    max_rows = 4
    x = ((torch.arange(local_rows * 8).reshape(local_rows, 8) % 7 - 3) / 8).to(
        torch.bfloat16
    )
    padded = torch.cat([x, torch.zeros(max_rows - local_rows, 8, dtype=x.dtype)])
    gathered = torch.cat([padded + i for i in range(4)])
    shards = [
        gathered @ h.weight[i * 64 : (i + 1) * 64].t() + h.bias[i * 64 : (i + 1) * 64]
        for i in range(4)
    ]
    expected = torch.cat(shards, 1)[rank * max_rows : rank * max_rows + local_rows]
    calls = []

    def gather(data, dim, use_custom):
        if dim == 0:
            torch.testing.assert_close(data, padded, rtol=0, atol=0)
            calls.append("hidden")
            return gathered
        torch.testing.assert_close(data, shards[rank], rtol=0, atol=0)
        calls.append("logits")
        return torch.cat(shards, 1)

    group = NS(
        world_size=4, rank_in_group=rank, all_gather=gather, device_group=object()
    )
    monkeypatch.setattr(mod, "get_dp_group", lambda: group)
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: NS(context=NS(running_tokens=max_rows))
    )
    monkeypatch.setattr(mod.envs, "ATOM_USE_CUSTOM_ALL_GATHER", True)

    def mm(x, w, b, out=None):
        result = x @ w.t() + b
        return result if out is None else out.copy_(result)

    monkeypatch.setattr(mod, "tgemm", NS(mm=mm))

    class Peer:
        max_rows = 4 if path != "capacity_fallback" else 3

        def produce(self, fill, rows, own_rows):
            # The head's GEMM writes this rank's shard into the pool slot.
            slot = torch.full((4 * rows, 64), float("nan"), dtype=torch.bfloat16)
            fill(slot)
            torch.testing.assert_close(slot, shards[rank], rtol=0, atol=0)
            assert (rows, own_rows) == (max_rows, local_rows)
            calls.append("peer")
            return expected.clone()

    if path != "native":
        h._dp_logits_exchange = Peer()

    def all2all(out, data, *, group):
        torch.testing.assert_close(data, shards[rank].flatten(), rtol=0, atol=0)
        calls.append("all2all")
        incoming = torch.stack(
            [s[rank * max_rows : (rank + 1) * max_rows] for s in shards]
        )
        out.copy_(incoming.flatten())

    monkeypatch.setattr(torch.distributed, "all_to_all_single", all2all)
    result = h._dp_sharded_logits(x, "all2all" if path == "all2all" else "allgather")
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.shape == (local_rows, 256)
    assert calls == [
        "hidden",
        "peer" if path == "peer" else "all2all" if path == "all2all" else "logits",
    ]


@pytest.mark.parametrize(
    "unsupported",
    ["mode", "tp", "dp", "dtype", "vocab", "no_comm", "disabled", "arch", "node"],
)
def test_unsupported_pool_keeps_native_head(modules, monkeypatch, unsupported):
    mod = modules.head
    h = head(
        modules,
        vocab=255 if unsupported == "vocab" else 256,
        tp=2 if unsupported == "tp" else 1,
        dtype=torch.float32 if unsupported == "dtype" else torch.bfloat16,
    )
    comm = NS(
        disabled=unsupported == "disabled",
        _is_gfx1250=unsupported != "arch",
        _same_node=unsupported != "node",
    )
    group = NS(
        world_size=2 if unsupported == "dp" else 4,
        device_communicator=NS(ca_comm=None if unsupported == "no_comm" else comm),
    )
    monkeypatch.setattr(mod, "get_dp_group", lambda: group)
    monkeypatch.setattr(
        mod.envs,
        "ATOM_DP_LM_HEAD_MODE",
        "all2all" if unsupported == "mode" else "allgather",
    )

    def allocate(*args, **kwargs):
        pytest.fail("unsupported head attempted to allocate peer memory")

    monkeypatch.setattr(modules.exchange, "PeerLogitsExchange", allocate)
    h.prepare_dp_logits_exchange([1, 4], "test_head")
    assert h.__dict__.get("_dp_logits_exchange") is None


def test_pool_prepares_all_row_buckets_before_capture(modules, monkeypatch):
    mod = modules.head
    h = head(modules)
    comm = NS(disabled=False, _is_gfx1250=True, _same_node=True, _use_vmm=False)
    group = NS(
        world_size=4,
        rank_in_group=2,
        cpu_group=object(),
        device_communicator=NS(ca_comm=comm),
    )
    monkeypatch.setattr(mod, "get_dp_group", lambda: group)
    monkeypatch.setattr(mod.envs, "ATOM_DP_LM_HEAD_MODE", "allgather")
    calls = []

    class Peer:
        def __init__(self, pool_group, device, max_rows, vocab_shard, **kwargs):
            assert pool_group is group.cpu_group
            assert device == h.weight.device
            assert (max_rows, vocab_shard) == (3, 64)
            assert kwargs == {
                "dtype": torch.bfloat16,
                "barrier_communicator": comm,
                "namespace": "test_head",
            }
            self.max_rows = max_rows

        def produce(self, fill, rows, own_rows):
            assert own_rows == rows
            slot = torch.zeros((4 * rows, 64), dtype=h.weight.dtype)
            fill(slot)
            # Warmup is transport only: nothing is written into the pool.
            assert not slot.any()
            calls.append(rows)

    monkeypatch.setattr(modules.exchange, "PeerLogitsExchange", Peer)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    h.prepare_dp_logits_exchange([3, 1, 2, 3], "test_head")
    assert calls == [3, 2, 1, "sync"]
    with pytest.raises(RuntimeError, match="already initialized"):
        h.prepare_dp_logits_exchange([1, 3], "test_head")


@pytest.mark.parametrize(
    "blocked",
    [
        "none",
        "context",
        "draft",
        "prefill",
        "ragged",
        "metadata",
        "tp",
        "dp",
        "vocab",
        "mode",
    ],
)
def test_dp_target_gate_respects_model_step_metadata(modules, monkeypatch, blocked):
    mod = modules.head
    h = head(
        modules,
        tp=2 if blocked == "tp" else 1,
        vocab=255 if blocked == "vocab" else 256,
    )
    context = NS(
        is_draft=blocked == "draft",
        is_prefill=blocked == "prefill",
        running_tokens_are_unified=blocked != "ragged",
    )
    if blocked == "context":
        context = None
    monkeypatch.setattr(
        mod, "get_dp_group", lambda: NS(world_size=1 if blocked == "dp" else 4)
    )
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: NS(dp_metadata=None if blocked == "metadata" else object()),
    )
    monkeypatch.setattr(
        mod.envs, "ATOM_DP_LM_HEAD_MODE", "off" if blocked == "mode" else "allgather"
    )
    assert h._can_use_dp_sharded_head(context) is (blocked == "none")


@pytest.mark.parametrize("value, enabled", [(None, False), ("0", False), ("1", True)])
def test_peer_flag_defaults_off(modules, monkeypatch, value, enabled):
    if value is None:
        monkeypatch.delenv("ATOM_DP_LOGITS_PEER_EXCHANGE", raising=False)
    else:
        monkeypatch.setenv("ATOM_DP_LOGITS_PEER_EXCHANGE", value)
    assert modules.head.envs.ATOM_DP_LOGITS_PEER_EXCHANGE is enabled


@pytest.mark.parametrize("enabled, tbo", [(True, True), (True, False), (False, True)])
def test_runner_prepares_reachable_verification_rows_and_rejects_tbo(
    modules, monkeypatch, enabled, tbo
):
    mod = modules.runner
    runner = mod.ModelRunner.__new__(mod.ModelRunner)
    runner.device = torch.device("cpu")
    runner._tr_hybrid_active = lambda: False
    runner.config = NS(
        capture_sizes=[64, 4, 2],
        max_num_seqs=4,
        max_num_batched_tokens=24,
        enable_tbo_decode=tbo,
    )
    runner.drafter = NS(mtp_k=5)
    calls = []
    heads = [head(modules), head(modules)]
    for h in heads:
        h.prepare_dp_logits_exchange = lambda rows, name: calls.append((rows, name))
    runner.model = NS(named_modules=lambda: zip(["target.head", "another.head"], heads))
    monkeypatch.setattr(modules.indexer_cp, "warmup_exchange", lambda device: None)
    monkeypatch.setattr(mod.envs, "ATOM_DP_LOGITS_PEER_EXCHANGE", enabled)

    class StopBeforeCapture(dict):
        def __contains__(self, key):
            # Pool preparation is done here; stop before any graph/GPU work.
            raise RuntimeError("stop before graph capture")

    runner.forward_vars = StopBeforeCapture()
    if enabled and tbo:
        with pytest.raises(ValueError, match="serial decode streams"):
            runner.capture_cudagraph()
        assert calls == []
    else:
        with pytest.raises(RuntimeError, match="stop before graph capture"):
            runner.capture_cudagraph()
        if enabled:
            assert len(calls) == 2
            assert [rows for rows, name in calls] == [[24, 12], [24, 12]]
            assert calls[0][1] != calls[1][1]
        else:
            assert calls == []
