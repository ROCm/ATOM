# SPDX-License-Identifier: MIT
# Core pipeline-parallel unit tests (GPU-free).

import argparse
import os
import sys
import tempfile
import types
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import zmq

# pp_comm imports aiter.dist.parallel_state and aiter.ops.communication, which
# need a GPU build; stub them so the wrapper logic can be tested on CPU.
sys.modules.setdefault("aiter", types.ModuleType("aiter"))
sys.modules.setdefault("aiter.ops", types.ModuleType("aiter.ops"))
sys.modules.setdefault("aiter.dist", types.ModuleType("aiter.dist"))

_ps_stub = types.ModuleType("aiter.dist.parallel_state")
for _fn in (
    "get_pp_group",
    "get_tp_group",
    "init_distributed_environment",
    "ensure_model_parallel_initialized",
    "_split_tensor_dict",
):
    setattr(_ps_stub, _fn, MagicMock())
sys.modules.setdefault("aiter.dist.parallel_state", _ps_stub)

_comm_stub = types.ModuleType("aiter.ops.communication")
_comm_stub.set_custom_all_reduce = lambda *a, **k: None
sys.modules.setdefault("aiter.ops.communication", _comm_stub)

from atom.distributed import pp_comm  # noqa: E402
from atom.distributed.pp_transport import PPStageTransport  # noqa: E402
from atom.model_engine.arg_utils import EngineArgs  # noqa: E402
from atom.model_engine.scheduler import (  # noqa: E402
    ScheduledBatch,
    ScheduledBatchOutput,
    Scheduler,
)
from atom.model_engine.sequence import Sequence, SequenceType  # noqa: E402
from atom.models.utils import (  # noqa: E402
    IntermediateTensors,
    PPMissingLayer,
    get_pp_indices,
    make_layers,
)

from tests.conftest import MockConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Layer partition
# ---------------------------------------------------------------------------


class _RealLayer(nn.Module):
    def __init__(self, layer_num):
        super().__init__()
        self._layer_num = layer_num


def _partitions(num_layers, pp_size):
    return [get_pp_indices(num_layers, r, pp_size) for r in range(pp_size)]


@pytest.mark.parametrize(
    "num_layers,pp_size",
    [(60, 8), (78, 8), (61, 4), (80, 3), (32, 1), (93, 8), (61, 2), (7, 7)],
)
def test_partition_covers_all_layers_contiguously(num_layers, pp_size):
    parts = _partitions(num_layers, pp_size)
    assert parts[0][0] == 0
    assert parts[-1][1] == num_layers
    for i in range(pp_size):
        start, end = parts[i]
        assert end > start, f"empty partition at rank {i}: {parts[i]}"
        if i > 0:
            assert start == parts[i - 1][1], f"gap/overlap at rank {i}: {parts}"
    assert sum(end - start for start, end in parts) == num_layers


@pytest.mark.parametrize("num_layers,pp_size", [(61, 4), (78, 8), (80, 3), (93, 8)])
def test_remainder_never_lands_on_last_partition(num_layers, pp_size):
    """Last partition holds the norm/lm_head; it must keep the base size."""
    base = num_layers // pp_size
    parts = _partitions(num_layers, pp_size)
    assert parts[-1][1] - parts[-1][0] == base


def test_layer_partition_env_override(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "15,15,15,16")
    assert get_pp_indices(61, 0, 4) == (0, 15)
    assert get_pp_indices(61, 1, 4) == (15, 30)
    assert get_pp_indices(61, 3, 4) == (45, 61)


def test_layer_partition_env_override_bad_count(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "15,15,31")
    with pytest.raises(ValueError):
        get_pp_indices(61, 0, 4)


def test_layer_partition_env_override_bad_sum(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "15,15,15,15")
    with pytest.raises(ValueError):
        get_pp_indices(61, 0, 4)


@pytest.mark.parametrize("pp_rank,pp_size", [(0, 4), (1, 4), (2, 4), (3, 4)])
def test_make_layers_places_real_and_missing(monkeypatch, pp_rank, pp_size):
    num_layers = 61
    grp = MagicMock()
    grp.rank_in_group = pp_rank
    grp.world_size = pp_size
    fake_ps = MagicMock()
    fake_ps.get_pp_group.return_value = grp
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", fake_ps)

    built = []

    def layer_fn(prefix, layer_num):
        built.append(layer_num)
        return _RealLayer(layer_num)

    start, end, modules = make_layers(num_layers, layer_fn, prefix="layers")

    assert (start, end) == get_pp_indices(num_layers, pp_rank, pp_size)
    assert len(modules) == num_layers
    for idx in range(num_layers):
        if start <= idx < end:
            assert not isinstance(modules[idx], PPMissingLayer)
            assert modules[idx]._layer_num == idx
        else:
            assert isinstance(modules[idx], PPMissingLayer)
    assert built == list(range(start, end))


# ---------------------------------------------------------------------------
# Inter-stage comm (pp_comm): proxy pack/unpack, routing, send-allgather
# ---------------------------------------------------------------------------


def _fake_pp_group(monkeypatch, rank_in_group=0, world_size=2):
    grp = MagicMock()
    grp.rank_in_group = rank_in_group
    grp.world_size = world_size
    sent = {}

    def send_tensor_dict(tensors, dst, all_gather_group=None):
        sent["tensors"] = tensors
        sent["dst"] = dst
        sent["all_gather_group"] = all_gather_group

    grp.send_tensor_dict.side_effect = send_tensor_dict
    monkeypatch.setattr(pp_comm, "get_pp_group", lambda: grp)
    # Default single-rank TP → pp_send_allgather_group() returns None, so send/recv
    # behave like full-tensor transfers. Allgather tests stub a wider TP group.
    tp = MagicMock()
    tp.world_size = 1
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    return grp, sent


def test_send_packs_only_proxy_keys_to_group_relative_next(monkeypatch):
    grp, sent = _fake_pp_group(monkeypatch, rank_in_group=0, world_size=2)
    it = IntermediateTensors(
        {
            "hidden_states": torch.ones(3, 4),
            "residual": torch.zeros(3, 4),
            "extra": torch.arange(5),  # must be dropped
        }
    )
    pp_comm.send_intermediate_tensors(it)
    assert set(sent["tensors"].keys()) == {"hidden_states", "residual"}
    assert sent["dst"] == 1
    assert torch.equal(sent["tensors"]["hidden_states"], torch.ones(3, 4))


def test_send_tolerates_missing_residual(monkeypatch):
    grp, sent = _fake_pp_group(monkeypatch)
    pp_comm.send_intermediate_tensors(
        IntermediateTensors({"hidden_states": torch.ones(2, 2)})
    )
    assert set(sent["tensors"].keys()) == {"hidden_states"}


def test_recv_wraps_from_group_relative_prev(monkeypatch):
    grp, _ = _fake_pp_group(monkeypatch, rank_in_group=1, world_size=2)
    payload = {"hidden_states": torch.ones(2, 2), "residual": torch.zeros(2, 2)}
    grp.recv_tensor_dict.return_value = payload

    out = pp_comm.recv_intermediate_tensors()

    grp.recv_tensor_dict.assert_called_once_with(src=0, all_gather_group=None)
    assert isinstance(out, IntermediateTensors)
    assert torch.equal(out["hidden_states"], torch.ones(2, 2))
    assert torch.equal(out["residual"], torch.zeros(2, 2))


def test_roundtrip_preserves_proxy_tensors(monkeypatch):
    grp, sent = _fake_pp_group(monkeypatch)
    it = IntermediateTensors(
        {"hidden_states": torch.randn(4, 8), "residual": torch.randn(4, 8)}
    )
    pp_comm.send_intermediate_tensors(it)
    grp.recv_tensor_dict.return_value = sent["tensors"]

    got = pp_comm.recv_intermediate_tensors()
    for k in ("hidden_states", "residual"):
        assert torch.equal(got[k], it[k])


def test_allgather_group_none_when_disabled(monkeypatch):
    tp = MagicMock()
    tp.world_size = 4
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    monkeypatch.setattr(pp_comm.envs, "ATOM_PP_SEND_ALLGATHER", False)
    assert pp_comm.pp_send_allgather_group() is None


def test_allgather_group_none_when_tp_size_one(monkeypatch):
    tp = MagicMock()
    tp.world_size = 1
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    monkeypatch.setattr(pp_comm.envs, "ATOM_PP_SEND_ALLGATHER", True)
    assert pp_comm.pp_send_allgather_group() is None


def test_allgather_group_selected_when_enabled(monkeypatch):
    tp = MagicMock()
    tp.world_size = 4
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    monkeypatch.setattr(pp_comm.envs, "ATOM_PP_SEND_ALLGATHER", True)
    assert pp_comm.pp_send_allgather_group() is tp


def test_recv_threads_allgather_group(monkeypatch):
    grp, _ = _fake_pp_group(monkeypatch, rank_in_group=1, world_size=2)
    tp = MagicMock()
    tp.world_size = 4
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    monkeypatch.setattr(pp_comm.envs, "ATOM_PP_SEND_ALLGATHER", True)
    grp.recv_tensor_dict.return_value = {
        "hidden_states": torch.ones(2, 2),
        "residual": torch.zeros(2, 2),
    }
    pp_comm.recv_intermediate_tensors()
    grp.recv_tensor_dict.assert_called_once_with(src=0, all_gather_group=tp)


def _fake_async_pp_group(monkeypatch, rank_in_group=0, world_size=2):
    grp = MagicMock()
    grp.rank_in_group = rank_in_group
    grp.world_size = world_size
    grp.ranks = list(range(world_size))
    grp.cpu_group = MagicMock()
    grp.device_group = MagicMock()
    monkeypatch.setattr(pp_comm, "get_pp_group", lambda: grp)
    return grp


def _capture_isend(monkeypatch):
    """Record the numel of every isend the send path posts."""
    sent = []

    def isend(buf, dst, group):
        sent.append(buf.numel())
        return MagicMock()

    monkeypatch.setattr(pp_comm.torch.distributed, "isend", isend)
    return sent


def test_async_send_slices_payload_when_allgather_on(monkeypatch):
    _fake_async_pp_group(monkeypatch, rank_in_group=0, world_size=2)
    tp = MagicMock()
    tp.world_size = 4
    tp.rank_in_group = 1
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    monkeypatch.setattr(pp_comm.envs, "ATOM_PP_SEND_ALLGATHER", True)
    sent = _capture_isend(monkeypatch)

    # [4, 8] → 32 elems, 32 % 4 == 0 → each rank sends an 8-elem slice.
    it = IntermediateTensors(
        {"hidden_states": torch.randn(4, 8), "residual": torch.randn(4, 8)}
    )
    pp_comm.async_send_intermediate_tensors(it)
    # First two isends are metadata (size + pickled object); the rest are slices.
    assert sent[2:] == [8, 8]


def test_async_send_full_tensor_when_allgather_off(monkeypatch):
    _fake_async_pp_group(monkeypatch, rank_in_group=0, world_size=2)
    tp = MagicMock()
    tp.world_size = 4
    tp.rank_in_group = 1
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    monkeypatch.setattr(pp_comm.envs, "ATOM_PP_SEND_ALLGATHER", False)
    sent = _capture_isend(monkeypatch)

    it = IntermediateTensors(
        {"hidden_states": torch.randn(4, 8), "residual": torch.randn(4, 8)}
    )
    pp_comm.async_send_intermediate_tensors(it)
    assert sent[2:] == [32, 32]


def test_async_send_full_tensor_when_not_divisible(monkeypatch):
    _fake_async_pp_group(monkeypatch, rank_in_group=0, world_size=2)
    tp = MagicMock()
    tp.world_size = 4
    tp.rank_in_group = 1
    monkeypatch.setattr(pp_comm, "get_tp_group", lambda: tp)
    monkeypatch.setattr(pp_comm.envs, "ATOM_PP_SEND_ALLGATHER", True)
    sent = _capture_isend(monkeypatch)

    # 5*7 == 35, 35 % 4 != 0 → guard fails, full tensor sent.
    pp_comm.async_send_intermediate_tensors(
        IntermediateTensors({"hidden_states": torch.randn(5, 7)})
    )
    assert sent[2:] == [35]


# ---------------------------------------------------------------------------
# Inter-stage transport (PPStageTransport ZMQ)
# ---------------------------------------------------------------------------


@dataclass
class _FakeBatch:
    """Stand-in for ScheduledBatch — any picklable object works."""

    req_ids: tuple
    positions: list


def _transport_addrs(tmpdir, pp_size):
    meta = [f"ipc://{os.path.join(tmpdir, f'meta{s}')}" for s in range(pp_size)]
    return meta, f"ipc://{os.path.join(tmpdir, 'token')}"


def test_head_last_metadata_and_token_roundtrip():
    ctx = zmq.Context()
    with tempfile.TemporaryDirectory() as tmp:
        meta_addrs, token_addr = _transport_addrs(tmp, 2)
        last = PPStageTransport(1, 2, meta_addrs, token_addr, ctx=ctx)
        head = PPStageTransport(0, 2, meta_addrs, token_addr, ctx=ctx)

        batch = _FakeBatch(req_ids=(1, 2, 3), positions=[0, 1, 2])
        head.send_metadata(batch)
        assert last.recv_metadata(timeout_ms=2000) == batch

        tokens = {"req1": (42,), "req2": (7,)}
        last.send_tokens(tokens)
        assert head.recv_tokens(timeout_ms=2000) == tokens

        head.close()
        last.close()
    ctx.term()


def test_metadata_fans_out_to_all_downstream():
    ctx = zmq.Context()
    with tempfile.TemporaryDirectory() as tmp:
        meta_addrs, token_addr = _transport_addrs(tmp, 3)
        s1 = PPStageTransport(1, 3, meta_addrs, token_addr, ctx=ctx)
        s2 = PPStageTransport(2, 3, meta_addrs, token_addr, ctx=ctx)
        head = PPStageTransport(0, 3, meta_addrs, token_addr, ctx=ctx)

        batch = _FakeBatch(req_ids=(9,), positions=[5])
        head.send_metadata(batch)
        assert s1.recv_metadata(timeout_ms=2000) == batch
        assert s2.recv_metadata(timeout_ms=2000) == batch

        head.close()
        s1.close()
        s2.close()
    ctx.term()


def test_recv_timeout_returns_none():
    ctx = zmq.Context()
    with tempfile.TemporaryDirectory() as tmp:
        meta_addrs, token_addr = _transport_addrs(tmp, 2)
        last = PPStageTransport(1, 2, meta_addrs, token_addr, ctx=ctx)
        head = PPStageTransport(0, 2, meta_addrs, token_addr, ctx=ctx)
        assert last.recv_metadata(timeout_ms=200) is None
        assert head.recv_tokens(timeout_ms=200) is None
        head.close()
        last.close()
    ctx.term()


# ---------------------------------------------------------------------------
# EngineArgs pipeline_parallel_size plumbing
# ---------------------------------------------------------------------------


def test_default_pp_is_one():
    assert EngineArgs().pipeline_parallel_size == 1


def test_pp_passthrough_to_engine_kwargs():
    kwargs = EngineArgs(pipeline_parallel_size=4)._get_engine_kwargs()
    assert kwargs["pipeline_parallel_size"] == 4


def test_pp_default_in_engine_kwargs():
    assert EngineArgs()._get_engine_kwargs()["pipeline_parallel_size"] == 1


def test_cli_parses_pp_short_and_long():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    assert parser.parse_args(["-pp", "4"]).pipeline_parallel_size == 4
    assert (
        parser.parse_args(["--pipeline-parallel-size", "2"]).pipeline_parallel_size == 2
    )
    assert parser.parse_args([]).pipeline_parallel_size == 1


def test_from_cli_args_roundtrip():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    args = EngineArgs.from_cli_args(parser.parse_args(["-pp", "8", "-tp", "8"]))
    assert args.pipeline_parallel_size == 8
    assert args.tensor_parallel_size == 8


# ---------------------------------------------------------------------------
# Schedule-time chunk advancement + page alignment (real Scheduler)
# ---------------------------------------------------------------------------


def _pp_config(**overrides):
    defaults = dict(
        pipeline_parallel_size=4,
        max_num_seqs=8,
        max_num_batched_tokens=100,
        max_model_len=131072,
        kv_cache_block_size=16,
        num_kvcache_blocks=4096,
    )
    defaults.update(overrides)
    return MockConfig(**defaults)


def _make_seq(prompt_len, block_size=16):
    return Sequence(token_ids=list(range(prompt_len)), block_size=block_size)


def test_advance_on_schedule_enabled_for_pp_gt1():
    assert Scheduler(_pp_config(pipeline_parallel_size=2)).advance_on_schedule is True


def test_advance_on_schedule_disabled_for_pp1():
    assert Scheduler(_pp_config(pipeline_parallel_size=1)).advance_on_schedule is False


def test_is_final_chunk_single_shot():
    cfg = _pp_config(max_num_batched_tokens=1000)
    sched = Scheduler(cfg)
    sched.add(_make_seq(50, block_size=cfg.kv_cache_block_size))
    batch, _ = sched.schedule()
    assert batch.is_final_chunk == [True]


def test_no_advancement_when_pp1():
    cfg = _pp_config(pipeline_parallel_size=1, max_num_batched_tokens=100)
    sched = Scheduler(cfg)
    seq = _make_seq(200, block_size=cfg.kv_cache_block_size)
    sched.add(seq)
    sched.schedule()
    # Legacy path: num_cached_tokens is not advanced at schedule time.
    assert seq.num_cached_tokens == 0


def test_decode_inflight_block():
    """A decode seq with an in-flight token is skipped by the next schedule()."""
    cfg = _pp_config(max_num_batched_tokens=1024)
    sched = Scheduler(cfg)
    seq = _make_seq(32, block_size=cfg.kv_cache_block_size)
    sched.add(seq)

    batch1, _ = sched.schedule()
    assert batch1.is_final_chunk == [True]
    sched.mark_pp_inflight(batch1)

    # Simulate postprocess: append token, transition to decode.
    seq.token_ids.append(42)
    seq.num_tokens += 1
    seq.type = SequenceType.DECODE

    batch2, _ = sched.schedule()
    assert len(batch2.req_ids) == 0  # blocked: token not yet received

    sched.release_pp_inflight(batch1)
    batch3, _ = sched.schedule()
    assert seq.id in batch3.req_ids


def test_multiple_seqs_partial_and_final():
    cfg = _pp_config(max_num_batched_tokens=200)
    sched = Scheduler(cfg)
    sched.add(_make_seq(50, block_size=cfg.kv_cache_block_size))
    sched.add(_make_seq(300, block_size=cfg.kv_cache_block_size))
    batch, _ = sched.schedule()
    assert len(batch.is_final_chunk) == 2
    assert batch.is_final_chunk[0] is True  # 50 tokens fit the budget
    assert batch.is_final_chunk[1] is False  # 300 tokens partial


def test_middle_chunk_output_discarded():
    """postprocess with advance_on_schedule discards a middle-chunk token."""
    import numpy as np

    cfg = _pp_config(max_num_batched_tokens=100)
    sched = Scheduler(cfg)
    seq = _make_seq(200, block_size=cfg.kv_cache_block_size)
    sched.add(seq)

    batch1, seqs1 = sched.schedule()
    assert batch1.is_final_chunk == [False]

    n = 1
    fwd_out = ScheduledBatchOutput(
        req_ids=[seq.id],
        token_ids=[(42,)],
        num_rejected=np.zeros(n, dtype=np.int32),
        num_bonus=np.zeros(n, dtype=np.int32),
        draft_token_ids=None,
        is_deferred_out=False,
    )
    initial_num_tokens = seq.num_tokens
    sched.postprocess(list(seqs1.values()), fwd_out, batch=batch1)
    assert seq.num_tokens == initial_num_tokens


@pytest.mark.parametrize(
    "num_new,budget,block_size,expected",
    [
        (200, 100, 16, 64),  # middle chunk aligns down to max(block,64)
        (50, 200, 16, 50),  # final chunk (fits) not aligned
        (200, 100, 4, 64),  # align granularity is max(block,64)
        (500, 300, 128, 256),  # block>64 uses block as alignment
        (500, 64, 128, 64),  # budget<align → no underflow, fall back to budget
        (200, 128, 16, 128),  # exact multiple unchanged
        (100, 100, 16, 100),  # budget == remaining → treated as final
        (200, 100, 1, 64),  # block_size=1 → align 64
    ],
)
def test_prefill_chunk_page_alignment(num_new, budget, block_size, expected):
    sched = Scheduler(_pp_config(kv_cache_block_size=block_size))
    assert sched._prefill_chunk_for_budget(num_new, budget, 0) == expected


def test_prefill_chunk_not_aligned_when_chunked_prefill_disabled():
    # num_new (200) > budget (100): with chunked prefill enabled this would
    # align down to 64; disabled, the whole request is returned unaligned.
    sched = Scheduler(_pp_config(enable_chunked_prefill=False))
    assert sched._prefill_chunk_for_budget(200, 100, 0) == 200


# ---------------------------------------------------------------------------
# Output skip for pure middle-chunk batches (real ScheduledBatch.produces_output)
# ---------------------------------------------------------------------------


def _batch(total_seqs_num_decode=0, is_final_chunk=None):
    b = object.__new__(ScheduledBatch)
    b.total_seqs_num_decode = total_seqs_num_decode
    b.is_final_chunk = is_final_chunk
    return b


@pytest.mark.parametrize(
    "decode,final,expected",
    [
        (4, None, True),  # decode batch always produces output
        (2, [False], True),  # decode dominates even with middle-chunk prefill
        (0, None, True),  # prefill with no chunk info → produces output
        (0, [False, False, False], False),  # all middle chunks → no output
        (0, [False, True], True),  # one final chunk → produces output
        (0, [True, True], True),  # all final chunks → produces output
    ],
)
def test_produces_output(decode, final, expected):
    assert _batch(decode, final).produces_output() is expected


# ---------------------------------------------------------------------------
# PP chunked-prefill prefix-hash registration (register_prefill_hashes)
# ---------------------------------------------------------------------------


def _drive_pp_prefill(sched, prompt_len, block_size=16):
    """Run one seq's prefill through the head defer/flush loop and return it."""
    import numpy as np

    seq = _make_seq(prompt_len, block_size=block_size)
    sched.add(seq)
    pending = []
    for _ in range(400):
        res = sched.schedule()
        if res is None:
            break
        batch, seqs = res
        if not batch.req_ids:
            break
        if batch.produces_output():
            for b in pending:
                sched.register_prefill_hashes(b)
            pending.clear()
            n = len(batch.req_ids)
            fwd = ScheduledBatchOutput(
                req_ids=list(batch.req_ids),
                token_ids=[(0,)] * n,
                num_rejected=np.zeros(n, dtype=np.int32),
                num_bonus=np.zeros(n, dtype=np.int32),
                draft_token_ids=None,
                is_deferred_out=False,
            )
            sched.postprocess(list(seqs.values()), fwd, batch=batch)
        else:
            pending.append(batch)
        if (
            seq.num_cached_tokens >= seq.num_prompt_tokens
            and not seq.is_partial_prefill
        ):
            break
    return seq


def test_pp_chunked_prefill_registers_prefix_hashes():
    bs = 16
    cfg = _pp_config(
        enable_prefix_caching=True,
        max_num_batched_tokens=64,
        kv_cache_block_size=bs,
        long_prefill_token_threshold=0,
    )
    sched = Scheduler(cfg)
    prompt = 512  # 32 blocks, 8 chunks of 64 -> 7 middle chunks
    _drive_pp_prefill(sched, prompt, block_size=bs)
    # A fresh same-prefix request must hit the whole prompt minus the last block.
    hit = sched.block_manager.can_allocate(_make_seq(prompt, block_size=bs))
    assert hit == prompt // bs - 1
