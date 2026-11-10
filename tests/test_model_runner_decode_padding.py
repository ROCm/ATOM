# SPDX-License-Identifier: MIT
"""Model execution must retain the step's padded height until sampling."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            parallel_state=load("aiter.dist.parallel_state"),
            ulysses_sp=load("atom.distributed.ulysses_sp"),
            runner=load("atom.model_engine.model_runner"),
            sp_wrapper=load("atom.utils.sp_wrapper"),
            forward_context=load("atom.utils.forward_context"),
            ubatching=load("atom.utils.tbo.ubatching"),
        )


@pytest.mark.parametrize("sp_rank", [None, 0, 3])
@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("dummy", [False, True])
@pytest.mark.parametrize(
    "scheduled,running,seqs,q,unified,prefill,token_graph",
    [
        (3, 4, 3, 1, True, False, False),
        (9, 12, 3, 3, True, False, False),
        (4, 9, 2, 3, True, False, False),  # packed/ragged, not seqs * q
        (3, 3, 3, 1, False, False, False),  # a peer is prefilling
        (7, 7, 2, 4, False, True, False),
        (4, 4, 4, 1, True, False, False),
        (3, 4, 3, 1, True, False, True),
    ],
)
def test_runner_model_height_and_sampling_rows(
    monkeypatch,
    modules,
    sp_rank,
    mrope,
    dummy,
    scheduled,
    running,
    seqs,
    q,
    unified,
    prefill,
    token_graph,
):
    ulysses_sp, sp_wrapper = modules.ulysses_sp, modules.sp_wrapper
    runner_module = modules.runner
    sp = SimpleNamespace(sequence_parallel_size=1 if sp_rank is None else 4)
    monkeypatch.setattr(ulysses_sp, "get_current_atom_config", lambda: sp)
    group = SimpleNamespace(rank_in_group=sp_rank)
    monkeypatch.setattr(modules.parallel_state, "get_pcp_group", lambda: group)
    runner = runner_module.ModelRunner.__new__(runner_module.ModelRunner)
    runner.use_mrope = mrope
    runner.sp_mega_moe = Mock()
    runner.config = SimpleNamespace(prefill_context_parallel_size=1)
    runner._detailed_label_suffix = lambda batch: ""
    runner._tr_hybrid_active = lambda: token_graph
    runner._dynamic_chunking = runner_module.DynamicChunkingWorker(runner)
    ids = torch.full((32,), -99, dtype=torch.int32)
    ids[:scheduled] = torch.arange(1, scheduled + 1)
    pos = torch.full((32,), -99, dtype=torch.int64)
    pos[:scheduled] = torch.arange(11, scheduled + 11)
    # The builder publishes the padding positions (`_pad_tokens`).
    pos[scheduled:running] = 0
    runner.forward_vars = {
        "input_ids": SimpleNamespace(gpu=ids),
        "positions": SimpleNamespace(gpu=pos),
        "mrope_positions": SimpleNamespace(gpu=torch.full((96,), -99)),
    }
    if mrope:
        # The mRoPE builder packs three planes at the running-token stride,
        # padding included.
        positions = runner._mrope_positions_view(running)
        positions[:] = pos[:running]
    else:
        positions = pos[:scheduled]
    mode = modules.forward_context.ForwardMode(
        use_cudagraph=token_graph,
        is_prefill=prefill,
        scheduled_bs=seqs,
        scheduled_tokens=scheduled,
        running_bs=4,
        running_tokens=running,
        running_tokens_are_unified=unified,
        max_seqlen_q=q,
        tbo_collective_active=False,
        sync=modules.ubatching.sync_dp_metadata(
            dp_group=None,
            dp_size=1,
            scheduled_tokens=scheduled,
            scheduled_bs=seqs,
            is_prefill=prefill,
            tbo_on=False,
            running_tokens_per_dp_rank=np.zeros(1, dtype=np.int32),
        ),
        token_graph=token_graph,
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.full((running,), -1),
        cu_seqlens_q=torch.arange(5),
        context_lens=torch.tensor([8, 8, 8, 0]),
    )
    original_metadata = {k: (v, v.clone()) for k, v in vars(metadata).items()}
    ctx = SimpleNamespace(
        context=SimpleNamespace(
            scheduled_bs=seqs,
            scheduled_tokens=scheduled,
            running_bs=4,
            is_prefill=prefill,
            is_dummy_run=dummy,
            positions=positions,
            forward_mode=mode,
        ),
        attn_metadata=metadata,
        ubatch_slices=None,
    )
    monkeypatch.setattr(runner_module, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(
        runner_module, "get_pp_group", lambda: SimpleNamespace(world_size=1)
    )
    expected_rows = running if unified and not prefill else scheduled
    sp_prefill = sp_rank is not None and prefill

    # SP runs on its range of the global ids, the address its graphs capture;
    # trailing ranges may run past the step into rows whose outputs are dropped.
    rows = (
        slice(0, expected_rows)
        if sp_rank is None
        else ulysses_sp.sp_local_slice(expected_rows)
    )

    def global_ids():
        full = torch.zeros(expected_rows, dtype=ids.dtype)
        full[:scheduled] = torch.arange(1, scheduled + 1)
        return full

    def local_rows():
        return ids[rows][:, None].float()

    class Model:
        def __call__(self, input_ids, model_positions):
            torch.testing.assert_close(ids[:expected_rows], global_ids())
            assert input_ids.data_ptr() == ids[rows].data_ptr()
            assert input_ids.shape[0] == rows.stop - rows.start
            assert model_positions.shape[-1] == expected_rows
            if expected_rows > scheduled:
                assert torch.all(model_positions[..., scheduled:] == 0)
            return input_ids[:, None].float()

        def compute_logits(self, hidden):
            if sp_prefill:
                # The LM head picks each prompt's last row from the local shard.
                torch.testing.assert_close(hidden, local_rows())
                hidden = global_ids()[:scheduled, None].float()
            else:
                # The LM head runs the running rows; `postprocess` cuts them.
                assert hidden.shape[0] == expected_rows
            return hidden + 1

    if sp_rank is not None:

        def gather_shards(local, total_tokens):
            assert total_tokens == expected_rows
            torch.testing.assert_close(local, local_rows())
            return global_ids()[:, None].float()

        monkeypatch.setattr(sp_wrapper, "sp_gather_tokens", gather_shards)
        monkeypatch.setattr(sp_wrapper, "get_forward_context", lambda: ctx)

    runner.model = Model() if sp_rank is None else sp_wrapper.SPModelWrapper(Model())
    logits, hidden = runner.run_model(ids[:scheduled])
    # MegaMoE epochs advance on graph replays as well as eager steps, by this
    # rank's share of the step's padded rows.
    world = 1 if sp_rank is None else 4
    runner.sp_mega_moe.before_step.assert_called_once_with(-(-running // world))
    # FusedMoE gathers SP's shards on this, dummy passes included.
    assert getattr(ctx.context, "pcp_split", False) == (sp_rank is not None)
    if sp_prefill:
        torch.testing.assert_close(hidden, local_rows())
    else:
        assert hidden.shape == (expected_rows, 1)
    torch.testing.assert_close(
        logits[:scheduled, 0], torch.arange(2, scheduled + 2).float()
    )
    assert torch.all(ids[running:] == -99)
    for name, (original, values) in original_metadata.items():
        assert getattr(metadata, name) is original
        torch.testing.assert_close(original, values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize(
    "query_lens", [(1,), (1, 1), (1, 1, 1), (3,), (3, 3, 3), (1, 3)]
)
def test_pa_preserves_complete_padded_layout(query_lens, capture):
    """Zero-context rows belong to the output; the guard starts after them."""
    import aiter

    from atom.model_ops.base_attention import run_pa_fwd_asm

    padded = 4
    scheduled = sum(query_lens)
    max_qlen = max(query_lens)
    running = 9 if query_lens == (1, 3) else padded * max_qlen
    q = torch.zeros((running, 32, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.ones((32, 8, 8, 16, 16), device="cuda", dtype=aiter.dtypes.fp8)
    v = torch.ones((32, 8, 1, 128, 16), device="cuda", dtype=aiter.dtypes.fp8)
    scale = torch.ones((32, 8, 16), device="cuda")
    blocks = torch.arange(32, device="cuda", dtype=torch.int32).repeat(padded, 1)
    lengths = torch.tensor(
        [284] * len(query_lens) + [0] * (padded - len(query_lens)),
        device="cuda",
        dtype=torch.int32,
    )
    cu = torch.tensor(
        [0, *query_lens, *([0] * (padded - len(query_lens)))],
        device="cuda",
        dtype=torch.int32,
    ).cumsum(0, dtype=torch.int32)
    storage = torch.full((running + 1, 32, 128), 123, device="cuda", dtype=q.dtype)
    output = storage[:running]

    def run():
        return run_pa_fwd_asm(
            q,
            k,
            v,
            blocks,
            lengths,
            scale,
            scale,
            out=output,
            qo_indptr=cu,
            max_qlen=max_qlen,
        )

    run()
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        storage.fill_(123)
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output[:scheduled], torch.ones_like(output[:scheduled]))
    assert torch.all(storage[running:] == 123)
    assert blocks.shape[0] == lengths.shape[0] == padded
    assert cu.shape[0] == padded + 1
