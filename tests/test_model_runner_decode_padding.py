# SPDX-License-Identifier: MIT
"""Model execution must retain the step's padded height until sampling."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.model_engine import model_runner as runner_module
from atom.utils.forward_context import ForwardMode


@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("dummy", [False, True])
@pytest.mark.parametrize(
    "scheduled,running,seqs,q,unified,prefill,piecewise",
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
    monkeypatch, mrope, dummy, scheduled, running, seqs, q, unified, prefill, piecewise
):
    runner = runner_module.ModelRunner.__new__(runner_module.ModelRunner)
    runner.use_mrope = mrope
    runner.config = SimpleNamespace(prefill_context_parallel_size=1)
    runner._detailed_label_suffix = lambda batch: ""
    runner._piecewise_cg_active = lambda: piecewise
    ids = torch.full((32,), -99, dtype=torch.int32)
    ids[:scheduled] = torch.arange(1, scheduled + 1)
    pos = torch.full((32,), -99, dtype=torch.int64)
    pos[:scheduled] = torch.arange(11, scheduled + 11)
    runner.forward_vars = {
        "input_ids": SimpleNamespace(gpu=ids),
        "positions": SimpleNamespace(gpu=pos),
        "mrope_positions": SimpleNamespace(gpu=torch.full((96,), -99)),
    }
    if mrope:
        # The mRoPE builder packs three planes at the running-token stride.
        positions = runner._mrope_positions_view(running)
        positions[:, :scheduled] = pos[:scheduled]
    else:
        positions = pos[:scheduled]
    mode = ForwardMode(
        use_cudagraph=piecewise,
        is_prefill=prefill,
        scheduled_bs=seqs,
        scheduled_tokens=scheduled,
        running_bs=4,
        running_tokens=running,
        running_tokens_are_unified=unified,
        max_seqlen_q=q,
        piecewise_captured=piecewise,
        tbo_collective_active=False,
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

    class Model:
        def __call__(self, input_ids, model_positions):
            assert input_ids.shape[0] == expected_rows
            assert model_positions.shape[-1] == expected_rows
            torch.testing.assert_close(input_ids[:scheduled], ids[:scheduled])
            if expected_rows > scheduled:
                assert torch.all(input_ids[scheduled:] == 0)
                assert torch.all(model_positions[..., scheduled:] == 0)
            return input_ids[:, None].float()

        def compute_logits(self, hidden):
            assert hidden.shape[0] == scheduled
            return hidden + 1

    runner.model = Model()
    logits, hidden = runner.run_model(ids[:scheduled])
    assert hidden.shape == (scheduled, 1)
    torch.testing.assert_close(logits[:, 0], torch.arange(2, scheduled + 2).float())
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
