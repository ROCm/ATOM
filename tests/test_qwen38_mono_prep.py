# SPDX-License-Identifier: MIT
"""Graph-stable Mono metadata must preserve accepted-token and padding contracts."""

import pytest
import torch

pytest.importorskip("triton")
from atom.models.qwen38.mono.kernels.prep import VerifyPrep, draft_meta

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


@pytest.mark.parametrize("batch", [1, 4])
def test_verify_prep_graph_replay_selects_accepted_state(batch):
    spec, hidden, streams = 4, 2560, 4
    ids = torch.arange(batch * spec, device="cuda", dtype=torch.int32)
    embedding = (
        (torch.arange(32 * hidden, device="cuda") % 251)
        .to(torch.bfloat16)
        .reshape(32, hidden)
    )
    out = torch.empty(
        batch * spec, hidden * streams, device="cuda", dtype=torch.bfloat16
    )
    # Strided slot tables and a permutation make request IDs different from slot IDs.
    idx = torch.arange(100, 100 + batch * 8, device="cuda", dtype=torch.int32).reshape(
        batch, 8
    )[:, :4]
    idx_in = torch.flip(idx.clone(), [0]) + 100
    accepted = torch.ones(batch, device="cuda", dtype=torch.int32)
    slots = torch.arange(batch * spec, device="cuda", dtype=torch.int64)
    # MRoPE positions [3, rows]: request b's rows sit `delta` past its cache rows (an image request's offset)
    context = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32) * 100
    delta = torch.tensor([0, -900, 7, -3][:batch], device="cuda", dtype=torch.int64)
    rows = (context.long() - spec)[:, None] + torch.arange(spec, device="cuda")
    positions = (rows + delta[:, None]).reshape(1, -1).repeat(3, 1).contiguous()
    prep = VerifyPrep(batch)

    def launch():
        return prep(ids, embedding, out, accepted, idx, idx_in, batch, streams, slots, spec,
                    positions, context)  # fmt: skip

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for count in (1, 4, 2, 3):
        accepted.fill_(count)
        slots.fill_(1)
        if batch > 1:
            slots[-spec] = -1
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, embedding[ids.long()].repeat(1, streams))
        assert torch.equal(prep.prev, accepted - 1)
        assert torch.equal(prep.slot, idx[:, 0])
        assert torch.equal(prep.ssm_read, idx_in[:, count - 1])
        assert torch.equal(prep.valid, slots[::spec] >= 0)
        assert torch.equal(
            prep.rope_off, torch.where(prep.valid, delta, 0).to(torch.int32)
        )


@pytest.mark.parametrize("spec", [1, 4])
@pytest.mark.parametrize("selected", [False, True])
def test_draft_metadata_replay_padding_and_epoch(spec, selected):
    batch, width = 4, 4096
    pos = torch.arange(batch * spec, device="cuda", dtype=torch.int64) + 16384
    slots = torch.ones_like(pos)
    # A strided source checks that table row stride is respected.
    table = torch.arange(batch * (width + 7), device="cuda", dtype=torch.int64).reshape(
        batch, width + 7
    )[:, :width]
    select = (
        torch.arange(batch, device="cuda", dtype=torch.int64) * spec
        if selected
        else None
    )
    metadata = {
        "seq_lens": torch.zeros(batch, device="cuda", dtype=torch.int32),
        "block_table": torch.zeros(batch, width, device="cuda", dtype=torch.int32),
        "sel": torch.full((batch,), -7, device="cuda", dtype=torch.int32),
    }
    epoch = torch.tensor(9, device="cuda", dtype=torch.int32)

    def launch():
        draft_meta(metadata, [epoch], pos, slots, table, select, batch, spec)

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    previous = epoch.item()
    for step in range(3):
        pos.add_(1)
        slots[spec] = -1
        pos[2 * spec] = -1
        graph.replay()
        torch.cuda.synchronize()
        assert epoch.item() == previous + step + 1
        assert metadata["seq_lens"].tolist() == [
            pos[0].item(),
            0,
            0,
            pos[3 * spec].item(),
        ]
        assert torch.equal(
            metadata["block_table"][[0, 3]], table[[0, 3]].to(torch.int32)
        )
        assert bool((metadata["block_table"][[1, 2]] == -1).all())
        expected = (
            select.to(torch.int32) if selected else torch.full_like(metadata["sel"], -7)
        )
        assert torch.equal(metadata["sel"], expected)


@pytest.mark.parametrize("batch", [1, 4])
def test_target_metadata_puts_match_the_reference_fill(batch):
    """The fused metadata launch writes what the reference ops would: a padded request gets slots -1 and a
    context of 0 over a block-table row of -1 (no pool touched), every counter + 1."""
    from atom.models.qwen38.mono.kernels.prep import fused_puts
    from atom.models.qwen38.mono.step_meta import StepMeta, meta_buffers, target_puts

    spec, width, dev = 4, 4096, "cuda"
    gen = torch.Generator().manual_seed(batch)

    def ints(*shape):
        return torch.randint(0, 1 << 20, shape, generator=gen, dtype=torch.int32).to(
            dev
        )

    valid = torch.ones(batch, dtype=torch.bool)
    valid[batch // 2] = batch == 1
    table = ints(batch, width + 7)[:, :width]
    meta = StepMeta(batch, valid.to(dev), table, ints(batch) + spec, ints(batch), ints(batch) % 4, ints(batch),
                    ints(batch, spec))  # fmt: skip
    m = meta_buffers(batch, spec, width, 16, dev)
    counters = [torch.full((1,), 5, dtype=torch.int32, device=dev) for _ in range(2)]
    fused_puts(meta.valid, target_puts(m, meta), counters)
    torch.cuda.synchronize()
    v, rows = meta.valid, meta.valid.repeat_interleave(spec)
    assert torch.equal(m["gdn.conv_slot"], torch.where(v, meta.conv_slot, -1))
    assert torch.equal(m["gdn.conv_off"], meta.conv_off)
    assert torch.equal(m["gdn.ssm_read"], torch.where(v, meta.ssm_read, -1))
    assert torch.equal(
        m["gdn.ssm_write"], torch.where(rows, meta.ssm_write.reshape(-1), -1)
    )
    assert torch.equal(m["seq_lens"], torch.where(v, meta.context_lens - spec, 0))
    assert torch.equal(m["block_table"], torch.where(v[:, None], table, -1))
    assert all(c.item() == 6 for c in counters)
