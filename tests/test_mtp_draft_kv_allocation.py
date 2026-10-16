# SPDX-License-Identifier: MIT
"""Every KV position an MTP step writes is allocated before its forward.

A forward writes the target's tokens and then the draft round's. With
`k = num_speculative_tokens`, `EagleProposer.propose` runs `k - 1` draft
mid-steps after step 0, each bumping `context_lens` and writing one token at
`context_lens - 1`; the builders' slot for it is
`block_tables[row, pos // B] * B + pos % B`. So a forward whose target context
ends at `C` writes up to `C + k - 2`, where

  prefill: `C` is the chunk end (`batch.context_lens`), and
  decode:  `C = batch.context_lens - num_rejected` of the previous verify
           (`AiterAttentionMetadataBuilder.prepare_decode`); the target writes
           `[C - k - 1, C)`.

A position whose block column is past the row's allocated blocks reads 0 from
the zeroed block-table tail, and the write lands in physical block 0.
`can_append` sizing the pool for the step is not the same as `may_append`
allocating it -- the original probe below is that gap.

These drive the real Scheduler / BlockManager through prefill and decode with
deferred output (the default) and check the runner's arithmetic against the
block table each forward sees.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager, draft_kv_lookahead
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams


def spec(k):
    return SimpleNamespace(num_speculative_tokens=k, use_dspark=lambda: False)


def make_scheduler(k, block, **overrides):
    cfg = {
        "speculative_config": spec(k),
        "kv_cache_block_size": block,
        "num_kvcache_blocks": 256,
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
    }
    cfg.update(overrides)
    return Scheduler(MockConfig(**cfg))


def output(seq, tokens, rejected, k):
    return ScheduledBatchOutput(
        [seq.id],
        [tuple(tokens)],
        np.array([rejected]),
        np.array([max(0, len(tokens) - 1)]),
        np.zeros((1, k), dtype=np.int32),
        is_deferred_out=True,
    )


def run(prompt_len, k, block, accepts):
    """Prefill then one decode per entry of `accepts` (tokens emitted, 1..k+1).

    Returns, per forward, the KV positions the runner writes and the number of
    blocks in the row's table when it does.
    """
    sched = make_scheduler(k, block)
    seq = Sequence(
        list(range(10, 10 + prompt_len)),
        block,
        SamplingParams(max_tokens=10_000, ignore_eos=True),
        num_draft_tokens=k,
    )
    sched.add(seq)
    steps = []
    pending = ScheduledBatchOutput([], [], None, None, None, is_deferred_out=True)
    r_prev = 0
    for acc in [None, *accepts]:
        batch, active = sched.schedule()
        ctx = int(batch.context_lens[0])
        if batch.total_seqs_num_prefill:
            target = range(ctx - int(batch.num_scheduled_tokens[0]), ctx)
            end = ctx
        else:
            end = ctx - r_prev
            target = range(end - k - 1, end)
        draft = range(end, end + k - 1)
        steps.append(
            SimpleNamespace(
                prefill=bool(batch.total_seqs_num_prefill),
                target=target,
                draft=draft,
                blocks=len(seq.block_table),
            )
        )
        # Deferred output: this forward returns the previous one's tokens.
        sched.postprocess(list(active.values()), pending, batch=batch)
        if acc is None:  # the prefill samples one token and verifies nothing
            pending, r_prev = output(seq, [99], 0, k), 0
        else:
            pending = output(seq, range(200, 200 + acc), k + 1 - acc, k)
            r_prev = k + 1 - acc
    return steps


def unallocated(step, block):
    return [p for p in (*step.target, *step.draft) if p // block >= step.blocks]


# ── The reviewer's probe, on the real functions ─────────────────────────────


def test_room_for_the_step_is_not_the_step_allocated():
    """Length 16, one block: `can_append(seq, 4)` says yes, `may_append` keeps
    the one block. The draft round past position 15 needs block 2."""
    bm = BlockManager(MockConfig(kv_cache_block_size=16, num_kvcache_blocks=8))
    seq = Sequence(list(range(16)), 16)
    bm.allocate(seq)
    assert bm.can_append(seq, 4)
    bm.may_append(seq, 4)
    assert len(seq.block_table) == 1
    bm.may_append(seq, 4, lookahead=draft_kv_lookahead(3))
    assert len(seq.block_table) == 2


@pytest.mark.parametrize("k, lookahead", [(0, 0), (1, 0), (2, 1), (3, 2), (5, 4)])
def test_lookahead_is_the_draft_mid_steps(k, lookahead):
    assert draft_kv_lookahead(k) == lookahead


# ── The real scheduler path ─────────────────────────────────────────────────

PATTERNS = {
    "full_accept": lambda k: [k + 1] * 12,
    "full_reject": lambda k: [1] * 12,
    "mixed": lambda k: [k + 1, 1, 2, k + 1, k + 1, 1] * 2,
}


@pytest.mark.parametrize("block", [16, 64, 128])
@pytest.mark.parametrize("k", [1, 2, 3, 5])
@pytest.mark.parametrize("pattern", sorted(PATTERNS))
def test_every_write_of_every_forward_is_allocated(block, k, pattern):
    """Every prompt length across two blocks, so each residue mod the block
    size is the prompt end once: prefill -> first draft round, then decode
    steps that fully accept, fully reject, or mix."""
    crossings = 0
    for prompt_len in range(1, 2 * block + 1):
        for i, step in enumerate(run(prompt_len, k, block, PATTERNS[pattern](k))):
            assert not unallocated(step, block), (
                f"prompt {prompt_len}, forward {i}: {step} writes "
                f"{unallocated(step, block)} past {step.blocks} blocks"
            )
            last = max(step.target.stop, step.draft.stop) - 1
            crossings += last // block > (step.target.stop - 1) // block
    if k > 1:
        # The draft round really did cross into a block past the target's.
        assert crossings > 0


@pytest.mark.parametrize("k", [2, 3, 5])
def test_prefill_ending_on_a_block_boundary_drafts_into_its_own_block(k):
    """The first draft round runs inside the prefill forward, from the prompt
    end -- which, for a prompt that fills its blocks exactly, is the first
    position of a block nobody allocated before."""
    block = 16
    prefill, first_decode = run(block, k, block, [k + 1])
    assert prefill.prefill and prefill.draft.start == block
    assert prefill.blocks == 2
    assert not unallocated(prefill, block)
    assert not unallocated(first_decode, block)


def test_full_reject_still_allocates_for_the_round_after_full_accept():
    """`C` depends on the previous verify; full accept leaves it at `len(seq)`,
    the case with the most draft positions past the scheduler's length."""
    block, k = 16, 3
    steps = run(13, k, block, [k + 1, 1, k + 1, k + 1, 1, 1])
    assert all(not unallocated(s, block) for s in steps)


# ── Admission charges the lookahead; the guard refuses a missing one ───────


@pytest.mark.parametrize("prefix", [False, True])
def test_admission_charges_the_lookahead_block(prefix):
    """A 16-token prompt at block 16 with k=3 takes two blocks: one for the
    prompt, one for the draft round past it. One free block is not enough."""
    for blocks, admitted in ((1, False), (2, True)):
        sched = make_scheduler(
            3, 16, num_kvcache_blocks=blocks, enable_prefix_caching=prefix
        )
        seq = Sequence(list(range(16)), 16, num_draft_tokens=3)
        sched.add(seq)
        out = sched.schedule()
        if admitted:
            assert out is not None and seq.id in out[1]
            assert len(seq.block_table) == 2
        else:
            assert out is None or seq.id not in out[1]
            assert not seq.block_table


def test_no_lookahead_without_drafts():
    sched = Scheduler(MockConfig(kv_cache_block_size=16, num_kvcache_blocks=8))
    seq = Sequence(list(range(16)), 16)
    sched.add(seq)
    sched.schedule()
    assert sched.draft_lookahead == 0
    assert len(seq.block_table) == 1


def test_a_path_that_skips_the_lookahead_fails_loudly(monkeypatch):
    sched = make_scheduler(3, 16)
    seq = Sequence(list(range(16)), 16, num_draft_tokens=3)
    sched.add(seq)
    monkeypatch.setattr(
        sched, "_allocate_prefill_draft_lookahead", lambda seq: None, raising=True
    )
    with pytest.raises(RuntimeError, match="would land in block 0"):
        sched.schedule()
