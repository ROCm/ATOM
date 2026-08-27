# SPDX-License-Identifier: MIT
"""The draft-pass machine's invariants, with no aiter and no GPU.

These are what the whole padding scheme rests on, so they belong where CI can
run them. `tests/test_dspark.py` keeps the DSpark-integration half, which needs
the compiled draft and is therefore skipped on a CPU runner -- if the invariants
lived only there, nothing would check them at all.
"""

import dataclasses
import types

import pytest
import torch

from atom.spec_decode.draft_graph import DraftGraph, StagedInput


def _graph(**kw):
    """A pass over one int32 input, bound to a stub config."""
    kw.setdefault("inputs", {"row": StagedInput(dtype=torch.int32)})
    kw.setdefault("pads", True)
    g = DraftGraph(
        forward=kw.pop("forward", lambda running_bs, **rows: running_bs), **kw
    )
    config = dataclasses.make_dataclass("C", [("max_num_seqs", int)])(256)
    return g.bind(config, torch.device("cpu"))


def _ctx(*, running_bs, use_cudagraph=True, dummy=False, running_tokens=None):
    """A forward context carrying only what `target_running_bs` reads."""
    return types.SimpleNamespace(
        is_dummy_run=dummy,
        running_bs=running_bs,
        running_tokens=running_bs if running_tokens is None else running_tokens,
        forward_mode=types.SimpleNamespace(
            use_cudagraph=use_cudagraph, running_bs=running_bs
        ),
    )


@pytest.mark.parametrize("bs,ran_at,want", [(44, 48, 48), (50, 64, 64), (64, 64, 64)])
def test_the_pad_batch_is_the_one_the_target_ran(bs, ran_at, want):
    """Never a batch the drafter picks. The target sized its per-sequence
    metadata with `running_bs`, so that is exactly how far a pad row may
    reach; anything wider reads a row nobody wrote this step."""
    assert _graph().target_running_bs(bs, _ctx(running_bs=ran_at)) == want


def test_an_empty_batch_is_never_widened():
    """A pad row is a copy of the last real row, and there is none.

    `stage` would reach `src[-1]` on a zero-row source and raise IndexError,
    which is a much worse way to learn that a rank was kept alive purely to
    reach the draft's collectives.
    """
    g = _graph()
    assert g.target_running_bs(0, _ctx(running_bs=48)) == 0
    assert g.stage(0, {"row": torch.zeros(0, dtype=torch.int32)})["row"].shape[0] == 0


def test_the_pad_batch_counts_sequences_and_never_rows():
    """The context carries the step's padded shape in both units, and only one
    of them is a batch.

    `running_tokens` is what MoE pads hidden_states to; on a DSpark ragged step
    it is a flat token bucket that no bs*q recovers, so reading it here would
    stage hundreds of fabricated sequences. Pinned with the two far apart --
    equal values would prove nothing about which one is read.
    """
    ctx = _ctx(running_bs=48, running_tokens=16336)
    assert _graph().target_running_bs(44, ctx) == 48


def test_nothing_is_padded_where_the_target_pinned_no_batch():
    g = _graph()
    # Planted first: owning a graph at 48 is NOT a licence to pad to it. Only a
    # replayed target sizes its per-sequence metadata past the real batch, and
    # a pad row that reaches further takes its ring slot -- which DSpark's block
    # SCATTERS draft KV through -- from a row nobody wrote this step.
    g._cuda_graphs[48] = ("graph", "out")
    ctx = _ctx(running_bs=256, use_cudagraph=False)
    assert g.target_running_bs(44, ctx) == 44, "eager: nobody sized metadata past bs"


def test_a_pass_that_cannot_pad_stays_at_the_real_batch():
    """`pads` gates the padding, not just the capture -- EPLB turns it off
    because pad rows would route through the draft's MoE and land in the
    expert-load histogram."""
    g = _graph(pads=False)
    assert g.target_running_bs(44, _ctx(running_bs=48)) == 44
    assert not g.will_capture


def test_declaring_pads_without_inputs_is_refused():
    """`pads` promises the fabricated rows land somewhere inert; with nothing
    staged there is no buffer for them to land in at all."""
    with pytest.raises(AssertionError, match="stages nothing"):
        _graph(inputs={}, pads=True)


def test_staged_tail_repeats_a_real_row_rather_than_zero_filling():
    """Zero-filling faults the GPU at the first padded decode step.

    Measured on V4-Flash-DSpark tp8: an all-zero tail gives
    `Memory access fault ... on address (nil)` on 8/8 ranks; repeating a real
    row is clean on the same workload. Which input cannot take a zero was not
    isolated, so this pins the repeat, not the reason.
    """
    g = _graph()
    first = g._stage_one("row", 8, torch.arange(3, dtype=torch.int32) + 100)
    assert first.tolist() == [100, 101, 102, 102, 102, 102, 102, 102]
    ptr = g._buffers["row"].data_ptr()
    second = g._stage_one("row", 4, torch.arange(2, dtype=torch.int32) + 5)
    assert second.tolist() == [5, 6, 6, 6]
    # Fixed storage: a per-step allocation would hand out a new address every
    # step, which a capture cannot follow.
    assert g._buffers["row"].data_ptr() == ptr


def test_staging_a_wrong_dtype_fails_loudly():
    g = _graph()
    with pytest.raises(AssertionError, match="baked"):
        g._stage_one("row", 4, torch.zeros(2, dtype=torch.int64))


def test_staging_a_source_whose_leading_axis_is_not_the_batch_fails_loudly():
    """Every staged source is indexed as `[rows, *StagedInput.shape]`.

    MRoPE positions are the live case: they keep the token axis LAST (`[3, N]`),
    so a drafter must not hand them to a pass at all. Without this the mismatch
    surfaced as a `copy_` size error two frames down, which reads like a
    batch-size bug rather than a layout one.
    """
    g = _graph()
    with pytest.raises(AssertionError, match="leading axis is not the batch"):
        g._stage_one("row", 8, torch.zeros(3, 8, dtype=torch.int32))


def test_padding_a_pass_that_never_called_its_pad_rows_inert_is_refused():
    """`pads` is the only part of the contract nothing else can check.

    A pass that lies about it still runs and still returns the right shape; the
    fabricated rows just land in another sequence's KV.
    """
    src = {"row": torch.zeros(3, dtype=torch.int32)}
    assert _graph(pads=True).stage(4, src)["row"].shape[0] == 4

    g = _graph(pads=False)
    with pytest.raises(AssertionError, match="fabricated rows are inert"):
        g.stage(4, src)
    # Unpadded entry stays legal -- the contract is about fabricated rows only.
    assert g.stage(3, src)["row"].shape[0] == 3


def test_sources_that_disagree_on_the_batch_are_refused():
    """They describe one step, so one of them is not this step's."""
    g = _graph(
        inputs={
            "a": StagedInput(dtype=torch.int32),
            "b": StagedInput(dtype=torch.int32),
        }
    )
    with pytest.raises(AssertionError, match="batches of"):
        g.stage(
            8,
            {
                "a": torch.zeros(3, dtype=torch.int32),
                "b": torch.zeros(5, dtype=torch.int32),
            },
        )


def test_warmup_runs_the_epilogue_too_not_just_the_capturable_forward(monkeypatch):
    """Warming only the capturable forward leaves the rest of the pass cold.

    Measured: `hipModuleLoadData` per rank on the 16k/20/50c reproducer went
    0 -> 4 when a block's warm was narrowed to the backbone, because the LM head
    has its own per-shape flydsl builder and it then fired mid-serve.
    """
    monkeypatch.setenv("ATOM_DRAFT_CUDAGRAPH", "0")  # capture needs a GPU
    ran = []
    g = _graph(
        forward=lambda running_bs, **rows: ran.append("forward"),
        epilogue=lambda out, running_bs, **rows: ran.append("epilogue"),
    )
    g.warmup(8)
    assert ran == ["forward", "epilogue"]


def test_warmup_happens_before_the_pad_and_capture_gates(monkeypatch):
    """A pass that pads nothing and captures nothing is still WARMED.

    That ordering is deliberate -- the JIT is worth paying at startup either
    way -- but it means declining to pad does not keep a flavor's forward off
    the startup sweep. A flavor whose model cannot answer it must decline the
    whole pass.
    """
    monkeypatch.setenv("ATOM_DRAFT_CUDAGRAPH", "0")
    ran = []
    g = _graph(pads=False, forward=lambda running_bs, **rows: ran.append("forward"))
    assert not g.will_capture
    g.warmup(8)
    assert ran == ["forward"]


def test_warmup_inputs_sees_the_pass_own_buffers(monkeypatch):
    """The seed writes into the staging buffers, not into copies of them: the
    warm batch has to be the one the forward then reads."""
    monkeypatch.setenv("ATOM_DRAFT_CUDAGRAPH", "0")  # capture needs a GPU
    g = _graph(warmup_inputs=lambda running_bs, **rows: rows["row"].fill_(7))
    g.warmup(4)
    assert g._buffers["row"][:4].tolist() == [7, 7, 7, 7]


def test_run_replays_the_recording_instead_of_the_forward():
    """The replay seam. A `run` that returned the recorded output WITHOUT
    replaying would serve every step the previous step's draft tokens, and
    nothing else in the suite would notice."""
    replays, forwards = [], []

    class _Graph:
        def replay(self):
            replays.append(1)

    g = _graph(forward=lambda running_bs, **rows: forwards.append(running_bs))
    src = {"row": torch.zeros(4, dtype=torch.int32)}

    real = _ctx(running_bs=4)
    assert not g.is_captured(4)
    g.run(4, real, **g.stage(4, src))
    assert (forwards, replays) == ([4], []), "no recording: must run the forward"

    g._cuda_graphs[4] = (_Graph(), "recorded")
    assert g.is_captured(4)
    assert g.run(4, real, **g.stage(4, src)) == "recorded"
    assert (forwards, replays) == ([4], [1]), "recorded: must replay, not re-run"


def test_a_dummy_replays_in_lockstep_with_the_real_ranks():
    """`is_dummy_run` is per-rank, so no decision that feeds a collective may
    read it.

    One DP step has the rank holding work running real while the others run
    dummies purely to reach the collectives. Gating the replay on dummy-ness
    therefore splits the group: the real rank replays a recorded collective
    while the rest issue it eagerly, and all of them wait forever. Measured on
    V4-Flash-DSpark tp8 + DPA -- one 32-token request hung 8/8 with exactly
    that split (rank 0 `dummy=False` replaying, ranks 1-7 `dummy=True` eager),
    and returned in 1.0s once the term was gone.

    A dummy is safe to replay because `warmup` asserts the recording was made
    on a REAL context: the graph holds the real branch, and doing a real rank's
    work alongside it is the dummy's whole purpose.
    """
    replays, forwards = [], []

    class _Graph:
        def replay(self):
            replays.append(1)

    g = _graph(forward=lambda running_bs, **rows: forwards.append(running_bs))
    g._cuda_graphs[1] = (_Graph(), "recorded")
    src = {"row": torch.zeros(1, dtype=torch.int32)}

    for dummy in (False, True):
        ctx = _ctx(running_bs=1, dummy=dummy)
        assert g.run(1, ctx, **g.stage(1, src)) == "recorded"
        assert " graph" in g.label(1, 1, ctx)
    assert (forwards, replays) == ([], [1, 1]), "both ranks replay, neither runs"


@pytest.mark.parametrize("use_cudagraph", [True, False])
def test_the_replay_decision_never_differs_between_a_dummy_and_a_real_step(
    use_cudagraph,
):
    """The property the case above pins, stated over the whole decision.

    Every term `_replays` reads has to be one the whole DP group agrees on;
    `use_cudagraph` and the batch are (``ForwardMode.decide`` derives both from
    DP-unified counts), and nothing else may enter. Asserting the equality
    rather than the two values is what makes a future third term fail here.
    """
    g = _graph()
    g._cuda_graphs[8] = ("graph", "out")
    real = _ctx(running_bs=8, use_cudagraph=use_cudagraph)
    dummy = _ctx(running_bs=8, use_cudagraph=use_cudagraph, dummy=True)
    assert g.target_running_bs(8, real) == g.target_running_bs(8, dummy)
    assert g.label(8, 8, real) == g.label(8, 8, dummy)


def test_the_capture_gate_reaches_the_env_that_names_it(monkeypatch):
    """The env has to be the answer, not merely documented as the answer.

    It once was not: the flag behind this was a literal False while
    `ATOM_DRAFT_CUDAGRAPH` sat in envs.py with no reader at all, so setting it
    enabled nothing -- and nothing noticed, because the default was off then too.
    """
    g = _graph()
    monkeypatch.setenv("ATOM_DRAFT_CUDAGRAPH", "0")
    assert not g.will_capture
    monkeypatch.setenv("ATOM_DRAFT_CUDAGRAPH", "1")
    assert g.will_capture
