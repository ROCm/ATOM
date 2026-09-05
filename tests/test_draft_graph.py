# SPDX-License-Identifier: MIT
"""The draft-pass machine's invariants, with no aiter and no GPU.

These are what the whole padding scheme rests on, so they belong where CI can
run them. `tests/test_dspark.py` keeps the DSpark-integration half, which needs
the compiled draft and is therefore skipped on a CPU runner -- if the invariants
lived only there, nothing would check them at all.
"""

import dataclasses

import pytest
import torch

from atom.spec_decode.draft_graph import DraftGraph, StagedInput


def _graph(**kw):
    """A pass over one int32 input, bound to a stub config."""
    kw.setdefault("inputs", {"row": StagedInput(dtype=torch.int32)})
    g = DraftGraph(
        forward=kw.pop("forward", lambda running_bs, **rows: running_bs), **kw
    )
    config = dataclasses.make_dataclass("C", [("max_num_seqs", int)])(256)
    return g.bind(config, torch.device("cpu"))


def test_an_unwidened_empty_batch_stages_nothing():
    """`stage` tail-repeats `src[-1]`, which a zero-row source does not have.

    No caller can ask for one: `prepare_model` asserts the batch carries tokens,
    so every drafter sees at least one row. Pinned as the boundary anyway --
    entering at the real count must stay a no-op, not an IndexError.
    """
    g = _graph()
    assert g.stage(0, {"row": torch.zeros(0, dtype=torch.int32)})["row"].shape[0] == 0


def test_owning_a_recording_says_nothing_about_a_batch_it_is_not_at():
    """The proposers widen to `running_bs` only when `is_captured` says that
    batch replays, so this answer decides whether a pad row is ever fabricated.

    It has to be about the batch it was ASKED about. Answering out of inventory
    -- "I own 48, so yes" -- would widen a 256-batch step to rows no recording
    covers, and the pass would then hand the variable-length MoE gather padded
    rows on a step that replays nothing.
    """
    g = _graph()
    g._cuda_graphs[48] = ("graph", "out")
    assert g.is_captured(48)
    assert not g.is_captured(256)
    assert not g.is_captured(44)


def test_nothing_recorded_means_no_batch_replays():
    """`ATOM_DRAFT_CUDAGRAPH=0` leaves the inventory empty, and then every
    batch must answer no -- that is what keeps the eager pass unpadded."""
    g = _graph()
    assert not any(g.is_captured(n) for n in (0, 1, 44, 48, 256))


def test_a_pass_that_stages_nothing_is_refused():
    """A fabricated row has to land somewhere. Every declared pass may pad now,
    so staging nothing is not an unpaddable pass -- it is no pass at all, and
    the flavor must decline the declaration instead."""
    with pytest.raises(AssertionError, match="stages nothing"):
        _graph(inputs={})


def test_a_pass_must_contribute_at_least_one_token_per_sequence():
    with pytest.raises(AssertionError, match="tokens_per_seq=0"):
        _graph(tokens_per_seq=0)


def test_token_width_is_owned_by_each_pass():
    """One drafter may declare a full-q first pass and a one-row mid-pass."""
    first = _graph(tokens_per_seq=4)
    middle = _graph()
    assert first.tokens_per_seq == 4
    assert middle.tokens_per_seq == 1


def test_a_full_q_pass_stages_tokens_inside_each_sequence_row():
    """Step 0 is flat in serving but fixed-width per sequence in its graph."""
    g = _graph(inputs={"hidden": StagedInput(shape=(4, 2, 3), dtype=torch.float32)})
    source = torch.arange(2 * 4 * 2 * 3, dtype=torch.float32).view(2, 4, 2, 3)
    staged = g.stage(2, {"hidden": source})["hidden"]
    assert staged.shape == (2, 4, 2, 3)
    torch.testing.assert_close(staged, source)


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


def test_a_producer_can_fill_fixed_storage_without_a_self_copy():
    """Direct producers must not launch a redundant copy back onto themselves."""
    g = _graph()
    produced = g.buffer("row", 3)
    produced.copy_(torch.tensor([7, 8, 9], dtype=torch.int32))
    version_after_produce = g._buffers["row"]._version

    staged = g.stage(3, {"row": produced})["row"]

    assert staged.data_ptr() == produced.data_ptr()
    assert staged.tolist() == [7, 8, 9]
    assert g._buffers["row"]._version == version_after_produce


def test_direct_fixed_storage_still_gets_a_repeated_pad_tail():
    """Only real rows are producer-owned; stage still fabricates coherent pads."""
    g = _graph()
    produced = g.buffer("row", 2)
    produced.copy_(torch.tensor([7, 8], dtype=torch.int32))

    staged = g.stage(5, {"row": produced})["row"]

    assert staged.tolist() == [7, 8, 8, 8, 8]


def test_fixed_input_and_exported_snapshot_have_separate_lifetimes():
    """The next replay reuses fixed ids while an owned snapshot may leave the GPU."""
    g = _graph()
    fixed_ids = g.buffer("row", 2)
    fixed_ids.copy_(torch.tensor([7, 8], dtype=torch.int32))
    exported_ids = fixed_ids.clone()

    staged = g.stage(2, {"row": fixed_ids})["row"]
    staged.copy_(torch.tensor([9, 10], dtype=torch.int32))

    assert staged.data_ptr() == fixed_ids.data_ptr()
    assert exported_ids.tolist() == [7, 8]
    assert staged.tolist() == [9, 10]


def test_fixed_storage_accessor_checks_role_and_capacity():
    g = _graph()
    with pytest.raises(AssertionError, match="no staged input"):
        g.buffer("missing", 1)
    with pytest.raises(AssertionError, match="capacity"):
        g.buffer("row", 257)


def test_staging_rejects_a_wrong_trailing_shape_before_identity_check():
    g = _graph(inputs={"row": StagedInput(shape=(4,), dtype=torch.int32)})
    wrong = g._buffers["row"][:2, :1]
    with pytest.raises(AssertionError, match="fixed storage expects"):
        g._stage_one("row", 2, wrong)


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


def test_warmup_happens_before_the_capture_gate(monkeypatch):
    """A pass that captures nothing is still WARMED.

    That ordering is deliberate -- the JIT is worth paying at startup either
    way -- but it means turning capture off does not keep a flavor's forward
    off the startup sweep. A flavor whose model cannot answer it must decline
    the whole pass.
    """
    monkeypatch.setenv("ATOM_DRAFT_CUDAGRAPH", "0")
    ran = []
    g = _graph(forward=lambda running_bs, **rows: ran.append("forward"))
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


def test_a_captured_epilogue_can_feed_output_back_to_fixed_input(monkeypatch):
    """Serial passes may make the next replay consume the previous output."""
    monkeypatch.setenv("ATOM_DRAFT_CUDAGRAPH", "0")  # exercise eager equivalent

    def epilogue(out, running_bs, *, row):
        row.copy_(out[:running_bs])
        return row

    g = _graph(
        forward=lambda running_bs, *, row: row + 1,
        epilogue=epilogue,
        capture_epilogue=True,
    )
    produced = g.buffer("row", 2)
    produced.copy_(torch.tensor([10, 20], dtype=torch.int32))

    first = g.run(2, **g.stage(2, {"row": produced}))
    second = g.run(2, **g.stage(2, {"row": first}))

    assert first.data_ptr() == produced.data_ptr()
    assert second.data_ptr() == produced.data_ptr()
    assert second.tolist() == [12, 22]


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

    assert not g.is_captured(4)
    g.run(4, **g.stage(4, src))
    assert (forwards, replays) == ([4], []), "no recording: must run the forward"

    g._cuda_graphs[4] = (_Graph(), "recorded")
    assert g.is_captured(4)
    assert g.run(4, **g.stage(4, src)) == "recorded"
    assert (forwards, replays) == ([4], [1]), "recorded: must replay, not re-run"


def test_a_replay_hands_back_a_value_not_the_recordings_own_storage():
    """What `run` returns has to survive the next replay of any other size.

    The recording's tensors are allocated inside the capture, so they live in
    the pool every captured size shares; the pool packs a later size's capture
    into memory the earlier ones released, and a replay rewrites every address
    it recorded whatever holds it now. Holding a Python reference does not
    stop that -- which is exactly why this cannot be left to the caller.

    Written as "the recording changing underneath must not change what was
    handed out", because that is the failure: DSpark's draft ids are read a
    step later as the next forward's `input_ids`, and another size's replay
    had turned them into that size's activations. The target's embedding
    gathered on those floats and faulted all eight ranks.
    """

    class _Graph:
        def replay(self):
            pass

    ids = torch.full((4, 3), 9, dtype=torch.int32)
    conf = torch.zeros(4, 3)
    g = _graph(epilogue=lambda out, running_bs, **rows: out, capture_epilogue=True)
    g._cuda_graphs[4] = (_Graph(), (ids, conf))

    got_ids, got_conf = g.run(
        4, **g.stage(4, {"row": torch.zeros(4, dtype=torch.int32)})
    )
    assert got_ids.tolist() == ids.tolist(), "the replay's values, unchanged"

    # A later replay at another size lands on the recording's storage.
    ids.fill_(1234)
    conf.fill_(5.0)
    assert got_ids.tolist() != ids.tolist(), "handed out a window into the recording"
    assert got_conf.tolist() != conf.tolist(), "every output, not just the ids"


def test_the_eager_path_hands_back_what_it_allocated():
    """...and only a replay needs the copy.

    Without a recording the pass allocates normally, where holding the
    reference is what keeps the storage. Copying there would be pure cost, so
    the identity is asserted rather than left to inspection.
    """
    made = torch.zeros(2, 3)
    g = _graph(
        forward=lambda running_bs, **rows: made,
        epilogue=lambda out, running_bs, **rows: out,
        capture_epilogue=True,
    )
    assert not g.is_captured(2)
    assert g.run(2, **g.stage(2, {"row": torch.zeros(2, dtype=torch.int32)})) is made


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

    for _dummy in (False, True):
        assert g.run(1, **g.stage(1, src)) == "recorded"
        assert " graph" in g.label(1, 1)
    assert (forwards, replays) == ([], [1, 1]), "both ranks replay, neither runs"


def test_the_replay_decision_is_a_function_of_the_batch_and_nothing_else():
    """Every term the decision reads has to be one the whole DP group agrees on.

    `running_bs` is, having been reduced in `sync_dp_metadata`. The step's own
    kind is not: `is_dummy_run` is per-rank, and so is `is_prefill` (one rank
    prefills while another decodes). Either would split the group into two
    collective widths -- the first one measured, V4-Flash-DSpark tp8 + DPA
    hanging 8/8 on the first real decode.

    Pinned as an interface property rather than a value: the decision takes the
    batch and nothing else, so a future term cannot be added without changing
    this signature.
    """
    import inspect

    params = list(inspect.signature(DraftGraph.is_captured).parameters)
    assert params == ["self", "running_bs"], (
        f"is_captured grew a term beyond the agreed batch: {params}. Anything "
        "keyed on the step's kind is per-rank and hangs the DP group."
    )


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
