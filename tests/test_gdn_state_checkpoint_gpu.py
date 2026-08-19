# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A checkpoint copied out of a forward must equal one taken by re-running.

The rest of the state-cache suite runs on CPU with AITER mocked, so it can
prove the scheduler reserves the right positions and stays green no matter
what the copy-out kernel does with them. This file is the other half: it
launches `write_state_checkpoints` for real and compares what lands in the
pool against a state computed the slow, obviously-correct way -- a second
forward over just the tokens up to that position.

Three things are under test, and they fail differently:

- **Interior targets** read `h[:, j]`, the chunked recurrence's per-chunk
  state. This is the whole point of the change: it is what lets a checkpoint
  sit mid-prompt without cutting the prefill in two. Every assertion here is
  *exact*, which is not what this file originally claimed -- see
  `test_interior_is_exact_not_approximate` for why the bf16 `h` costs nothing
  once the state reaches a bf16 pool.

- **End targets** read the runtime slot instead, because `h` holds boundaries
  strictly *before* the end and `chunk_offsets[row] + T // 64` is the next
  sequence's first chunk. Reading `h` there would silently pick up another
  sequence's state, so this one is checked for exact equality.

- **Varlen bases.** `h` and the conv input are batch-concatenated, so every
  target's indices are relative to `chunk_offsets[row]` and `cu_seqlens[row]`.
  A missing base still writes a correctly-shaped state -- just the wrong
  sequence's -- which nothing downstream can detect. The multi-sequence cases
  use *different* lengths so a dropped base cannot coincidentally land right.

The conv half is checked alongside every SSM assertion rather than in its own
test: the two halves share `slots`, so a checkpoint whose state is at P and
whose window is at Q is exactly the corruption the single entry point exists
to prevent, and it only shows up when both are read together.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "launches the checkpoint copy-out kernel; needs a real GPU",
        allow_module_level=True,
    )

from atom.model_ops.fla_ops.chunk import (
    CHUNK_SIZE,
    chunk_gated_delta_rule,
    pop_last_intermediate_states,
)
from atom.model_ops.fla_ops.index import prepare_chunk_offsets
from atom.model_ops.fla_ops.state_checkpoint import (
    write_state_checkpoints,
)

H, K, V = 4, 64, 64
STATE_LEN = 3  # linear_conv_kernel_dim - 1
D = 2 * H * K + H * V  # the conv width: q, k packed at K plus v at V
FUSED_EXTRA = 17  # junk columns, so `x` is a non-contiguous slice like the real one


def _inputs(lens, seed=0):
    """Varlen GDN inputs for sequences of the given lengths, plus their conv input.

    `x` is deliberately a *column slice* of a wider tensor. In the model the
    conv input is a slice of the fused in-projection, so its row stride is the
    fused width rather than D; a kernel that derives the stride from the shape
    reads the right number of bytes from the wrong place, and every value it
    copies is plausible. Handing it a contiguous tensor here would hide that.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = sum(lens)
    kw = {"device": "cuda", "dtype": torch.bfloat16, "generator": g}
    q = torch.randn(1, t, H, K, **kw)
    k = torch.nn.functional.normalize(torch.randn(1, t, H, K, **kw), p=2, dim=-1)
    v = torch.randn(1, t, H, V, **kw)
    beta = torch.rand(1, t, H, **kw).sigmoid()
    gate = torch.nn.functional.logsigmoid(torch.rand(1, t, H, **kw).float())
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device="cuda")
    fused = torch.randn(t, D + FUSED_EXTRA, device="cuda", dtype=torch.bfloat16)
    return q, k, v, gate, beta, cu, fused[:, FUSED_EXTRA:]


def _pools(num_slots):
    """The GDN state pool, at the dtype production allocates it.

    bf16 for *both* families, not fp32: `GDNStateMixin._state_dtypes` returns
    `config.torch_dtype` twice, and only `kimi_linear` overrides the v side to
    fp32 -- which is excluded from this path anyway (see
    `_KimiMLAGDNCommon.state_transfer`). Getting this wrong is not cosmetic. An
    fp32 pool here would record `h`'s bf16 rounding as a difference from the
    fp32 reference and make the copy-out look approximate, when in production
    the same rounding happens on the way into the pool no matter which source
    it came from. That mistake is what the docstrings this file replaced said.
    """
    ssm = torch.zeros(num_slots, H, K, V, device="cuda", dtype=torch.bfloat16)
    # [slot, state_len, D] allocated, [slot, D, state_len] as the impl sees it,
    # so the kernel meets the same transposed view it does in production.
    conv = torch.zeros(num_slots, STATE_LEN, D, device="cuda", dtype=torch.bfloat16)
    return ssm, conv.transpose(-1, -2)


def _run(q, k, v, gate, beta, cu, initial, keep):
    return chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
        keep_intermediate_states=keep,
    )


def _slice(t, cu, row, lo=0, hi=None):
    a, b = int(cu[row]), int(cu[row + 1])
    return t[:, a + lo : (b if hi is None else a + hi)]


def _reference_state(q, k, v, gate, beta, cu, row, upto, initial, pool_dtype=None):
    """The state after `upto` tokens of `row`, by re-running only those tokens.

    Deliberately the dumb way: a fresh forward whose sequence simply ends where
    the checkpoint is, so nothing about how `h` is laid out or indexed can be
    wrong in both the reference and the thing it checks. This is exactly what
    the *old* design did to get a checkpoint -- cut the prefill so the position
    became a step end -- so it is not merely a correct answer, it is the answer
    the 17.5% throughput cost was buying.

    `pool_dtype` casts the result the way storing it would. Compare against the
    cast version: the pool is bf16, so a re-run's state is rounded on the way
    in just as `h`'s is, and comparing the raw fp32 final state instead would
    charge the copy-out for a rounding both paths pay.
    """
    args = [_slice(t, cu, row, hi=upto) for t in (q, k, v, gate, beta)]
    cu2 = torch.tensor([0, upto], device="cuda")
    _, final = _run(*args, cu2, initial[row : row + 1].contiguous(), keep=False)
    return final[0] if pool_dtype is None else final[0].to(pool_dtype)


def _targets(entries, device="cuda"):
    """`_checkpoint_targets`' output, built by hand.

    Each entry is (row, slot, off, is_end, runtime_slot) -- the same five
    columns `GDNStateMixin._checkpoint_targets` zips out of the scheduler's
    reservations. Built here rather than driving the scheduler so a failure
    points at the kernel and not at the engine layer above it.
    """
    cols = zip(*entries)
    return [torch.tensor(c, dtype=torch.int32, device=device) for c in cols]


def _write(h, ssm, x, conv, entries, cu):
    rows, slots, offs, is_end, runtime = _targets(entries)
    write_state_checkpoints(
        h,
        ssm,
        x,
        conv,
        rows,
        slots,
        offs,
        is_end,
        runtime,
        prepare_chunk_offsets(cu, CHUNK_SIZE),
        cu,
        CHUNK_SIZE,
    )


def _expected_conv(x, cu, row, off):
    """The STATE_LEN tokens ending at the target, in the pool's [D, L] layout."""
    end = int(cu[row]) + off
    return x[end - STATE_LEN : end].transpose(0, 1)


def _forward_and_write(lens, entries, num_slots, seed=0):
    """One prefill step with checkpoints, returning everything worth asserting on."""
    q, k, v, gate, beta, cu, x = _inputs(lens, seed)
    n = len(lens)
    initial = torch.randn(n, H, K, V, device="cuda", dtype=torch.float32)
    ssm, conv = _pools(num_slots)
    # The runtime slots hold the incoming state, exactly as the pool does when
    # the impl gathers `initial_state` out of it.
    for row in range(n):
        ssm[_runtime_of(entries, row)] = initial[row]

    _, final = _run(q, k, v, gate, beta, cu, initial, keep=True)
    h = pop_last_intermediate_states()
    assert h is not None, "keep_intermediate_states=True must leave states to pop"
    # The impl's scatter, which must land before the write so an `is_end`
    # target reads this step's final state rather than the previous one's.
    for row in range(n):
        ssm[_runtime_of(entries, row)] = final[row]

    _write(h, ssm, x, conv, entries, cu)
    return {
        "q": q,
        "k": k,
        "v": v,
        "gate": gate,
        "beta": beta,
        "cu": cu,
        "x": x,
        "initial": initial,
        "ssm": ssm,
        "conv": conv,
        "final": final,
    }


def _runtime_of(entries, row):
    for r, _slot, _off, _end, runtime in entries:
        if r == row:
            return runtime
    raise AssertionError(f"no target for row {row}")


def _assert_conv(res, row, off, slot):
    torch.testing.assert_close(
        res["conv"][slot], _expected_conv(res["x"], res["cu"], row, off)
    )


def _want(res, row, off):
    """What a shortened re-run would have left in the pool at this position."""
    return _reference_state(
        res["q"],
        res["k"],
        res["v"],
        res["gate"],
        res["beta"],
        res["cu"],
        row,
        off,
        res["initial"],
        pool_dtype=res["ssm"].dtype,
    )


def _assert_state(res, slot, want):
    """Exact, deliberately: see `test_interior_is_exact_not_approximate`."""
    torch.testing.assert_close(res["ssm"][slot], want, rtol=0, atol=0)


def test_interior_checkpoint_matches_a_shortened_rerun():
    """The state at a mid-prompt boundary, without ending the forward there."""
    off = 2 * CHUNK_SIZE
    res = _forward_and_write([5 * CHUNK_SIZE], [(0, 7, off, 0, 3)], num_slots=8)
    _assert_state(res, 7, _want(res, 0, off))
    _assert_conv(res, 0, off, 7)


@pytest.mark.parametrize("cut", [1, 2, 3, 4])
def test_interior_is_exact_not_approximate(cut):
    """A checkpoint out of `h` equals a cut prefill's, bit for bit.

    Worth its own test because the obvious reasoning says otherwise, and this
    file was first written believing it. `h` is bf16 while the recurrence
    carries fp32, so reading a state out of it looks like it must lose
    precision relative to ending the forward at that position. It does not:
    the pool is bf16 too (`_pools`), so the *stored* checkpoint is rounded on
    the way in whichever source it came from, and the two roundings are of the
    same fp32 value. The rounding is real; the *difference* is not.

    This matters beyond tidiness. "Interior checkpoints are approximate" is an
    argument for keeping the prefill cut that costs 17.5% throughput -- pay the
    cut, get the exact state. There is nothing to buy. Anyone who reaches for
    that argument should land here.

    Swept across every interior boundary rather than one: the equality holds
    per chunk, and a source of drift that accumulated with distance from the
    start would show at cut=4 while cut=1 stayed clean.
    """
    off = cut * CHUNK_SIZE
    res = _forward_and_write([5 * CHUNK_SIZE], [(0, 7, off, 0, 3)], num_slots=8)
    assert res["ssm"].dtype == torch.bfloat16, "the premise above; see `_pools`"
    _assert_state(res, 7, _want(res, 0, off))


def test_resuming_from_an_interior_checkpoint_reproduces_the_tail():
    """End to end: the tokens after a checkpoint decode to the same outputs.

    The state-equality tests above compare against a re-run of the *prefix*.
    This one closes the loop the way a cache hit actually does -- restore the
    checkpoint, run only the remaining tokens, and compare that tail to what
    the uncut forward produced for the same positions. If the checkpoint were
    subtly wrong in a way that state comparison tolerated, the recurrence would
    carry it forward and it would show up here amplified, not damped.
    """
    t, cut = 8 * CHUNK_SIZE, 3 * CHUNK_SIZE
    res = _forward_and_write([t], [(0, 7, cut, 0, 3)], num_slots=8, seed=3)
    q, k, v, g, b, cu = (res[x] for x in ("q", "k", "v", "gate", "beta", "cu"))
    o_full, _ = _run(q, k, v, g, b, cu, res["initial"], keep=False)
    tail = [x[:, cut:t] for x in (q, k, v, g, b)]
    cu2 = torch.tensor([0, t - cut], device="cuda")
    o_tail, _ = _run(*tail, cu2, res["ssm"][7:8].float().contiguous(), keep=False)
    torch.testing.assert_close(o_tail, o_full[:, cut:t], rtol=0, atol=0)


def test_end_checkpoint_is_an_exact_copy_of_the_runtime_slot():
    """`h` cannot serve the end; the runtime slot can, and byte for byte."""
    t = 4 * CHUNK_SIZE
    res = _forward_and_write([t], [(0, 7, t, 1, 3)], num_slots=8)
    _assert_state(res, 7, res["ssm"][3])
    _assert_state(res, 7, res["final"][0].to(res["ssm"].dtype))
    _assert_conv(res, 0, t, 7)


def test_end_checkpoint_does_not_read_the_next_sequences_chunk():
    """The end of row 0 is where row 1's first chunk lives in `h`.

    This is the case that makes `is_end` necessary rather than a convenience:
    with two sequences packed together, `chunk_offsets[0] + T0 // 64` is a
    valid index into `h` that belongs to row 1. An implementation that dropped
    the `is_end` branch would read it, get a real state, and be wrong.
    """
    t0, t1 = 3 * CHUNK_SIZE, 4 * CHUNK_SIZE
    res = _forward_and_write(
        [t0, t1], [(0, 6, t0, 1, 2), (1, 7, t1, 1, 3)], num_slots=8
    )
    _assert_state(res, 6, res["final"][0].to(res["ssm"].dtype))
    _assert_state(res, 7, res["final"][1].to(res["ssm"].dtype))
    assert not torch.equal(res["ssm"][6], res["ssm"][7])


@pytest.mark.parametrize("row", [0, 1, 2])
def test_varlen_interior_uses_its_own_sequences_base(row):
    """Ragged lengths, so a dropped `chunk_offsets` base cannot land right.

    The lengths are distinct multiples of the chunk grid and the checkpoint
    sits at a different offset in each, so reading row r's state with row 0's
    base picks up a state from the wrong sequence entirely.
    """
    lens = [2 * CHUNK_SIZE, 5 * CHUNK_SIZE, 3 * CHUNK_SIZE]
    off = CHUNK_SIZE
    entries = [(r, 4 + r, off, 0, r) for r in range(3)]
    res = _forward_and_write(lens, entries, num_slots=8, seed=1)
    _assert_state(res, 4 + row, _want(res, row, off))
    _assert_conv(res, row, off, 4 + row)


def test_several_checkpoints_in_one_sequence():
    """One row can hold a grid rung, a demand and its prompt end at once.

    `state_save_all` is a list *per sequence*, so this is the ordinary case,
    not an edge one -- and it is where a kernel that assumed one target per
    row would write every state into the last slot.
    """
    t = 6 * CHUNK_SIZE
    entries = [
        (0, 4, 1 * CHUNK_SIZE, 0, 0),
        (0, 5, 3 * CHUNK_SIZE, 0, 0),
        (0, 6, t, 1, 0),
    ]
    res = _forward_and_write([t], entries, num_slots=8)
    for slot, off, is_end in ((4, CHUNK_SIZE, 0), (5, 3 * CHUNK_SIZE, 0), (6, t, 1)):
        want = res["final"][0].to(res["ssm"].dtype) if is_end else _want(res, 0, off)
        _assert_state(res, slot, want)
        _assert_conv(res, 0, off, slot)
    assert not torch.equal(res["ssm"][4], res["ssm"][5])


def test_mixed_interior_and_end_across_rows():
    """Both source paths in one launch, which is how a real step arrives.

    The grid is split by `program_id(1)`, not by target, so every program in
    the launch takes the `is_end` branch independently. A branch that leaked
    across targets shows up here and nowhere in the single-kind tests.
    """
    t0, t1 = 4 * CHUNK_SIZE, 3 * CHUNK_SIZE
    entries = [(0, 5, 2 * CHUNK_SIZE, 0, 0), (1, 6, t1, 1, 1)]
    res = _forward_and_write([t0, t1], entries, num_slots=8, seed=2)
    _assert_state(res, 5, _want(res, 0, 2 * CHUNK_SIZE))
    _assert_state(res, 6, res["final"][1].to(res["ssm"].dtype))
    _assert_conv(res, 0, 2 * CHUNK_SIZE, 5)
    _assert_conv(res, 1, t1, 6)


def test_checkpoints_do_not_perturb_the_forward():
    """Asking for `h` must not change `o` or the final state.

    `keep_intermediate_states` only parks a reference, but it is the flag that
    stops `SUPPRESS_LEVEL` from dropping `h`, so it is worth showing the
    kernel's own outputs are untouched -- a regression here would be an
    accuracy loss on every request, not just cached ones.
    """
    q, k, v, gate, beta, cu, _ = _inputs([5 * CHUNK_SIZE])
    initial = torch.randn(1, H, K, V, device="cuda", dtype=torch.float32)
    o_off, f_off = _run(q, k, v, gate, beta, cu, initial, keep=False)
    assert pop_last_intermediate_states() is None
    o_on, f_on = _run(q, k, v, gate, beta, cu, initial, keep=True)
    assert pop_last_intermediate_states() is not None
    torch.testing.assert_close(o_on, o_off, rtol=0, atol=0)
    torch.testing.assert_close(f_on, f_off, rtol=0, atol=0)


def test_pop_consumes_so_a_later_step_cannot_read_a_stale_h():
    """Second pop returns None; a forward that did not ask leaves nothing.

    The failure this guards is quiet: a step with no checkpoints popping the
    *previous* step's `h` would index it with this step's offsets, which is a
    different token count and possibly a different shape.
    """
    q, k, v, gate, beta, cu, _ = _inputs([2 * CHUNK_SIZE])
    initial = torch.randn(1, H, K, V, device="cuda", dtype=torch.float32)
    _run(q, k, v, gate, beta, cu, initial, keep=True)
    assert pop_last_intermediate_states() is not None
    assert pop_last_intermediate_states() is None
    _run(q, k, v, gate, beta, cu, initial, keep=False)
    assert pop_last_intermediate_states() is None


def test_untargeted_slots_are_left_alone():
    """A checkpoint writes its own slot and nothing near it.

    Neighbouring slots belong to other requests' live state, so an off-by-one
    in the slot arithmetic corrupts a sequence that has nothing to do with the
    checkpoint -- and does it to both pools at once.
    """
    off = 2 * CHUNK_SIZE
    res = _forward_and_write([4 * CHUNK_SIZE], [(0, 5, off, 0, 0)], num_slots=8)
    for slot in (1, 2, 3, 4, 6, 7):
        assert not res["ssm"][slot].any(), f"ssm slot {slot} was written"
        assert not res["conv"][slot].any(), f"conv slot {slot} was written"
