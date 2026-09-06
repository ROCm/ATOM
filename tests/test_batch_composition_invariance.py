"""A sequence's per-seq work must not depend on who shares its batch.

Motivated by a measured accuracy loss whose only known trigger is batch
COMPOSITION: on DeepSeek-V4-Flash-DSpark the checkpoint ladder loses ~3.9pp on
GSM8K when a prefill batch mixes a prompt long enough to be cut with one that
finishes in a single chunk, and recovers when the scheduler keeps those two
apart -- with the chunk boundaries themselves unchanged. Within the forward
those two sequences differ only in token count, so any component whose per-seq
output moves when a batch-mate changes is a candidate.

These are property tests, not golden values: each case builds the same sequence
alone and alongside a different-length one, and asserts the rows it owns are the
same. No GPU -- `make_compress_plans` is numpy plus a buffer stand-in.
"""

import importlib.util
import pathlib

import numpy as np
import pytest
import torch

_CP_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "atom/model_ops/v4_kernels/compress_plan.py"
)
_spec = importlib.util.spec_from_file_location("_compress_plan_invariance", _CP_PATH)
_cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cp)
make_compress_plans = _cp.make_compress_plans

# CSA r=4 overlap (K_pool=8), HCA r=128 non-overlap (K_pool=128).
RATIOS_OVERLAP = [(4, True), (128, False)]


class _FakeBuf:
    def __init__(self, rows):
        self.np = np.full((rows, 4), 7, dtype=np.int32)
        self._t = torch.from_numpy(self.np)

    def copy_to_gpu(self, n=None):
        return self._t if n is None else self._t[:n]


def _buffers(rows=8192):
    return {
        ratio: {"compress": _FakeBuf(rows), "write": _FakeBuf(rows)}
        for ratio, _ in RATIOS_OVERLAP
    }


def _plans(extend, context):
    return make_compress_plans(
        np.asarray(extend, dtype=np.int32),
        np.asarray(context, dtype=np.int32),
        RATIOS_OVERLAP,
        plan_buffers=_buffers(),
    )


def _rows_for_seq(plan_gpu, n_rows, extend, seq_idx):
    """The plan rows belonging to `seq_idx`, in a batch-independent form.

    Column 0 is the token's index into the batch's concatenated kv tensor, so it
    legitimately shifts with the sequences ahead of it; subtract that offset.
    Column 1 is the sequence's slot in the batch, equally legitimate. What is
    left -- which tokens were selected, their absolute positions, and how many
    state-cache rows each reads -- must not move.
    """
    rows = plan_gpu.numpy()[:n_rows]
    rows = rows[rows[:, 1] == seq_idx].copy()
    cu = int(np.sum(extend[:seq_idx]))
    rows[:, 0] -= cu
    rows[:, 1] = 0
    return rows


def _assert_seq_invariant(extend, context, seq_idx, *, label):
    """`seq_idx`'s rows, batched vs alone."""
    batched = _plans(extend, context)
    alone = _plans([extend[seq_idx]], [context[seq_idx]])
    for ratio, _ in RATIOS_OVERLAP:
        for kind in ("compress", "write"):
            b = getattr(batched[ratio], f"{kind}_plan_gpu")
            a = getattr(alone[ratio], f"{kind}_plan_gpu")
            nb = getattr(batched[ratio], f"num_{kind}")
            na = getattr(alone[ratio], f"num_{kind}")
            got = _rows_for_seq(b, nb, np.asarray(extend), seq_idx)
            want = _rows_for_seq(a, na, np.asarray([extend[seq_idx]]), 0)
            assert got.shape == want.shape, (
                f"{label} ratio={ratio} {kind}: seq {seq_idx} got "
                f"{got.shape[0]} rows batched vs {want.shape[0]} alone"
            )
            assert np.array_equal(got, want), (
                f"{label} ratio={ratio} {kind}: seq {seq_idx} rows differ by "
                f"batch composition\nbatched:\n{got}\nalone:\n{want}"
            )


class TestCompressPlanBatchComposition:
    # The measured-toxic shape: a prompt cut at the ladder rung (512, so its
    # context ends mid-prompt) beside one that finishes here.
    CUT = (512, 512)  # (extend, context) -- chunk 1 of a 543-token prompt
    WHOLE = (400, 400)  # a prompt short enough to need no cut
    RESUME = (88, 600)  # chunk 2 of that 543-token prompt, cut at 512

    @pytest.mark.parametrize("seq_idx", [0, 1])
    def test_cut_beside_whole(self, seq_idx):
        extend = [self.CUT[0], self.WHOLE[0]]
        context = [self.CUT[1], self.WHOLE[1]]
        _assert_seq_invariant(extend, context, seq_idx, label="cut+whole")

    @pytest.mark.parametrize("seq_idx", [0, 1])
    def test_cut_beside_resume(self, seq_idx):
        # The composition the isolation probe allows, and which scores clean.
        extend = [self.CUT[0], self.RESUME[0]]
        context = [self.CUT[1], self.RESUME[1]]
        _assert_seq_invariant(extend, context, seq_idx, label="cut+resume")

    def test_order_does_not_matter(self):
        """Same two sequences, swapped: each still owns the same rows."""
        a = _plans([self.CUT[0], self.WHOLE[0]], [self.CUT[1], self.WHOLE[1]])
        b = _plans([self.WHOLE[0], self.CUT[0]], [self.WHOLE[1], self.CUT[1]])
        for ratio, _ in RATIOS_OVERLAP:
            for kind in ("compress", "write"):
                cut_first = _rows_for_seq(
                    getattr(a[ratio], f"{kind}_plan_gpu"),
                    getattr(a[ratio], f"num_{kind}"),
                    np.asarray([self.CUT[0], self.WHOLE[0]]),
                    0,
                )
                cut_second = _rows_for_seq(
                    getattr(b[ratio], f"{kind}_plan_gpu"),
                    getattr(b[ratio], f"num_{kind}"),
                    np.asarray([self.WHOLE[0], self.CUT[0]]),
                    1,
                )
                assert np.array_equal(cut_first, cut_second), (
                    f"ratio={ratio} {kind}: the cut sequence's rows depend on "
                    f"its slot in the batch"
                )

    @pytest.mark.parametrize("n_short", [1, 4, 16])
    def test_one_cut_among_many_whole(self, n_short):
        """Scale the batch the way the scheduler does — the loss grew with the
        number of batch-mates, so a single short partner is not enough."""
        extend = [self.CUT[0]] + [self.WHOLE[0]] * n_short
        context = [self.CUT[1]] + [self.WHOLE[1]] * n_short
        _assert_seq_invariant(extend, context, 0, label=f"cut+{n_short}xwhole")

    def test_ragged_lengths_like_gsm8k(self):
        """The real distribution: prompts 289..922, the >=512 ones cut at 512."""
        lens = [289, 425, 511, 512, 512, 512, 543, 681, 922]
        extend = [min(n, 512) for n in lens]
        context = list(extend)  # every sequence's context ends at this chunk
        for i in range(len(lens)):
            _assert_seq_invariant(extend, context, i, label="gsm8k-shaped")


# --- swa_write: the ring a resumed chunk reads back -------------------------
#
# Needs a GPU (Triton kernel). Same property: the rows a sequence writes into
# its own ring slot must not move because a different-length sequence shares the
# forward. `write_per_batch` is the thing to watch -- the model passes
# `min(window_size, max_seqlen_q)`, and `max_seqlen_q` is a BATCH maximum, so it
# is the one per-seq input that a batch-mate can change.

_HEAD_DIM = 16
_RING_SLOTS = 11
_NUM_SLOTS = 5
_BLOCK_SIZE = 256


def _swa_env():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("swa_write is a Triton kernel; needs a real GPU")
    from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
        CSA_RATIO,
        DENSE_RATIO,
        HCA_RATIO,
        UnifiedPoolGeometry,
    )
    from atom.model_ops.v4_kernels.state_writes import swa_write

    geo = UnifiedPoolGeometry(
        [DENSE_RATIO, CSA_RATIO, HCA_RATIO],
        num_blocks=2,
        num_slots=_NUM_SLOTS,
        ring_slots=_RING_SLOTS,
        block_size=_BLOCK_SIZE,
    )
    return torch, geo, swa_write, (DENSE_RATIO, CSA_RATIO, HCA_RATIO)


def _run_swa(torch, geo, swa_write, ratio, seqs, window):
    """`seqs` = [(start_pos, n_tok, slot, kv)] -> the written plane."""
    dev = "cuda"
    cu = torch.zeros(len(seqs) + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.cumsum(
        torch.tensor([n for _, n, _, _ in seqs], dtype=torch.int32), 0
    ).to(dev)
    positions = torch.cat(
        [torch.arange(s, s + n, dtype=torch.int32) for s, n, _, _ in seqs]
    ).to(dev)
    kv = torch.cat([t for _, _, _, t in seqs]).to(dev)
    slots = torch.tensor([sl for _, _, sl, _ in seqs], dtype=torch.int32, device=dev)
    plane = torch.zeros(geo.plane_rows, _HEAD_DIM, dtype=torch.bfloat16, device=dev)
    # What deepseek_v4.py passes: the window, capped by the batch's longest seq.
    wpb = min(window, max(n for _, n, _, _ in seqs))
    swa_write(kv, positions, cu, slots, plane, geo.window_params(ratio), wpb)
    torch.cuda.synchronize()
    return plane


@pytest.mark.parametrize("mate_len", [4, 40, 512])
def test_swa_write_ring_is_independent_of_batch_mates(mate_len):
    """The cut sequence's ring rows, alone vs beside a shorter/longer mate."""
    torch, geo, swa_write, ratios = _swa_env()
    torch.manual_seed(0)
    n_cut = 64  # the sequence whose state a later chunk resumes from
    cut_kv = torch.randn(n_cut, _HEAD_DIM, dtype=torch.bfloat16)
    mate_kv = torch.randn(mate_len, _HEAD_DIM, dtype=torch.bfloat16)
    cut = (0, n_cut, 3, cut_kv)
    mate = (0, mate_len, 0, mate_kv)
    window = 8

    for ratio in ratios:
        alone = _run_swa(torch, geo, swa_write, ratio, [cut], window)
        with_mate = _run_swa(torch, geo, swa_write, ratio, [cut, mate], window)
        rows = [geo.window_params(ratio).index(3, p) for p in range(n_cut)]
        rows = sorted(set(rows))
        a = alone[rows]
        b = with_mate[rows]
        assert torch.equal(a, b), (
            f"ratio={ratio} mate_len={mate_len}: the cut sequence's ring rows "
            f"changed because a {mate_len}-token sequence shared the batch; "
            f"max|diff|={(a.float() - b.float()).abs().max()}"
        )


def test_swa_write_masks_programs_a_long_mate_adds():
    """A sequence shorter than `write_per_batch` must no-op the extra programs.

    The grid is `(bs, write_per_batch)` and `write_per_batch` rises with the
    batch's longest sequence, so a short sequence gets programs it does not own
    the moment a long one joins -- alone it gets none. Those programs index
    `cu + tok_n - write_per_batch + j`, which runs NEGATIVE for them; unmasked
    they would read another sequence's kv and scatter it into this one's ring.
    This is the one place `write_per_batch` is not algebraically inert, so it is
    the case worth pinning.
    """
    torch, geo, swa_write, ratios = _swa_env()
    torch.manual_seed(1)
    window = 8
    n_short = 4  # < window, so a long mate strictly raises write_per_batch
    n_long = 512
    short_kv = torch.randn(n_short, _HEAD_DIM, dtype=torch.bfloat16)
    long_kv = torch.randn(n_long, _HEAD_DIM, dtype=torch.bfloat16)
    # Put the short sequence second so its kv is preceded by the long one's:
    # a negative index would land in real data rather than off the tensor.
    short = (0, n_short, 3, short_kv)
    long_ = (0, n_long, 0, long_kv)

    for ratio in ratios:
        alone = _run_swa(torch, geo, swa_write, ratio, [short], window)
        mixed = _run_swa(torch, geo, swa_write, ratio, [long_, short], window)
        rows = sorted({geo.window_params(ratio).index(3, p) for p in range(n_short)})
        a, b = alone[rows], mixed[rows]
        assert torch.equal(a, b), (
            f"ratio={ratio}: a {n_short}-token sequence's ring changed when a "
            f"{n_long}-token sequence raised write_per_batch from {n_short} to "
            f"{window}; max|diff|={(a.float() - b.float()).abs().max()}"
        )
