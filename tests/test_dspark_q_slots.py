# SPDX-License-Identifier: MIT
"""A request with no live draft q reads NO q -- never another request's row.

The accept kernel reaches a request's q by indirection: `draft_probs_row_base`
answers the first row, and `coin * q < p` is tested there. The fallback for a
request the map does not know used to be row 0. Row 0 is in range and holds
q > 0, so it passes the kernel's probability guard and the ratio test then
accepts UNCONDITIONALLY against somebody else's distribution -- measured at
24.4% of inline accuracy lost while acceptance ROSE to 3.01, a failure shaped
like a win. The answer is the -1 sentinel, and the second half of the answer is
that a request should not BE unknown: persistent per-request slots, so a q
survives the prefill-only steps between the propose that drew it and the verify
that tests it.

This file runs on a plain CPU runner, which is the only kind CI has. That is
deliberate and it is why the methods are lifted out of the proposer's source
rather than imported: `atom.spec_decode.dspark_proposer` pulls in the model
loader and cannot be imported without an AITER build, so importing it would put
this file behind `importorskip("aiter")` and the regression above would have
ZERO CI coverage -- see `.github/scripts/run_unit_tests.sh` and the note in
`tests/conftest.py` about four suites that went quiet exactly that way.
"""

import ast
import pathlib

import pytest
import torch

from atom.utils import CpuGpuBuffer

PROPOSER_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1] / "atom/spec_decode/dspark_proposer.py"
)
Q_SLOT_METHODS = (
    "_claim_q_slot",
    "record_draft_probs_slots",
    "draft_probs_row_base",
)


def _q_slot_methods():
    """`DSparkProposer`'s q bookkeeping, compiled out of the shipped file.

    Not a copy: the source read here is the one the engine imports, so a change
    to either method arrives in this test by itself. `torch` is the only
    module-level name the three of them reach for -- a return annotation, which
    is evaluated at definition -- so the namespace they run in carries that and
    nothing else, and a name added later raises `NameError` here rather than
    passing quietly against a stale transcription.
    """
    tree = ast.parse(PROPOSER_SOURCE.read_text(), str(PROPOSER_SOURCE))
    classes = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DSparkProposer"
    ]
    assert len(classes) == 1, "DSparkProposer moved or is defined twice"
    picked = [
        n
        for n in classes[0].body
        if isinstance(n, ast.FunctionDef) and n.name in Q_SLOT_METHODS
    ]
    assert sorted(n.name for n in picked) == sorted(Q_SLOT_METHODS), (
        f"DSparkProposer no longer defines all of {Q_SLOT_METHODS}; this test "
        "is asserting on methods that have been renamed or moved"
    )
    namespace: dict = {"torch": torch}
    exec(  # noqa: S102 - the source is this repository's own, read above
        compile(ast.Module(body=picked, type_ignores=[]), str(PROPOSER_SOURCE), "exec"),
        namespace,
    )
    return namespace


class QSlots:
    """The proposer's persistent-q seam, on CPU, over the real methods.

    Everything below the seam is stood in for: `propose` plays the part of the
    captured block head, which writes q into the scratch at BATCH SLOT rows
    `[0, bs * T)` and nothing else. That is the whole reason the persistent
    store exists -- a batch slot belongs to one step, and ATOM never mixes
    prefill and decode, so a prefill-only propose rewrites rows that belong to
    decode requests whose q has not been tested yet.

    A q row is a constant vector carrying a marker unique to the
    (request, position) that drew it, so "this request read its OWN q" is an
    exact comparison rather than a plausible one.
    """

    _methods = _q_slot_methods()
    _claim_q_slot = _methods["_claim_q_slot"]
    record_draft_probs_slots = _methods["record_draft_probs_slots"]
    draft_probs_row_base = _methods["draft_probs_row_base"]

    def __init__(self, max_num_seqs, tokens_per_seq, vocab=4, oversubscribe=2):
        # Only tested against None, so the sentinel for "lossless sampling is
        # armed" can be anything that is not None.
        self.draft_sampling = object()
        self.draft_tokens_per_seq = tokens_per_seq
        self.vocab = vocab
        slots = oversubscribe * max_num_seqs
        self._q_slot_of = {}
        self._q_free = list(range(slots - 1, -1, -1))
        self._q_store = torch.zeros(slots * tokens_per_seq, vocab)
        self._draft_probs_flat = torch.zeros(max_num_seqs * tokens_per_seq, vocab)
        self._q_rows_buf = CpuGpuBuffer(
            max_num_seqs * tokens_per_seq,
            dtype=torch.int64,
            device="cpu",
            pin_memory=False,
        )
        self._draft_probs_row_base_buf = CpuGpuBuffer(
            max_num_seqs, dtype=torch.int32, device="cpu", pin_memory=False
        )
        self._marker = 0
        self.drawn = {}

    def propose(self, req_ids):
        """One propose over ``req_ids``, then the copy-out into their slots."""
        t = self.draft_tokens_per_seq
        # The scratch is this step's and nothing else's: a slot not written by
        # this propose holds whatever the previous one left, which is exactly
        # the trap a request that read the scratch directly would fall into.
        for i, req_id in enumerate(req_ids):
            for k in range(t):
                self._marker += 1
                self._draft_probs_flat[i * t + k].fill_(self._marker)
                self.drawn[req_id, k] = self._marker
        self.record_draft_probs_slots(req_ids)

    def q_rows(self, req_id):
        """The rows the accept kernel would read for ``req_id``, or None for
        the -1 sentinel -- the two answers the kernel's `has_q` distinguishes."""
        base = int(self.draft_probs_row_base([req_id])[0].item())
        if base < 0:
            return None
        t = self.draft_tokens_per_seq
        return self._q_store[base : base + t]

    def reads_its_own_q(self, req_id):
        rows = self.q_rows(req_id)
        if rows is None:
            return False
        return all(
            torch.equal(
                rows[k], torch.full((self.vocab,), float(self.drawn[req_id, k]))
            )
            for k in range(self.draft_tokens_per_seq)
        )


@pytest.mark.parametrize("tokens_per_seq", (1, 5))
def test_a_request_the_q_map_never_saw_reads_the_sentinel_and_not_row_zero(
    tokens_per_seq,
):
    slots = QSlots(max_num_seqs=4, tokens_per_seq=tokens_per_seq)
    slots.propose(["a", "b"])
    bases = slots.draft_probs_row_base(["a", "b", "never-drafted"]).tolist()
    # Armed: the first free slot is 0, so row 0 IS a live row holding another
    # request's q -- which is what made the historic fallback pass the kernel's
    # `0 < q <= 1` guard and accept unconditionally.
    assert bases[0] == 0 and (slots._q_store[0] > 0).all()
    assert bases[1] == tokens_per_seq
    assert bases[2] == -1
    assert slots.reads_its_own_q("a") and slots.reads_its_own_q("b")
    assert slots.q_rows("never-drafted") is None


def test_an_expired_slot_is_a_sentinel_rather_than_its_new_owners_rows():
    """The only other way to be absent: a q slot handed to somebody else.

    `_claim_q_slot` recycles the least recently drafted slot when the free list
    runs out, and it DROPS the victim from the map on the way. Leaving the
    entry would point the kernel at a row that is in range, holds q > 0 and
    belongs to another request -- the 24.4% bug with one more step in front
    of it.
    """
    slots = QSlots(max_num_seqs=2, tokens_per_seq=2, oversubscribe=2)
    slots.propose(["stale"])
    stale_rows = slots.q_rows("stale").clone()
    for i in range(4):
        slots.propose([f"later-{i}"])
    assert slots.q_rows("stale") is None
    # The rows themselves were reused, which is what the sentinel is protecting
    # against: answering the old base would have read a live stranger's q.
    assert not torch.equal(slots._q_store[0:2], stale_rows)
    assert all(slots.reads_its_own_q(f"later-{i}") for i in range(4))


# The verify batch below is `decode_bs + 1`, and a batch never exceeds
# `max_num_seqs` -- which is what the row-base buffer is sized at.
@pytest.mark.parametrize("decode_bs", (1, 7, 15))
def test_a_prefill_only_propose_leaves_every_decode_request_its_own_q(decode_bs):
    """The root cause, as a sequence of proposes: decode, prefill, verify.

    ATOM never mixes prefill and decode in one batch, and a prefill-only step
    still runs the drafter -- over batch slots `[0, bs)`, which are the DECODE
    requests' scratch rows. Measured on hardware before persistent slots, that
    left 7.03% of verify rows with no live q, each forced to reject at position
    1 and accept zero drafts; 0.0000% after. So: zero sentinels, and every
    request reading the q IT drew rather than merely some q.
    """
    slots = QSlots(max_num_seqs=16, tokens_per_seq=3)
    decode = [f"decode-{i}" for i in range(decode_bs)]
    slots.propose(decode)
    before = slots.draft_probs_row_base(decode).tolist()
    # The prefill batch writes scratch rows [0, T), which decode-0 owns.
    slots.propose(["prefill"])
    bases = slots.draft_probs_row_base(decode + ["prefill"])
    assert (bases >= 0).all(), "a bystander lost its q to a prefill-only step"
    assert len(set(bases.tolist())) == decode_bs + 1, "two requests share q rows"
    # A decode request's rows are the ones it was given, unmoved.
    assert slots.draft_probs_row_base(decode).tolist() == before
    for req_id in decode + ["prefill"]:
        assert slots.reads_its_own_q(req_id), req_id
