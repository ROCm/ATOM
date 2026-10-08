"""The draft-graph warmup must replay the metadata `propose` builds.

`propose` follows `_enter_decode_metadata` with `_advance_decode_metadata`,
whose `prepare_mtp_decode` replaces the target's verify-width persistent MLA
work plan with the one-row-per-sequence plan the draft's attention reads.
`_step_warmup_inputs` replays both for the draft graphs; without the second
half a DPA (persistent-mode) MLA draft walked the target's plan over a one-row
query and faulted at CUDA graph capture.

Replaying it means bumping `kv_indptr` and `context_lens` in place, once per
capture size, so `warmup_draft_graphs` has to hand every size the state the
target's capture left -- or a later size's `kv_indptr` runs backwards.
"""

import types
from unittest.mock import patch

import pytest

# Named on the module, not on "aiter", for the reason tests/test_dspark.py gives:
# a test collected after one that stubs `sys.modules["aiter"]` dies on a symbol
# rather than the name, and `exc_type` turns that ImportError into a skip too.
mod_eagle = pytest.importorskip(
    "atom.spec_decode.eagle_proposer",
    reason="the drafter imports aiter at module load",
    exc_type=ImportError,
)

import torch

from atom.spec_decode.drafter import Drafter

EagleProposer = mod_eagle.EagleProposer
PAD_SLOT_ID = mod_eagle.PAD_SLOT_ID

TARGET_QUERY_WIDTH = 5  # mtp_k=4 verify: 1 + 4 rows per sequence
CAPTURE_MAX_SEQLEN_K = 9


def _fake_proposer(num_attention_heads, *, fused, returns_slots=False):
    calls = []

    def prepare_mtp_decode(
        bs, max_seqlen_q, max_seqlen_k, positions, only_update=False, **kwargs
    ):
        calls.append(
            {
                "bs": bs,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "positions": positions.clone(),
                "only_update": only_update,
                "num_reject_tokens": kwargs.get("num_reject_tokens"),
                "update_context_lens": kwargs.get("update_context_lens", False),
                "positions_out": kwargs.get("positions_out"),
            }
        )
        out = {"work_indptr": "draft-plan", "work_info_set": "draft-info"}
        if returns_slots:
            # A backend that recomputes the draft's write slot (MHA, QSA).
            out["slot_mapping"] = torch.tensor([11, 12, 13], dtype=torch.int32)
        return out

    builder = types.SimpleNamespace(
        num_attention_heads=num_attention_heads,
        fuse_mtp_decode_position_update=fused,
        prepare_mtp_decode=prepare_mtp_decode,
    )
    proposer = types.SimpleNamespace(
        _share_mtp_indices=False,
        device=torch.device("cpu"),
        runner=types.SimpleNamespace(attn_metadata_builder=builder),
        # The real rewrite drops the metadata to one row per sequence and
        # hands back the target's verify width.
        _enter_decode_metadata=lambda *a: (TARGET_QUERY_WIDTH, None),
    )
    proposer._advance_decode_metadata = types.MethodType(
        EagleProposer._advance_decode_metadata, proposer
    )
    return proposer, calls


def _warm_one_size(proposer, running_bs=3):
    attn_metadata = types.SimpleNamespace(
        max_seqlen_q=1,
        max_seqlen_k=CAPTURE_MAX_SEQLEN_K,
        context_lens=torch.tensor([7, 8, 9], dtype=torch.int32),
        work_indptr="target-plan",
        work_info_set="target-info",
    )
    fc = types.SimpleNamespace(attn_metadata=attn_metadata)
    positions = torch.zeros(running_bs, dtype=torch.int64)
    with patch("atom.spec_decode.eagle_proposer.get_forward_context", return_value=fc):
        EagleProposer._step_warmup_inputs(proposer, running_bs, positions=positions)
    return attn_metadata, positions


@pytest.mark.parametrize(
    "num_attention_heads, expect_update, expect_q",
    [(64, True, TARGET_QUERY_WIDTH), (32, False, 1)],
)
def test_step_warmup_installs_draft_work_plan(
    num_attention_heads, expect_update, expect_q
):
    proposer, calls = _fake_proposer(num_attention_heads, fused=True)
    attn_metadata, _ = _warm_one_size(proposer)

    # Same call `propose` makes at step 0, mirroring its only_update rule.
    (prepare,) = calls
    assert prepare["bs"] == 3
    assert prepare["only_update"] is expect_update
    assert prepare["max_seqlen_q"] == expect_q
    assert prepare["num_reject_tokens"].shape == (3,)
    assert not prepare["num_reject_tokens"].any()
    # The target's plan is replaced by the draft's.
    assert attn_metadata.work_indptr == "draft-plan"
    assert attn_metadata.work_info_set == "draft-info"


def test_step_warmup_advances_like_propose_fused():
    proposer, calls = _fake_proposer(64, fused=True)
    attn_metadata, positions = _warm_one_size(proposer)

    prepare = calls[0]
    # One drafted token further than the capture's context.
    assert prepare["max_seqlen_k"] == CAPTURE_MAX_SEQLEN_K + 1
    assert attn_metadata.max_seqlen_k == CAPTURE_MAX_SEQLEN_K + 1
    # The fused backend bumps positions / context_lens itself.
    assert prepare["update_context_lens"] is True
    assert prepare["positions_out"] is positions
    assert prepare["positions"].tolist() == [7, 8, 9]


def test_step_warmup_advances_like_propose_unfused():
    proposer, calls = _fake_proposer(64, fused=False)
    attn_metadata, positions = _warm_one_size(proposer)

    prepare = calls[0]
    assert prepare["max_seqlen_k"] == CAPTURE_MAX_SEQLEN_K + 1
    # Bumped by the caller, before the backend reads them.
    assert prepare["update_context_lens"] is False
    assert prepare["positions"].tolist() == [8, 9, 10]
    assert attn_metadata.context_lens.tolist() == [8, 9, 10]
    assert positions.tolist() == [8, 9, 10]


def test_step_warmup_reblanks_recomputed_slots():
    # Capture blanks every cache write target so the warmup forward writes
    # nowhere; a backend that recomputes the slot must not undo that.
    proposer, _ = _fake_proposer(64, fused=True, returns_slots=True)
    attn_metadata, _ = _warm_one_size(proposer)
    assert attn_metadata.slot_mapping.tolist() == [PAD_SLOT_ID] * 3


class _Buf:
    def __init__(self, t):
        self.gpu = t


class _AdvancingPass:
    """Stands in for a DraftGraph whose warmup runs `prepare_mtp_decode`.

    Applies what the MLA kernel does to `kv_indptr` at page size 1
    (`kv_indptr += cu_seqlens_q`, the latter an arange after the rewrite) and
    what the fused update does to `context_lens`, then records the result.
    """

    name = "step"

    def __init__(self, var):
        self.var = var
        self.seen = {}

    def warmup(self, bs, *, pool=None, stream=None):
        kv_indptr = self.var["kv_indptr"].gpu
        kv_indptr[: bs + 1] += torch.arange(bs + 1, dtype=kv_indptr.dtype)
        self.var["context_lens"].gpu[:bs] += 1
        self.seen[bs] = (
            kv_indptr[: bs + 1].clone(),
            self.var["context_lens"].gpu[:bs].clone(),
        )
        return pool


def _warmup_all_sizes(warmup_draft_graphs, capture_sizes=(4, 2, 1)):
    var = {
        "kv_indptr": _Buf(torch.zeros(9, dtype=torch.int32)),
        "context_lens": _Buf(torch.full((8,), 7, dtype=torch.int32)),
    }
    pass_ = _AdvancingPass(var)
    proposer = object.__new__(EagleProposer)
    proposer.runner = types.SimpleNamespace(
        forward_vars=var,
        capture_sizes=list(capture_sizes),
        graph_pool=None,
        rank=0,
    )
    proposer.draft_graphs = (pass_,)
    proposer.config = None

    def build_context(bs):
        return types.SimpleNamespace(), types.SimpleNamespace()

    with (
        patch("atom.spec_decode.drafter.set_forward_context"),
        patch("torch.cuda.synchronize"),
    ):
        warmup_draft_graphs(proposer, build_context, stream=None)
    return var, pass_.seen


def test_accumulated_warmup_runs_kv_indptr_backwards():
    # The hazard, shown on the base loop: sizes warm smallest first, and each
    # one's in-place bump lands on the rows every earlier size already bumped.
    _, seen = _warmup_all_sizes(Drafter.warmup_draft_graphs)
    kv_indptr, _ = seen[4]
    assert kv_indptr.tolist() == [0, 3, 4, 3, 4]
    assert (kv_indptr.diff() < 0).any()


def test_every_size_warms_from_the_capture_state():
    var, seen = _warmup_all_sizes(EagleProposer.warmup_draft_graphs)
    for bs, (kv_indptr, context_lens) in seen.items():
        # One drafted token per sequence on top of an untouched capture.
        assert kv_indptr.tolist() == list(range(bs + 1))
        assert context_lens.tolist() == [8] * bs
    # And the buffers are left the way the capture left them.
    assert not var["kv_indptr"].gpu.any()
    assert var["context_lens"].gpu.tolist() == [7] * 8
