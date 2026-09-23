"""The draft-graph warmup must install the draft's own attention work plan.

`propose` follows `_enter_decode_metadata` with `prepare_mtp_decode`, which
replaces the target's verify-width persistent MLA work plan with the
one-row-per-sequence plan the draft's attention reads. `_step_warmup_inputs`
replays that rewrite for the draft graphs; without the second half a DPA
(persistent-mode) MLA draft walked the target's plan over a one-row query and
faulted at CUDA graph capture.
"""

import types
from unittest.mock import patch

import pytest
import torch

from atom.spec_decode.eagle_proposer import EagleProposer

TARGET_QUERY_WIDTH = 5  # mtp_k=4 verify: 1 + 4 rows per sequence


def _fake_proposer(num_attention_heads):
    calls = []

    def prepare_mtp_decode(
        bs, max_seqlen_q, max_seqlen_k, positions, only_update=False, **kwargs
    ):
        calls.append(
            {
                "bs": bs,
                "max_seqlen_q": max_seqlen_q,
                "only_update": only_update,
                "num_reject_tokens": kwargs.get("num_reject_tokens"),
            }
        )
        return {"work_indptr": "draft-plan", "work_info_set": "draft-info"}

    builder = types.SimpleNamespace(
        num_attention_heads=num_attention_heads,
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
    return proposer, calls


@pytest.mark.parametrize(
    "num_attention_heads, expect_update, expect_q",
    [(64, True, TARGET_QUERY_WIDTH), (32, False, 1)],
)
def test_step_warmup_installs_draft_work_plan(
    num_attention_heads, expect_update, expect_q
):
    running_bs = 3
    proposer, calls = _fake_proposer(num_attention_heads)
    attn_metadata = types.SimpleNamespace(
        max_seqlen_q=1,
        max_seqlen_k=9,
        context_lens=torch.tensor([7, 8, 9], dtype=torch.int32),
        work_indptr="target-plan",
        work_info_set="target-info",
    )
    fc = types.SimpleNamespace(attn_metadata=attn_metadata)
    positions = torch.zeros(running_bs, dtype=torch.int64)

    with patch("atom.spec_decode.eagle_proposer.get_forward_context", return_value=fc):
        EagleProposer._step_warmup_inputs(proposer, running_bs, positions=positions)

    # Same call `propose` makes at step 0, mirroring its only_update rule.
    assert len(calls) == 1
    call = calls[0]
    assert call["bs"] == running_bs
    assert call["only_update"] is expect_update
    assert call["max_seqlen_q"] == expect_q
    assert call["num_reject_tokens"].shape == (running_bs,)
    assert not call["num_reject_tokens"].any()
    # The target's plan is replaced by the draft's.
    assert attn_metadata.work_indptr == "draft-plan"
    assert attn_metadata.work_info_set == "draft-info"
    assert positions.tolist() == [7, 8, 9]
