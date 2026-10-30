import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

xgr = pytest.importorskip("xgrammar")

from atom.model_engine.structured_output import (
    StructuredOutput,
    structured_stop_token_ids,
)


def test_missing_generation_config_uses_engine_and_tokenizer_stops():
    assert structured_stop_token_ids(
        NS(generation_config=None, stop_token_ids=[]), NS(eos_token_id=127)
    ) == [127]
    assert structured_stop_token_ids(
        NS(stop_token_ids=[125, 127]), NS(eos_token_id=127)
    ) == [125, 127]


@pytest.fixture
def controller():
    obj = StructuredOutput.__new__(StructuredOutput)
    obj.xgr = xgr
    obj.stop_ids = [127]
    obj.vocab_size = 128
    info = xgr.TokenizerInfo([chr(i) for i in range(128)], stop_token_ids=[127])
    obj.compiler = xgr.GrammarCompiler(info, max_threads=1)
    obj.compile = obj.compiler.compile_json_schema
    obj.pending = {}
    return obj


def batch(history=(), marker=(), prefill=0, schema=None, mixed=False):
    if schema is None:
        schema = {
            "type": "object",
            "properties": {"x": {"const": 1}},
            "required": ["x"],
            "additionalProperties": False,
        }
    return NS(
        structured_requests={0: (json.dumps(schema), marker, list(history))},
        req_ids=[11, 12] if mixed else [11],
        is_final_chunk=None,
        total_seqs_num_prefill=prefill,
    )


def test_masks_only_requested_rows(controller):
    b = batch(mixed=True)
    logits = torch.zeros(2, 128)
    controller.apply(logits, b, None)
    assert torch.isneginf(logits[0, ord("x")])
    assert logits[0, ord("{")] == 0
    assert torch.equal(logits[1], torch.zeros(128))


def test_deferred_output_and_replay(controller):
    b = batch(prefill=1)
    prefixes = controller.apply(torch.zeros(1, 128), b, None)
    controller.commit(torch.tensor([[ord("{")]]), b, prefixes)
    # Scheduler history is one delivery behind; worker must not open twice.
    b = batch()
    logits = torch.zeros(1, 128)
    controller.apply(logits, b, None)
    assert torch.isneginf(logits[0, ord("{")])
    assert logits[0, ord('"')] == 0
    # A replay must reconstruct the scheduler history, not keep a stale suffix.
    b = batch(prefill=1)
    logits = torch.zeros(1, 128)
    controller.apply(logits, b, None)
    assert logits[0, ord("{")] == 0


@pytest.mark.parametrize("draft", ['{"', "xx", "{x"])
def test_speculative_rows_follow_drafts_without_committing_them(controller, draft):
    b = batch()
    metadata = NS(
        target_logits_indices=torch.tensor([0, 1]),
        bonus_logits_indices=torch.tensor([2]),
        num_draft_tokens_np=np.array([2]),
        input_ids=torch.tensor([0, ord(draft[0]), ord(draft[1])]),
    )
    logits = torch.zeros(3, 128)
    prefixes = controller.apply(logits, b, metadata)
    assert torch.isneginf(logits[0, ord("x")])
    if draft[0] == "{":
        assert torch.isneginf(logits[1, ord("{")])
    # Commit only the chosen replacement, not the invalid draft branch.
    controller.commit(torch.tensor([[ord("{"), -1, -1]]), b, prefixes)
    logits = torch.zeros(1, 128)
    controller.apply(logits, batch(), None)
    assert logits[0, ord('"')] == 0
    assert torch.isneginf(logits[0, ord("{")])


def test_reasoning_closes_before_schema_starts(controller):
    marker = tuple(map(ord, "</t>"))
    logits = torch.zeros(1, 128)
    controller.apply(logits, batch(list(map(ord, "abc")), marker), None)
    assert logits[0, ord("x")] == 0
    assert torch.isneginf(logits[0, 127])
    logits = torch.zeros(1, 128)
    controller.apply(logits, batch(list(map(ord, "abc</t>")), marker), None)
    assert torch.isneginf(logits[0, ord("x")])
    assert logits[0, ord("{")] == 0


def test_valid_output_and_stop(controller):
    expected = '{"x": 1}'
    history = []
    for token in list(map(ord, expected)) + [127]:
        b = batch(history)
        logits = torch.zeros(1, 128)
        prefixes = controller.apply(logits, b, None)
        assert torch.isfinite(logits[0, token]), chr(token)
        controller.commit(torch.tensor([[token]]), b, prefixes)
        history.append(token)
    assert json.loads(expected) == {"x": 1}
