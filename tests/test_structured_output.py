import json
from types import SimpleNamespace as NS
from typing import ClassVar

import numpy as np
import pytest
import torch

llg = pytest.importorskip("llguidance")

from atom.model_engine import structured_grammar as xgr
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

    class AsciiTokenizer:
        eos_token_id = 127
        bos_token_id = None
        tokens: ClassVar[list[bytes]] = [bytes([i]) for i in range(128)]
        special_token_ids: ClassVar[list[int]] = [127]

        def __call__(self, value):
            return list(value if isinstance(value, bytes) else value.encode())

    info = llg.LLTokenizer(llg.TokenizerWrapper(AsciiTokenizer()))
    obj.compiler = xgr.GrammarCompiler(info)
    obj.compile = obj.compiler.compile_json_schema
    obj.pending = {}
    obj.states = {}
    obj.current = {}
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


def test_incremental_masks_match_replay_without_reconsuming_history(controller):
    original = controller.matcher
    replayed = []

    def counted(schema, marker, history):
        replayed.append(list(history))
        return original(schema, marker, history)

    controller.matcher = counted
    history = []
    for token in list(map(ord, '{"x": 1}')) + [127]:
        b = batch(history)
        logits = torch.zeros(1, 128)
        prefixes = controller.apply(logits, b, None)
        reference, active = original(*b.structured_requests[0])
        mask = xgr.allocate_token_bitmask(1, 128)
        reference.fill_next_token_bitmask(mask)
        allowed = (
            (mask[0, torch.arange(128) // 32] >> (torch.arange(128) % 32)) & 1
        ).bool()
        assert active and torch.equal(torch.isfinite(logits[0]), allowed)
        controller.commit(torch.tensor([[token]]), b, prefixes)
        history.append(token)
    assert replayed == [[]]


def test_speculative_reasoning_marker_can_end_inside_a_draft(controller):
    marker = tuple(map(ord, "</t>"))
    b = batch(list(map(ord, "abc</t")), marker)
    metadata = NS(
        target_logits_indices=torch.tensor([0, 1]),
        bonus_logits_indices=torch.tensor([2]),
        num_draft_tokens_np=np.array([2]),
        input_ids=torch.tensor([0, ord(">"), ord("{")]),
    )
    logits = torch.zeros(3, 128)
    prefixes = controller.apply(logits, b, metadata)
    assert torch.isfinite(logits[0, ord("x")])
    assert torch.isneginf(logits[1, ord("x")])
    assert torch.isfinite(logits[1, ord("{")])
    assert torch.isfinite(logits[2, ord('"')])
    # Reject the entire draft: committed reasoning has not ended.
    controller.commit(torch.tensor([[ord("x"), -1, -1]]), b, prefixes)
    logits = torch.zeros(1, 128)
    controller.apply(logits, batch(list(map(ord, "abc</t")), marker), None)
    assert torch.isfinite(logits[0, ord("x")])
    assert torch.isneginf(logits[0, 127])


def test_completed_reasoning_state_advances_only_committed_tokens(controller):
    marker = tuple(map(ord, "</t>"))
    b = batch(list(map(ord, "abc</t")), marker)
    prefixes = controller.apply(torch.zeros(1, 128), b, None)
    controller.commit(torch.tensor([[ord(">"), ord("{")]]), b, prefixes)
    logits = torch.zeros(1, 128)
    controller.apply(logits, batch(list(map(ord, "abc</t")), marker), None)
    assert torch.isfinite(logits[0, ord('"')])
    assert torch.isneginf(logits[0, ord("{")])


def test_state_does_not_leak_across_schema_change_or_request_retirement(controller):
    b = batch()
    prefixes = controller.apply(torch.zeros(1, 128), b, None)
    controller.commit(torch.tensor([[ord("{")]]), b, prefixes)
    changed = batch(schema={"const": 3}, prefill=1)
    logits = torch.zeros(1, 128)
    controller.apply(logits, changed, None)
    assert torch.isfinite(logits[0, ord("3")])
    assert torch.isneginf(logits[0, ord('"')])
    controller.clear()
    assert not controller.pending and not controller.states and not controller.current


def test_empty_batch_does_not_touch_gpu_metadata_or_logits(controller):
    b = batch()
    b.structured_requests = {}
    # Any tensor access would raise on these sentinels.
    assert controller.apply(object(), b, object()) == {}


@pytest.mark.parametrize("required", [False, True])
def test_native_tools_are_an_alternative_to_schema_text(controller, required):
    spec = json.dumps(
        {"prefix": "", "section": "calls", "names": ["weather"], "required": required}
    )
    compiled = controller.compile('{"const":{"answer":42}}', spec)
    examples = [
        ('{"answer":42}', not required),
        ('{"answer":41}', False),
        (
            '<calls><invoke name="weather"><parameter name="city">Paris</parameter></invoke></calls>',
            True,
        ),
        ('<calls><invoke name="unknown"></invoke></calls>', False),
        ("plain non-JSON text", False),
    ]
    for text, expected in examples:
        matcher = xgr.GrammarMatcher(compiled)
        assert (
            all(matcher.accept_token(t) for t in [*text.encode(), 127]) == expected
        ), text


def test_tool_format_is_part_of_incremental_state_identity(controller):
    b = batch(schema={"const": {"answer": 42}})
    fmt = json.dumps(
        {"prefix": "", "section": "calls", "names": ["weather"], "required": False}
    )
    schema, marker, history = b.structured_requests[0]
    b.structured_requests[0] = ((schema, fmt), marker, history)
    logits = torch.zeros(1, 128)
    p = controller.apply(logits, b, None)
    assert torch.isfinite(logits[0, ord("<")])
    controller.commit(torch.tensor([[ord("<")]]), b, p)
    logits = torch.zeros(1, 128)
    controller.apply(logits, batch(schema={"const": {"answer": 42}}, prefill=1), None)
    assert torch.isneginf(logits[0, ord("<")])


@pytest.mark.parametrize(
    "value_schema,good,bad",
    [
        ({"allOf": [{"type": "integer", "minimum": 2}, {"maximum": 5}]}, "3", "6"),
        (
            {"type": "integer", "multipleOf": 7, "minimum": 10, "maximum": 30},
            "14",
            "15",
        ),
        ({"type": "integer", "not": {"const": 0}}, "2", "0"),
        (
            {
                "type": "array",
                "items": {"enum": [1, 2, 3]},
                "uniqueItems": True,
                "minItems": 3,
            },
            "[1,2,3]",
            "[1,1,3]",
        ),
    ],
)
def test_supported_predicates_reject_invalid_tokens(
    controller, value_schema, good, bad
):
    schema = {
        "type": "object",
        "properties": {"v": value_schema},
        "required": ["v"],
        "additionalProperties": False,
    }
    compiled = controller.compile(json.dumps(schema))
    for value, expected in [(good, True), (bad, False)]:
        matcher = xgr.GrammarMatcher(compiled)
        tokens = list(map(ord, '{"v":' + value + "}")) + [127]
        assert all(matcher.accept_token(t) for t in tokens) == expected


def test_unsupported_unique_cannot_enable_lenient_compilation(controller):
    schema = {
        "type": "array",
        "items": {"type": "integer"},
        "uniqueItems": True,
        "x-guidance": {"lenient": True},
    }
    with pytest.raises(ValueError, match="finite item/prefix domains"):
        controller.compile(json.dumps(schema))


@pytest.mark.parametrize(
    "kind,lo,hi,small,large",
    [
        ("string", "minLength", "maxLength", '"x"', '"xxx"'),
        ("array", "minItems", "maxItems", "[1]", "[1,2,3]"),
        ("object", "minProperties", "maxProperties", '{"a":1}', '{"a":1,"b":2,"c":3}'),
    ],
)
@pytest.mark.parametrize("negate_min", [False, True])
def test_float_cardinality_complement_is_enforced_during_decoding(
    controller, kind, lo, hi, small, large, negate_min
):
    schema = {"type": kind, "not": {lo if negate_min else hi: 2.0}}
    compiled = controller.compile(json.dumps(schema))
    for text, expected in [(small, negate_min), (large, not negate_min)]:
        matcher = xgr.GrammarMatcher(compiled)
        assert all(matcher.accept_token(t) for t in [*text.encode(), 127]) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU transfer test")
def test_gpu_deferred_commit_and_draft_copy(controller, monkeypatch):
    """No CUDA tolist, and commit must not synchronously consume its own copy."""
    original = torch.Tensor.tolist

    def cpu_only(tensor):
        assert tensor.device.type == "cpu", "synchronous GPU tolist"
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", cpu_only)
    b = batch(prefill=1)
    p = controller.apply(torch.zeros(1, 128, device="cuda"), b, None)
    sampled = torch.tensor([[ord("{")]], device="cuda")
    with monkeypatch.context() as m:

        def no_wait(*args, **kwargs):
            raise AssertionError("commit waited for D2H")

        m.setattr(torch.cuda.Event, "synchronize", no_wait)
        controller.commit(sampled, b, p)
    assert not controller.pending  # CPU grammar not advanced yet
    # Simulate the next RPC with a fresh, reordered batch and lagging history.
    b = batch(mixed=True)
    b.req_ids = [12, 11]
    b.structured_requests = {1: b.structured_requests[0]}
    ids = torch.tensor([0, 0, 0, 0, ord('"'), ord("x")], device="cuda")
    md = NS(
        target_logits_indices_np=np.array([0, 1, 3, 4]),
        bonus_logits_indices_np=np.array([2, 5]),
        num_draft_tokens_np=np.array([2, 2]),
        input_ids=ids,
    )
    # GPU row metadata must never be needed by the CPU grammar path.
    md.target_logits_indices = md.bonus_logits_indices = object()
    controller.prepare(b, md)
    # Next forward may reuse input staging. The async copy owns its snapshot.
    ids.fill_(ord("!"))
    md.target_logits_indices_np.fill(99)
    md.bonus_logits_indices_np.fill(99)
    logits = torch.zeros(6, 128, device="cuda")
    p = controller.apply(logits, b, md)
    assert torch.isfinite(logits[3, ord('"')])
    assert torch.isneginf(logits[3, ord("{")])
    assert torch.isfinite(logits[4, ord("x")])
    assert torch.isfinite(logits[5, ord('"')])
    assert torch.equal(logits[:3], torch.zeros(3, 128, device="cuda"))
    # Reject the draft; commit a different legal token (whitespace).
    controller.commit(torch.tensor([[-1, -1, -1], [32, -1, -1]], device="cuda"), b, p)
    # Reorder again and deliver the previous result to the scheduler.
    logits = torch.zeros(1, 128, device="cuda")
    controller.prepare(batch([ord("{")]), None)
    controller.apply(logits, batch([ord("{")]), None)
    assert torch.isfinite(logits[0, ord('"')])
    assert torch.isneginf(logits[0, ord("x")])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU transfer test")
def test_async_commit_replay_cancel_and_buffer_reuse(controller):
    # Replay wins over a still-in-flight old grammar state.
    b = batch(prefill=1)
    p = controller.apply(torch.zeros(1, 128, device="cuda"), b, None)
    controller.commit(torch.tensor([[ord("{")]], device="cuda"), b, p)
    logits = torch.zeros(1, 128, device="cuda")
    p = controller.apply(logits, batch(prefill=1), None)
    assert torch.isfinite(logits[0, ord("{")])
    controller.commit(torch.tensor([[ord("{")]], device="cuda"), b, p)
    # Cancellation drops state without waiting. Reuse the copy slot afterwards.
    controller.clear()
    for token in [3, 4]:
        b = batch(schema={"const": token}, prefill=1)
        p = controller.apply(torch.zeros(1, 128, device="cuda"), b, None)
        controller.commit(torch.tensor([[ord(str(token))]], device="cuda"), b, p)
        controller.clear()
    assert controller.commit_records is None
    assert not controller.pending and not controller.states


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU transfer test")
def test_gpu_metadata_requires_prefetch(controller):
    md = NS(input_ids=torch.zeros(3, device="cuda"))
    with pytest.raises(ValueError, match="not prefetched"):
        controller.apply(torch.zeros(1, 128, device="cuda"), batch(), md)
