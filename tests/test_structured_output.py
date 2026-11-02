# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from collections import OrderedDict
from types import SimpleNamespace as NS
from typing import ClassVar

import numpy as np
import pytest
import torch
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from atom.structured_output import StructuredOutput, StructuredRequest
from atom.structured_output import backend_guidance as xgr
from atom.structured_output.backend_guidance import grammar_tokenizer
from atom.structured_output.utils import StructuredOutputMask

llg = pytest.importorskip("llguidance")


@pytest.fixture
def controller(monkeypatch):
    class AsciiTokenizer:
        eos_token_id = 127
        bos_token_id = None
        tokens: ClassVar[list[bytes]] = [bytes([i]) for i in range(128)]
        special_token_ids: ClassVar[list[int]] = [127]

        def __call__(self, value):
            return list(value if isinstance(value, bytes) else value.encode())

    info = llg.LLTokenizer(llg.TokenizerWrapper(AsciiTokenizer()))
    monkeypatch.setattr(xgr, "grammar_tokenizer", lambda *args: info)
    return StructuredOutput(None, 128, [127])


def batch(
    controller,
    history=(),
    *,
    start=0,
    initial=True,
    marker="",
    schema=None,
    request_id=11,
    prefill=0,
):
    schema = json.dumps(schema or {"const": {"x": 1}})
    key = (schema, None, marker)
    if initial:
        controller.compile(*key)
    return NS(
        structured_requests={
            0: StructuredRequest(key if initial else None, start, tuple(history))
        },
        req_ids=[request_id],
        is_final_chunk=None,
        total_seqs_num_prefill=prefill,
    )


def prepare_reference(controller, b, drafts):
    """Stage CPU draft snapshots using the same row layout as runner metadata."""
    controller.draft_plan = {0: (list(range(len(drafts) + 1)), 0, len(drafts))}
    controller.draft_copy = NS(wait=lambda drafts=drafts: torch.tensor(drafts))
    return NS()


def test_valid_output_uses_only_acknowledged_deltas(controller):
    tokens = [*map(ord, '{"x":1}'), 127]
    identity = None
    for i, token in enumerate(tokens):
        b = batch(
            controller,
            tokens[i - 1 : i] if i else (),
            start=max(0, i - 1),
            initial=i == 0,
        )
        logits = torch.zeros(1, 128)
        p = controller.apply(logits, b, None)
        assert torch.isfinite(logits[0, token])
        state = controller.states[11]
        identity = identity or state.matcher
        assert state.matcher is identity
        controller.commit(torch.tensor([[token]]), b, p)
        assert len(state.pending) <= 1
    assert controller.states[11].matcher.is_terminated()
    assert not controller.errors


def test_unrelated_batch_preserves_matcher_and_pending_tokens(controller):
    b = batch(controller)
    p = controller.apply(torch.zeros(1, 128), b, None)
    controller.commit(torch.tensor([[ord("{")]]), b, p)
    state = controller.states[11]
    assert controller.apply(object(), NS(structured_requests={}), object()) == {}
    logits = torch.zeros(1, 128)
    controller.apply(logits, batch(controller, initial=False), None)
    assert controller.states[11] is state
    assert torch.isfinite(logits[0, ord('"')])


@pytest.mark.parametrize("draft", ['{"', "xx", "{x"])
def test_speculative_masks_roll_back_without_copying(controller, draft, monkeypatch):
    b = batch(controller)
    md = prepare_reference(controller, b, list(map(ord, draft)))
    monkeypatch.setattr(
        xgr.GrammarMatcher,
        "copy",
        lambda *args: pytest.fail("per-step matcher copy"),
        raising=False,
    )
    logits = torch.zeros(3, 128)
    p = controller.apply(logits, b, md)
    assert torch.isneginf(logits[0, ord("x")])
    controller.commit(torch.tensor([[ord("{"), -1, -1]]), b, p)
    logits = torch.zeros(1, 128)
    controller.apply(logits, batch(controller, initial=False), None)
    assert torch.isfinite(logits[0, ord('"')])
    assert not controller.errors


def test_reasoning_end_is_matched_as_text(controller):
    b = batch(controller, map(ord, "abc</t>"), marker="</t>")
    logits = torch.zeros(1, 128)
    controller.apply(logits, b, None)
    assert torch.isneginf(logits[0, ord("x")])
    assert torch.isfinite(logits[0, ord("{")])
    assert not controller.errors


def test_reasoning_end_inside_speculative_branch_rolls_back(controller):
    b = batch(controller, map(ord, "abc</t"), marker="</t>")
    md = prepare_reference(controller, b, list(map(ord, ">{")))
    logits = torch.zeros(3, 128)
    p = controller.apply(logits, b, md)
    assert torch.isfinite(logits[0, ord("x")])
    assert torch.isneginf(logits[1, ord("x")])
    assert torch.isfinite(logits[2, ord('"')])
    controller.commit(torch.tensor([[ord("x"), -1, -1]]), b, p)
    logits = torch.zeros(1, 128)
    controller.apply(
        logits, batch(controller, start=6, initial=False, marker="</t>"), None
    )
    assert torch.isfinite(logits[0, ord("x")])
    assert torch.isneginf(logits[0, 127])


def test_replay_discards_only_unacknowledged_output(controller):
    b = batch(controller)
    p = controller.apply(torch.zeros(1, 128), b, None)
    controller.commit(torch.tensor([[ord("{")]]), b, p)
    logits = torch.zeros(1, 128)
    controller.apply(logits, batch(controller, initial=False, prefill=1), None)
    assert torch.isfinite(logits[0, ord("{")])
    assert not controller.states[11].pending


def test_bad_request_does_not_fail_neighbor(controller):
    bad = batch(controller, map(ord, "invalid"))
    good = batch(controller, request_id=12)
    b = NS(
        structured_requests={
            0: bad.structured_requests[0],
            1: good.structured_requests[0],
        },
        req_ids=[11, 12],
        is_final_chunk=None,
        total_seqs_num_prefill=0,
    )
    logits = torch.zeros(2, 128)
    p = controller.apply(logits, b, None)
    assert set(controller.errors) == {11}
    assert list(p) == [1]
    assert torch.isfinite(logits[0, 127])
    assert torch.isfinite(logits[1, ord("{")])
    controller.release([11])
    assert 11 not in controller.states and not controller.errors
    assert 12 in controller.states


def test_partial_prefill_initializes_but_does_not_sample(controller):
    b = batch(controller, prefill=1)
    b.is_final_chunk = [False]
    assert controller.apply(torch.zeros(1, 128), b, None) == {}
    logits = torch.zeros(1, 128)
    controller.apply(logits, batch(controller, initial=False, prefill=1), None)
    assert torch.isfinite(logits[0, ord("{")])


def test_runtime_uses_exact_preflight_key(controller, monkeypatch):
    b = batch(controller)
    monkeypatch.setattr(
        controller.compiler,
        "compile_json_schema",
        lambda *args: pytest.fail("compiled during forward"),
    )
    controller.apply(torch.zeros(1, 128), b, None)
    assert not controller.errors


def test_native_tool_header_does_not_allow_undeclared_whitespace(controller):
    fmt = json.dumps(
        {
            "prefix": "|DSML|",
            "section": "function_calls",
            "names": ["get_weather"],
            "required": True,
        }
    )
    compiled = controller.compile("{}", fmt)
    good = '<|DSML|function_calls><|DSML|invoke name="get_weather">'
    for text, expected in [
        (good, True),
        (good.replace('"get_weather"', '" get_weather "'), False),
    ]:
        matcher = xgr.GrammarMatcher(compiled)
        assert all(matcher.accept_token(t) for t in map(ord, text)) is expected


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


def test_long_history_transports_only_new_tokens_and_freezes_snapshot():
    seq = NS(
        output_tokens=[1] * 16000,
        num_placeholder_tokens=0,
        structured_cursor=0,
        structured_started=False,
        json_schema="{}",
        structured_tool_format=None,
        structured_reasoning_end="",
    )
    first = StructuredRequest.capture(seq)
    assert len(first.tokens) == 16000 and first.schema == ("{}", None, "")
    seq.output_tokens.extend([2, 3, 127, 127])
    seq.num_placeholder_tokens = 2
    second = StructuredRequest.capture(seq)
    assert second.schema is None and second.start == 16000 and second.tokens == (2, 3)
    seq.output_tokens[-4] = 99
    assert second.tokens == (2, 3)
    assert StructuredRequest.capture(seq).tokens == ()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU transfer test")
def test_gpu_snapshot_overlap_reorder_and_replay(controller):
    first = batch(controller)
    p = controller.apply(torch.zeros(1, 128, device="cuda"), first, None)
    controller.commit(torch.tensor([[ord("{")]], device="cuda"), first, p)
    assert not controller.states[11].pending
    second = batch(controller, initial=False)
    md = NS(
        target_logits_indices_np=np.array([0, 1]),
        bonus_logits_indices_np=np.array([2]),
        num_draft_tokens_np=np.array([2]),
        input_ids=torch.tensor([0, ord('"'), ord("x")], device="cuda"),
    )
    controller.prepare(second, md)
    md.input_ids.fill_(ord("!"))
    md.target_logits_indices_np.fill(99)
    logits = torch.zeros(3, 128, device="cuda")
    p = controller.apply(logits, second, md)
    assert torch.isfinite(logits[0, ord('"')]) and torch.isfinite(logits[1, ord("x")])
    controller.commit(
        torch.tensor([[ord('"'), ord("x"), -1]], device="cuda"), second, p
    )
    # Scheduler acknowledges only '{'; replay must roll back the rest.
    replay = batch(controller, [ord("{")], initial=False, prefill=1)
    logits = torch.zeros(1, 128, device="cuda")
    controller.apply(logits, replay, None)
    assert torch.isfinite(logits[0, ord('"')])
    controller.release([11])
    assert not controller.states


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU transfer test")
def test_cancelled_async_snapshot_does_not_reuse_live_buffer(controller):
    b = batch(controller)
    p = controller.apply(torch.zeros(1, 128, device="cuda"), b, None)
    controller.commit(torch.tensor([[ord("{")]], device="cuda"), b, p)
    controller.release([11])
    b = batch(controller, request_id=12, schema={"const": 3})
    p = controller.apply(torch.zeros(1, 128, device="cuda"), b, None)
    controller.commit(torch.tensor([[ord("3")]], device="cuda"), b, p)
    controller._finish_commit()
    assert set(controller.states) == {12} and not controller.errors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_large_row_stride_uses_64_bit_addresses():
    stride = 2**30 + 17
    if torch.cuda.mem_get_info()[0] < 6 * 2**30:
        pytest.skip("requires 6 GiB free GPU memory for a >2**31 element offset")
    logits = torch.empty_strided(
        (3, 33), (stride, 1), device="cuda", dtype=torch.float16
    )
    logits.fill_(7)
    buffer = StructuredOutputMask(33)
    masks = buffer.prepare(2, logits.device)
    masks.fill_(-1)
    masks[0, 1] = 0  # mask row 2, token 32, beyond the signed-int32 offset limit
    masks[1, 0] = 0  # also exercise row zero
    buffer.apply(logits, [2, 0])
    actual = logits.cpu()
    expected = torch.full((3, 33), 7, dtype=torch.float16)
    expected[2, 32] = -float("inf")
    expected[0, :32] = -float("inf")
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab", [1, 31, 32, 33, 127, 129, 131073])
def test_packed_mask_preserves_exact_logits_and_unrequested_rows(device, dtype, vocab):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    torch.manual_seed(19)
    # Strided vocabulary and batch axes, and reordered speculative target/bonus
    # rows. High bits of signed int32 words must work as well as low bits.
    storage = torch.randn(14, vocab * 2, dtype=dtype, device=device)
    logits = storage[::2, ::2]
    expected = storage.clone()
    rows = [5, 0, 3]
    buffer = StructuredOutputMask(vocab)
    masks = buffer.prepare(len(rows), logits.device)
    masks.random_(-(2**31), 2**31 - 1)
    masks[0].zero_()
    masks[1].fill_(-1)
    tokens = torch.arange(vocab)
    for i, row in enumerate(rows):
        allowed = ((masks[i, tokens // 32] >> (tokens % 32)) & 1).bool()
        expected[::2, ::2][row].masked_fill_(~allowed.to(device), -float("inf"))
    buffer.apply(logits, rows)
    assert torch.equal(storage, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_buffer_reuse_growth_and_changing_batches(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    buffer = StructuredOutputMask(65)
    pointers = {}
    for size in [1, 8, 2, 0, 7, 16, 1]:
        logits = torch.ones(max(size, 1), 65, device=device)
        masks = buffer.prepare(size, logits.device)
        masks.fill_(-1 if size % 2 else 0)
        pointer = buffer.host_masks.data_ptr()
        if buffer.capacity in pointers:
            assert pointer == pointers[buffer.capacity]
        pointers[buffer.capacity] = pointer
        buffer.apply(logits, list(range(size)))
        assert torch.all(logits == (1 if size % 2 or size == 0 else -float("inf")))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_async_reuse_across_streams_keeps_each_batch_mask():
    buffer = StructuredOutputMask(4097)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for i in range(40):
        with torch.cuda.stream(streams[i % 2]):
            logits = torch.ones(8, 4097, device="cuda")
            masks = buffer.prepare(8, logits.device)
            masks.fill_(-1 if i % 2 else 0)
            buffer.apply(logits, list(range(8)))
            outputs.append(logits)
    for stream in streams:
        stream.synchronize()
    for i, logits in enumerate(outputs):
        assert torch.all(logits == (1 if i % 2 else -float("inf")))


pytest.importorskip("llguidance")


@pytest.fixture
def tokenizer():
    vocab = {s: i for i, s in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab["[EOS]"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_tokens([AddedToken("<think>"), AddedToken("</think>")])
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]")


@pytest.mark.parametrize("speculative", [False, True])
def test_visible_added_tokens_agree_between_masks_and_commits(tokenizer, speculative):
    text = '{"v":"<think>literal</think>"}'
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.convert_tokens_to_ids("<think>") in ids
    assert tokenizer.decode(ids, skip_special_tokens=True) == text
    ids.append(tokenizer.eos_token_id)
    controller = StructuredOutput(tokenizer, len(tokenizer), [tokenizer.eos_token_id])
    schema = json.dumps({"const": {"v": "<think>literal</think>"}})
    controller.compile(schema, None, "")
    history = []
    cursor = 0
    width = 6 if speculative else 1
    for start in range(0, len(ids), width):
        selected = ids[start : start + width]
        drafts = selected[:-1]
        batch = NS(
            structured_requests={
                0: StructuredRequest(
                    (schema, None, "") if start == 0 else None,
                    cursor,
                    tuple(history[cursor:]),
                )
            },
            req_ids=[1],
            is_final_chunk=None,
            total_seqs_num_prefill=0,
        )
        metadata = (
            NS(
                target_logits_indices=torch.arange(len(drafts)),
                bonus_logits_indices=torch.tensor([len(drafts)]),
                num_draft_tokens_np=np.array([len(drafts)]),
                input_ids=torch.tensor([0, *drafts]),
            )
            if speculative
            else None
        )
        cursor = len(history)
        if speculative:
            controller.draft_plan = {0: (list(range(len(selected))), 0, len(drafts))}
            controller.draft_copy = NS(wait=lambda drafts=drafts: torch.tensor(drafts))
        logits = torch.zeros(len(selected), len(tokenizer))
        prefixes = controller.apply(logits, batch, metadata)
        for row, token in enumerate(selected):
            assert torch.isfinite(logits[row, token])
        controller.commit(torch.tensor([selected]), batch, prefixes)
        history.extend(selected)
    assert controller.states[1].matcher.is_terminated()


def test_added_token_adapter_preserves_eos_and_original_tokenizer(tokenizer):
    original = tokenizer.backend_tokenizer.to_str()
    info = grammar_tokenizer(tokenizer, len(tokenizer), [tokenizer.eos_token_id])
    for text in ("<think>", "</think>"):
        token = tokenizer.convert_tokens_to_ids(text)
        assert not info.is_special_token(token)
        assert info.decode_bytes([token]) == text.encode()
    assert info.is_special_token(tokenizer.eos_token_id)
    assert tokenizer.backend_tokenizer.to_str() == original


@pytest.mark.parametrize("split_marker", [False, True])
def test_reasoning_closer_accepts_different_tokenizations(tokenizer, split_marker):
    controller = StructuredOutput(tokenizer, len(tokenizer), [tokenizer.eos_token_id])
    schema = json.dumps({"const": {"x": 1}})
    key = (schema, None, "</think>")
    controller.compile(*key)
    text = 'reasoning</think>{"x":1}'
    ids = (
        [t for ch in text for t in tokenizer.encode(ch, add_special_tokens=False)]
        if split_marker
        else tokenizer.encode(text, add_special_tokens=False)
    )
    assert tokenizer.decode(ids) == text
    b = NS(
        structured_requests={0: StructuredRequest(key, 0, tuple(ids))},
        req_ids=[1],
        is_final_chunk=None,
        total_seqs_num_prefill=0,
    )
    logits = torch.zeros(1, len(tokenizer))
    controller.apply(logits, b, None)
    assert not controller.errors
    assert torch.isfinite(logits[0, tokenizer.eos_token_id])


def test_admitted_matcher_survives_compilation_cache_eviction(controller):
    from atom.structured_output.manager import prepare_worker_grammar

    schema = '{"const":{"x":1}}'
    controller.fingerprint = "test-fingerprint"
    controller.compile(schema)
    runner = NS(structured_output=controller, structured_compilations=OrderedDict())
    assert prepare_worker_grammar(
        runner, schema, None, "", "test-fingerprint", 128, (11,)
    )["ready"]
    state = controller.states[11]
    for value in range(129):
        controller.compile(json.dumps({"const": value}))
    assert (schema, None, "") not in controller.grammars
    b = NS(
        structured_requests={0: StructuredRequest((schema, None, ""), 0, ())},
        req_ids=[11],
        is_final_chunk=None,
        total_seqs_num_prefill=0,
    )
    logits = torch.zeros(1, 128)
    assert controller.apply(logits, b, None)
    assert controller.states[11] is state
    assert torch.isfinite(logits[0, ord("{")])
    controller.release([11])
    assert not controller.states
