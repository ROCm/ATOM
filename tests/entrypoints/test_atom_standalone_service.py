# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for the ATOM standalone service's streaming chat state."""

import json
import pathlib
import queue
from typing import ClassVar

import pytest
from import_guard import skip_if_dependency_missing

try:
    from atom.entrypoints.atomesh.atom_standalone_service import (
        ChatCompletionStreamState,
    )
except Exception as exc:  # noqa: BLE001 pragma: no cover
    skip_if_dependency_missing(exc, "atomesh service import unavailable")
    ChatCompletionStreamState = None  # type: ignore[assignment]
    _import_error = exc
else:
    _import_error = None

pytestmark = pytest.mark.skipif(
    ChatCompletionStreamState is None,
    reason=f"atom_standalone_service import unavailable: {_import_error!r}",
)


class _StubTokenizer:
    """Minimal tokenizer stub: only .encode() is used by ChatCompletionStreamState.__init__."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text.split())

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return ""


def _make_state(n: int) -> ChatCompletionStreamState:
    return ChatCompletionStreamState(
        request_id="chatcmpl-test",
        model_name="model",
        prompt="hello",
        tokenizer=_StubTokenizer(),
        stream_queue=queue.Queue(),
        n=n,
    )


class TestChatCompletionStreamStateRoleChunkContent:
    """Regression test: AtomStandaloneRouter::route_chat()'s streaming path
    must emit content="" with role="assistant", matching the fix in serving_chat.py's
    stream_chat_response/_fanout."""

    def test_single_sequence_role_chunk_has_empty_content(self):
        state = _make_state(n=1)

        chunks = state.drain(max_items=16)

        assert len(chunks) == 1
        assert chunks[0].startswith("data: ")
        data = json.loads(chunks[0][6:])
        delta = data["choices"][0]["delta"]
        assert delta["role"] == "assistant"
        assert delta["content"] == ""

    def test_fanout_role_chunks_have_empty_content(self):
        state = _make_state(n=3)

        chunks = state.drain(max_items=16)

        assert len(chunks) == 3
        for expected_index, raw_chunk in enumerate(chunks):
            assert raw_chunk.startswith("data: ")
            data = json.loads(raw_chunk[6:])
            choice = data["choices"][0]
            assert choice["index"] == expected_index
            delta = choice["delta"]
            assert delta["role"] == "assistant"
            assert delta["content"] == ""

    def test_role_chunks_not_resent_on_subsequent_drain(self):
        state = _make_state(n=1)

        first = state.drain(max_items=16)
        assert len(first) == 1

        second = state.drain(max_items=16, timeout=0.01)
        assert second == []


class TestTheDrainKeepsWhatItCannotHandOutYet:
    """`max_items` is a batch size, not a licence to discard.

    The build loop `break`ed when it reached that count -- after the parser
    had already yielded the events -- so everything past the first was gone
    permanently. At `max_items=1` a tool call lost its arguments and, once
    `has_tool_calls` moved onto the argument event, reported
    `finish_reason: stop` for a call it had just announced.
    """

    TOOLS: ClassVar[list] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    CALL = (
        "<tool_call><function=get_weather><parameter=city>Paris</parameter>"
        "</function></tool_call>"
    )

    def _state(self):
        from atom.entrypoints.openai.tool_parser.qwen3_tool_parser import QwenXmlParser

        return ChatCompletionStreamState(
            request_id="chatcmpl-test",
            model_name="model",
            prompt="hello",
            tokenizer=_StubTokenizer(),
            stream_queue=queue.Queue(),
            n=1,
            tool_parser_cls=QwenXmlParser,
            tools=self.TOOLS,
        )

    @staticmethod
    def _drain_all(state, event, max_items):
        out = state._event_to_chunks(event, max_items)
        idle = {"index": 0, "text": "", "token_ids": [], "finished": True}
        for _ in range(20):
            more = state._event_to_chunks(idle, max_items)
            if not more:
                break
            out += more
        return out

    @pytest.mark.parametrize("max_items", [1, 2, 16])
    def test_the_arguments_survive_any_batch_size(self, max_items):
        state = self._state()
        event = {"index": 0, "text": self.CALL, "token_ids": [1], "finished": True}
        payloads = [
            json.loads(c.split("data: ", 1)[1])
            for c in self._drain_all(state, event, max_items)
            if c.split("data: ", 1)[1].strip() != "[DONE]"
        ]
        arguments = "".join(
            tc.get("function", {}).get("arguments", "")
            for p in payloads
            for tc in (
                (p.get("choices") or [{}])[0].get("delta", {}).get("tool_calls") or []
            )
        )
        assert '"city"' in arguments, f"arguments lost at max_items={max_items}"

    @pytest.mark.parametrize("max_items", [1, 2, 16])
    def test_and_the_finish_reason_says_a_tool_was_called(self, max_items):
        state = self._state()
        event = {"index": 0, "text": self.CALL, "token_ids": [1], "finished": True}
        self._drain_all(state, event, max_items)
        assert state.has_tool_calls == [True]

    @pytest.mark.parametrize("max_items", [1, 2, 16])
    def test_the_stream_still_closes_at_any_batch_size(self, max_items):
        """Queueing the overflow made the early return the common path, and
        it returned before the final chunks -- a fully drained stream with no
        `finish_reason`, no usage and no `[DONE]`."""
        state = self._state()
        event = {"index": 0, "text": self.CALL, "token_ids": [1], "finished": True}
        chunks = self._drain_all(state, event, max_items)
        assert chunks[-1].split("data: ", 1)[1].strip() == "[DONE]"
        reasons = [
            c["finish_reason"]
            for raw in chunks
            if raw.split("data: ", 1)[1].strip() != "[DONE]"
            for c in json.loads(raw.split("data: ", 1)[1]).get("choices", [])
            if c.get("finish_reason")
        ]
        assert reasons == ["tool_calls"], reasons


class TestDrainDeliversWhatItBuiltBeforeItFinishes:
    """Driven through `drain`, which is what the Rust router calls.

    "Every sibling finished" is not the same fact as "everything built has
    been sent", and three places used the first as a proxy for the second --
    the top of `drain`, its `queue.Empty` arm, and the `done` event. Each
    built and handed out the terminal chunks while chunks were still queued,
    and handing those out sets `completed`, after which the router pops the
    state and the queued ones are gone.

    At the `max_items=1` the single-chunk poll uses, a response with one tool
    call reported `finish_reason: tool_calls` having sent no `tool_calls`
    delta at all.
    """

    CALL = (
        "<tool_call><function=get_weather><parameter=city>Paris</parameter>"
        "</function></tool_call>"
    )
    TOOLS: ClassVar[list] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]

    def _drain_everything(self, max_items: int) -> list[dict]:
        from atom.entrypoints.openai.tool_parser.qwen3_tool_parser import (
            QwenXmlParser,
        )

        stream_queue: queue.Queue = queue.Queue()
        state = ChatCompletionStreamState(
            request_id="req",
            model_name="m",
            prompt="hi",
            tokenizer=_StubTokenizer(),
            stream_queue=stream_queue,
            n=1,
            tool_parser_cls=QwenXmlParser,
            tools=self.TOOLS,
        )
        stream_queue.put(
            {
                "index": 0,
                "text": "Sure. " + self.CALL,
                "token_ids": [1],
                "finished": True,
                "finish_reason": "eos",
            }
        )
        stream_queue.put({"done": True})
        frames: list[str] = []
        for _ in range(64):
            got = state.drain(max_items=max_items, timeout=0.0)
            if not got:
                break
            frames.extend(got)
            if state.completed:
                break
        return [
            json.loads(part[6:])
            for frame in frames
            for part in frame.split("\n\n")
            if part.startswith("data: {")
        ]

    @pytest.mark.parametrize("max_items", [1, 2, 16])
    def test_the_tool_call_is_delivered_before_the_stream_is_finished(self, max_items):
        payloads = self._drain_everything(max_items)
        args = [
            tc
            for p in payloads
            for c in p.get("choices", [])
            for tc in c.get("delta", {}).get("tool_calls", [])
            if "arguments" in tc.get("function", {})
        ]
        reasons = [
            c["finish_reason"]
            for p in payloads
            for c in p.get("choices", [])
            if c.get("finish_reason")
        ]
        assert args, f"at max_items={max_items} no arguments were ever sent"
        assert reasons == ["tool_calls"], reasons

    @pytest.mark.parametrize("max_items", [1, 2, 16])
    def test_the_answer_before_the_call_is_delivered_too(self, max_items):
        payloads = self._drain_everything(max_items)
        content = "".join(
            c.get("delta", {}).get("content", "")
            for p in payloads
            for c in p.get("choices", [])
        )
        assert content == "Sure. ", repr(content)


class TestEveryMethodSaysHowItIsCalled:
    """A method whose first parameter is not `self`/`cls` and which declares no
    decorator is a decorator that went missing.

    `_json_safe` lost its `@staticmethod` to a mechanical re-indent, and both
    non-streaming endpoints then raised `TypeError` on their final `return`:
    every POST to /v1/chat/completions and /v1/completions on this entrypoint
    became a 500, with the engine result built correctly and then discarded.
    Nothing about the model output mattered, so no behavioural test in this
    file could see it.
    """

    def test_no_method_has_lost_its_decorator(self):
        import ast
        import inspect

        from atom.entrypoints.atomesh import atom_standalone_service

        source = pathlib.Path(
            inspect.getsourcefile(atom_standalone_service)
        ).read_text()
        offenders = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            for fn in node.body:
                if not isinstance(fn, ast.FunctionDef):
                    continue
                decorated = {
                    getattr(d, "id", getattr(d, "attr", "")) for d in fn.decorator_list
                }
                if decorated & {"staticmethod", "classmethod", "property"}:
                    continue
                args = fn.args.posonlyargs + fn.args.args
                if not args or args[0].arg not in ("self", "cls"):
                    offenders.append(f"{node.name}.{fn.name}")
        assert not offenders, f"missing @staticmethod/@classmethod on {offenders}"

    def test_json_safe_is_callable_off_the_class(self):
        """The call the endpoints actually make, on the shape they make it on."""
        from atom.entrypoints.atomesh.atom_standalone_service import (
            AtomStandaloneService,
        )

        assert AtomStandaloneService._json_safe({"a": [1, {"b": 2}]}) == {
            "a": [1, {"b": 2}]
        }


class TestAStreamClosedBeforeItStartedIsStillStopped:
    """`start_stream` only enqueues; the sequences do not exist until the
    worker thread runs `_submit_stream_request`.

    A close in that window had nothing to abort, and the sequence was then
    added to the engine *afterwards* -- decoding to `max_tokens` on the GPU,
    putting into a queue nobody would ever read, and holding its KV blocks for
    the life of the process. That is the exact case the abort exists for, and
    the one it missed.
    """

    class _Engine:
        def __init__(self):
            self.added = []
            self.aborted = []
            self.preprocessed = 0
            self.io_processor = self
            self.core_mgr = self
            self.requests = {}

        def preprocess(self, *a, **kw):
            self.preprocessed += 1
            return type("Seq", (), {"id": self.preprocessed})()

        def add_request(self, seqs):
            self.added.extend(seqs)

        def abort_request(self, seq_id):
            self.aborted.append(seq_id)

    def _service(self):
        from atom.entrypoints.atomesh.atom_standalone_service import AtomEngineService

        return AtomEngineService(self._Engine(), _StubTokenizer())

    def test_closing_before_submission_stops_it_being_submitted(self):
        """The close lands before the worker thread picks the request up."""
        from atom.entrypoints.atomesh.atom_standalone_service import (
            EngineStreamRequest,
        )

        service = self._service()
        request = EngineStreamRequest(
            request_id="r1",
            prompt="hi",
            sampling_params=None,
            effective_n=1,
            stream_queue=queue.Queue(),
        )
        service.abort_stream("r1")  # the close arrives first
        service._submit_stream_request(request)  # then the worker gets to it
        assert service.engine.added == [], "a closed stream reached the engine"
        assert service.engine.preprocessed == 0, (
            "the sequences were built for a stream already closed; each one "
            "registers with the engine's io_processor"
        )
        service.close()

    def test_closing_during_preprocessing_stops_it_too(self):
        """And the close lands *while* the sequences are being built.

        The two checks guard different moments and neither covers the other:
        the first keeps the work from starting, the second keeps its result
        from reaching the engine. Driven with a real thread, because the race
        is the whole point.
        """
        import threading

        from atom.entrypoints.atomesh.atom_standalone_service import (
            EngineStreamRequest,
        )

        service = self._service()
        building = threading.Event()
        may_finish = threading.Event()
        original = service.engine.preprocess

        def slow_preprocess(*a, **kw):
            building.set()
            may_finish.wait(timeout=5)
            return original(*a, **kw)

        service.engine.preprocess = slow_preprocess
        request = EngineStreamRequest(
            request_id="r4",
            prompt="hi",
            sampling_params=None,
            effective_n=1,
            stream_queue=queue.Queue(),
        )
        worker = threading.Thread(
            target=service._submit_stream_request, args=(request,)
        )
        worker.start()
        assert building.wait(timeout=5), "preprocessing never started"
        service.abort_stream("r4")
        may_finish.set()
        worker.join(timeout=5)
        assert (
            service.engine.added == []
        ), "sequences built before the close reached the engine after it"
        service.close()

    def test_closing_after_submission_aborts_what_was_added(self):
        from atom.entrypoints.atomesh.atom_standalone_service import (
            EngineStreamRequest,
        )

        service = self._service()
        request = EngineStreamRequest(
            request_id="r2",
            prompt="hi",
            sampling_params=None,
            effective_n=1,
            stream_queue=queue.Queue(),
        )
        service._submit_stream_request(request)
        assert service.engine.added, "the premise is that it was submitted"
        service.abort_stream("r2")
        assert service.engine.aborted == [s.id for s in service.engine.added]
        service.close()

    def test_a_stream_that_ends_on_its_own_is_forgotten(self):
        """Nothing else drops it, so the map grows for the process lifetime."""
        from atom.entrypoints.atomesh.atom_standalone_service import (
            EngineStreamRequest,
        )

        service = self._service()
        service._submit_stream_request(
            EngineStreamRequest(
                request_id="r3",
                prompt="hi",
                sampling_params=None,
                effective_n=1,
                stream_queue=queue.Queue(),
            )
        )
        assert "r3" in service._stream_seqs
        service.forget_stream("r3")
        assert "r3" not in service._stream_seqs and "r3" not in service._abandoned
        service.close()

    def test_closing_between_add_request_and_publication_aborts(self):
        """The third window, and the one publication order decides.

        Publishing the seqs into `_stream_seqs` *before* `add_request` left a
        gap where a close popped the list and aborted sequences the engine had
        not been told about -- the abort raised into a swallowed `except`, the
        worker then added them, and they decoded to `max_tokens` into a queue
        nobody would read, holding their KV blocks for the process lifetime.
        """
        import threading

        from atom.entrypoints.atomesh.atom_standalone_service import (
            EngineStreamRequest,
        )

        service = self._service()
        added = threading.Event()
        may_publish = threading.Event()
        original = service.engine.add_request

        def slow_add(seqs):
            original(seqs)
            added.set()
            may_publish.wait(timeout=5)

        service.engine.add_request = slow_add
        request = EngineStreamRequest(
            request_id="r5",
            prompt="hi",
            sampling_params=None,
            effective_n=1,
            stream_queue=queue.Queue(),
        )
        worker = threading.Thread(
            target=service._submit_stream_request, args=(request,)
        )
        worker.start()
        assert added.wait(timeout=5), "add_request never ran"
        service.abort_stream("r5")  # lands after the engine has them
        may_publish.set()
        worker.join(timeout=5)
        assert service.engine.aborted == [
            s.id for s in service.engine.added
        ], "sequences the engine was given were never aborted"
        assert "r5" not in service._stream_seqs
        service.close()
