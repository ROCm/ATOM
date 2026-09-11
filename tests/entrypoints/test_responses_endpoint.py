# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the Python /v1/responses adapter, including Codex tool shapes.

Format translation only — no GPU or running server.
"""

from atom.entrypoints.openai.serving_responses import (
    CodexToolContext,
    ResponsesRequest,
    ResponseStreamEmitter,
    build_responses_response,
    custom_tool_input,
    normalize_codex_request,
    responses_to_openai_messages,
    responses_tools_to_openai,
    unsupported_responses_parameter,
)


class TestNormalizeCodexRequest:
    def test_custom_namespace_and_history(self):
        raw = {
            "model": "glm",
            "input": [
                {"role": "user", "content": "edit it"},
                {
                    "type": "custom_tool_call",
                    "call_id": "call_1",
                    "name": "apply_patch",
                    "input": "*** Begin Patch",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "output": "Done",
                },
            ],
            "tools": [
                {
                    "type": "namespace",
                    "name": "editing",
                    "description": "tools",
                    "tools": [
                        {
                            "type": "custom",
                            "name": "apply_patch",
                            "description": "patch",
                            "format": {
                                "type": "grammar",
                                "syntax": "lark",
                                "definition": "start: /.+/",
                            },
                        }
                    ],
                }
            ],
        }
        body, context = normalize_codex_request(raw)
        assert context.kind("apply_patch") == "custom"
        assert context.namespace("apply_patch") == "editing"
        assert len(body["tools"]) == 1
        tool = body["tools"][0]
        assert tool["type"] == "function"
        assert tool["name"] == "apply_patch"
        assert "[editing]" in tool["description"]
        assert "lark" in tool["description"]
        assert body["input"][1]["type"] == "function_call"
        assert body["input"][1]["name"] == "apply_patch"
        assert '"input"' in body["input"][1]["arguments"]
        assert body["input"][2]["type"] == "function_call_output"

    def test_deferred_tools_from_tool_search_output(self):
        raw = {
            "model": "glm",
            "input": [
                {"role": "user", "content": "delegate"},
                {
                    "type": "tool_search_call",
                    "call_id": "search_1",
                    "execution": "client",
                    "arguments": {"query": "agent"},
                },
                {
                    "type": "tool_search_output",
                    "call_id": "search_1",
                    "tools": [
                        {
                            "type": "function",
                            "name": "spawn_agent",
                            "description": "spawn",
                            "parameters": {"type": "object"},
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "tool_search",
                    "execution": "client",
                    "parameters": {"type": "object"},
                }
            ],
        }
        body, context = normalize_codex_request(raw)
        assert len(body["tools"]) == 2
        assert context.kind("tool_search") == "tool_search"
        names = {
            t.get("name") or (t.get("function") or {}).get("name")
            for t in body["tools"]
        }
        assert names == {"tool_search", "spawn_agent"}


class TestRewriteCodexResponse:
    def test_restores_custom_tool_response_shape(self):
        context = CodexToolContext(
            kinds={"apply_patch": "custom"},
            namespaces={"apply_patch": "editing"},
        )
        item = {
            "id": "fc_1",
            "type": "function_call",
            "call_id": "call_1",
            "name": "apply_patch",
            "arguments": '{"input":"patch"}',
            "status": "completed",
        }
        context.rewrite_output_item(item)
        assert item["type"] == "custom_tool_call"
        assert item["input"] == "patch"
        assert item["namespace"] == "editing"
        assert "arguments" not in item

    def test_restores_tool_search_call(self):
        context = CodexToolContext(kinds={"tool_search": "tool_search"})
        item = {
            "id": "fc_1",
            "type": "function_call",
            "call_id": "search_1",
            "name": "tool_search",
            "arguments": '{"query":"agent"}',
            "status": "completed",
        }
        context.rewrite_output_item(item)
        assert item["type"] == "tool_search_call"
        assert item["execution"] == "client"
        assert item["arguments"] == {"query": "agent"}
        assert "name" not in item

    def test_rewrite_response_restores_original_tools(self):
        original = [{"type": "custom", "name": "apply_patch"}]
        context = CodexToolContext(
            kinds={"apply_patch": "custom"},
            original_tools=original,
        )
        response = {
            "output": [
                {
                    "type": "function_call",
                    "name": "apply_patch",
                    "arguments": '{"input":"x"}',
                }
            ],
            "tools": [{"type": "function", "name": "apply_patch"}],
        }
        context.rewrite_response(response)
        assert response["tools"] == original
        assert response["output"][0]["type"] == "custom_tool_call"


class TestResponsesToChat:
    def test_text_input_and_instructions(self):
        request = ResponsesRequest(
            input="Hello, world!",
            instructions="You are a helpful assistant.",
            model="glm",
            temperature=0.7,
        )
        messages = responses_to_openai_messages(request)
        assert messages[0] == {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        assert messages[1] == {"role": "user", "content": "Hello, world!"}

    def test_function_call_history(self):
        request = ResponsesRequest(
            input=[
                {"role": "user", "content": "edit"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "apply_patch",
                    "arguments": '{"input":"p"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "Done",
                },
            ]
        )
        messages = responses_to_openai_messages(request)
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"][0]["id"] == "call_1"
        assert messages[2] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Done",
        }

    def test_flat_function_tools_wrap_for_template(self):
        tools = responses_tools_to_openai(
            [
                {
                    "type": "function",
                    "name": "tool_search",
                    "description": "search",
                    "parameters": {"type": "object"},
                }
            ]
        )
        assert tools == [
            {
                "type": "function",
                "function": {
                    "name": "tool_search",
                    "description": "search",
                    "parameters": {"type": "object"},
                },
            }
        ]


class TestUnsupportedParameters:
    def test_background(self):
        assert "Background mode" in unsupported_responses_parameter(
            {"background": True}
        )

    def test_previous_response_id(self):
        msg = unsupported_responses_parameter({"previous_response_id": "resp_1"})
        assert "previous_response_id" in msg

    def test_conversation(self):
        msg = unsupported_responses_parameter({"conversation": "conv_1"})
        assert "conversation" in msg

    def test_ok(self):
        assert unsupported_responses_parameter({"input": "hi"}) is None


class TestBuildResponse:
    def test_custom_tool_from_events(self):
        context = CodexToolContext(kinds={"apply_patch": "custom"})
        events = [
            (
                "tool_call_start",
                {
                    "id": "call_1",
                    "index": 0,
                    "function": {"name": "apply_patch"},
                },
            ),
            (
                "tool_call_args",
                {"index": 0, "function": {"arguments": '{"input":"patch"}'}},
            ),
        ]
        response = build_responses_response(
            "resp_x",
            "glm",
            events,
            context=context,
            request=ResponsesRequest(),
        )
        assert response["output"][0]["type"] == "custom_tool_call"
        assert response["output"][0]["input"] == "patch"
        assert response["store"] is False


class TestCustomToolInput:
    def test_object_input_field(self):
        assert custom_tool_input('{"input":"patch"}') == "patch"

    def test_raw_string(self):
        assert custom_tool_input("not-json") == "not-json"


class TestResponseStreamEmitter:
    def test_custom_tool_uses_codex_event_shape(self):
        _, context = normalize_codex_request(
            {
                "model": "m",
                "input": "edit",
                "tools": [
                    {"type": "custom", "name": "apply_patch", "description": "patch"}
                ],
            }
        )
        emitter = ResponseStreamEmitter("resp_x", "m", context=context, created_at=0)
        frames = []
        frames.extend(emitter.opening_frames())
        frames.extend(
            emitter.frames_for_events(
                [
                    (
                        "tool_call_start",
                        {
                            "id": "call_patch",
                            "index": 0,
                            "type": "function",
                            "function": {"name": "apply_patch"},
                        },
                    ),
                    (
                        "tool_call_args",
                        {
                            "index": 0,
                            "function": {"arguments": '{"input":"*** Begin Patch"}'},
                        },
                    ),
                ]
            )
        )
        frames.extend(emitter.finish_frames(has_tool_calls=True))
        joined = "".join(frames)
        assert '"type":"custom_tool_call"' in joined.replace(" ", "")
        assert "response.custom_tool_call_input.delta" in joined
        assert "response.custom_tool_call_input.done" in joined
        assert "*** Begin Patch" in joined
        assert "response.function_call_arguments" not in joined
        assert "data: [DONE]" in joined

    def test_function_tool_emits_argument_deltas(self):
        emitter = ResponseStreamEmitter("resp_x", "m", created_at=0)
        frames = emitter.frames_for_events(
            [
                (
                    "tool_call_start",
                    {
                        "id": "c1",
                        "index": 0,
                        "function": {"name": "f"},
                    },
                ),
                (
                    "tool_call_args",
                    {"index": 0, "function": {"arguments": '{"a":1}'}},
                ),
            ]
        )
        frames.extend(emitter.finish_frames(has_tool_calls=True))
        joined = "".join(frames)
        assert "response.function_call_arguments.delta" in joined
        assert "response.function_call_arguments.done" in joined
        assert "response.custom_tool_call_input" not in joined

    def test_text_emits_output_text_delta(self):
        emitter = ResponseStreamEmitter("resp_x", "m", created_at=0)
        frames = emitter.opening_frames()
        frames.extend(emitter.frames_for_events([("content", "hi")]))
        frames.extend(emitter.finish_frames())
        joined = "".join(frames)
        assert "response.created" in joined
        assert "response.output_item.added" in joined
        assert "response.content_part.added" in joined
        assert "response.output_text.delta" in joined
        assert "response.completed" in joined
