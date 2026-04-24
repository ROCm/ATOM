# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for OpenAI-compatible protocol (Pydantic request/response models)."""

import time

import pytest

from atom.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
)

# ============================================================================
# ChatMessage Tests
# ============================================================================


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_string_content(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.get_content_text() == "Hello"

    def test_multimodal_content_single_text(self):
        msg = ChatMessage(
            role="user",
            content=[{"type": "text", "text": "What is this?"}],
        )
        assert msg.get_content_text() == "What is this?"

    def test_multimodal_content_multiple_text_parts(self):
        msg = ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "First part"},
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/img.jpg"},
                },
                {"type": "text", "text": "Second part"},
            ],
        )
        assert msg.get_content_text() == "First part\nSecond part"

    def test_multimodal_content_no_text(self):
        msg = ChatMessage(
            role="user",
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/img.jpg"},
                }
            ],
        )
        assert msg.get_content_text() == ""

    def test_none_content(self):
        """Tool role messages may have content=None."""
        msg = ChatMessage(role="assistant", content=None)
        assert msg.get_content_text() == ""

    def test_extra_fields_allowed(self):
        """ChatMessage should accept extra fields (e.g., 'name')."""
        msg = ChatMessage(role="user", content="Hi", name="Alice")
        assert msg.role == "user"

    def test_to_template_dict_basic(self):
        msg = ChatMessage(role="user", content="Hello")
        d = msg.to_template_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_to_template_dict_with_tool_calls(self):
        """Assistant message with tool_calls should preserve them."""
        msg = ChatMessage.model_validate(
            {
                "role": "assistant",
                "content": "I'll run that.",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "exec", "arguments": '{"cmd": "ls"}'},
                    }
                ],
            }
        )
        d = msg.to_template_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "I'll run that."
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["function"]["name"] == "exec"

    def test_to_template_dict_tool_message(self):
        """Tool result message should preserve tool_call_id."""
        msg = ChatMessage.model_validate(
            {
                "role": "tool",
                "content": "file1.txt\nfile2.txt",
                "tool_call_id": "call_0",
            }
        )
        d = msg.to_template_dict()
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "call_0"
        assert "file1.txt" in d["content"]

    def test_to_template_dict_with_name(self):
        """Message with name field should preserve it."""
        msg = ChatMessage.model_validate(
            {"role": "user", "content": "Hi", "name": "Alice"}
        )
        d = msg.to_template_dict()
        assert d["name"] == "Alice"


# ============================================================================
# ChatCompletionRequest Tests
# ============================================================================


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest model."""

    def test_basic_request(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert req.model == "test-model"
        assert len(req.get_messages()) == 1

    def test_prompt_alias(self):
        """'prompt' field should be accepted as alias for 'messages'."""
        req = ChatCompletionRequest(
            prompt=[ChatMessage(role="user", content="Hello")],
        )
        assert len(req.get_messages()) == 1
        assert req.get_messages()[0].content == "Hello"

    def test_no_messages_raises(self):
        req = ChatCompletionRequest()
        with pytest.raises(ValueError, match="messages.*prompt"):
            req.get_messages()

    def test_extra_fields_ignored(self):
        """Unknown fields should be silently ignored (not cause 422)."""
        req = ChatCompletionRequest.model_validate(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream_options": {"include_usage": True},
                "tools": [],
                "tool_choice": "auto",
                "unknown_field": "value",
            }
        )
        assert req.model == "test"

    def test_defaults(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")],
        )
        assert req.temperature == 1.0
        assert req.max_tokens == 8192
        assert req.stream is False
        assert req.top_p == 1.0
        assert req.top_k == -1
        assert req.n == 1

    def test_n_greater_than_one(self):
        req = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "n": 4,
            }
        )
        assert req.n == 4

    def test_multimodal_messages(self):
        """Request with multimodal content should parse correctly."""
        req = ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Describe this"}],
                    }
                ],
            }
        )
        msg = req.get_messages()[0]
        assert msg.get_content_text() == "Describe this"


# ============================================================================
# CompletionRequest Tests
# ============================================================================


class TestCompletionRequest:
    """Tests for CompletionRequest model."""

    def test_basic_request(self):
        req = CompletionRequest(prompt="Hello world")
        assert req.prompt == "Hello world"
        assert req.max_tokens == 8192
        assert req.n == 1

    def test_extra_fields_ignored(self):
        req = CompletionRequest.model_validate(
            {"prompt": "Hello", "unknown": "ignored"}
        )
        assert req.prompt == "Hello"

    def test_n_parameter_accepted(self):
        req = CompletionRequest.model_validate({"prompt": "Hi", "n": 3})
        assert req.n == 3


# ============================================================================
# Response Model Tests
# ============================================================================


class TestResponseModels:
    """Tests for response models."""

    def test_chat_completion_response(self):
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            created=int(time.time()),
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        )
        assert resp.id == "chatcmpl-123"
        assert resp.choices[0]["message"]["content"] == "Hello!"

    def test_completion_response(self):
        resp = CompletionResponse(
            id="cmpl-123",
            created=int(time.time()),
            model="test-model",
            choices=[{"index": 0, "text": "world", "finish_reason": "stop"}],
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        )
        assert resp.choices[0]["text"] == "world"

    def test_model_card(self):
        card = ModelCard(id="test-model")
        assert card.id == "test-model"
        assert card.object == "model"
        assert card.owned_by == "atom"

    def test_model_list(self):
        model_list = ModelList(data=[ModelCard(id="model-a"), ModelCard(id="model-b")])
        assert model_list.object == "list"
        assert len(model_list.data) == 2

    def test_error_response(self):
        err = ErrorResponse(
            error={"message": "Not found", "type": "invalid_request_error", "code": 404}
        )
        assert err.error["message"] == "Not found"
