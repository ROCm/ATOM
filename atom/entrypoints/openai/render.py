# SPDX-License-Identifier: MIT
"""Request preparation shared by inference and exact token rendering."""

from atom.entrypoints.openai.chat_encoders import apply_chat_template
from atom.entrypoints.openai.protocol import ChatCompletionRequest
from atom.entrypoints.openai.serving_chat import (
    normalize_chat_tools,
    resolve_thinking,
    validate_chat_request,
)


def prepare_chat_fields(request, default_chat_template_kwargs, reasoning_toggle):
    request.tools = normalize_chat_tools(request.tools)
    validate_chat_request(request)
    messages = request.get_messages()

    merged_kwargs = dict(default_chat_template_kwargs)
    if request.chat_template_kwargs:
        merged_kwargs.update(request.chat_template_kwargs)
    # Forward K3 template controls the chat template needs but that pydantic
    # does not otherwise thread through: structured-output response_format,
    # a string tool_choice ("auto"/"none"/"required"), and thinking/effort.
    if request.response_format is not None:
        merged_kwargs["response_format"] = request.response_format
    if isinstance(request.tool_choice, str):
        merged_kwargs["tool_choice"] = request.tool_choice
    _th_enabled, _th_effort = resolve_thinking(request)
    if request.thinking is not None or request.reasoning_effort is not None:
        # By the name this template actually reads. `thinking` was
        # hardcoded, which is right for Kimi-K3 and a silent no-op for the
        # whole Qwen family, whose templates read `enable_thinking` --
        # measured, `thinking=False` left the `<think>` prefill in place.
        # A template ignores a kwarg it does not know, so the failure was
        # invisible: the model reasoned anyway.
        # Only when the request said something about it. An effort is
        # not an opt-in, and this is merged after the server defaults and
        # after the client's own `chat_template_kwargs` -- so writing it
        # unconditionally overrode both.
        if reasoning_toggle is not None and _th_enabled is not None:
            name, off_value, on_value = reasoning_toggle
            merged_kwargs[name] = on_value if _th_enabled else off_value
        if _th_effort is not None:
            merged_kwargs["thinking_effort"] = _th_effort

    return messages, merged_kwargs


def prepare_text_chat(request, tokenizer, encoder, defaults, reasoning_toggle):
    if isinstance(request, dict):
        request = ChatCompletionRequest.model_validate(request)
    messages, kwargs = prepare_chat_fields(request, defaults, reasoning_toggle)
    if any(
        isinstance(message.content, list)
        and any(part.get("type") != "text" for part in message.content)
        for message in messages
    ):
        raise ValueError("exact cache rendering currently supports text only")
    prompt = apply_chat_template(
        tokenizer,
        encoder,
        [message.to_template_dict() for message in messages],
        tools=request.tools,
        **kwargs,
    )
    return prompt, kwargs
