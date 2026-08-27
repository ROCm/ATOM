# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Request policy for the OpenAI chat endpoint.

Everything that happens between "pydantic parsed the request body" and
"render the chat template / build SamplingParams":

* :func:`validate_chat_request` — reject malformed input with ``ValueError`` so
  the endpoint answers **400**, instead of letting bad input reach the chat
  template (500) or the engine (200 with garbage).
* :func:`normalize_chat_messages` — role handling, including MiniMax's
  ``role="root"`` protocol extension and OpenAI's ``developer`` role.

Tool and ``thinking`` policy are not here: ``serving_chat`` owns
``validate_chat_request``, ``validate_tool_list`` and ``resolve_thinking``, and
``tool_parser.registry.forbids_tool_calls`` owns ``tool_choice: "none"``.

This module deliberately imports nothing heavier than :mod:`.protocol` (pydantic
only), so it is unit-testable without a GPU, an engine or FastAPI.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from atom.sampling_params import MAX_TEMPERATURE

from .protocol import ChatCompletionRequest, ChatMessage, _fix_invalid_json_escapes

# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

#: Roles defined by the OpenAI chat API.
OPENAI_ROLES = ("system", "developer", "user", "assistant", "tool", "function")

#: Vendor protocol extensions ATOM accepts. MiniMax's ``root`` carries
#: instructions that outrank ``system``.
EXTENSION_ROLES = ("root",)

#: Anything outside this set is a client error (400), not something to drop
#: silently in the chat template.
VALID_ROLES = frozenset(OPENAI_ROLES + EXTENSION_ROLES)

#: Roles that carry instructions rather than conversation turns.
SYSTEM_LIKE_ROLES = ("system", "developer", "root")

#: Prepended to a ``root`` message when it is folded into ``system`` and has to
#: outrank a competing ``system`` message.
ROOT_INSTRUCTION_HEADER = (
    "The following are root-level instructions. They have the highest priority "
    "and override any conflicting system, developer or user instruction:"
)

#: Roles ATOM can rewrite when the loaded chat template does not handle them.
_REWRITABLE_ROLES = ("root", "developer")


# ---------------------------------------------------------------------------
# Chat-template role support probe
# ---------------------------------------------------------------------------


def template_supported_roles(chat_template: Any) -> frozenset:
    """Which rewritable roles the loaded Jinja chat template branches on.

    A template that never mentions ``'root'`` silently drops root messages, so
    the instruction they carry would be lost; :func:`normalize_chat_messages`
    rewrites those. Templates that do handle the role keep it verbatim.

    ``chat_template`` may be the Jinja source, a ``{name: source}`` mapping
    (tokenizers with named templates), or ``None``.
    """
    if isinstance(chat_template, dict):
        source = "\n".join(str(v) for v in chat_template.values())
    elif isinstance(chat_template, (list, tuple)):
        source = "\n".join(str(v) for v in chat_template)
    else:
        source = chat_template if isinstance(chat_template, str) else ""
    if not source:
        return frozenset()
    return frozenset(
        role
        for role in _REWRITABLE_ROLES
        if f"'{role}'" in source or f'"{role}"' in source
    )


# ---------------------------------------------------------------------------
# Message normalization
# ---------------------------------------------------------------------------


def _merge_messages(messages: Sequence[ChatMessage]) -> ChatMessage:
    """Collapse several same-role messages into one, joining their text."""
    if len(messages) == 1:
        return messages[0]
    joined = "\n\n".join(m.get_content_text() for m in messages if m.get_content_text())
    return messages[0].model_copy(update={"content": joined})


def normalize_chat_messages(
    messages: Sequence[ChatMessage],
    *,
    supported_roles: frozenset = frozenset(),
) -> list[ChatMessage]:
    """Map extension roles onto roles the chat template understands.

    ``root`` (MiniMax) and ``developer`` (OpenAI) are dropped silently by
    templates that only branch on system/user/assistant/tool, which loses the
    instructions they carry. When the loaded template does not mention the role,
    rewrite it to ``system``.

    A rewritten ``root`` message is additionally hoisted into the last
    system-level slot — directly after any ``system``/``developer`` messages at
    the head of the conversation — so it is both explicitly labelled as
    higher-priority and the most recent instruction the model reads. That is
    what makes ``root`` override ``system`` rather than the other way round.

    ``supported_roles`` comes from :func:`template_supported_roles`.
    """
    template_handles_root = "root" in supported_roles
    kept: list[ChatMessage] = []
    roots: list[ChatMessage] = []
    competing_system = any(
        (m.role or "").strip() in ("system", "developer") for m in messages
    )

    for message in messages:
        role = (message.role or "").strip()
        if role == "root":
            roots.append(message)
            continue
        if role == "developer" and "developer" not in supported_roles:
            kept.append(message.model_copy(update={"role": "system"}))
            continue
        kept.append(message)

    if not roots:
        return kept

    if template_handles_root:
        # MiniMax-M3's template implements the role, but reads it only at
        # messages[0] — a root message anywhere else is silently dropped, and
        # only the first of several is seen. Merge and hoist to match that
        # contract exactly; the template itself renders it as the high-priority
        # system prompt, so no header text is needed.
        return [_merge_messages(roots)] + kept

    # Folded into system: place it last among the leading system-level messages
    # so it is both flagged as higher priority and the most recent instruction.
    text = _merge_messages(roots).get_content_text()
    content = f"{ROOT_INSTRUCTION_HEADER}\n{text}" if competing_system else text
    folded = roots[0].model_copy(update={"role": "system", "content": content})
    insert_at = 0
    for index, message in enumerate(kept):
        if (message.role or "").strip() in ("system", "developer"):
            insert_at = index + 1
        else:
            break
    return kept[:insert_at] + [folded] + kept[insert_at:]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_json_object_string(raw: str) -> bool:
    for candidate in (raw, _fix_invalid_json_escapes(raw)):
        try:
            return isinstance(json.loads(candidate), dict)
        except (ValueError, TypeError):
            continue
    return False


def _validate_sampling(request: ChatCompletionRequest) -> None:
    temperature = request.temperature
    if temperature is not None and not 0.0 <= temperature <= MAX_TEMPERATURE:
        raise ValueError(
            f"temperature must be in [0, {MAX_TEMPERATURE:g}], got {temperature}"
        )
    for name in ("max_tokens", "max_completion_tokens"):
        value = getattr(request, name, None)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    # torch.Generator.manual_seed takes a signed 64-bit value.
    seed = getattr(request, "seed", None)
    if seed is not None and not -(2**63) <= seed < 2**63:
        raise ValueError("seed must fit in a signed 64-bit integer")


def _validate_assistant_tool_calls(tool_calls: Any) -> list[str]:
    """Validate one assistant message's ``tool_calls``; return their ids."""
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ValueError("'tool_calls' must be a non-empty array")
    ids: list[str] = []
    for entry in tool_calls:
        if not isinstance(entry, dict):
            # ValueError, not TypeError: the endpoint maps it to HTTP 400.
            raise ValueError(  # noqa: TRY004
                "each entry of 'tool_calls' must be an object"
            )
        call_type = entry.get("type", "function")
        if call_type != "function":
            raise ValueError(f"unsupported tool_call type '{call_type}'")
        fn = entry.get("function")
        if not isinstance(fn, dict):
            raise ValueError(  # noqa: TRY004
                "each tool_call requires a 'function' object"
            )
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("each tool_call requires a non-empty 'function.name'")
        arguments = fn.get("arguments", "{}")
        if isinstance(arguments, str):
            if arguments.strip() and not _is_json_object_string(arguments):
                raise ValueError(
                    f"tool_call '{name}' has malformed 'function.arguments': "
                    "expected a JSON object string"
                )
        elif not isinstance(arguments, dict):
            raise ValueError(  # noqa: TRY004
                f"tool_call '{name}': 'function.arguments' must be a JSON object "
                "string"
            )
        call_id = entry.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("each tool_call requires a non-empty 'id'")
        ids.append(call_id)
    return ids


def _validate_message_sequence(messages: Sequence[ChatMessage]) -> None:
    """Validate roles and the assistant-tool_calls / tool-response pairing."""
    announced_ids: set = set()
    # tool_call_id -> index of the assistant message that requested it
    unanswered: dict[str, int] = {}
    last_index = len(messages) - 1

    for index, message in enumerate(messages):
        role = (message.role or "").strip()
        if not role:
            raise ValueError(f"messages[{index}] requires a 'role'")
        if role not in VALID_ROLES:
            raise ValueError(
                f"messages[{index}] has invalid role '{role}'. Supported roles: "
                + ", ".join(sorted(VALID_ROLES))
            )

        if role != "tool" and unanswered:
            missing = ", ".join(sorted(unanswered))
            raise ValueError(
                "An assistant message with 'tool_calls' must be followed by a "
                f"'tool' message for every tool_call_id; missing: {missing}"
            )

        extras = message.model_extra or {}

        if role == "assistant" and extras.get("tool_calls") is not None:
            ids = _validate_assistant_tool_calls(extras["tool_calls"])
            announced_ids.update(ids)
            unanswered = {call_id: index for call_id in ids}
        elif role == "tool":
            call_id = extras.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(
                    f"messages[{index}]: a 'tool' message requires a non-empty "
                    "'tool_call_id'"
                )
            # Only enforced when the same request also carries the assistant
            # turn that requested the call: clients that trim history and send a
            # bare tool result are tolerated.
            if announced_ids and call_id not in announced_ids:
                raise ValueError(
                    f"messages[{index}]: tool_call_id '{call_id}' does not match "
                    "any tool call from a preceding assistant message"
                )
            unanswered.pop(call_id, None)

    # Unanswered calls are fine only when the assistant turn that made them is
    # the final message (the client is asking the model to continue from there).
    if unanswered and max(unanswered.values()) != last_index:
        missing = ", ".join(sorted(unanswered))
        raise ValueError(
            "An assistant message with 'tool_calls' must be followed by a "
            f"'tool' message for every tool_call_id; missing: {missing}"
        )


def validate_request_messages(
    request: ChatCompletionRequest,
) -> list[ChatMessage]:
    """Validate the parts of a chat request ``serving_chat`` does not, and
    return its messages.

    ``serving_chat.validate_chat_request`` covers tools, ``tool_choice`` and
    ``response_format``; this covers the conversation itself and the sampling
    ranges.

    Raises:
        ValueError: for any client error, which the endpoint maps to HTTP 400.
            Bad input must not reach the chat template (500) or be accepted
            silently (200).
    """
    messages = request.get_messages()
    _validate_message_sequence(messages)
    _validate_sampling(request)
    choice = request.tool_choice
    forced = choice in ("required", "any") or isinstance(choice, dict)
    if forced and not request.tools:
        raise ValueError(f"tool_choice {choice!r} requires a non-empty 'tools' list")
    return messages
