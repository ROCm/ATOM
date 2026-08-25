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
* :func:`resolve_tool_choice` — turn ``tool_choice`` into the tool list the
  chat template advertises plus the output-parsing policy.
* :func:`resolve_thinking` / :func:`disable_primed_thinking` — MiniMax's
  ``thinking: {"type": "enabled"|"disabled"|"adaptive"}`` extension.

This module deliberately imports nothing heavier than :mod:`.protocol` (pydantic
only), so it is unit-testable without a GPU, an engine or FastAPI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from atom.sampling_params import MAX_TEMPERATURE

from .protocol import ChatCompletionRequest, ChatMessage, _fix_invalid_json_escapes
from .reasoning import THINK_MARKER_PAIRS

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


def _append_text(message: ChatMessage, extra: str) -> ChatMessage:
    """Return a copy of ``message`` with ``extra`` appended to its text content."""
    if isinstance(message.content, list):
        content: Any = list(message.content) + [{"type": "text", "text": extra}]
    else:
        base = message.content or ""
        content = f"{base}\n\n{extra}" if base else extra
    return message.model_copy(update={"content": content})


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
) -> List[ChatMessage]:
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
    kept: List[ChatMessage] = []
    roots: List[ChatMessage] = []
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


def apply_system_directive(
    messages: Sequence[ChatMessage], directive: Optional[str]
) -> List[ChatMessage]:
    """Attach ``directive`` to the conversation's system-level instructions.

    Appended to the last system message when there is one (safest for chat
    templates that require a specific message order), otherwise inserted as a
    new leading ``system`` message.
    """
    out = list(messages)
    if not directive:
        return out
    for index in range(len(out) - 1, -1, -1):
        if (out[index].role or "").strip() in SYSTEM_LIKE_ROLES:
            out[index] = _append_text(out[index], directive)
            return out
    return [ChatMessage(role="system", content=directive)] + out


# ---------------------------------------------------------------------------
# tool_choice
# ---------------------------------------------------------------------------

REQUIRED_TOOL_DIRECTIVE = (
    "You must call at least one of the available tools to answer this request. "
    "Do not reply without calling a tool."
)


def named_tool_directive(name: str) -> str:
    return (
        f'You must call the tool "{name}" to answer this request. '
        "Do not call any other tool and do not reply without calling it."
    )


@dataclass(frozen=True)
class ResolvedToolChoice:
    """How one request's ``tools``/``tool_choice`` pair is served.

    Attributes:
        mode: ``"auto"``, ``"none"``, ``"required"`` or ``"function"``.
        function_name: the requested function for ``mode == "function"``.
        template_tools: the tool list handed to the chat template (``None``
            advertises no tools at all, which is how ``"none"`` is served).
        parse_output: whether model output is scanned for tool calls. False for
            ``"none"`` so a model that emits tool syntax anyway cannot smuggle
            ``tool_calls`` into a response the client asked to be tool-free.
        directive: extra system instruction used to steer ``required`` /
            ``function`` mode. ATOM has no constrained decoding, so forcing a
            call is best-effort prompting rather than a hard guarantee.
    """

    mode: str = "auto"
    function_name: Optional[str] = None
    template_tools: Optional[List[Dict[str, Any]]] = None
    parse_output: bool = True
    directive: Optional[str] = None


def tool_name(tool: Any) -> Optional[str]:
    """Extract the function name from an OpenAI or bare tool entry."""
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function", tool)
    if not isinstance(fn, dict):
        return None
    name = fn.get("name")
    return name if isinstance(name, str) else None


def parse_tool_choice(tool_choice: Any) -> Tuple[str, Optional[str]]:
    """Normalize ``tool_choice`` to ``(mode, function_name)``.

    Raises:
        ValueError: on a value outside the OpenAI grammar.
    """
    if tool_choice is None:
        return "auto", None
    if isinstance(tool_choice, str):
        mode = tool_choice.strip().lower()
        if mode in ("auto", "none", "required"):
            return mode, None
        if mode == "any":  # Anthropic spelling of "required"
            return "required", None
        raise ValueError(
            f"Invalid tool_choice '{tool_choice}'. Expected 'auto', 'none', "
            "'required', or {'type': 'function', 'function': {'name': ...}}"
        )
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") not in (None, "function"):
            raise ValueError(
                f"Unsupported tool_choice type '{tool_choice.get('type')}'; "
                "only 'function' is supported"
            )
        name = tool_name(tool_choice)
        if not name:
            raise ValueError("tool_choice.function.name is required")
        return "function", name
    raise ValueError(f"Invalid tool_choice of type {type(tool_choice).__name__}")


def resolve_tool_choice(
    tools: Optional[List[Dict[str, Any]]], tool_choice: Any
) -> ResolvedToolChoice:
    """Build the :class:`ResolvedToolChoice` for one request.

    Assumes :func:`validate_chat_request` already ran; unresolvable
    combinations degrade to ``auto`` rather than raising.
    """
    mode, name = parse_tool_choice(tool_choice)

    if mode == "none":
        return ResolvedToolChoice(mode="none", template_tools=None, parse_output=False)

    if not tools:
        return ResolvedToolChoice(mode="auto", template_tools=None)

    if mode == "function":
        selected = [t for t in tools if tool_name(t) == name]
        if not selected:
            return ResolvedToolChoice(mode="auto", template_tools=list(tools))
        return ResolvedToolChoice(
            mode="function",
            function_name=name,
            template_tools=selected,
            directive=named_tool_directive(name),
        )

    if mode == "required":
        return ResolvedToolChoice(
            mode="required",
            template_tools=list(tools),
            directive=REQUIRED_TOOL_DIRECTIVE,
        )

    return ResolvedToolChoice(mode="auto", template_tools=list(tools))


# ---------------------------------------------------------------------------
# thinking (MiniMax extension)
# ---------------------------------------------------------------------------

_THINKING_ON = ("enabled", "enable", "on", "true")
_THINKING_OFF = ("disabled", "disable", "off", "false")
# MiniMax-M3's third mode: the model decides per turn. Leaving the template
# switch unset is exactly that, so these resolve to None rather than a bool.
_THINKING_AUTO = ("adaptive", "auto")


def resolve_thinking(thinking: Any) -> Optional[bool]:
    """Normalize the ``thinking`` request extension.

    Returns True (think), False (do not think) or None (unspecified — leave the
    template default alone).

    Accepts MiniMax's ``{"type": "enabled"|"disabled"|"adaptive"}`` object plus
    the plain bool / string spellings other providers use.

    Raises:
        ValueError: on a value outside that grammar.
    """
    if thinking is None:
        return None
    value: Any = thinking
    if isinstance(thinking, dict):
        if "type" not in thinking:
            raise ValueError("'thinking' object requires a 'type' field")
        value = thinking["type"]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _THINKING_ON:
            return True
        if normalized in _THINKING_OFF:
            return False
        if normalized in _THINKING_AUTO:
            return None
    raise ValueError(
        "'thinking.type' must be 'enabled', 'disabled' or 'adaptive', got "
        f"{value!r}"
    )


def thinking_template_kwargs(enabled: Optional[bool]) -> Dict[str, Any]:
    """Chat-template kwargs that express the thinking toggle.

    Templates spell the switch differently — ``thinking_mode`` with the string
    values MiniMax-M3 expects, ``enable_thinking`` for Qwen3/GLM, plain
    ``thinking`` for others. Extra kwargs reach the template as unused Jinja
    variables (verified against transformers' ``apply_chat_template``, which
    merges ``**kwargs`` straight into the render context), so passing all three
    is safe and covers whichever the loaded template reads.

    Left empty when unset, so the template keeps its own default — MiniMax-M3's
    is ``adaptive``, i.e. the model decides per turn.
    """
    if enabled is None:
        return {}
    return {
        "enable_thinking": enabled,
        "thinking": enabled,
        "thinking_mode": "enabled" if enabled else "disabled",
    }


def disable_primed_thinking(prompt: str) -> str:
    """Close a ``<think>`` block the chat template primed at the end of a prompt.

    A template that primes thinking unconditionally would make the model reason
    even when the request asked it not to. Turning the open marker into an empty,
    already-closed block (``<think></think>``, the trick Qwen3's template applies
    natively) tells the model thinking is done and the answer starts now.

    Only a fallback: MiniMax-M3's template handles ``thinking_mode="disabled"``
    itself, emitting the *closing* marker as the generation prefix, in which case
    there is no open marker here and the prompt is returned unchanged.
    """
    for open_marker, close_marker in THINK_MARKER_PAIRS:
        index = prompt.rfind(open_marker)
        if index == -1:
            continue
        tail = prompt[index + len(open_marker) :]
        if tail.strip() or close_marker in tail:
            continue
        return prompt[: index + len(open_marker)] + close_marker + tail
    return prompt


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
    top_p = request.top_p
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    top_k = request.top_k
    if top_k is not None and top_k != -1 and top_k < 1:
        raise ValueError(f"top_k must be -1 (disabled) or >= 1, got {top_k}")
    for name in ("presence_penalty", "frequency_penalty"):
        value = getattr(request, name, None)
        if value is not None and not -2.0 <= value <= 2.0:
            raise ValueError(f"{name} must be in [-2, 2], got {value}")
    for name in ("max_tokens", "max_completion_tokens"):
        value = getattr(request, name, None)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")


def _validate_tools(tools: Optional[List[Dict[str, Any]]]) -> None:
    if not tools:
        return
    seen: set = set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("each entry of 'tools' must be an object")
        tool_type = tool.get("type", "function")
        if tool_type != "function":
            raise ValueError(
                f"unsupported tool type '{tool_type}'; only 'function' is supported"
            )
        fn = tool.get("function", tool if "name" in tool else None)
        if not isinstance(fn, dict):
            raise ValueError("each tool requires a 'function' object")
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("each tool requires a non-empty 'function.name'")
        if name in seen:
            raise ValueError(f"duplicate tool name '{name}'")
        seen.add(name)
        schema = fn.get("parameters", fn.get("input_schema"))
        if schema is not None and not isinstance(schema, dict):
            raise ValueError(
                f"tool '{name}': 'parameters' must be a JSON Schema object"
            )


def _validate_tool_choice(
    tools: Optional[List[Dict[str, Any]]], tool_choice: Any
) -> None:
    mode, name = parse_tool_choice(tool_choice)
    if mode in ("required", "function") and not tools:
        raise ValueError(f"tool_choice '{mode}' requires a non-empty 'tools' list")
    if mode == "function":
        available = {tool_name(t) for t in tools or []}
        if name not in available:
            raise ValueError(
                f"tool_choice requests function '{name}', which is not in 'tools'"
            )


def _validate_assistant_tool_calls(tool_calls: Any) -> List[str]:
    """Validate one assistant message's ``tool_calls``; return their ids."""
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ValueError("'tool_calls' must be a non-empty array")
    ids: List[str] = []
    for entry in tool_calls:
        if not isinstance(entry, dict):
            raise ValueError("each entry of 'tool_calls' must be an object")
        call_type = entry.get("type", "function")
        if call_type != "function":
            raise ValueError(f"unsupported tool_call type '{call_type}'")
        fn = entry.get("function")
        if not isinstance(fn, dict):
            raise ValueError("each tool_call requires a 'function' object")
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
            raise ValueError(
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
    unanswered: Dict[str, int] = {}
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


def validate_chat_request(request: ChatCompletionRequest) -> List[ChatMessage]:
    """Validate a chat request and return its messages.

    Raises:
        ValueError: for any client error. The endpoint maps this to HTTP 400,
            which is what the OpenAI API returns for malformed input — bad
            input must not reach the chat template (500) or be silently
            accepted (200).
    """
    messages = request.get_messages()
    _validate_message_sequence(messages)
    _validate_sampling(request)
    _validate_tools(request.tools)
    _validate_tool_choice(request.tools, request.tool_choice)
    resolve_thinking(request.thinking)
    return messages


# ---------------------------------------------------------------------------
# One-shot request preparation
# ---------------------------------------------------------------------------


@dataclass
class PreparedChatRequest:
    """Validated, template-ready view of a chat request."""

    messages: List[ChatMessage] = field(default_factory=list)
    tool_choice: ResolvedToolChoice = field(default_factory=ResolvedToolChoice)
    template_kwargs: Dict[str, Any] = field(default_factory=dict)
    thinking_enabled: Optional[bool] = None

    @property
    def parse_tool_calls(self) -> bool:
        return self.tool_choice.parse_output

    @property
    def parse_reasoning(self) -> bool:
        """Reasoning separation is skipped when thinking was turned off.

        The prompt already closes the thinking block, so any ``</think>`` in the
        output would be model noise — treating it as a reasoning delimiter would
        resurrect the ``reasoning_content`` the client asked not to get.
        """
        return self.thinking_enabled is not False


def prepare_chat_request(
    request: ChatCompletionRequest,
    *,
    default_template_kwargs: Optional[Dict[str, Any]] = None,
    supported_roles: frozenset = frozenset(),
) -> PreparedChatRequest:
    """Validate ``request`` and derive everything the endpoint needs from it.

    Raises:
        ValueError: for malformed input (mapped to HTTP 400 by the endpoint).
    """
    messages = validate_chat_request(request)
    thinking_enabled = resolve_thinking(request.thinking)
    tool_choice = resolve_tool_choice(request.tools, request.tool_choice)

    messages = normalize_chat_messages(messages, supported_roles=supported_roles)
    messages = apply_system_directive(messages, tool_choice.directive)

    template_kwargs = dict(default_template_kwargs or {})
    template_kwargs.update(thinking_template_kwargs(thinking_enabled))
    if request.chat_template_kwargs:
        # An explicit chat_template_kwargs from the client wins over both the
        # server default and the derived thinking switch.
        template_kwargs.update(request.chat_template_kwargs)

    return PreparedChatRequest(
        messages=messages,
        tool_choice=tool_choice,
        template_kwargs=template_kwargs,
        thinking_enabled=thinking_enabled,
    )
