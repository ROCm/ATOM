# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Frontend contract choices, separate from prompt encoder capabilities."""

from dataclasses import dataclass

from .deepseek_v41_encoder import DEFAULT_EFFORT_LEVEL, SUPPORTED_EFFORT_LEVELS


@dataclass(frozen=True)
class APICompatibility:
    chat_usage_on_finish: bool = False
    chat_forced_tools_with_thinking: bool = True
    responses_ignored_fields: frozenset[str] = frozenset()
    anthropic_output_effort: bool = False
    display_name: str | None = None
    output_token_limit: int | None = None
    effort_levels: tuple[str, ...] = ()
    default_effort: str | None = None

    def responses_body(self, body: dict) -> dict:
        return {k: v for k, v in body.items() if k not in self.responses_ignored_fields}

    def validate_chat_tool_choice(self, choice, *, thinking_enabled: bool) -> None:
        if (
            not self.chat_forced_tools_with_thinking
            and thinking_enabled
            and (choice == "required" or isinstance(choice, dict))
        ):
            raise ValueError(
                "This Chat API contract does not support required or named tool_choice in thinking mode"
            )


DEFAULT_API_COMPATIBILITY = APICompatibility()
DEEPSEEK_API_COMPATIBILITY = APICompatibility(
    display_name="DeepSeek-V4.1-Flash",
    output_token_limit=393216,
    effort_levels=SUPPORTED_EFFORT_LEVELS,
    default_effort=DEFAULT_EFFORT_LEVEL,
    chat_usage_on_finish=True,
    chat_forced_tools_with_thinking=False,
    anthropic_output_effort=True,
    # Explicit unsupported/ignored fields from DeepSeek's Responses support table.
    # Unknown extension fields are not globally stripped.
    responses_ignored_fields=frozenset(
        {
            "background",
            "previous_response_id",
            "conversation",
            "store",
            "parallel_tool_calls",
            "max_tool_calls",
            "metadata",
            "include",
            "prompt",
            "truncation",
            "service_tier",
            "safety_identifier",
            "prompt_cache_key",
            "prompt_cache_retention",
            "context_management",
            "stream_options",
        }
    ),
)


def api_compatibility_for(encoder) -> APICompatibility:
    # Preserve the current deployment selection in one place. Wire behavior is
    # a frontend contract, not an encoder operation or a served-model alias.
    return {"encoding_dsv41": DEEPSEEK_API_COMPATIBILITY}.get(
        getattr(encoder, "name", None), DEFAULT_API_COMPATIBILITY
    )
