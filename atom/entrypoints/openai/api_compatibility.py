# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Frontend contract choices, separate from prompt encoder capabilities."""

from dataclasses import dataclass


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


def api_compatibility_for(encoder) -> APICompatibility:
    # Model adapters carry their wire contract; unknown adapters retain the
    # generic defaults without importing or recognizing a particular model.
    return getattr(encoder, "api_compatibility", DEFAULT_API_COMPATIBILITY)
