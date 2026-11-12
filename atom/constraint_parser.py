# SPDX-License-Identifier: MIT
"""Declarative constraint parsing without importing execution backends."""

from dataclasses import dataclass
import logging


@dataclass(frozen=True)
class ChannelRule:
    after: str
    opening: str
    channel_start: str
    end: str
    controls: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class GrammarRule:
    schema: str
    tool_format: object = None
    reasoning_end: str = ""


@dataclass(frozen=True)
class ToolGrammarRule:
    builder: object
    functions: list[dict]
    required: bool
    starts_in_reasoning: bool


class ConstraintParser:
    """Compile format declarations through one public interface.

    The adapter is a tokenizer for channel rules and a grammar compiler for
    grammar rules. Each keeps its native representation and execution backend.
    """

    @classmethod
    def parse(cls, rule, adapter):
        if rule is None:
            return None
        if isinstance(rule, ToolGrammarRule):
            return rule.builder(rule.functions, required=rule.required,
                                starts_in_reasoning=rule.starts_in_reasoning)
        if isinstance(rule, GrammarRule):
            return adapter.compile_json_schema(
                rule.schema, rule.tool_format, rule.reasoning_end
            )
        if not isinstance(rule, ChannelRule):
            raise TypeError(f"Unsupported constraint rule: {type(rule).__name__}")
        from atom.sampling_params import RequiredChannel

        def ids(text):
            return tuple(adapter.encode(text, add_special_tokens=False))

        def unheld(reason):
            logging.getLogger("atom").warning(
                '%s tool calls: %s, so tool_choice="required" will not '
                "hold output to the tools channel on this server.",
                rule.label,
                reason,
            )

        split = [c for c in rule.controls if len(ids(c)) != 1]
        if split:
            unheld(f"this tokenizer splits {', '.join(split)} into several ids")
            return None
        start, opening = ids(rule.channel_start), ids(rule.opening)
        if opening[: len(start)] != start:
            unheld(
                f"this tokenizer does not encode {rule.channel_start!r} at the "
                "head of the forced opening as it encodes it alone"
            )
            return None
        return RequiredChannel(
            after=ids(rule.after), opening=opening,
            opens_at=len(start), end_ids=ids(rule.end),
        )
