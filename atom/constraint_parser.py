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


@dataclass(frozen=True)
class ConstraintBinding:
    """A native rule value tagged for its execution adapter.

    Bindings are rebuilt at the engine boundary; provider objects never cross
    sequence IPC and do not replace native RequiredChannel/grammar objects.
    """
    backend: str
    value: object


class ConstraintParser:
    """Compile format declarations through one public interface.

    The adapter is a tokenizer for channel rules and a grammar compiler for
    grammar rules. Each keeps its native representation and execution backend.
    """

    _adapters = {}

    @classmethod
    def register(cls, rule_type, backend, adapter):
        if rule_type in cls._adapters:
            raise ValueError(f"Duplicate constraint rule: {rule_type.__name__}")
        cls._adapters[rule_type] = (backend, adapter)

    @classmethod
    def parse(cls, rule, context):
        if rule is None:
            return None
        try:
            _, adapter = cls._adapters[type(rule)]
        except KeyError:
            raise TypeError(f"Unsupported constraint rule: {type(rule).__name__}") from None
        return adapter(rule, context)

    @classmethod
    def bind(cls, rule, context):
        if rule is None:
            return None
        value = cls.parse(rule, context)
        if value is None:
            return None
        backend, _ = cls._adapters[type(rule)]
        return ConstraintBinding(backend, value)


def _parse_grammar(rule, adapter):
    return adapter.compile_json_schema(rule.schema, rule.tool_format, rule.reasoning_end)


def _parse_tool_grammar(rule, adapter):
    return rule.builder(rule.functions, required=rule.required,
                        starts_in_reasoning=rule.starts_in_reasoning)


def _parse_channel(rule, adapter):
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


ConstraintParser.register(ChannelRule, "channel", _parse_channel)
ConstraintParser.register(GrammarRule, "grammar", _parse_grammar)
ConstraintParser.register(ToolGrammarRule, "grammar-format", _parse_tool_grammar)
