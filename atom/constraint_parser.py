# SPDX-License-Identifier: MIT
"""Declarative constraint parsing without importing execution backends."""

import logging
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ChannelRule:
    after: str
    opening: str
    channel_start: str
    end: str
    controls: tuple[str, ...]
    label: str


class ConstraintParser:
    """Compile format declarations through one public interface.

    A rule's adapter receives the rule and a context (a tokenizer for channel
    rules) and returns the native object its execution backend consumes.
    """

    _adapters: ClassVar[dict] = {}

    @classmethod
    def register(cls, rule_type, adapter):
        if rule_type in cls._adapters:
            raise ValueError(f"Duplicate constraint rule: {rule_type.__name__}")
        cls._adapters[rule_type] = adapter

    @classmethod
    def parse(cls, rule, context):
        if rule is None:
            return None
        try:
            adapter = cls._adapters[type(rule)]
        except KeyError:
            raise TypeError(
                f"Unsupported constraint rule: {type(rule).__name__}"
            ) from None
        return adapter(rule, context)


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
        after=ids(rule.after),
        opening=opening,
        opens_at=len(start),
        end_ids=ids(rule.end),
    )


ConstraintParser.register(ChannelRule, _parse_channel)
