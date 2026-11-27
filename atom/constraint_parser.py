# SPDX-License-Identifier: MIT
"""Channel rule declarations, compiled without importing execution backends."""

import logging
from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelRule:
    after: str
    opening: str
    channel_start: str
    end: str
    controls: tuple[str, ...]
    label: str


def parse_channel_rule(rule, tokenizer):
    """`rule` as a `RequiredChannel` in the tokenizer's ids, or None (logged)."""
    from atom.sampling_params import RequiredChannel

    def ids(text):
        return tuple(tokenizer.encode(text, add_special_tokens=False))

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
