# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's channel tokens, owned in one place.

Two readers strip these: the reasoning stage, which separates the think
channel from the answer, and the tool-call parser, which reads the tools
section. Each used to carry its own list and they drifted apart: tokens only
the parser listed were removed only when a K3 tool parser happened to be
resolved, so a deployment whose template refused the tools probe leaked them
to `content` on both delivery paths. Which literals are framing is a property
of the model's wire format, not of whether tool calling was configured.

The split below is by *owner*, and it is the one thing this module asserts:

- :data:`CHANNEL_FRAMING` wraps every answer whether or not a tool was called,
  so the reasoning stage removes it and the tool parser does too.
- :data:`TOOL_REGION_FRAMING` brackets a call. Those are the tool parser's
  alone: the reasoning stage cannot remove the payload between them -- the
  tool name, the argument keys and their values -- so stripping only the
  brackets would hand the client mangled half-text instead of either a call
  or an honest quotation.
"""

THINK_START = "<|open|>think<|sep|>"
THINK_END = "<|close|>think<|sep|>"
RESPONSE_START = "<|open|>response<|sep|>"
RESPONSE_END = "<|close|>response<|sep|>"
MESSAGE_START = "<|open|>message<|sep|>"
MESSAGE_END = "<|close|>message<|sep|>"
END_OF_MSG = "<|end_of_msg|>"
TOOLS_START = "<|open|>tools<|sep|>"


# Wrappers around the answer itself. Paired spellings only; the bare closers
# are in `UNPAIRED_FRAMING` below, for the reason given there.
CHANNEL_FRAMING: tuple[str, ...] = (
    RESPONSE_START,
    RESPONSE_END,
    MESSAGE_START,
    MESSAGE_END,
    END_OF_MSG,
)

# The unpaired closers, and the separator on its own. These are framing too,
# but they cannot move into `CHANNEL_FRAMING`, for two independent reasons,
# even though the move looks obviously right.
#
# `<|sep|>` occurs *inside* a tool region (`<|open|>call tool="x"<|sep|>`),
# and the reasoning stage runs first. Removing it there turns a region the
# reasoning stage is only passing through into mangled half-text before the
# parser sees it.
#
# The bare closers are proper prefixes of the paired ones, and `MarkerScanner`
# reports a complete short marker rather than waiting to see whether the long
# one follows (see `_plan`, which documents the gap and the test that holds
# formats inside it). In this parser's list that is harmless -- strip
# `<|close|>think` and the `<|sep|>` it leaves is also framing, so both
# chunkings converge. Add the bare ones to the reasoning dialect *without*
# `<|sep|>` and they stop converging: the shorter one fires, the separator
# survives, and `<|close|>think<|sep|>` -- the token that ends the reasoning
# channel -- never matches, so at four bytes per chunk the whole answer stays
# in `reasoning_content`. They are a set or they are nothing.
# Named on its own as well, because it is also what sits between a call's
# closer and the section's -- the wrapper walk has to step over it there.
BARE_SEPARATOR = "<|sep|>"

UNPAIRED_FRAMING: tuple[str, ...] = (
    "<|close|>response",
    "<|close|>think",
    "<|close|>message",
    BARE_SEPARATOR,
)

# Brackets around a tool call. The tool parser's, for the reason in the module
# docstring.
TOOL_REGION_FRAMING: tuple[str, ...] = (
    TOOLS_START,
    "<|close|>tools",
    "<|close|>call",
    "<|close|>argument",
)
