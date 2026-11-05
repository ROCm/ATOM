# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Client stop strings, found in the generated text rather than its tokens.

The engine stops on a stop string's own tokenization, but in context the model
often spells the same text with other tokens -- "```" inside a "\\n```" token --
and that match never fires; when it does fire, the stop string's tokens are
already part of the output. The APIs return the text before the first stop
string and nothing after it, so the check is made here on decoded text: what
may still turn out to begin a stop string is held back until it does not, and
the first stop string found ends the text and the request.
"""

from collections.abc import Iterable

from .marker_scanner import MarkerScanner

# DeepSeek's limit on the number. The length bound is ours: the hold-back
# plan keeps every prefix of every stop string, so its size grows with the
# square of the length.
MAX_STOP_STRINGS = 16
MAX_STOP_STRING_CHARS = 256


def stop_list(stop: str | Iterable[str] | None) -> list[str]:
    """The non-empty stop strings, from either spelling the APIs accept."""
    if stop is None:
        return []
    if isinstance(stop, str):
        stop = [stop]
    return [s for s in stop if s]


def check_stop_strings(stop: str | Iterable[str] | None) -> None:
    """Refuse more stop strings, or longer ones, than are watched for."""
    stops = stop_list(stop)
    if len(stops) > MAX_STOP_STRINGS:
        raise ValueError(f"at most {MAX_STOP_STRINGS} stop strings are accepted")
    if any(len(s) > MAX_STOP_STRING_CHARS for s in stops):
        raise ValueError(
            f"a stop string may be at most {MAX_STOP_STRING_CHARS} characters"
        )


class StopStrings:
    """Releases streamed text once it cannot belong to a stop string.

    The stop strings are the markers of a `MarkerScanner`, which holds back
    the longest end that may still begin one and reports the first one
    complete; that one ends the text.
    """

    def __init__(self, stops: list[str]):
        self.stopped = False
        self._scanner = MarkerScanner(tuple(stops))

    def feed(self, text: str) -> str:
        """What of ``text`` can be released now; nothing after a stop string."""
        if self.stopped:
            return ""
        scan = self._scanner.feed(text)
        if scan.hit is not None:
            self.stopped = True
            self._scanner.take_rest()
        return scan.released

    def flush(self) -> str:
        """The held-back end, once the text is over without a stop string."""
        return "" if self.stopped else self._scanner.flush()
