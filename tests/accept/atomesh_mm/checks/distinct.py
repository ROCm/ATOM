"""Red/blue image distinction through ATOMesh: one text, sent once with a red
and once with a blue image, one request after the other, with the engine's
prefix cache on and every routing decision logged (checks/distinct.env).

Passes when each answer names its own color and not the other, and ATOMesh's
log ($RUN_DIR/mesh.log) holds one `Selected worker` event (target
`mesh::route`) per request, each with `media_count=1`, the blue one with a
`match_ratio` below 1. A media request routes on cache_aware's prefix tree,
which by then holds the red request's routing key: a blue key equal to it
would match whole. With the text shared, only the images' media digests can
tell the two keys apart, and only the images' cache_seed keeps the second
request off the first one's cached KV.

Needs ROCm/ATOM#3407, which puts `media_count` on the route log and keys a
media chat's route by its images; on main the blue request's key equals the
red one's and the check fails.
"""

import json
import re
import struct
import time
import urllib.request
import zlib
from base64 import b64encode
from pathlib import Path

from _common import env, finish, run_check

COLORS = {"red": (255, 0, 0), "blue": (0, 0, 255)}
SIDE = 448  # large enough that the image's tokens fill whole KV blocks
PROMPT = "What color is this image? Answer with one word."
EVENT_WAIT_S = 30

ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELD = re.compile(r"(\w+)=(\"[^\"]*\"|\S+)")


def png(rgb, side=SIDE):
    """A solid side x side RGB PNG."""

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * side
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * side))
        + chunk(b"IEND", b"")
    )


def image_url(color):
    return "data:image/png;base64," + b64encode(png(COLORS[color])).decode()


def ask(color):
    """The answer (content, without reasoning) to the prompt with color's image."""
    body = {
        "model": env("SERVED_MODEL"),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url(color)}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 8192,
    }
    request = urllib.request.Request(
        env("ENDPOINT") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        reply = json.loads(response.read())
    return reply["choices"][0]["message"]["content"] or ""


def decision_events(log):
    """The `Selected worker` events in a router log, as dicts of the event's
    own fields (not its span's)."""
    events = []
    for line in log.splitlines():
        _, mark, fields = ANSI.sub("", line).partition("Selected worker")
        if mark:
            events.append({k: v.strip('"') for k, v in FIELD.findall(fields)})
    return events


def read_log(path):
    return path.read_text(errors="replace") if path.exists() else ""


def new_events(path, seen):
    """The decision events after the first seen, waiting for at least one."""
    deadline = time.monotonic() + EVENT_WAIT_S
    while True:
        events = decision_events(read_log(path))[seen:]
        if events or time.monotonic() > deadline:
            return events
        time.sleep(1)


def names(answer, color):
    return re.search(rf"\b{color}\b", answer, re.IGNORECASE) is not None


def main():
    log = Path(env("RUN_DIR"), "mesh.log")
    seen = len(decision_events(read_log(log)))
    result, failures = {}, []
    for color in COLORS:
        answer = ask(color)
        events = new_events(log, seen)
        seen += len(events)
        event = events[0] if len(events) == 1 else {}
        result[color] = {
            "answer": answer,
            "branch": event.get("branch"),
            "match_ratio": event.get("match_ratio"),
            "media_count": event.get("media_count"),
            "events": events,
        }
        other = next(c for c in COLORS if c != color)
        if not names(answer, color) or names(answer, other):
            failures.append(
                f"the {color} image's answer {answer!r} does not name {color} alone"
            )
        if len(events) != 1:
            failures.append(
                f"{len(events)} Selected worker events in {log} for the {color} request"
            )
        elif event.get("media_count") != "1":
            failures.append(
                f"the {color} request routed with media_count "
                f"{event.get('media_count')}, not 1"
            )
    blue = result["blue"]
    if len(blue["events"]) == 1:
        if blue["match_ratio"] is None:
            failures.append(
                f"the blue request's route ({blue['branch']}) logged no match_ratio"
            )
        elif float(blue["match_ratio"]) >= 1:
            failures.append(
                "the blue request matched the red one's routing key whole "
                f"(match_ratio {blue['match_ratio']})"
            )
    finish(result, failures)


if __name__ == "__main__":
    run_check(main)
