"""Red/blue image distinction through ATOMesh: one text, sent with a red
image, then a blue one, then the red one again, one request after the other,
with the engine's prefix cache on (checks/distinct.env).

Passes when each answer names its own color and not the other, ATOMesh's log
($RUN_DIR/mesh.log) holds one routing decision event per request, the red
and blue ones with different key hashes, and the second red request starts
its prefill further into the engine's prefix cache than the blue one did
(its first prefill batch's "done" cursor in $RUN_DIR/atom.log). With the
text shared, only the images' media digests can tell the two routing keys
apart, and only the images' cache_seed keeps the blue request off the red
one's cached KV; the second red request shows that KV was there to reuse.
"""

import json
import re
import struct
import time
import urllib.request
import zlib
from base64 import b64encode
from pathlib import Path

from mmstar import env, finish, run_check

COLORS = {"red": (255, 0, 0), "blue": (0, 0, 255)}
# Large enough that the image's tokens fill whole prefix-cache hash blocks
# (block size x DCP size tokens): 2304 tokens on Kimi-K3.1 (14-pixel patches
# merged 2 x 2), whose hash block at block size 128 and DCP 8 is 1024 tokens.
SIDE = 1344
PROMPT = "What color is this image? Answer with one word."
# The second red request shows what the blue one would have reused.
REQUESTS = [("red", "red"), ("blue", "blue"), ("red_again", "red")]
EVENT_WAIT_S = 30

ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELD = re.compile(r"(\w+)=(\"[^\"]*\"|\S+)")
ARRIVED = re.compile(r"Request (\d+) arrived")
PREFILL = re.compile(
    r"Scheduled prefill batch: .*\(done: \[(\d+)\].*req_ids: \((\d+),\)"
)


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
    """The routing decision events in a router log, as dicts of the event's
    own fields (not its span's)."""
    events = []
    for line in log.splitlines():
        _, mark, fields = ANSI.sub("", line).partition("routing decision")
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


def prefix_hit(log):
    """The prompt tokens a request's prefill took from the prefix cache: where
    the engine's first prefill batch for it starts, in its log lines."""
    arrived = ARRIVED.search(log)
    if arrived is None:
        return None
    for done, req_id in PREFILL.findall(log):
        if req_id == arrived.group(1):
            return int(done)
    return None


def names(answer, color):
    return re.search(rf"\b{color}\b", answer, re.IGNORECASE) is not None


def main():
    log = Path(env("RUN_DIR"), "mesh.log")
    atom_log = Path(env("RUN_DIR"), "atom.log")
    seen = len(decision_events(read_log(log)))
    result, failures = {}, []
    for name, color in REQUESTS:
        start = len(read_log(atom_log))
        answer = ask(color)
        events = new_events(log, seen)
        seen += len(events)
        result[name] = {
            "answer": answer,
            "prefix_hit": prefix_hit(read_log(atom_log)[start:]),
            "key_hash": events[0].get("key_hash") if len(events) == 1 else None,
            "events": events,
        }
        other = next(c for c in COLORS if c != color)
        if not names(answer, color) or names(answer, other):
            failures.append(
                f"the {color} image's answer {answer!r} does not name {color} alone"
            )
        if len(events) != 1:
            failures.append(
                f"{len(events)} routing decision events in {log} for the {name} request"
            )
    hashes = [result[color]["key_hash"] for color in COLORS]
    if None not in hashes and hashes[0] == hashes[1]:
        failures.append(f"both colors routed on key hash {hashes[0]}")
    blue, again = result["blue"]["prefix_hit"], result["red_again"]["prefix_hit"]
    if blue is None or again is None:
        failures.append(f"no prefill start for a request in {atom_log}")
    elif again <= blue:
        failures.append(
            f"the second red request took {again} prompt tokens from the prefix "
            f"cache, the blue one {blue}: the red image's KV was not there to "
            "tell them apart"
        )
    finish(result, failures)


if __name__ == "__main__":
    run_check(main)
