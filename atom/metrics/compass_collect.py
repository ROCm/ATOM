"""Pull every ATOM Compass record from running servers into local files.

    python -m atom.metrics.compass_collect --target host:8000 [--target ...] \\
        --output DIR [--interval 2] [-- benchmark command ...]

Polls ``/compass/files`` and ``/compass/records`` on each target, appending new
whole lines to ``DIR/<host_port>/<rank>/<file>``. Offsets are saved after each
append, so a restarted collector resumes without gaps or duplicates. With a
command, collection runs for the command's lifetime and the command's exit code
is returned; without one, it runs until SIGINT/SIGTERM. Either way it ends with
a drain that waits for the ranks' last flush.
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger("compass_collect")

# Ranks flush at most once a second and the engine polls pending events about
# as often; a drain round that finds nothing new for this long is final.
_DRAIN_QUIET_S = 3.0
_DRAIN_MAX_S = 60.0


class Target:
    def __init__(self, address, output):
        self.base = address if "://" in address else f"http://{address}"
        name = urllib.parse.urlsplit(self.base).netloc.replace(":", "_")
        self.dir = os.path.join(output, name)
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, "_offsets.json")
        self.offsets = {}
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                self.offsets = json.load(f)
        self.records = 0
        self.errors = 0

    def _get(self, path, **params):
        url = f"{self.base}{path}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read(), resp.headers

    def pull(self):
        """Fetch everything new on this target; return bytes appended."""
        body, _ = self._get("/compass/files")
        appended = 0
        for entry in json.loads(body)["files"]:
            rel, size = entry["file"], entry["size"]
            while self.offsets.get(rel, 0) < size:
                offset = self.offsets.get(rel, 0)
                data, headers = self._get("/compass/records", file=rel, offset=offset)
                next_offset = int(headers["X-Compass-Next-Offset"])
                if next_offset == offset:
                    break  # only a partial line so far
                path = os.path.join(self.dir, *rel.split("/"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                # A file rewritten whole (meta.json) restarts at offset 0.
                with open(path, "ab" if offset else "wb") as f:
                    f.write(data)
                self.offsets[rel] = next_offset
                self._save_offsets()
                appended += len(data)
                self.records += data.count(b"\n")
        return appended

    def _save_offsets(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.offsets, f)
        os.replace(tmp, self.state_path)

    def safe_pull(self):
        try:
            return self.pull()
        except (OSError, ValueError, KeyError) as e:
            # Unreachable or restarting server: keep polling, offsets hold.
            self.errors += 1
            logger.warning("%s: pull failed: %s", self.base, e)
            return 0


def collect(targets, interval, keep_running):
    while keep_running():
        began = time.monotonic()
        for t in targets:
            t.safe_pull()
        time.sleep(max(0.0, interval - (time.monotonic() - began)))
    deadline = time.monotonic() + _DRAIN_MAX_S
    quiet_since = time.monotonic()
    while time.monotonic() < deadline:
        if sum(t.safe_pull() for t in targets):
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= _DRAIN_QUIET_S:
            break
        time.sleep(0.5)


def main():
    argv = sys.argv[1:]
    command = []
    if "--" in argv:
        i = argv.index("--")
        argv, command = argv[:i], argv[i + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[compass_collect] %(message)s")

    targets = [Target(a, args.output) for a in args.target]
    stop = {"signal": None}

    def on_signal(signum, _frame):
        stop["signal"] = signum
        if proc is not None and proc.poll() is None:
            proc.send_signal(signum)

    proc = None
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    if command:
        proc = subprocess.Popen(command)

        def keep_running():
            return proc.poll() is None

    else:

        def keep_running():
            return stop["signal"] is None

    collect(targets, args.interval, keep_running)
    for t in targets:
        logger.info(
            "%s: %d records in %s (%d failed pulls)",
            t.base,
            t.records,
            t.dir,
            t.errors,
        )
    if proc is not None:
        rc = proc.wait()
        return 128 - rc if rc < 0 else rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
