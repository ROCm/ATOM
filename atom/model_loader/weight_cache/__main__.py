# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""python -m atom.model_loader.weight_cache status|stop"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import glob
import os
import sys
import time

from atom.model_loader.weight_cache.preload import stop_daemon
from atom.model_loader.weight_cache.transport import (
    _ping,
    daemon_alive,
    daemon_state,
    socket_dir,
)


def _running_daemons() -> dict:
    """{main pid: [(socket path, state), ...]} of this user's daemons here."""
    launches: dict = {}
    for lock in sorted(glob.glob(os.path.join(socket_dir(), "gpu_*.sock.lock"))):
        path = lock[: -len(".lock")]
        if daemon_alive(path):
            st = daemon_state(path) or {}
            launches.setdefault(
                st.get("daemon_pid") or f"? (runner {st.get('pid')})", []
            ).append((path, st))
    return launches


def _age(seconds: float) -> str:
    return str(datetime.timedelta(seconds=max(int(seconds), 0)))


def _status(launches: dict) -> None:
    now = time.time()
    for main_pid, gpus in launches.items():
        st0 = gpus[0][1]
        timeout = st0.get("idle_timeout_s")
        print(f"daemon pid {main_pid}: {st0.get('model', '?')}")
        print(
            f"  up {_age(now - st0.get('started', now))}, exits after "
            f"{f'{timeout}s idle' if timeout else 'no idle timeout'}; log {st0.get('log') or '-'}"
        )
        for path, st in sorted(gpus, key=lambda g: g[1].get("rank", 0)):
            line = f"  rank {st.get('rank', '?')}: {st.get('state', '?')}"
            if st.get("state") == "serving":
                try:
                    r = _ping(path, timeout=5)
                    line += f", {r['summary']['bytes'] / (1 << 30):.1f} GiB, {r.get('clients', '?')} server(s) connected"
                    if r.get("clients") == 0:
                        line += f", idle {_age(r.get('idle_s', 0))}"
                    if r.get("poisoned"):
                        line += ", WEIGHTS MODIFIED by a client (will refuse to serve)"
                except Exception as e:  # noqa: BLE001
                    line += f", not answering ({e})"
            else:
                line += f" for {_age(now - st.get('since', now))}"
            print(f"{line}; gpu {st.get('gpu', '?')}")


def _stop(launches: dict, force: bool) -> int:
    rc = 0
    for main_pid, gpus in launches.items():
        servers = 0  # each server connects once per GPU
        for path, _ in gpus:
            with contextlib.suppress(Exception):
                servers = max(servers, _ping(path, timeout=5).get("clients", 0))
        if servers and not force:
            # Their mappings outlive the daemon in practice, but HIP does not
            # promise that for memory whose exporter has exited.
            print(
                f"daemon pid {main_pid}: {servers} running server(s) use its weights; "
                "stop them first, or pass --force"
            )
            rc = 1
            continue
        try:
            for path, _ in gpus:
                stop_daemon(path)  # the first one stops the whole launch
        except Exception as e:  # noqa: BLE001
            print(f"daemon pid {main_pid}: {e}")
            rc = 1
            continue
        note = (
            f"; {servers} running server(s) keep its weights until they exit"
            if servers
            else ""
        )
        print(f"stopped daemon pid {main_pid} ({len(gpus)} GPU(s)){note}")
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m atom.model_loader.weight_cache",
        description="Show or stop this user's ATOM weight cache daemons on this machine.",
    )
    parser.add_argument("command", choices=("status", "stop"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="stop: also stop daemons that running servers use",
    )
    args = parser.parse_args(argv)
    launches = _running_daemons()
    if not launches:
        print(f"no weight cache daemon running (in {socket_dir()})")
        return 0
    if args.command == "status":
        _status(launches)
        return 0
    return _stop(launches, args.force)


if __name__ == "__main__":
    sys.exit(main())
