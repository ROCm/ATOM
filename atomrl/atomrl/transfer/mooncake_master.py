# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Mooncake master process launcher."""

from __future__ import annotations

import atexit
import ctypes
import logging
import os
import random
import shutil
import signal
import socket
import subprocess
import threading
import time

logger = logging.getLogger(__name__)


def _resolve_mooncake_bin() -> str:
    if "MOONCAKE_BUILD_DIR" in os.environ:
        return os.path.join(
            os.environ["MOONCAKE_BUILD_DIR"], "mooncake-store/src/mooncake_master"
        )
    which = shutil.which("mooncake_master")
    if which:
        return which
    return os.path.join(
        os.path.expanduser("~"), "build/mooncake-store/src/mooncake_master"
    )


def _preexec():
    os.setpgrp()
    PR_SET_PDEATHSIG = 1
    ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


class MooncakeMaster:
    """Manages the mooncake_master subprocess lifecycle."""

    def __init__(self):
        self._process = None
        self._info = {}

    def start(
        self,
        port: int = 0,
        http_port: int = 0,
        http_host: str = "0.0.0.0",
        kv_lease_ttl_s: float = 5.0,
    ) -> dict:
        binary = _resolve_mooncake_bin()
        if not os.path.exists(binary):
            raise FileNotFoundError(f"mooncake_master not found at {binary}")

        if port == 0:
            port = _find_free_port(51000, 52000)
        if http_port == 0:
            http_port = _find_free_port(8100, 9100)

        cmd = [
            binary,
            f"--port={port}",
            f"--http_metadata_server_port={http_port}",
            f"--http_metadata_server_host={http_host}",
            "--enable_http_metadata_server=true",
            f"--default_kv_lease_ttl={int(kv_lease_ttl_s * 1000)}",
        ]

        logger.info("Starting mooncake_master: port=%d, http=%d", port, http_port)
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_preexec,  # noqa: PLW1509  # Linux parent-death signal requires pre-exec setup.
        )

        self._stream_logs(self._process.stdout, "stdout")
        self._stream_logs(self._process.stderr, "stderr")

        time.sleep(2)
        if self._process.poll() is not None:
            raise RuntimeError(
                f"mooncake_master failed to start (exit={self._process.returncode})"
            )

        host = socket.gethostbyname(socket.gethostname())
        self._info = {
            "master_addr": f"{host}:{port}",
            "metadata_server": f"http://{host}:{http_port}/metadata",
            "http_port": http_port,
        }
        logger.info(
            "mooncake_master started: %s (PID=%d)", self._info, self._process.pid
        )
        return self._info

    def get_info(self) -> dict:
        return self._info

    def shutdown(self) -> None:
        if self._process is not None and self._process.poll() is None:
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
                self._process.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except OSError:  # SIGKILL can race with process-group exit.
                    pass
            self._process = None

    def _stream_logs(self, stream, name: str) -> None:
        def _reader():
            for line in stream:
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                line = line.rstrip("\n")
                if line:
                    logger.debug("[mooncake_master %s] %s", name, line)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

    def __del__(self):
        proc = getattr(self, "_process", None)
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:  # Destructor cleanup can race with process-group exit.
                    pass


def launch_mooncake_master(config) -> MooncakeMaster:
    """Launch mooncake master and return the handle."""
    master = MooncakeMaster()

    master_addr = getattr(config, "master_server_address", None)
    port = 0
    http_port = 0

    if master_addr and ":" in master_addr:
        port = int(master_addr.split(":")[1])

    info = master.start(
        port=port,
        http_port=http_port,
        kv_lease_ttl_s=getattr(config, "kv_lease_ttl_s", 5.0),
    )

    config.master_server_address = info["master_addr"]
    config.metadata_server = info["metadata_server"]

    atexit.register(master.shutdown)
    return master


def _find_free_port(start: int = 50000, end: int = 60000) -> int:
    port = random.randint(start, end)
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                port = random.randint(start, end)
    raise RuntimeError(f"Cannot find free port in [{start}, {end}]")
