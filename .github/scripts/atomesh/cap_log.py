#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Copy stdin to a log file, stopping at a byte limit and killing a container.

A wedged RDMA engine can log the same error millions of times a minute; one
such container log reached 91 GB on the shared log volume. Past the limit the
service is broken anyway, so stop writing, kill the container to end the cell,
and keep draining stdin so the writer never blocks on a full pipe.
"""

import subprocess
import sys


def main() -> int:
    path, limit, container = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    written = 0
    killed = False
    source = sys.stdin.buffer
    with open(path, "wb") as log:
        while True:
            chunk = source.read1(1 << 20)
            if not chunk:
                break
            if written < limit:
                part = chunk[: limit - written]
                log.write(part)
                written += len(part)
                if written < limit:
                    continue
            if not killed:
                killed = True
                log.write(
                    f"\n[logs] container log reached {limit} bytes; "
                    f"killing {container} and discarding further output\n".encode()
                )
                log.flush()
                print(
                    f"ERROR: {container} log exceeded {limit} bytes; killing it",
                    file=sys.stderr,
                )
                subprocess.run(
                    ["docker", "kill", container],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=60,
                    check=False,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
