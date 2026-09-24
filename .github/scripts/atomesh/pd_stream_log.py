# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Print a bounded increment of a shared log to stderr and its offset to stdout."""

import os
import sys

# Limit each file per poll, including progress bars/JSON without newlines.
MAX_BYTES = 64 * 1024


def stream_log(path, offset, prefix):
    try:
        with open(path, "rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size < offset:
                offset = 0
            start = max(offset, size - MAX_BYTES)
            stream.seek(start)
            data = stream.read(size - start)
    except OSError as exc:
        # A transient NFS read failure must not cancel a healthy benchmark.
        print(f"{prefix}Cannot read {path}: {exc}; will retry", file=sys.stderr)
        return offset

    if start > offset:
        print(
            f"{prefix}Skipped {start - offset} bytes of CI output; "
            f"full log: {path}",
            file=sys.stderr,
        )
    if data:
        # One prefix per chunk keeps output bounded even with many short lines.
        print(prefix + data.decode("utf-8", errors="replace"), end="", file=sys.stderr)
        if not data.endswith(b"\n"):
            print(file=sys.stderr)
    return start + len(data)


if __name__ == "__main__":
    print(stream_log(sys.argv[1], int(sys.argv[2]), sys.argv[3]))
