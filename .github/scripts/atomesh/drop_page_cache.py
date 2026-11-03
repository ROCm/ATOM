#!/usr/bin/env python3
"""Evict the clean page cache of every file under the given directories.

posix_fadvise(DONTNEED) needs no privilege and only drops clean pages. Weight
loads fill the page cache of the NUMA node the loader runs on; a process bound
there with MPOL_BIND then reclaims that cache on every allocation instead of
spilling to the other node.

Usage: drop_page_cache.py [--every SECONDS --for SECONDS] <dir> [<dir> ...]
"""

from __future__ import annotations

import argparse
import os
import stat
import time


def drop_page_cache(root: str) -> None:
    files_advised = bytes_advised = errors = 0
    for directory, _, files in os.walk(root):
        for name in files:
            try:
                fd = os.open(
                    os.path.join(directory, name), os.O_RDONLY | os.O_NONBLOCK
                )
            except OSError:
                errors += 1
                continue
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    continue
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                files_advised += 1
                bytes_advised += info.st_size
            except OSError:
                errors += 1
            finally:
                os.close(fd)
    tag = (
        "[page-cache]" if files_advised and not errors else "[page-cache][WARN]"
    )
    print(
        f"{tag} root={root} files_advised={files_advised} "
        f"file_bytes_advised={bytes_advised} errors={errors}; "
        "advice covers clean file pages only; bytes reclaimed are not measured",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="+")
    parser.add_argument(
        "--every", type=float, default=0.0, help="repeat interval in seconds"
    )
    parser.add_argument(
        "--for",
        dest="duration",
        type=float,
        default=0.0,
        help="keep repeating for this many seconds",
    )
    args = parser.parse_args()
    deadline = time.monotonic() + args.duration
    while True:
        for root in args.dirs:
            drop_page_cache(root)
        if args.every <= 0 or time.monotonic() + args.every > deadline:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
