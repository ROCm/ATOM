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
import json
import os
import time


def drop_page_cache(root: str, *, stats: bool = False) -> dict:
    """Report advice outcomes, not an estimate of bytes actually reclaimed."""
    began = time.monotonic()
    result = {
        "root": root,
        "files_seen": 0,
        "files_opened": 0,
        "files_advised": 0,
        "advised_file_bytes": 0,
        "open_errors": 0,
        "advice_errors": 0,
        "stat_errors": 0,
        "walk_errors": 0,
        "skipped_symlink_directories": 0,
        "errors_by_errno": {},
        "error_examples": [],
    }

    def report(event):
        print(
            "[file-cache] "
            + json.dumps(
                {
                    **result,
                    "event": event,
                    "seconds": round(time.monotonic() - began, 3),
                }
            ),
            flush=True,
        )

    def error(operation, path, exc):
        result[f"{operation}_errors"] += 1
        key = str(exc.errno)
        result["errors_by_errno"][key] = result["errors_by_errno"].get(key, 0) + 1
        if len(result["error_examples"]) < 10:
            result["error_examples"].append(
                {"operation": operation, "path": str(path), "error": str(exc)}
            )

    if stats:
        report("start")
    last_report = began
    for directory, directories, files in os.walk(
        root, onerror=lambda exc: error("walk", exc.filename, exc)
    ):
        if stats:
            result["skipped_symlink_directories"] += sum(
                os.path.islink(os.path.join(directory, name)) for name in directories
            )
        for name in files:
            path = os.path.join(directory, name)
            result["files_seen"] += 1
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError as exc:
                error("open", path, exc)
            else:
                result["files_opened"] += 1
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    result["files_advised"] += 1
                    if stats:
                        try:
                            result["advised_file_bytes"] += os.fstat(fd).st_size
                        except OSError as exc:
                            error("stat", path, exc)
                except OSError as exc:
                    error("advice", path, exc)
                finally:
                    os.close(fd)
            if stats and time.monotonic() - last_report >= 10:
                report("progress")
                last_report = time.monotonic()
    if stats:
        report("complete")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="+")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="log per-directory advice counts and errors",
    )
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
            drop_page_cache(root, stats=args.stats)
        if args.every <= 0 or time.monotonic() + args.every > deadline:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
