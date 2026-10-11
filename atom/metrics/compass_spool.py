"""Read-side of the ATOM Compass spool, served over HTTP for remote collectors.

Ranks append JSONL to ``$ATOM_COMPASS_DIR/<rank>/`` (see ``compass.py``). A
collector pulls each file incrementally by byte offset, so every record is
fetched exactly once however often it polls; the spool itself is the buffer.
"""

import os

_SUFFIXES = (".jsonl", ".json")


def list_files(root):
    files = []
    for rank in sorted(os.listdir(root)) if os.path.isdir(root) else ():
        rank_dir = os.path.join(root, rank)
        if not os.path.isdir(rank_dir):
            continue
        for name in sorted(os.listdir(rank_dir)):
            if name.endswith(_SUFFIXES):
                path = os.path.join(rank_dir, name)
                files.append({"file": f"{rank}/{name}", "size": os.path.getsize(path)})
    return files


def resolve(root, rel):
    """Map a listed ``<rank>/<name>`` back to its path; refuse anything else."""
    parts = rel.split("/")
    if len(parts) != 2 or not parts[1].endswith(_SUFFIXES):
        raise ValueError(f"not a compass file: {rel!r}")
    if any(p in ("", ".", "..") or os.sep in p for p in parts):
        raise ValueError(f"not a compass file: {rel!r}")
    return os.path.join(root, *parts)


def read_lines(path, offset, max_bytes):
    """Return ``(data, next_offset)`` holding only whole lines from ``offset``.

    A line still being written stays for the next read. A single line longer
    than ``max_bytes`` is returned whole rather than never.
    """
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(max_bytes)
        cut = data.rfind(b"\n") + 1
        if cut == 0 and len(data) == max_bytes:
            rest = f.readline()
            if rest.endswith(b"\n"):
                data, cut = data + rest, len(data) + len(rest)
    return data[:cut], offset + cut
