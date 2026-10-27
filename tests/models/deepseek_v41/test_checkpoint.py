# SPDX-License-Identifier: MIT
"""Optional local-checkpoint readiness check; never reads entire weight payloads."""

import json
import os
import struct
from pathlib import Path

import pytest

from .reference import MANIFEST, check_reference_sources


def test_local_checkpoint_headers_and_download_revision():
    model_dir = os.environ.get("ATOM_DSV41_REFERENCE")
    if not model_dir:
        pytest.skip("Set ATOM_DSV41_REFERENCE to the pinned HF snapshot")
    root = Path(model_dir)
    check_reference_sources(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    assert len(shard_names) == 48
    found = {}
    payload_bytes = 0
    for shard in shard_names:
        path = root / shard
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            assert 0 < header_size < 64 * 1024**2
            header = json.loads(stream.read(header_size))
            entries = {k: v for k, v in header.items() if k != "__metadata__"}
            end = max(v["data_offsets"][1] for v in entries.values())
            assert path.stat().st_size == 8 + header_size + end, shard
            # A final-page read additionally checks that the mounted payload is readable.
            stream.seek(max(8 + header_size, path.stat().st_size - 4096))
            assert stream.read()
        intervals = sorted(v["data_offsets"] for v in entries.values())
        previous = 0
        for start, stop in intervals:
            assert start == previous and stop >= start, shard
            previous = stop
        for key in entries:
            assert key not in found, key
            found[key] = shard
        payload_bytes += end
        metadata = root / ".cache/huggingface/download" / f"{shard}.metadata"
        if metadata.exists():
            assert metadata.read_text().splitlines()[0] == MANIFEST["revision"], shard
    assert found == weight_map
    assert len(found) == 96085
    assert payload_bytes == index["metadata"]["total_size"] == 510286023000
