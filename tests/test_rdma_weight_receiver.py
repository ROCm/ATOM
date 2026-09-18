# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The RDMA weight receiver: wire format and transaction semantics.

The wire format is frozen by the sender
(``lumenrl.engine.inference.rdma_weight_transfer``) and is reproduced here from
its own constants rather than copied by eye, so a drift on either side shows up
as a test failure instead of a 61 GB transfer that decodes to garbage.

The transaction exists because weights are applied *in place*. A stream that
fails halfway leaves the model a mix of two versions, and inference would keep
serving -- quietly wrong. So a failure must fence serving, not just log.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.rollout.rdma_weight_receiver import (
        _CMD_BUCKET,
        _CMD_END,
        _HEADER_WORDS,
        _decode_bucket,
    )


# ── the frozen wire format ─────────────────────────────────────────────────


def test_the_command_and_header_constants_match_the_sender():
    """These three numbers are the contract. LumenRL's sender hardcodes the
    same values; a change on either side silently corrupts every transfer."""
    assert _CMD_END == 0
    assert _CMD_BUCKET == 1
    assert _HEADER_WORDS == 4  # command, metadata_bytes, payload_bytes, version


def _encode(entries_and_bytes):
    """Build a (metadata, payload) pair exactly as the sender does."""
    import torch

    entries = []
    blobs = []
    offset = 0
    for name, tensor in entries_and_bytes:
        raw = tensor.contiguous().view(torch.uint8).reshape(-1)
        entries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                # The sender strips the "torch." prefix; getattr(torch, ...) on
                # the receiving side is what has to match.
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "offset": offset,
                "nbytes": raw.numel(),
            }
        )
        blobs.append(raw)
        offset += raw.numel()

    payload = torch.cat(blobs) if blobs else torch.empty(0, dtype=torch.uint8)
    meta = torch.tensor(
        list(json.dumps(entries, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )
    return meta, payload


def test_a_bucket_round_trips_through_the_real_encoding():
    import torch

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    b = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float32)
    meta, payload = _encode([("layer.a", a), ("layer.b", b)])

    decoded = _decode_bucket(meta, payload)

    assert [name for name, _ in decoded] == ["layer.a", "layer.b"]
    got_a, got_b = decoded[0][1], decoded[1][1]
    assert got_a.dtype == torch.bfloat16 and tuple(got_a.shape) == (2, 2)
    assert got_b.dtype == torch.float32 and tuple(got_b.shape) == (3,)
    assert torch.equal(got_a, a)
    assert torch.equal(got_b, b)


def test_the_decoded_tensors_are_views_not_copies():
    """A 61 GB transfer cannot afford to double its peak footprint."""
    import torch

    a = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    meta, payload = _encode([("w", a)])
    ((_, view),) = _decode_bucket(meta, payload)
    assert view.data_ptr() == payload.data_ptr()


def test_an_out_of_bounds_offset_is_rejected_not_truncated():
    """Silently clamping would hand a neighbouring tensor's bytes to the model."""
    import torch

    meta, payload = _encode([("w", torch.tensor([1.0], dtype=torch.float32))])
    entries = json.loads(bytes(meta.tolist()).decode())
    entries[0]["nbytes"] = 4096  # past the end
    bad = torch.tensor(list(json.dumps(entries).encode()), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="out of bounds"):
        _decode_bucket(bad, payload)


@pytest.mark.parametrize(
    ("mutate", "expect"),
    [
        # torch's own view() rejects this before the explicit size check does,
        # which is fine -- what matters is that it is refused, not silently
        # reshaped into whatever fits.
        (lambda e: e.update({"shape": [99]}), "invalid for input of size"),
        (lambda e: e.pop("name"), "invalid RDMA weight metadata entry"),
        (
            lambda e: e.update({"dtype": "not_a_dtype"}),
            "invalid RDMA weight metadata entry",
        ),
        (lambda e: e.update({"nbytes": 0}), "out of bounds"),
        (lambda e: e.update({"offset": -1}), "out of bounds"),
    ],
)
def test_corrupt_metadata_is_rejected(mutate, expect):
    import torch

    meta, payload = _encode([("w", torch.tensor([1.0, 2.0], dtype=torch.float32))])
    entries = json.loads(bytes(meta.tolist()).decode())
    mutate(entries[0])
    bad = torch.tensor(list(json.dumps(entries).encode()), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match=expect):
        _decode_bucket(bad, payload)


def test_an_empty_metadata_list_is_rejected():
    import torch

    bad = torch.tensor(list(b"[]"), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="non-empty list"):
        _decode_bucket(bad, torch.empty(0, dtype=torch.uint8))


def test_the_sender_and_receiver_agree_on_the_constants():
    """Read LumenRL's sender from source. It is a separate repo, so an import
    would couple the two -- but the numbers still have to match."""
    import re

    sender = Path(
        "/home/cchen104/openxla/Lumen-RL/lumenrl/engine/inference/rdma_weight_transfer.py"
    )
    if not sender.exists():
        pytest.skip("LumenRL checkout not present")
    src = sender.read_text()
    for const, value in (
        ("_CMD_END", _CMD_END),
        ("_CMD_BUCKET", _CMD_BUCKET),
        ("_HEADER_WORDS", _HEADER_WORDS),
    ):
        m = re.search(rf"^{const}\s*=\s*(\d+)", src, re.MULTILINE)
        assert m, f"{const} not found in the sender"
        assert (
            int(m.group(1)) == value
        ), f"{const}: sender says {m.group(1)}, receiver says {value}"
