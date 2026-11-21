"""The RDMA sender against ATOM's receiver of its stream.

ATOM's atom/rollout/rdma_weight_receiver.py decodes what
atomrl/engine/inference/rdma_weight_transfer.py sends. Both are read as text,
since importing ATOM pulls in GPU-only code. The receiver is the ATOM in this
repository unless ATOM_ROOT names another checkout; until the receiver lands
here (ROCm/ATOM#3058) the in-tree check skips.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_ATOM_REPO = _REPO.parent
_SENDER = _REPO / "atomrl/engine/inference/rdma_weight_transfer.py"
_RECEIVER = Path("atom/rollout/rdma_weight_receiver.py")
_PROTOCOL = Path("atom/rollout/rdma_protocol.py")
# decode_bucket moved from weight_updater.py to weight_bucket.py in ROCm/ATOM#3058.
_DECODERS = (
    Path("atom/rollout/weight_bucket.py"),
    Path("atom/rollout/weight_updater.py"),
)
_HEADER = ("command", "metadata_bytes", "payload_bytes", "version")


def _atom(relative: Path) -> str:
    root = os.environ.get("ATOM_ROOT")
    if not root:
        if not (_ATOM_REPO / _RECEIVER).is_file():
            pytest.skip(f"this repository's ATOM has no {_RECEIVER} yet")
        return (_ATOM_REPO / relative).read_text()
    path = Path(root, relative)
    # Named but missing fails rather than skips.
    assert path.is_file(), f"ATOM_ROOT={root} has no {relative}"
    return path.read_text()


def _decoder() -> str:
    root = Path(os.environ.get("ATOM_ROOT") or _ATOM_REPO)
    for relative in _DECODERS:
        path = root / relative
        if path.is_file() and "def decode_bucket(" in path.read_text():
            return path.read_text()
    raise AssertionError(f"no decode_bucket() in {[str(d) for d in _DECODERS]}")


def _constant(src: str, name: str) -> int | None:
    m = re.search(rf"^{name}\s*=\s*(\d+)", src, re.MULTILINE)
    return int(m[1]) if m else None


def _function(src: str, name: str) -> str:
    # Up to the next top-level statement; a signature's closing ")" is not one.
    m = re.search(
        rf"^def {name}\(.*?(?=^(?:def |class |@|[A-Za-z_])|\Z)",
        src,
        re.MULTILINE | re.DOTALL,
    )
    assert m, f"no top-level {name}()"
    return m[0]


def test_the_commands_and_header_width_agree():
    sender, receiver = _SENDER.read_text(), _atom(_RECEIVER)
    protocol = _atom(_PROTOCOL)
    assert "from atom.rollout.rdma_protocol import" in sender
    assert "from atom.rollout.rdma_protocol import" in receiver
    assert _constant(protocol, "CMD_END") == 0
    assert _constant(protocol, "CMD_BUCKET") == 1
    assert _constant(protocol, "HEADER_WORDS") == 4


def test_the_header_is_read_in_the_order_it_is_sent():
    """Swapped sizes make the two sides post broadcasts of different lengths,
    which RCCL does not check."""
    packs = _function(_SENDER.read_text(), "_broadcast_header")
    unpacks = _function(_atom(_RECEIVER), "_recv_header")
    sent = re.search(r"\[\s*(\w+),\s*(\w+),\s*(\w+),\s*(\w+)\s*,?\s*\]", packs)
    read = re.search(r"(\w+),\s*(\w+),\s*(\w+),\s*(\w+)\s*=\s*\(", unpacks)
    assert sent and sent.groups() == _HEADER
    assert read and read.groups() == _HEADER


def test_each_tensor_is_described_by_what_the_decoder_reads():
    entry = re.search(
        r"entries\.append\(\s*\{(.*?)\}\s*\)", _SENDER.read_text(), re.DOTALL
    )
    assert entry, "the sender builds no entries"
    _atom(_RECEIVER)
    decoder = _function(_decoder(), "decode_bucket")
    sent = set(re.findall(r'"(\w+)":', entry[1]))
    read = set(re.findall(r'fields\.get\("(\w+)"\)', decoder))
    assert sent == read == {"dtype", "name", "nbytes", "offset", "shape"}
