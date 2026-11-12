"""The CPU half of the Engram GTT path (`host_memory._gtt_rows`).

The GEM allocation and the HIP import need a GPU; what is tested here is the
part that decides who copies and how the others get the copy: the share name a
NUMA node's ranks agree on, the dma-buf hand-off over the abstract socket, and
`device_address` on the device rows handed out.
"""

import mmap
import os
import socket
import threading

import numpy as np
import pytest
import torch

from atom.model_ops.engram import host_memory


@pytest.fixture
def table_file(tmp_path):
    path = tmp_path / "table.bin"
    path.write_bytes(np.arange(1 << 16, dtype=np.uint8).tobytes())
    return path


def _mapped(path, offset, nbytes):
    """`nbytes` of `path` from `offset`, as a fresh read-only mapping."""
    raw = torch.from_numpy(np.memmap(path, dtype=np.uint8, mode="r"))
    return raw[offset : offset + nbytes]


def test_share_name_is_the_checkpoint_range(table_file):
    # Two mappings of the same bytes -- two processes' views -- agree.
    a = _mapped(table_file, 4096, 8192)
    b = _mapped(table_file, 4096, 8192)
    assert a.data_ptr() != b.data_ptr()
    assert host_memory._share_name(a, 1) == host_memory._share_name(b, 1)
    # Another node, offset or size is another copy.
    assert host_memory._share_name(a, 0) != host_memory._share_name(a, 1)
    assert host_memory._share_name(a, 1) != host_memory._share_name(
        _mapped(table_file, 8192, 8192), 1
    )
    assert host_memory._share_name(a, 1) != host_memory._share_name(
        _mapped(table_file, 4096, 4096), 1
    )
    assert host_memory._share_name(a, 1).startswith("\0")


def test_share_name_needs_a_file_mapping():
    assert host_memory._share_name(torch.zeros(4096, dtype=torch.uint8), 0) is None


def test_file_offset(table_file):
    t = _mapped(table_file, 12345, 100)
    path, offset = host_memory._file_offset(t.data_ptr())
    assert os.path.samefile(path, table_file)
    assert offset == 12345


def _unique_name():
    return f"\0atom-engram-gtt-test-{os.getpid()}-{threading.get_ident()}"


def test_receive_without_a_leader():
    assert host_memory._receive_dmabuf(_unique_name()) is None


def test_receive_while_the_leader_copies():
    # Bound, not yet listening: the followers wait.
    name = _unique_name()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind(name)
        assert host_memory._receive_dmabuf(name) is None


def test_server_hands_out_the_descriptor(tmp_path):
    name = _unique_name()
    payload = tmp_path / "payload"
    payload.write_bytes(b"engram" * 1000)
    fd = os.open(payload, os.O_RDONLY)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(name)
    server = host_memory._GttServer(sock, fd, 6000)
    os.close(fd)  # the server keeps its own
    try:
        for _ in range(3):  # every rank of the node, one after another
            got = host_memory._receive_dmabuf(name)
            assert got is not None
            rfd, size = got
            try:
                assert size == 6000
                with mmap.mmap(rfd, 6000, prot=mmap.PROT_READ) as m:
                    assert m[:6] == b"engram"
            finally:
                os.close(rfd)
    finally:
        server.close()
    assert host_memory._receive_dmabuf(name) is None


def test_device_address_of_device_rows():
    base, size = 0x7F0000000000, 1 << 20
    with host_memory._DEVICE_SPANS_LOCK:
        host_memory._DEVICE_SPANS[base] = base + size
    try:
        assert host_memory.device_address(base + 123, base) == base + 123
    finally:
        with host_memory._DEVICE_SPANS_LOCK:
            host_memory._DEVICE_SPANS.pop(base)


def test_no_gpu_no_gtt_rows(table_file):
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        pytest.skip("a GPU process takes the GTT path")
    leases = []
    src = _mapped(table_file, 0, 8192)
    assert host_memory._gtt_rows([src], leases, shared=True) is None
    assert leases == []
