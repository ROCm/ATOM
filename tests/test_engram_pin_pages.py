# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""engine.engram_pin_pages: the page-locked copy of an Engram shard is also
pinned against page migration (`host_memory._pin_pages`), and unpinned when it
is freed. Off by default: the allocator then does what it did."""

import ctypes
import mmap
import os
import types

import pytest

from atom import plugin_host
from atom.model_ops.engram import host_memory
from atom.plugin_recipe import EngineOptions


class _Hip:
    """`hipHostMalloc` / `hipHostFree` over one anonymous mapping."""

    def __init__(self, nbytes):
        self.buffer = mmap.mmap(-1, nbytes)
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.buffer))
        self.freed = []
        self.hipHostMalloc = types.SimpleNamespace(__call__=None)
        self.hipHostFree = types.SimpleNamespace(__call__=None)

    def allocator(self):
        def malloc(ref, nbytes, flags):
            ref._obj.value = self.address
            return 0

        def free(ptr):
            self.freed.append(ptr)
            return 0

        libc = types.SimpleNamespace(prctl=lambda *args: 0, madvise=lambda *args: 0)
        allocator = host_memory._HipHostAllocator(self, libc)
        allocator._malloc, allocator._free = malloc, free
        return allocator


@pytest.fixture
def plan(monkeypatch):
    def set_plan(**engine):
        monkeypatch.setattr(
            plugin_host,
            "_plan",
            types.SimpleNamespace(engine=EngineOptions(**engine)),
        )

    monkeypatch.setattr(plugin_host, "_plan", None)
    return set_plan


def test_off_by_default_nothing_is_pinned(plan, monkeypatch):
    assert not EngineOptions().engram_pin_pages
    assert not host_memory._pin_pages_wanted()
    monkeypatch.setattr(
        host_memory, "_pin_pages", lambda *a: pytest.fail("pinned with the option off")
    )
    hip = _Hip(2 * host_memory._PAGE)
    allocator = hip.allocator()
    ptr = allocator.alloc(2 * host_memory._PAGE)
    assert ptr == hip.address and not allocator._pins
    allocator.free(ptr)
    assert hip.freed == [ptr]


def test_on_pins_the_block_and_free_unpins_it(plan, monkeypatch):
    plan(engram_pin_pages=True)
    assert host_memory._pin_pages_wanted()
    read_end, write_end = os.pipe()
    pinned = []

    def pin(ptr, nbytes):
        pinned.append((ptr, nbytes))
        return write_end

    monkeypatch.setattr(host_memory, "_pin_pages", pin)
    hip = _Hip(2 * host_memory._PAGE)
    allocator = hip.allocator()
    ptr = allocator.alloc(2 * host_memory._PAGE)
    assert pinned == [(hip.address, 2 * host_memory._PAGE)]
    assert allocator._pins == {ptr: write_end}
    allocator.free(ptr)
    assert not allocator._pins and hip.freed == [ptr]
    # the descriptor that held the pin is closed: the pipe reads end-of-file
    assert os.read(read_end, 1) == b""
    os.close(read_end)


def test_a_refused_pin_leaves_the_copy_as_it_was(plan, monkeypatch):
    plan(engram_pin_pages=True)
    monkeypatch.setattr(host_memory, "_pin_pages", lambda ptr, nbytes: None)
    hip = _Hip(host_memory._PAGE)
    allocator = hip.allocator()
    ptr = allocator.alloc(host_memory._PAGE)
    assert ptr == hip.address and not allocator._pins
    allocator.free(ptr)


def test_pin_pages_pins_or_declines_without_raising():
    """The real call on two anonymous pages: a descriptor that holds the pin,
    or None where the kernel or the container refuses io_uring."""
    buffer = mmap.mmap(-1, 2 * host_memory._PAGE)
    address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    fd = host_memory._pin_pages(address, 2 * host_memory._PAGE)
    if fd is not None:
        os.close(fd)
