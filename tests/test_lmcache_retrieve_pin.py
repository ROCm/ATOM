# SPDX-License-Identifier: MIT
"""A retrieved chunk's lookup pin is released once, not twice.

LMCache 0.5.5rc3 ``retrieve`` unpins every object it copied, and ATOM then
calls ``lookup_unpin``, which unpins them again. The second decrement is the
``Double unpin`` warning, and it hands the buffer back to the allocator.
"""

from atom.kv_transfer.offload._offload_common import (
    _RETRIEVE_HOLD_PIN,
    _unpin_unless_retrieve_holds_it,
)


class _Buf:
    def __init__(self) -> None:
        self.pin_count = 1
        self.freed = False

    def unpin(self) -> bool:
        self.pin_count -= 1
        if self.pin_count <= 0:
            self.freed = True
        self.pin_count = max(self.pin_count, 0)
        return True


def test_unpin_during_retrieve_is_deferred_to_lookup_unpin():
    buf = _Buf()
    unpin = _unpin_unless_retrieve_holds_it(_Buf.unpin)

    token = _RETRIEVE_HOLD_PIN.set(True)
    try:
        unpin(buf)  # retrieve's unpin
    finally:
        _RETRIEVE_HOLD_PIN.reset(token)

    assert buf.pin_count == 1
    assert buf.freed is False

    unpin(buf)  # lookup_unpin, after the copy has synchronized

    assert buf.pin_count == 0
    assert buf.freed is True


def test_unpin_outside_retrieve_still_releases():
    buf = _Buf()
    unpin = _unpin_unless_retrieve_holds_it(_Buf.unpin)

    unpin(buf)

    assert buf.pin_count == 0
    assert buf.freed is True
