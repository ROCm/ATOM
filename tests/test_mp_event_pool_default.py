"""The interprocess step-event pool is opt-in, and the reason is measured.

Re-recording one `hipEvent_t` while another process still waits on it raises
`HIP error: invalid argument`. The engine_driven path was already bypassed
for that; the lmcache_driven path was assumed immune because "the server
waits on its own event, imported from the handle". It does not --
`lmcache_driven_transfer.retrieve` waits on the producer event -- and on
DeepSeek-V4.1 TP4 that assumption cost 2 of 4 rounds of the long-prompt
accuracy harness, one of them fatally, against 6 of 6 clean unpooled.

The refcount cannot close it: it counts the transfers in THIS process that
named the event, and a server holding an imported handle is a consumer it
cannot see.
"""

import importlib
import os

import pytest


def _reload(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ATOM_MP_EVENT_POOL", raising=False)
    else:
        monkeypatch.setenv("ATOM_MP_EVENT_POOL", value)
    import atom.kv_transfer.offload.mp.worker as w

    return importlib.reload(w)


@pytest.fixture(autouse=True)
def _restore():
    before = os.environ.get("ATOM_MP_EVENT_POOL")
    yield
    import atom.kv_transfer.offload.mp.worker as w

    if before is None:
        os.environ.pop("ATOM_MP_EVENT_POOL", None)
    else:
        os.environ["ATOM_MP_EVENT_POOL"] = before
    importlib.reload(w)


def test_the_pool_is_off_when_nobody_asks(monkeypatch):
    """Default-off is the contract. An unset variable must not enable it."""
    assert _reload(monkeypatch, None)._EVENT_POOL_ON is False


def test_only_an_explicit_one_turns_it_on(monkeypatch):
    assert _reload(monkeypatch, "1")._EVENT_POOL_ON is True


@pytest.mark.parametrize("value", ["0", "", "true", "yes", "on", "2"])
def test_nothing_else_turns_it_on(monkeypatch, value):
    """It used to be "anything but 0", so `ATOM_MP_EVENT_POOL=false` enabled
    the thing it was written to disable."""
    assert _reload(monkeypatch, value)._EVENT_POOL_ON is False
