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

Asked of the predicate rather than by reloading the module. The reload left
a second copy of `LMCacheMPConnector` alive, since `native_state_worker`
had already imported the first, and that makes `is` and `==` disagree for
every test that runs afterwards.
"""

import pytest

from atom.kv_transfer.offload.mp.worker import _EVENT_POOL_ON, event_pool_enabled


def test_the_pool_is_off_when_nobody_asks():
    """Default-off is the contract. An unset variable must not enable it."""
    assert event_pool_enabled({}) is False


def test_only_an_explicit_one_turns_it_on():
    assert event_pool_enabled({"ATOM_MP_EVENT_POOL": "1"}) is True


@pytest.mark.parametrize("value", ["0", "", "true", "yes", "on", "2", "false"])
def test_nothing_else_turns_it_on(value):
    """It used to be "anything but 0", so `ATOM_MP_EVENT_POOL=false` enabled
    the thing it was written to disable."""
    assert event_pool_enabled({"ATOM_MP_EVENT_POOL": value}) is False


def test_the_module_constant_comes_from_this_predicate():
    """Otherwise the predicate could be right while the module is not."""
    import os

    assert _EVENT_POOL_ON is event_pool_enabled(os.environ)
