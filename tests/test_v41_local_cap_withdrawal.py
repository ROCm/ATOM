"""The connector half of the V4.1 local prefix-cache cap.

`cap_local_hit` shortens vLLM's own hit by one block on the promise that the
tier will supply that block and the CSA2 state behind it. Only the connector
learns whether the tier actually did, and only before allocation -- vLLM
forbids parking a request with no external tokens
(`assert num_external_computed_tokens > 0`), so a hit that keeps its shortened
local prefix with nothing loaded behind it reaches the forward and
`_report_stale_state` raises.

So an unbacked cap has to be withdrawn in `get_num_new_matched_tokens`, which
returns None to defer the request a step. These tests pin that: the None, the
withdrawal, and the fact that a backed cap is left alone.
"""

from types import SimpleNamespace

from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector


class _Planner:
    def __init__(self):
        self.withdrawn = []
        self.confirmed = []
        self.has_cap = True

    def begin_lookup(self, request):
        pass

    def end_lookup(self):
        pass

    def withdraw_local_cap(self, request):
        if not self.has_cap:
            return False
        self.withdrawn.append(request.request_id)
        return True

    def confirm_local_cap(self, request):
        self.confirmed.append(request.request_id)


class _Seqs:
    def get_or_create(self, request):
        return SimpleNamespace(
            id=request.request_id, set_num_cached_tokens=lambda n: None
        )


def _connector(planner, *, need, park):
    stub = SimpleNamespace(
        _seqs=_Seqs(),
        _requests={},
        _kda_planner=None,
        _v41_planner=planner,
        _promised_loads={},
        _scheduler=SimpleNamespace(
            get_num_new_matched_tokens=lambda seq: (need, False),
            should_park_for_load_after_alloc=lambda seq: park,
        ),
    )
    return AtomLMCacheOffloadConnector.get_num_new_matched_tokens.__get__(stub), stub


def test_an_unbacked_cap_is_withdrawn_and_the_request_deferred():
    planner = _Planner()
    call, _ = _connector(planner, need=0, park=False)
    assert call(SimpleNamespace(request_id="r0"), 512) == (None, False)
    assert planner.withdrawn == ["r0"]


def test_a_cap_unbacked_because_the_tier_declined_to_park_is_also_withdrawn():
    """A hit the tier holds but will not park on is just as unbacked.

    `should_park_for_load_after_alloc` declines below the save floor and off a
    chunk boundary. Either way no load is issued, so the shortened local hit
    has nothing behind it.
    """
    planner = _Planner()
    call, _ = _connector(planner, need=256, park=False)
    assert call(SimpleNamespace(request_id="r1"), 512) == (None, False)
    assert planner.withdrawn == ["r1"]


def test_a_backed_cap_is_confirmed_and_the_request_parks():
    planner = _Planner()
    call, stub = _connector(planner, need=256, park=True)
    assert call(SimpleNamespace(request_id="r2"), 512) == (256, True)
    assert planner.confirmed == ["r2"] and planner.withdrawn == []
    # Unparked until vLLM allocates: `update_state_after_alloc` confirms it.
    assert stub._promised_loads["r2"] == [0, 512, 256, False]


def test_a_request_that_never_capped_is_answered_zero_not_deferred():
    """Most lookups never cap. Deferring them all would be a livelock."""
    planner = _Planner()
    planner.has_cap = False
    call, _ = _connector(planner, need=0, park=False)
    assert call(SimpleNamespace(request_id="r3"), 0) == (0, False)
    assert planner.withdrawn == []


def test_a_model_with_no_v41_planner_is_untouched():
    """GLM, M3 and K3 share this method and must not learn a new return type."""
    stub = SimpleNamespace(
        _seqs=_Seqs(),
        _requests={},
        _kda_planner=None,
        _v41_planner=None,
        _promised_loads={},
        _scheduler=SimpleNamespace(
            get_num_new_matched_tokens=lambda seq: (0, False),
            should_park_for_load_after_alloc=lambda seq: False,
        ),
    )
    call = AtomLMCacheOffloadConnector.get_num_new_matched_tokens.__get__(stub)
    assert call(SimpleNamespace(request_id="r4"), 0) == (0, False)
