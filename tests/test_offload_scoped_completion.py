from types import SimpleNamespace

from atom.kv_transfer.offload import atom_lmcache_staging as staging


def test_completion_waits_recorded_point_not_whole_stream(monkeypatch):
    calls = []
    stream = SimpleNamespace(synchronize=lambda: calls.append("whole_stream"))

    class Event:
        def __init__(self, **kwargs):
            calls.append(("new", kwargs))

        def record(self, target):
            assert target is stream
            calls.append("record")

        def synchronize(self):
            calls.append("event_wait")

    monkeypatch.setattr(staging.torch.cuda, "Event", Event)
    staging.fence_submitted_work(stream)
    staging.fence_submitted_work(stream)
    assert calls == [("new", {"blocking": False}), "record", "event_wait"] * 2
