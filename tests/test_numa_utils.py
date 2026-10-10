from __future__ import annotations

import re

from atom.utils.numa_utils import _partition_physical_cores


def test_partition_physical_cores_keeps_smt_siblings_together(monkeypatch):
    siblings = {
        0: "0,4",
        1: "1,5",
        2: "2,6",
        3: "3,7",
        4: "0,4",
        5: "1,5",
        6: "2,6",
        7: "3,7",
    }

    class _File:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return self.value

    def fake_open(path):
        cpu = int(re.search(r"/cpu(\d+)/", path).group(1))
        return _File(siblings[cpu])

    monkeypatch.setattr("builtins.open", fake_open)
    left = _partition_physical_cores(set(range(8)), 0, 2)
    right = _partition_physical_cores(set(range(8)), 1, 2)

    assert left == {0, 1, 4, 5}
    assert right == {2, 3, 6, 7}
    assert left.isdisjoint(right)


def test_partition_physical_cores_leaves_remainder_for_control_plane(monkeypatch):
    monkeypatch.setattr(
        "atom.utils.numa_utils._physical_core_groups",
        lambda _cpus: [{i} for i in range(11)],
    )

    partitions = [
        _partition_physical_cores(set(range(11)), rank, 4) for rank in range(4)
    ]

    assert partitions == [{0, 1}, {2, 3}, {4, 5}, {6, 7}]
    assert set().union(*partitions) == set(range(8))
