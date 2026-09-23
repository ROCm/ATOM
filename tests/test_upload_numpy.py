# SPDX-License-Identifier: MIT

"""`upload_numpy` picks a transfer route by size, and the size that matters.

The helper exists because a pageable host-to-device copy larger than the
driver's staging buffer stops being a hand-off: the host waits for it, on the
current stream, behind everything already queued. Under that size it costs the
same as anything else and allocates nothing.

Two things are worth holding here. The obvious one is that the bytes arrive
whichever branch runs. The other is that the threshold is not decoration -- a
test that only checked correctness would pass just as well against a helper
that always took one branch, so the cliff itself is measured.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from atom.utils import _PAGEABLE_H2D_LIMIT, upload_numpy

if not torch.cuda.is_available():
    pytest.skip("host-to-device transfer needs a real GPU", allow_module_level=True)

DEV = torch.device("cuda:0")


def _arr(nbytes, dtype=np.int32):
    n = nbytes // np.dtype(dtype).itemsize
    return np.arange(n, dtype=dtype) % 30011


@pytest.mark.parametrize(
    "nbytes",
    [4, 4096, _PAGEABLE_H2D_LIMIT - 4, _PAGEABLE_H2D_LIMIT, _PAGEABLE_H2D_LIMIT * 3],
    ids=["tiny", "4k", "just-under", "at-limit", "over"],
)
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32])
def test_the_bytes_arrive_on_either_side_of_the_cliff(nbytes, dtype):
    arr = _arr(nbytes, dtype)
    got = upload_numpy(arr, DEV)
    torch.cuda.synchronize()
    assert got.device.type == "cuda"
    assert np.array_equal(got.cpu().numpy(), arr)


def test_a_non_contiguous_source_still_arrives():
    """`from_numpy` rejects a negative stride and copies a gapped one; either
    way the caller gets the values it passed."""
    arr = np.arange(4096, dtype=np.int32).reshape(64, 64)[:, ::2].copy()
    got = upload_numpy(arr, DEV)
    torch.cuda.synchronize()
    assert np.array_equal(got.cpu().numpy(), arr)


_FILLER = None


def _queue_filler():
    global _FILLER
    if _FILLER is None:
        _FILLER = torch.randn(4096, 4096, dtype=torch.bfloat16, device=DEV)
    return _FILLER


def _cost_behind_queued_work(upload, depth=6, nreps=40):
    """Host ms charged to `upload` with `depth` matmuls already in flight.

    The queue is what makes the difference visible: a synchronizing copy waits
    out everything on the stream, not just its own bytes -- so its cost grows
    with `depth` and a real async one's does not.
    """
    import timeit

    # Built once for the module, not per call: a fresh 32 MB tensor each time
    # churns the caching allocator, and that noise lands on whichever route is
    # measured next -- it read as a 10x worse slope for the helper.
    a = _queue_filler()

    def enqueue():
        for _ in range(depth):
            a @ a

    def both():
        enqueue()
        upload()

    for fn in (enqueue, both):
        fn()
    torch.cuda.synchronize()
    base = sorted(timeit.repeat(enqueue, number=1, repeat=nreps))[nreps // 2]
    torch.cuda.synchronize()
    got = sorted(timeit.repeat(both, number=1, repeat=nreps))[nreps // 2]
    torch.cuda.synchronize()
    return (got - base) * 1e3


def test_the_helper_is_what_keeps_a_large_upload_off_the_cliff():
    """Positive control for the threshold.

    Past the limit a plain `.to()` blocks until the queue drains and the
    helper's route does not. Without this the parametrisation above would pass
    against a constant of any value, including one that never takes the second
    branch.
    """
    big = _arr(_PAGEABLE_H2D_LIMIT * 3)

    def raw():
        torch.from_numpy(big).to(DEV, non_blocking=True)

    def helper():
        upload_numpy(big, DEV)

    # Slope against queue depth, not cost at one depth: a copy that
    # synchronizes inherits the queue, so its cost has a slope and an async
    # one's does not. A ratio at a single depth would be another unexplained
    # threshold sitting on top of the one under test.
    lo, hi = 4, 24
    raw_slope = (
        _cost_behind_queued_work(raw, depth=hi) - _cost_behind_queued_work(raw, lo)
    ) / (hi - lo)
    helper_slope = (
        _cost_behind_queued_work(helper, depth=hi)
        - _cost_behind_queued_work(helper, lo)
    ) / (hi - lo)
    print(
        f"\n{big.nbytes:,} B  host us per queued matmul: "
        f"pageable {raw_slope * 1000:+.1f}, helper {helper_slope * 1000:+.1f}"
    )
    assert raw_slope > 0.01, (
        "pageable cost did not grow with queue depth, so nothing here is "
        f"synchronizing and the threshold is untested: {raw_slope * 1000:+.1f} "
        "us per queued matmul"
    )
    # A loose gate on purpose. Over the cliff the helper can only reach for a
    # fresh pinned block, which measured about 2.4x better than pageable rather
    # than free -- a shared reusable block does no better, for the reason in
    # `upload_numpy`'s docstring. Asserting more would be asserting a number
    # this route cannot deliver.
    assert helper_slope < raw_slope / 1.5, (
        "the helper inherits the queue nearly as much as a plain `.to()`: "
        f"{helper_slope * 1000:+.1f} vs {raw_slope * 1000:+.1f} us each"
    )
