# SPDX-License-Identifier: MIT
"""A tensor-parallel group faked with one thread per rank on one device.

`run_ranks(world, fn)` calls `fn(rank, group)` on `world` threads; the group's
`all_gather` concatenates every rank's tensor along dim 0. Works on a GPU and
under the Triton interpreter (TRITON_INTERPRET=1) on the CPU.
"""

import threading

import torch


class FakeGroup:
    """All-gather across rank threads on one device; launches are serialized
    by a global lock that all_gather releases while it waits."""

    def __init__(self, world):
        self.world_size = world
        self.bar = threading.Barrier(world)
        self.slots = [None] * world
        self.lock = threading.Lock()

    def view(self, rank):
        g = self

        class _R:
            world_size = g.world_size
            rank_in_group = rank

            def all_gather(self, t, dim=0, use_custom=True):
                g.slots[rank] = t
                g.lock.release()
                g.bar.wait()
                g.lock.acquire()
                out = torch.cat(list(g.slots), dim=0)
                g.lock.release()
                g.bar.wait()
                g.lock.acquire()
                return out

        return _R()


def run_ranks(world, fn):
    g = FakeGroup(world)
    outs = [None] * world
    errs = []

    def body(rk):
        g.lock.acquire()
        try:
            outs[rk] = fn(rk, g.view(rk))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:  # noqa
            errs.append(e)
            g.bar.abort()
        finally:
            g.lock.release()

    th = [threading.Thread(target=body, args=(rk,)) for rk in range(world)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errs:
        raise errs[0]
    return outs
