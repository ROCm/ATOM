"""C1 q4: 160 HC producers in256 CTAs; paired four-wave G1 tiles in each eight-wave CTA."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    rows: int = 4
    grid: int = 256
    threads: int = 512
    hidden: int = 2560
    inter: int = 640
    topk: int = 11
    shared_slot: int = 0
    shared_first: int = 240
    shared_ctas: int = 5
    hc_ready_offset: int = 16448
    tasks_ready_offset: int = 17728
    flags_bytes: int = 17736

    def check(self):
        # Each expert's ten BN128 tiles form five complete CTA pairs. All
        # runtime branches containing barriers are uniform across a CTA.
        for routes in range(1, 41):
            pairs = [
                i for cta in range(self.grid) for i in range(cta, routes * 5, self.grid)
            ]
            assert sorted(pairs) == list(range(routes * 5))
            tiles = [
                ((i // 5 + 1) * 10 + (i % 5) * 2 + half)
                for i in pairs
                for half in range(2)
            ]
            assert sorted(tiles) == list(range(10, (routes + 1) * 10))
        assert (
            self.shared_first >= 48
            and self.shared_first + self.shared_ctas <= self.grid
        )
        assert self.hc_ready_offset + 160 * 8 == self.tasks_ready_offset
        assert self.tasks_ready_offset + 8 == self.flags_bytes
        return self


PLAN = Plan().check()
