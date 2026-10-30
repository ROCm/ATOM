# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx950 c1 execution model for the gfx1250 structural port.

Logical N16 tasks and their 256-CTA placement belong to the model's schedule,
independently of the matrix instruction's physical wave32 lane layout. This
plan preserves the mother schedule, including multiple WQB/WOB tasks and the
down stage's arrival-assigned extra task. It does not select the old 320-CTA
backend or enable a runner path by itself.
"""

from dataclasses import dataclass

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels.dims import Dims
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES, first_task
from atom.mono.plan.layout import pair_layout

PHYSICAL_WAVE = 32
PHYSICAL_WAVES = THREADS // PHYSICAL_WAVE


@dataclass(frozen=True)
class StageTasks:
    base: int
    count: int

    def first(self, bid):
        return first_task(bid, self.base)

    def owner(self, task):
        return (task + self.base) % BLOCKS

    def tasks(self, bid):
        """Host inspection of one CTA's work; device loops use first + BLOCKS."""
        return tuple(range(self.first(bid), self.count, BLOCKS))


@dataclass(frozen=True)
class C1ReferencePlan:
    tp: int = 2
    feeds_index: bool = False

    def __post_init__(self):
        if self.tp not in (2, 4):
            raise ValueError("c1 reference plan supports TP2/TP4 dimensions")

    @property
    def dims(self):
        return Dims(self.tp)

    @property
    def stages(self):
        d = self.dims
        # The original c1 runner chooses ug_groups=1 (rows < fine_ug_below=7).
        router0, router_tasks = 8, 384 // 16 * 4
        route0 = router0 + router_tasks
        shared0 = route0 + 8
        return {
            "k1.slice": StageTasks(0, 160),
            "k1.norm": StageTasks(168, 1),
            "k1.wqkv_a": StageTasks(176, 112),
            "k1.gate": StageTasks(160, 1),
            "k1.qkv": StageTasks(32, 1),
            "k1.wq_b": StageTasks(40, d.wqb_tasks),
            # IQ executes inside WQB's token loop, before index_w.
            "k1.iq": StageTasks(32, 256 if self.feeds_index else 0),
            "k1.index_w": StageTasks(0, 32 if self.feeds_index else 0),
            "k1.index_quant": StageTasks(0, 32 if self.feeds_index else 0),
            "k2a.attn_score": StageTasks(0, d.head_tiles.count * 8),
            "k2a.irq": StageTasks(64, d.head_tiles.count * 16),
            "k2a.wo_a": StageTasks(160, d.woa_tasks),
            "k2a.wo_b": StageTasks(160 + d.woa_tasks, 320),
            "k2a.slice": StageTasks(0, 160),
            "k2a.norm": StageTasks(168, 1),
            "k2b.xq": StageTasks(0, 1),
            "k2b.router": StageTasks(router0, router_tasks),
            "k2b.route": StageTasks(route0, 1),
            "k2b.shared": StageTasks(shared0, d.shared_tasks),
            "k2a.gate": StageTasks(160, 1),
            "k2b.ug": StageTasks(shared0 + d.shared_tasks, 6 * (d.inter_real // 32)),
            "k2b.down": StageTasks(0, BLOCKS),
            "k2b.reduce": StageTasks(0, 160),
        }

    def down_tasks(self, bid, arrival):
        """Inspection of the original DQ rule: the first 64 arrivals take extras."""
        if not 0 <= bid < BLOCKS or not 0 <= arrival < BLOCKS:
            raise ValueError("down CTA and arrival rank must be in [0, 256)")
        return (bid, BLOCKS + arrival) if arrival < 320 - BLOCKS else (bid,)

    @staticmethod
    def k1_scratch_layout():
        """The mother's complete c1 scratch, including selecting-layer regions."""
        return pair_layout({
            "lin": 5120 // 2, "pmix": 160 * 25,
            "x8": 5120 // 4, "x8s": 5120 // 32,
            "qkv": 1792, "qx8": 1280 // 4, "qx8s": 1280 // 32,
            "nb": 5120 // 2, "iq": 4096 // 2, "iw": 32,
        }.items())

    def merged_scratch_layout(self, index_bound_max=0):
        """Original disjoint K2a/K2b storage, with intentional plain-word aliases.

        K2a reuses LIN/PMIX. Its unused X8 names are replaced by K2b's X8
        aliases in the merged context, as in layer_post.scratch_layout.
        """
        if index_bound_max < 0:
            raise ValueError("index_bound_max must be nonnegative")
        d = self.dims
        base = self.k1_scratch_layout()
        pairs = {
            "xo": d.xo_words, "xos": d.heads * 512 // 32,
            "am": 64 * d.heads, "al": 64 * d.heads,
            "ap": d.head_tiles.count * 64 * 64 * 2,
            "x8b": d.yb_words, "x8bs": d.o_rows // 32,
        }
        if index_bound_max:
            pairs.update(isf=ip.chunks(index_bound_max), irdy=1)
            for kind, k in ip.KINDS.items():
                for level in range(ip.top_level(kind, index_bound_max)):
                    pairs[f"i{kind}{level}"] = ip.list_slots(kind, level, index_bound_max) * k * 2
        layout = {name: base[name] for name in ("lin", "pmix", "x8", "x8s")}
        layout.update(pair_layout(pairs.items(), start=max(o+n for o,n in base.values())))
        moe_base = max(o+n for o,n in layout.values())
        moe = pair_layout({
            "x8r": 5120 // 4 // 2, "x8rs": 5120 // 32 // 2,
            "x8": 5120 // 4 // 2, "x8s": 5120 // 32 // 2, "x8rdy": 1,
            "logit": 384 * 4, "route": 8 * 2 * 6,
            "smid": d.sh_inter, "smq": d.sh_inter // 4, "smc": d.sh_inter // 32,
            "mid": 6 * d.mid_words, "mids": 6 * d.inter // 32,
            "ugf": 6 * (d.inter_real // 32), "ugq": 256 // 2, "dq": 256 // 2,
        }.items(), start=moe_base)
        layout.update(moe)
        layout["xrdy"] = (max(o+n for o,n in moe.values()), 8)
        return layout

    @staticmethod
    def tags(layer):
        if not 0 <= layer < 40:
            raise ValueError("reference backbone layer must be in [0, 40)")
        return {"k1": layer + 1, "k2a": 65 + layer, "k2b": 129 + layer}
