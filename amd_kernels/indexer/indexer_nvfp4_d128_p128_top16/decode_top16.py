# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The decode top-16 of a step in one call: the four-GPU scorer on the rank-owned or replicated index cache
(``split4.host.IndexSplit4RankOwned.score_topk``) or the one-GPU scorer + top-16
(``score_decode.nvfp4_index_score_top16``), chosen by request count; the ids are the same on both routes
(``reference.topk_reference``).
"""

import torch

from .reference import BLK, score_width
from .score_decode import nvfp4_index_score_rows, nvfp4_index_score_top16

# the four-GPU route from this many requests (by rows per request): the decode block (index top-16 + attention,
# benchmark_index_decode_ranks4.py) at an agentic workload's decode-time contexts, us: 8 rows x 4 / 5 requests one GPU
# 24.30 / 27.92 vs four 25.46 / 26.64; 4 rows x 7 / 8 23.45 / 28.36 vs 24.59 / 24.75
SPLIT_MIN_REQS = {8: 5, 4: 8}
WARM_CONTEXT = 65536  # context of the four-GPU route's warm-up calls


def split_min_reqs(query_len: int) -> int:
    return SPLIT_MIN_REQS.get(query_len, 1)


def use_split(
    num_reqs: int,
    query_len: int,
    num_index_heads: int,
    *,
    split_query_len: int,
    split_max_rows: int,
) -> bool:
    """Whether a step of ``num_reqs`` x ``query_len`` rows takes the four-GPU route built for ``split_query_len``
    rows per request and ``split_max_rows`` rows."""
    return (
        num_index_heads == 1
        and query_len == split_query_len
        and num_reqs * query_len <= split_max_rows
        and num_reqs >= split_min_reqs(query_len)
    )


class IndexDecodeTop16:
    """The decode top-16 of every (index head, row) on this GPU; built once per process, at the same point on
    every rank when ``group`` is given (a collective: IPC buffers, then the warm-up's launches).

    Args:
        group: the four ranks' gloo group for the four-GPU route, or None (one GPU only). The route needs four
            ranks and 1-``decode_score.MAX_QL`` rows per request; otherwise ``split_off`` says why it is off.
        sharded: the index cache holds only this rank's pages (False: the replicated cache).
        warm_up: compile every variant now.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        query_len: int,
        max_reqs: int,
        max_seq_len: int,
        num_index_heads: int = 1,
        rank: int | None = None,
        group=None,
        sharded: bool = False,
        warm_up: bool = True,
    ):
        self.device = torch.device(device)
        self.query_len, self.max_reqs, self.max_seq_len = query_len, max_reqs, max_seq_len
        self.num_index_heads, self.sharded = num_index_heads, sharded
        self.split = None
        self.split_off = None if group is not None else "no process group"
        if group is not None:
            import torch.distributed as dist

            from .split4 import decode_score
            from .split4.host import WORLD, IndexSplit4RankOwned

            world = dist.get_world_size(group)
            if world != WORLD:
                self.split_off = f"needs {WORLD} ranks ({world})"
            elif not 1 <= query_len <= decode_score.MAX_QL:
                self.split_off = f"{query_len} decode rows per request (1-{decode_score.MAX_QL})"
            else:
                reqs = min(decode_score.MAX_REQS, max_reqs)
                self.split = IndexSplit4RankOwned(
                    rank, group, self.device, reqs * query_len, score_width(max_seq_len), query_len
                )
        if warm_up:
            self.warm_up()

    def route(self, num_reqs: int, query_len: int, num_index_heads: int, split: bool = True) -> str:
        """``"split4"`` or ``"one_gpu"``; ``split`` False keeps one GPU (DBO micro-batches, where no collective
        orders the four-GPU push)."""
        cp = self.split
        if (
            split
            and cp is not None
            and use_split(
                num_reqs, query_len, num_index_heads,
                split_query_len=cp.query_len, split_max_rows=cp.max_rows,
            )
        ):
            return "split4"
        return "one_gpu"

    def __call__(
        self,
        iq: torch.Tensor,  # [rows, index heads, 128] e4m3
        index_cache: torch.Tensor,  # NVFP4 index pages [num_blocks, 128, 72] uint8
        block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
        seq_lens: torch.Tensor,  # [num_reqs] int32
        init_blocks: int,
        local_blocks: int,
        topk_idx: torch.Tensor,  # [index heads, rows, 16] int32, written
        *,
        query_len: int,
        max_seq_len: int,
        split: bool = True,
    ) -> None:
        """The top-16 block ids of every decode row; graph-safe (lengths are read on the device)."""
        heads = iq.shape[1]
        if self.route(seq_lens.shape[0], query_len, heads, split) == "split4":
            self.split.score_topk(
                iq[:, 0],
                index_cache,
                block_table,
                seq_lens,
                init_blocks,
                local_blocks,
                score_width(max_seq_len),
                topk_idx,
                max_seq_len,
                sharded=self.sharded,
            )
            return
        nvfp4_index_score_top16(
            iq,
            index_cache,
            block_table,
            seq_lens,
            init_blocks,
            local_blocks,
            topk_idx,
            query_len=query_len,
            max_seq_len=max_seq_len,
        )

    def warm_up(self) -> None:
        """Compiles every variant either route launches (root README rule 4): the one-GPU entries' first calls
        warm their sizes (``amd_kernels.warm_up``); the four-GPU route's fused (<= ``TK_MAX_ROWS`` rows), four-wave
        (<= 512) and one-wave top-k are real launches on every rank in the same order."""
        dev, ql, heads = self.device, self.query_len, self.num_index_heads
        with torch.inference_mode():
            ix = torch.zeros((2, BLK, 72), dtype=torch.uint8, device=dev)
            for reqs, ctx in ((1, 1024), (min(64, self.max_reqs), self.max_seq_len)):
                rows = reqs * ql
                args = (
                    torch.zeros((rows, heads, 128), dtype=torch.float8_e4m3fn, device=dev),
                    ix,
                    torch.zeros((reqs, max(1, -(-ctx // BLK))), dtype=torch.int32, device=dev),
                    torch.full((reqs,), ql, dtype=torch.int32, device=dev),
                    0,
                    1,
                )
                topk = torch.empty((heads, rows, 16), dtype=torch.int32, device=dev)
                nvfp4_index_score_top16(*args, topk, query_len=ql, max_seq_len=ctx)
                # the scorer alone, for the attention's prologue top-16
                score = torch.empty((heads, rows, score_width(ctx)), dtype=torch.float32, device=dev)
                q_idx, cache, table, seq, init, local = args
                nvfp4_index_score_rows(q_idx, cache, score, table, seq, init, local, query_len=ql, max_seq_len=ctx)
            if self.split is None or heads != 1:
                return
            from .split4 import decode_score

            above = (decode_score.TK_MAX_ROWS // ql + 1) * ql
            for rows in sorted({ql, 64 // ql * ql, above, 1024 // ql * ql}):
                reqs = rows // ql
                if reqs > self.split.max_rows // ql:
                    continue
                self.split.score_topk(
                    torch.zeros((rows, 128), dtype=torch.float8_e4m3fn, device=dev),
                    ix,
                    torch.zeros((reqs, -(-WARM_CONTEXT // BLK)), dtype=torch.int32, device=dev),
                    torch.full((reqs,), WARM_CONTEXT, dtype=torch.int32, device=dev),
                    0,
                    1,
                    score_width(WARM_CONTEXT),
                    torch.empty((1, rows, 16), dtype=torch.int32, device=dev),
                    WARM_CONTEXT,
                    sharded=self.sharded,
                )
