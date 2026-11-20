"""Exact TP selection while one captured graph switches request work modes."""

import json
import os
from dataclasses import replace
from types import SimpleNamespace

import torch
from aiter.dist.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)

from atom.models.deepseek_v41.mono.indexer import MonoIndexer
from atom.models.deepseek_v41.mono.indexer_workspace import IndexerWorkspace
from atom.models.deepseek_v41.mono.runner import V41MonoDecodeRunner
from tests.models.deepseek_v41.test_mono_indexer import make_case


class SwitchIndexer(MonoIndexer):
    def _key(self, rows):
        key = super()._key(rows)
        return (
            replace(key, world=self.world, peer_plain=self.peer_plain)
            if os.environ.get("TP_FORCE_COOP") == "1"
            else key
        )


rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
initialize_model_parallel(tensor_model_parallel_size=world)
group = get_tp_group().cpu_group
tested = 0

for candidate, reindex in ((False, False), (True, False), (False, True)):
    case = make_case(8, 163840, 1, "mfma", candidate=candidate, reindex=reindex)
    runner = SwitchIndexer(
        case.runner.model,
        163968,
        True,
        "mfma",
        group,
        workspace=IndexerWorkspace(case.runner.model, 163968, world),
    )
    assert runner.prepare(8)
    runner.active = True

    def call(case=case, runner=runner):
        case.step.candidate_blocks.clear()
        runner.begin_step(8, case.cache, case.step)
        assert runner.run(
            case.indexer,
            case.hidden,
            case.qr,
            case.qrs,
            case.cache,
            case.step,
            case.rope,
        )

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()

    for cycle in range(3):
        for lengths in (
            [8192] * 8,
            [8193] * 8,
            [8192, 8193, 163840, 513, 512, 1, 0, 0],
            [8170 + i for i in range(6)] + [163839, 163840],
            [163839, 163840] + [8170 + i for i in range(6)],
            [163840] * 8,
            [8192] * 8,
        ):
            bounds = torch.tensor(lengths, dtype=torch.int32, device="cuda")
            case.step.visible[1].copy_(bounds)
            case.step.positions.copy_((bounds.long() - 1).clamp_min(0))
            ids = torch.arange(8, dtype=torch.int32, device="cuda")
            if lengths[0] == 8170:
                ids = torch.tensor([0] * 6 + [1] * 2, dtype=torch.int32, device="cuda")
            elif lengths[0] == 163839 and lengths[2] == 8170:
                ids = torch.tensor([0] * 2 + [1] * 6, dtype=torch.int32, device="cuda")
            ids.masked_fill_(bounds == 0, -1)
            case.step.batch_ids.copy_(ids)
            if reindex:
                selected = case.step.candidates[20]
                for row, bound in enumerate(lengths):
                    count = min(2048, (bound + 7) // 8)
                    selected[row].fill_(-1)
                    if count:
                        selected[row, :count].copy_(
                            torch.linspace(
                                0, (bound - 1) // 8, count, device="cuda"
                            ).int()
                        )
            for _ in range(20):
                graph.replay()
            torch.cuda.synchronize()
            query = V41MonoDecodeRunner.index_query(
                SimpleNamespace(
                    index_fp4=True,
                    index_layout="mfma",
                    iq=runner.iq[:8],
                    iq_scale=runner.iqs[:8],
                )
            )
            native = case.native(query, (runner.iw[:8] * 64).bfloat16())[2]
            assert torch.equal(
                runner.selected[case.indexer.spec.layer_id][:8], native[0]
            ), (rank, candidate, reindex, cycle, lengths, "selected")
            if candidate:
                assert torch.equal(runner.candidates[:8], native[1]), (
                    rank,
                    cycle,
                    lengths,
                    "candidates",
                )
            tested += 1
            if rank == 0:
                print(
                    "SWITCH_EXACT",
                    json.dumps(
                        {
                            "candidate": candidate,
                            "reindex": reindex,
                            "cycle": cycle,
                            "bounds": lengths,
                        }
                    ),
                    flush=True,
                )
            torch.distributed.barrier(group=group)
    runner._finalizer()

if rank == 0:
    print("PASSED", tested, flush=True)
torch.distributed.destroy_process_group()
