"""All-rank differential qualification of the integrated FP4 indexer."""

import json
import os
from dataclasses import replace

import torch
from aiter.dist.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)

from atom.models.deepseek_v41.mono.indexer import MonoIndexer
from atom.models.deepseek_v41.mono.indexer_workspace import IndexerWorkspace
from atom.mono.runtime.compile import launcher_resources
from tests.models.deepseek_v41.test_mono_indexer import make_case


class ProbeIndexer(MonoIndexer):
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
tests = 0
points = [
    (1, 8192, 1, 1, False, False, False),
    (1, 8193, 1, 1, False, False, False),
    (1, 163840, 1, 1, False, False, False),
    (8, 163840, 2, 1, False, False, True),
    (6, 163840, 2, 6, False, False, False),
    (12, 163840, 2, 6, False, False, True),
    (1, 163840, 1, 1, True, False, False),
    (6, 163840, 1, 6, True, False, False),
    (8, 163840, 1, 1, True, False, True),
    (1, 163840, 1, 1, False, True, False),
    (8, 163840, 1, 1, False, True, True),
    (40, 8192, 2, 1, False, False, False),
]
for rows, context, ratio, qlen, candidate, reindex, pad in points:
    case = make_case(
        rows,
        context,
        ratio,
        "mfma",
        qlen=qlen,
        candidate=candidate,
        reindex=reindex,
        pad=pad,
    )
    r = ProbeIndexer(
        case.runner.model,
        context + 128,
        True,
        "mfma",
        group,
        workspace=IndexerWorkspace(case.runner.model, context + 128, world),
    )
    assert r.world == world and r.prepare(rows)
    r.active = True

    def call(case=case, r=r, rows=rows):
        r.begin_step(rows, case.cache, case.step)
        assert r.run(
            case.indexer,
            case.hidden,
            case.qr,
            case.qrs,
            case.cache,
            case.step,
            case.rope,
        )

    local_step = r._key(rows).world == 1
    if local_step:
        r.peers.bytes.fill_(7)
    call()
    torch.cuda.synchronize()
    if local_step:
        assert torch.all(r.peers.bytes == 7), (rank, "local step reset peer mailboxes")
    # Mono uses live fused BF16 IW; compare native score selection off that IW.
    from types import SimpleNamespace

    from atom.models.deepseek_v41.mono.runner import V41MonoDecodeRunner

    mono_q = V41MonoDecodeRunner.index_query(
        SimpleNamespace(
            index_fp4=True, index_layout="mfma", iq=r.iq[:rows], iq_scale=r.iqs[:rows]
        )
    )
    native = case.native(mono_q, (r.iw[:rows] * 64).bfloat16())
    expected = native[2]
    # Native score helper returns (selection, candidates).
    sel = expected[0] if isinstance(expected, tuple) else expected
    got = case.step.selected[case.indexer.spec.layer_id][0]
    assert torch.equal(got, sel), (
        rank,
        rows,
        context,
        ratio,
        qlen,
        candidate,
        reindex,
        pad,
        "eager",
        got.shape,
        sel.shape,
    )
    if candidate:
        assert torch.equal(case.step.candidates[20], expected[1])
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    for rep in range(20):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(got, sel), (
        rank,
        rows,
        context,
        ratio,
        qlen,
        candidate,
        reindex,
        pad,
        "graph",
    )
    tests += 1
    if rank == 0:
        print(
            "INDEXER_EXACT",
            rows,
            context,
            ratio,
            qlen,
            candidate,
            reindex,
            pad,
            json.dumps(
                launcher_resources(r._launch_context(case.indexer.spec, rows)[0])
            ),
            flush=True,
        )
    torch.distributed.barrier(group=group)
    r._finalizer()
if rank == 0:
    print("PASSED", tests, flush=True)
torch.distributed.destroy_process_group()
