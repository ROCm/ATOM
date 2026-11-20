"""Paired real-weight FP4 baseline/optimized FULL indexer benchmark."""

import json
import os
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from aiter.dist.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)

from atom.models.deepseek_v41.mono.index_tp_plan import reference_work
from atom.models.deepseek_v41.mono.indexer import MonoIndexer
from atom.models.deepseek_v41.mono.indexer_workspace import IndexerWorkspace
from atom.models.deepseek_v41.mono.runner import V41MonoDecodeRunner
from atom.mono.runtime.compile import launcher_resources
from tests.models.deepseek_v41.test_mono_indexer import make_case


class BaselineIndexer(MonoIndexer):
    def _bind(self):
        super()._bind()
        self.full_scratch = None


assert os.environ.get("ATOM_DSV41_CHECKPOINT"), "Real checkpoint weights required"
rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
assert world in (2, 4)
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
initialize_model_parallel(tensor_model_parallel_size=world)
group = get_tp_group().cpu_group
batch = 10
replays = 50
points = os.environ.get(
    "TP_POINTS", "1:8192:1,1:163840:1,8:8192:1,8:163840:1,40:8192:1,40:163840:1"
).split(",")
reports = []
out = Path(
    os.environ.get(
        "TP_OUTPUT",
        f"validation/tp{world}-full-final-performance.json",
    )
)
out.parent.mkdir(parents=True, exist_ok=True)


def capture(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(batch):
            fn()
    for _ in range(50):
        g.replay()
    torch.cuda.synchronize()
    return g


def timing(g):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(replays):
        g.replay()
    b.record()
    b.synchronize()
    local = torch.tensor(
        a.elapsed_time(b) * 1000 / (batch * replays), device="cuda", dtype=torch.float64
    )
    torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.MAX)
    return local.item()


def loaded_clocks(g):
    estimate = timing(g)
    for _ in range(max(30, int(600000 / (estimate * batch)))):
        g.replay()
    result = (
        subprocess.run(
            ["rocm-smi", "--showclocks", "--showpower", "--json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if rank == 0
        else None
    )
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)
    return result


for point in points:
    rows, context, qlen = map(int, point.split(":"))
    for kind, ratio, candidate, reindex in (
        ("full", 2, False, False),
        ("candidate", 1, True, False),
        ("reindex", 1, False, True),
    ):
        case = make_case(
            rows,
            context,
            ratio,
            "mfma",
            qlen=qlen,
            candidate=candidate,
            reindex=reindex,
            pad=False,
        )
        local = BaselineIndexer(
            case.runner.model,
            context + 128,
            True,
            "mfma",
            group,
            workspace=IndexerWorkspace(case.runner.model, context + 128, world),
        )
        coop = MonoIndexer(
            local.model,
            context + 128,
            True,
            "mfma",
            group,
            workspace=IndexerWorkspace(local.model, context + 128, world),
        )
        for r in (local, coop):
            assert r.prepare(rows)
            r.active = True

        def run(r, case=case, rows=rows):
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

        baseline_selection = None
        for r in (local, coop):
            run(r)
            torch.cuda.synchronize()
            query = V41MonoDecodeRunner.index_query(
                SimpleNamespace(
                    index_fp4=True,
                    index_layout="mfma",
                    iq=r.iq[:rows],
                    iq_scale=r.iqs[:rows],
                )
            )
            expected = case.native(query, (r.iw[:rows] * 64).bfloat16())[2]
            got = r.selected[case.indexer.spec.layer_id][:rows]
            assert torch.equal(got, expected[0]), (rank, point, kind, "selected")
            if candidate:
                assert torch.equal(r.candidates[:rows], expected[1]), (
                    rank,
                    point,
                    kind,
                    "candidates",
                )
            if baseline_selection is None:
                baseline_selection = got.clone()
            else:
                assert torch.equal(got, baseline_selection), (
                    rank,
                    point,
                    kind,
                    "baseline selected",
                )

        gs = {
            "native": capture(case.native),
            "baseline": capture(lambda r=local, fn=run: fn(r)),
            "optimized": capture(lambda r=coop, fn=run: fn(r)),
        }
        for _ in range(100):
            gs["native"].replay()
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)
        before = loaded_clocks(gs["native"])
        samples = {name: [] for name in gs}
        for rep in range(7):
            for name in list(gs) if rep % 2 == 0 else list(reversed(gs)):
                torch.distributed.barrier(group=group)
                for _ in range(5):
                    gs[name].replay()
                samples[name].append(timing(gs[name]))
        median = {name: statistics.median(values) for name, values in samples.items()}
        selected_world = 1 if reindex else coop._key(rows).world
        work = reference_work(
            case.step.visible[ratio].tolist(),
            case.step.batch_ids.tolist(),
            selected_world,
            0,
            force_local=reindex,
        )
        modes = sorted(
            {("local", "owned", "split")[w.kind] for w in work if w.owner >= 0}
        )
        item = {
            "checkpoint": os.environ["ATOM_DSV41_CHECKPOINT"],
            "split_segment": 8192,
            "tp": world,
            "kernel_tp": selected_world,
            "execution": "+".join(modes),
            "execution_source": "reference_work on static benchmark inputs",
            "rows": rows,
            "bs": rows // qlen,
            "qlen": qlen,
            "context": context,
            "kind": kind,
            "median_us": median,
            "samples_us": samples,
            "spread_us": {
                name: max(values) - min(values) for name, values in samples.items()
            },
            "reduction_vs_native": 100 * (1 - median["optimized"] / median["native"]),
            "reduction_vs_baseline": 100
            * (1 - median["optimized"] / median["baseline"]),
            "resources": {
                "baseline": launcher_resources(
                    local._launch_context(case.indexer.spec, rows)[0]
                ),
                "optimized": launcher_resources(
                    coop._launch_context(case.indexer.spec, rows)[0]
                ),
            },
            "clocks_before": before,
            "clocks_after": loaded_clocks(gs["optimized"]),
        }
        reports.append(item)
        if rank == 0:
            out.write_text(json.dumps(reports, indent=2))
            print(
                "PERF",
                json.dumps(
                    {
                        k: v
                        for k, v in item.items()
                        if k
                        not in (
                            "resources",
                            "clocks_before",
                            "clocks_after",
                            "samples_us",
                        )
                    }
                ),
                flush=True,
            )
        torch.distributed.barrier(group=group)
        local._finalizer()
        coop._finalizer()
if rank == 0:
    print("COMPLETE", len(reports), flush=True)
torch.distributed.destroy_process_group()
