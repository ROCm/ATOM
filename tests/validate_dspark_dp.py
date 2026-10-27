"""Real TP x DP collectives and optional GPU-event/graph component validation.

torchrun --standalone --nproc-per-node=4 -m tests.validate_dspark_dp
  --tp-size 2 --device cuda --output /tmp/dspark-dp.json

This is a synthetic component probe, not a model performance measurement.
"""

import argparse
import json
import os
import statistics
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
import torch.distributed as dist

from atom.config import CUDAGraphMode, DSparkConfig
from atom.spec_decode.dspark_dp import DPVerifyPlanner, SPSMesh, executable_graph
from atom.spec_decode.dspark_runtime import DSparkRuntime
from atom.spec_decode.dspark_verify import VerifyScheduler
from atom.utils.region_graph import TokenGraph


class CPUEvent:
    def record(self, *args):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 2.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world % args.tp_size == 0 and world // args.tp_size > 1
    dp_size, tp_rank = world // args.tp_size, rank % args.tp_size
    dp_rank = rank // args.tp_size
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    tp_group = dp_group = None
    for d in range(dp_size):
        group = dist.new_group(
            list(range(d * args.tp_size, (d + 1) * args.tp_size)), backend="gloo"
        )
        if d == dp_rank:
            tp_group = group
    for t in range(args.tp_size):
        group = dist.new_group(
            [d * args.tp_size + t for d in range(dp_size)], backend="gloo"
        )
        if t == tp_rank:
            dp_group = group
    mesh = SPSMesh(tp_group, dp_group)
    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if args.device == "cuda"
        else torch.device("cpu")
    )
    graphs, tensors = {}, []
    sizes = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24]
    baseline = [1000 / (0.4 + i * 0.12) for i in range(25)]
    if device.type == "cuda":
        torch.cuda.set_device(device)
        for rows in sizes:
            x = torch.randn((rows, 1024), device=device)
            w = torch.randn((1024, 1024), device=device)
            out = torch.empty_like(x)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(4):
                    torch.mm(x, w, out=out)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8):
                    torch.mm(x, w, out=out)
            graphs[rows] = graph
            tensors.append((x, w, out))
    else:
        torch.cuda.is_current_stream_capturing = lambda: False
    TokenGraph.captured_sizes = classmethod(lambda cls, phase: tuple(sizes))
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "component-only.json"
        path.write_text(json.dumps({"sps_table": baseline}))
        config = NS(
            dspark=DSparkConfig(
                online_sps=True,
                bucket_global=True,
                confidence_schedule=True,
                ragged=True,
                ragged_graph_sizes="1,3,6",
                calibration_profile=str(path),
                online_sps_window_steps=64,
                online_sps_sample_interval=2,
                online_sps_update_interval=8,
            ),
            max_num_seqs=4,
            speculative_config=NS(num_speculative_tokens=5),
            parallel_config=NS(data_parallel_size=dp_size),
            compilation_config=NS(cudagraph_mode=CUDAGraphMode.TR_HYBRID),
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=False,
            enable_tbo=False,
            enforce_eager=False,
        )
        runner = NS(config=config, tokenID_processor=NS(prev_batch=None))
        if device.type == "cuda":
            runner.tokenID_processor.async_copy_stream = torch.cuda.Stream()
        scheduler = VerifyScheduler.__new__(VerifyScheduler)
        scheduler.runner = runner
        scheduler.sts_temperatures = None
        scheduler._ell_map = {}
        scheduler.runtime = DSparkRuntime(
            runner, mesh=mesh, event_factory=CPUEvent if device.type == "cpu" else None
        )
        scheduler.dp = DPVerifyPlanner(scheduler, mesh=mesh)
        planner, runtime = scheduler.dp, scheduler.runtime
        cases, costs, graphs_seen = {}, [], set()
        pure_steps = 0
        for step in range(176):
            dummy = 48 <= step < 56 and dp_rank == 1
            prefill = 56 <= step < 64 and dp_rank == 0
            n = 1 if dummy else (2 if dp_rank == 0 else 4)
            ids = (-1,) if dummy else tuple(100 * dp_rank + i for i in range(n))
            if 64 <= step < 72:
                ids = tuple(i + step for i in ids)  # cancellations and arrivals
            if 72 <= step < 80:
                ids = ids[::-1] if step % 2 else ids
            batch = NS(
                req_ids=ids,
                total_seqs_num=n,
                is_dummy_run=dummy,
                total_tokens_num_prefill=n * 6 if prefill else 0,
                total_tokens_num_decode=0 if prefill else n * 6,
                num_spec_step=5,
                num_scheduled_tokens=np.full(n, 6),
                num_bonus=np.full(n, 2 if 96 <= step < 112 else 0),
            )
            before = time.perf_counter()
            planner.prepare(batch)
            costs.append((time.perf_counter() - before) * 1000)
            pure_steps += int(planner.state.pure_decode)
            if prefill:
                assert not scheduler._ell_map
            # Both TP peers must use leader confidence even when this probe
            # deliberately feeds very different probabilities to the other.
            own = torch.tensor(
                [scheduler._ell_map.get(i, 5) for i in ids], device="cpu"
            )
            tp_values = mesh.gather(own, tp_group, args.tp_size)
            assert torch.all(tp_values == own)
            lens = [
                max(scheduler._ell_map.get(i, 5), int(batch.num_bonus[j])) + 1
                for j, i in enumerate(ids)
            ]
            if dummy:
                lens = [6]
            local = torch.tensor([sum(lens), planner.selected_q or 6], device="cpu")
            shapes = mesh.gather_dp(local)
            rows = executable_graph(
                int(shapes[:, 0].max()), int(shapes[:, 1].max()), sizes
            )
            mode = NS(token_graph=planner.state.pure_decode, running_tokens=rows or 24)
            timing = runtime.begin(batch, mode)
            if device.type == "cuda":
                graphs[rows or 24].replay()
                if rank % 2:
                    graphs[rows or 24].replay()
            runtime.target_done(timing)
            # Middle-prefill returns without a confidence record. The next
            # forward must not reinterpret an older entry as generation N-2.
            if not prefill:
                confidence = torch.full(
                    (n, 5), 0.15 if tp_rank == 0 else 0.99, device=device
                )
                scheduler.compute_ell(confidence)
                if device.type == "cuda":
                    planner.record(ids)
                elif mesh.tp_rank == 0:
                    planner.pending.append(
                        (
                            planner.epoch,
                            CPUEvent(),
                            planner.survival,
                            ids,
                            planner.survival,
                        )
                    )
                    planner.pending = planner.pending[-4:]
                    planner.survival = None
            if device.type == "cuda":
                graphs[rows or 24].replay()
            runtime.finish(timing)
            # Test-only rendezvous simulating a consumer of this step's output.
            if device.type == "cuda":
                torch.cuda.synchronize()
            runner.tokenID_processor.prev_batch = batch
            label = (
                "mixed"
                if 56 <= step < 64
                else (
                    "idle"
                    if 48 <= step < 56
                    else "boundary" if 64 <= step < 80 else "decode"
                )
            )
            cases[label] = cases.get(label, 0) + 1
            graphs_seen.add(rows)
        snapshot = runtime.model.snapshot()
        snapshots = [None] * world
        dist.all_gather_object(snapshots, snapshot)
        assert all(x == snapshots[0] for x in snapshots)
        assert runtime.step == pure_steps
        assert runtime.model.accepted > 0 and runtime.model.version > 1
        assert planner.contracted > 0
        assert runtime.model.current_step - runtime.model.samples[0][0] < 64
        report = {
            "completed": True,
            "kind": "synthetic component",
            "device": args.device,
            "tp": args.tp_size,
            "dp": dp_size,
            "cases": cases,
            "group_decode_steps": pure_steps,
            "planned": planner.planned,
            "contracted": planner.contracted,
            "graphs": sorted(g for g in graphs_seen if g is not None),
            "median_planner_ms": statistics.median(costs),
            "model": snapshot,
            "all_worker_models_equal": True,
        }
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
