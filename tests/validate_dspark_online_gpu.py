"""Two-rank real-event/CUDA-graph integration probe, not a model benchmark.

Run with torchrun --standalone --nproc-per-node=2 -m tests.validate_dspark_online_gpu
--output /path/result.json. Profiles use measured small GEMM graphs solely for
this component probe; they must never be used as a V4.1 deployment profile.
"""

import argparse
import json
import os
import statistics
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from atom.config import CUDAGraphMode, DSparkConfig
from atom.spec_decode.dspark_runtime import DSparkRuntime
from atom.spec_decode.dspark_verify import VerifyScheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    device = torch.device("cuda", rank)
    x = torch.randn((1024, 1024), dtype=torch.float32, device=device)
    y = torch.randn_like(x)
    z = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(8):
            torch.mm(x, y, out=z)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(4):
            torch.mm(x, y, out=z)
    # Explicit synchronization is confined to this probe's baseline measurement
    # and final assertions, never the online runtime being tested.
    times = []
    for _ in range(12):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    measured = torch.tensor([statistics.median(times)], device="cpu")
    dist.all_reduce(measured, op=dist.ReduceOp.MAX)
    baseline = [1000 / (measured.item() * (0.5 + i / 12)) for i in range(13)]
    with tempfile.TemporaryDirectory() as folder:
        profile = Path(folder) / "component-only.json"
        profile.write_text(json.dumps({"sps_table": baseline}))
        config = SimpleNamespace(
            dspark=DSparkConfig(
                online_sps=True,
                confidence_schedule=True,
                ragged=True,
                calibration_profile=str(profile),
                online_sps_sample_interval=2,
                online_sps_update_interval=16,
            ),
            max_num_seqs=2,
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.TR_HYBRID),
            parallel_config=SimpleNamespace(data_parallel_size=1),
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=False,
            enable_tbo=False,
            enforce_eager=False,
        )
        runner = SimpleNamespace(config=config)
        runtime = DSparkRuntime(runner, group=dist.group.WORLD)
        vs = VerifyScheduler.__new__(VerifyScheduler)
        vs.runner, vs.runtime = runner, runtime
        vs.sps_table = torch.tensor(baseline, device=device)
        vs.sts_temperatures = None
        confidence = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]], device=device)
        initial_ell = vs.compute_ell(confidence).cpu().tolist()
        batch = SimpleNamespace(
            is_dummy_run=False,
            total_tokens_num_prefill=0,
            total_tokens_num_decode=6,
            total_seqs_num=1,
        )
        mode = SimpleNamespace(token_graph=True, running_tokens=6)
        for step in range(192):
            sample = runtime.begin(batch, mode)
            graph.replay()
            if rank == 1:
                graph.replay()  # Deliberately asymmetric worker timing.
            runtime.target_done(sample)
            graph.replay()  # Sample/draft phase for this component test.
            ell = vs.compute_ell(confidence)
            runtime.finish(sample)
            # Match the runner's ordinary host/output rendezvous without making
            # the runtime rely on an added device synchronize.
            ell.cpu()
        torch.cuda.synchronize()
        snapshot = runtime.snapshot()
        snapshots = [None] * dist.get_world_size()
        dist.all_gather_object(snapshots, snapshot)
        keys = [
            "version",
            "accepted",
            "slope_ms_per_row_by_batch",
            "fixed_ms_by_batch",
            "fixed_ms_global",
            "slope_support_by_batch",
            "post_ms_by_batch",
        ]
        assert all(
            {k: s[k] for k in keys} == {k: snapshots[0][k] for k in keys}
            for s in snapshots
        )
        assert snapshot["accepted"] >= 30, snapshot
        assert snapshot["version"] >= 5, snapshot
        assert snapshot["pending"] <= 8
        assert snapshot["slope_ms_per_row"] == 0
        assert snapshot["fixed_ms_by_batch"][1] > 0
        effective = runtime.table(vs.sps_table, 1)
        assert torch.all(effective <= vs.sps_table)
        assert torch.all(effective[1:] <= effective[:-1])
        final_ell = vs.compute_ell(confidence).cpu().tolist()
        if rank == 0:
            Path(args.output).write_text(
                json.dumps(
                    {
                        "passed": True,
                        "scope": "two-rank component integration, not V4.1 performance",
                        "gpu": torch.cuda.get_device_name(),
                        "baseline_forward_ms": measured.item(),
                        "initial_ell": initial_ell,
                        "final_ell": final_ell,
                        "ranks": snapshots,
                    },
                    indent=2,
                )
            )
            print("DSPARK_ONLINE_GPU_PASS", initial_ell, final_ell)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
