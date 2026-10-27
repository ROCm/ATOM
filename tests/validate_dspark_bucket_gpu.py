"""GPU/CPU decision parity, TP consistency and host-read guard; no model payload."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from atom.spec_decode.dspark_bucket import (
    graph_row_indices,
    schedule_bucket_prefix_lengths_tensor,
)
from atom.spec_decode.dspark_scheduler import schedule_prefix_lengths_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    device = torch.device("cuda", rank)
    qs = [1, 3, 6]
    sizes = sorted({b * q for b in [1, 2, 4, 8, 16, 32, 64, 128] for q in qs})
    rows = torch.tensor(graph_row_indices(sizes, qs, 768))
    sps = 1000 / (10 + torch.arange(769) * 0.15)
    gpu_rows, gpu_sps = rows.to(device), sps.to(device)
    records = []
    for batch in [1, 2, 8, 16, 64, 128]:
        generator = torch.Generator().manual_seed(batch)
        conf = torch.rand((batch, 5), generator=generator) * 0.98 + 0.01
        expected = schedule_bucket_prefix_lengths_tensor(conf, sps, rows, qs)
        gpu_conf = conf.to(device)
        actual = schedule_bucket_prefix_lengths_tensor(gpu_conf, gpu_sps, gpu_rows, qs)
        assert torch.equal(expected, actual.cpu())
        signatures = [None] * dist.get_world_size()
        dist.all_gather_object(signatures, actual.cpu().tolist())
        assert all(s == signatures[0] for s in signatures)
        timings = {}
        for name, fn in [
            ("legacy", lambda c=gpu_conf: schedule_prefix_lengths_tensor(c, gpu_sps)),
            (
                "bucket",
                lambda c=gpu_conf: schedule_bucket_prefix_lengths_tensor(
                    c, gpu_sps, gpu_rows, qs
                ),
            ),
        ]:
            for _ in range(8):
                fn()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                fn()
            torch.cuda.synchronize()
            timings[name] = (time.perf_counter() - start) * 1000 / 100
        records.append({"batch": batch, "mean_call_wall_ms": timings})
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        schedule_bucket_prefix_lengths_tensor(gpu_conf, gpu_sps, gpu_rows, qs)
    assert not any(e.key == "aten::_local_scalar_dense" for e in prof.key_averages())
    if rank == 0:
        Path(args.output).write_text(
            json.dumps(
                {
                    "passed": True,
                    "cpu_gpu_tp_parity": True,
                    "additional_host_scalar_reads": 0,
                    "scope": "component microbenchmark, not model throughput",
                    "records": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
