"""Bounded replay of captured migration traffic; not a serving benchmark."""

import argparse
from contextlib import contextmanager
from datetime import timedelta
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import time


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fingerprint(operations):
    return hashlib.sha256(
        json.dumps(operations, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@contextmanager
def annotated_phases(module, torch):
    originals = {}
    for name in (
        "_plan_single_layer_migration", "_execute_batched_p2p_ops",
        "_migrate_single_layer", "move_from_buffer", "commit_layer",
    ):
        original = getattr(module, name)
        originals[name] = original

        def wrap(fn, label):
            @functools.wraps(fn)
            def annotated(*args, **kwargs):
                with torch.profiler.record_function("eplb_replay::" + label):
                    return fn(*args, **kwargs)
            return annotated

        setattr(module, name, wrap(original, name))
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 8
    # Set debug paths before importing or initializing the GPU runtime.
    os.environ["ATOM_EPLB_DIAGNOSTICS_DIR"] = str(args.output.resolve())
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,GRAPH,P2P,ENV"
    os.environ["NCCL_DEBUG_FILE"] = str(args.output.resolve() / "rccl-%h-%p.log")
    os.environ["NCCL_TOPO_DUMP_FILE"] = str(args.output.resolve() / f"topology-{rank}.xml")
    os.environ["NCCL_GRAPH_DUMP_FILE"] = str(args.output.resolve() / f"graph-{rank}.xml")

    import torch
    import torch.distributed as dist
    from atom.benchmarks.results.metadata import installed_package
    from atom.model_ops import eplb

    assert Path(importlib.util.find_spec("atom").origin).resolve().parent == Path("/workspace/atom")
    sha = subprocess.check_output(
        ["git", "-c", "safe.directory=/workspace", "rev-parse", "HEAD"], text=True
    ).strip()
    assert sha == os.environ["EXPECTED_ATOM_SHA"]
    aiter = installed_package("amd-aiter")
    assert aiter["sha"] == os.environ["EXPECTED_AITER_SHA"]
    assert torch.cuda.device_count() == 8
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    captured = json.loads(args.input.read_text(encoding="utf-8"))
    assert captured["world_size"] == 8 and captured["nnodes"] == 1
    assert captured["p2p_batch_chunk_size"] == 32
    assert [c["rebalance"] for c in captured["cases"]] == [1, 2, 3]
    layouts = captured["cases"][0]["weight_layout"]
    assert all(c["weight_layout"] == layouts for c in captured["cases"])
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=3))
    migration_group = dist.new_group(ranks=list(range(8)), timeout=timedelta(minutes=3))
    dist.barrier(group=migration_group)
    torch.cuda.synchronize()

    if rank == 0:
        libs = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                if any(name in line for name in ("librccl.so", "libhsa-runtime64.so", "libamdhip64.so"))}
        marker = Path("/opt/rocm/.info/atom-rocm-runtime-commit")
        dump(args.output / "runtime.json", {
            "atom_sha": sha, "aiter": aiter, "torch": torch.__version__,
            "hip": torch.version.hip, "nccl": torch.cuda.nccl.version(),
            "rocm_runtime_overlay_commit": marker.read_text().strip() if marker.is_file() else None,
            "loaded_gpu_libraries": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(libs) if Path(p).is_file()},
            "devices": [str(torch.cuda.get_device_properties(i)) for i in range(8)],
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "diagnostic_only": True, "original_serving_workload": False,
            "limitations": [
                "Single layer and fresh communicator; no model inference, other communicators or model-weight residency.",
                "Initialization and rank alignment synchronize outside measurement; prior GPU work and entry skew are intentionally removed.",
                "P2P events include communication-stream dependencies and peer waits.",
                "A fast replay does not prove the serving regression fixed.",
            ],
        })

    weights = [torch.empty_strided(
        x["shape"], x["stride"], dtype=getattr(torch, x["dtype"].split(".")[-1]), device=device
    ) for x in layouts]
    buffers = [torch.empty_like(w) for w in weights]
    assert all(list(w.stride()) == x["stride"] and w.element_size() == x["element_size"]
               for w, x in zip(weights, layouts))

    def metadata(permutation):
        assert sorted(permutation) == list(range(384))
        inverse = [0] * 384
        for physical, logical in enumerate(permutation):
            inverse[logical] = physical
        return eplb.ExpertLocationMetadata.from_rebalance_result(
            physical_to_logical_map=torch.tensor([permutation], dtype=torch.int32, device=device),
            logical_to_physical_map=torch.tensor(inverse, dtype=torch.int32, device=device).reshape(1, 384, 1),
            logical_replica_count=torch.ones((1, 384), dtype=torch.int32, device=device),
            ep_size=8, ep_rank=rank, max_num_replicas=1,
        )

    def pattern(permutation, index):
        # Four-byte expert ID distinguishes all 384 experts; full storage is checked.
        codes = [list(struct.pack("<IIII", expert, index, 0xFEEDBEEF, expert ^ 0xC0DECAFE))
                 for expert in permutation[rank * 48:(rank + 1) * 48]]
        return torch.tensor(codes, dtype=torch.uint8, device=device).unsqueeze(1)

    diagnostic = args.output / f"migration-{os.getpid()}.jsonl"
    results = []

    def execute(case, trace):
        label = f"rebalance{case['rebalance']}-layer{case['source_layer']}" + ("-trace" if trace else "")
        live = metadata(case["old_physical_to_logical"])
        new = metadata(case["new_physical_to_logical"])
        for index, (w, buf) in enumerate(zip(weights, buffers)):
            w.view(torch.uint8).reshape(48, -1, 16).copy_(pattern(case["old_physical_to_logical"], index))
            buf.view(torch.uint8).fill_(0xA5)
        expected = [pattern(case["new_physical_to_logical"], i) for i in range(len(weights))]
        torch.cuda.synchronize()
        dist.barrier(group=migration_group)
        torch.cuda.synchronize()

        def migrate():
            begin = time.perf_counter()
            eplb.migrate_and_commit_chunk(
                layer_ids=[0], old_meta=live, new_meta=new,
                expert_weights_of_layer={0: weights}, temp_buffers=buffers,
                ep_group=migration_group, nnodes=1, rank=rank, live_meta=live,
                p2p_batch_chunk_size=32, cuda_stream=None,
            )
            return time.perf_counter() - begin

        profile = None
        if trace:
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=activities, record_shapes=False,
                                        profile_memory=False, with_stack=False) as profile:
                with annotated_phases(eplb, torch):
                    elapsed = migrate()
            profile.export_chrome_trace(str(args.output / f"trace-rank{rank}.json"))
        else:
            elapsed = migrate()
        row = json.loads(diagnostic.read_text(encoding="utf-8").splitlines()[-1])
        source = case["expected_ranks"][rank]
        assert row["p2p_ops"] == source["p2p_ops"]
        assert row["p2p_bytes_send_plus_receive"] == source["p2p_bytes_send_plus_receive"]
        assert fingerprint(row["p2p_operations"]) == source["p2p_operations_sha256"]
        assert row["old_physical_to_logical"] == case["old_physical_to_logical"]
        assert row["new_physical_to_logical"] == case["new_physical_to_logical"]
        assert row["weight_layout"] == layouts
        # Check every migrated byte and every metadata map, outside the timed region.
        for index, (w, code) in enumerate(zip(weights, expected)):
            assert bool((w.view(torch.uint8).reshape(48, -1, 16) == code).all()), (label, rank, index)
        for name in ("physical_to_logical_map", "logical_to_physical_map", "logical_replica_count",
                     "expert_map", "logical_to_rank_dispatch_physical_map"):
            assert torch.equal(getattr(live, name), getattr(new, name)), (label, rank, name)
        result = {"label": label, "rank": rank, "pid": os.getpid(), "profiled": trace,
                  "host_seconds": elapsed, "diagnostic": row,
                  "full_byte_validation": True, "metadata_validation": True, "operation_fingerprint_match": True}
        if profile is not None:
            events = profile.events()
            result["profiler_event_count"] = len(events)
            result["profiler_cuda_event_count"] = sum(e.device_type == torch.autograd.DeviceType.CUDA for e in events)
        results.append(result)
        # Keep each first result even if a later case fails or profiling is unavailable.
        dump(args.output / f"results-rank{rank}.json", results)
        print(json.dumps({k: v for k, v in result.items() if k != "diagnostic"}), flush=True)
        dist.barrier(group=migration_group)
        torch.cuda.synchronize()

    for case in captured["cases"]:
        execute(case, trace=False)
    execute(captured["cases"][1], trace=True)
    dump(args.output / f"completed-rank{rank}.json", {"cases": 4, "success": True})
    dist.destroy_process_group(migration_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
