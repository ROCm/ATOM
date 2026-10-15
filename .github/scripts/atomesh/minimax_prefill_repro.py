#!/usr/bin/env python3
"""CI-only, model-free candidate for the M3 prefill TP/offload stall.

Uses the checkout's unmodified gather definitions. This is a reduction experiment,
not a confirmed reproducer: no target/draft attention, aiter custom allreduce,
CUDA graphs, full LMCache connector, Mooncake, or decode process is included.
A completed run means only that this reduced workload did not hang.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import faulthandler
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import threading
import time

import torch
import torch.distributed as dist

HEADS, DIM, PAGE = 16, 128, 128
GIB = 1024**3
# Actual consecutive prefill batches from run 38664663067, attempt 1.
BATCHES = [
    ([101888, 93568, 118272, 100224, 126208, 117120, 104064],
     [968, 1025, 82, 189, 998, 2495, 27008]),
    ([131072, 146688, 165376, 176000, 171776, 171648, 190720, 164864],
     [10943, 2384, 142, 2412, 336, 3970, 12409, 128]),
]
RANK = int(os.environ["LOCAL_RANK"])
LOG_LOCK = threading.Lock()


def log(stage, **facts):
    with LOG_LOCK:
        print(json.dumps(dict(rank=RANK, stage=stage, t=time.time(), **facts)), flush=True)


def load_gather(out):
    source = Path("/workspace/ATOM/atom/model_ops/base_attention.py")
    text = source.read_text()
    lines = text.splitlines(keepends=True)
    names = {"cp_mha_gather_cache_kernel", "cp_mha_gather_cache"}
    definitions = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            definitions.append("".join(lines[start - 1:node.end_lineno]))
    assert len(definitions) == 2
    code = "import torch\nimport triton\nimport triton.language as tl\n\n" + "\n".join(definitions)
    assert "token_id = tl.program_id(0).to(tl.int64)" in code
    path = out / f"gather_rank{RANK}.py"
    path.write_text(code)
    spec = importlib.util.spec_from_file_location(f"gather_rank{RANK}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    log("gather_source", sha256=hashlib.sha256(code.encode()).hexdigest())
    return module.cp_mha_gather_cache


def metadata(done, new):
    lengths = [a + b for a, b in zip(done, new)]
    pages = [(n + PAGE - 1) // PAGE for n in lengths]
    table = torch.zeros((len(lengths), max(pages)), dtype=torch.int32)
    offset = 0
    for i, n in enumerate(pages):
        table[i, :n] = torch.arange(offset, offset + n)
        offset += n
    # CPU construction avoids a device-side repeat_interleave synchronization.
    ids = torch.cat([torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lengths)])
    cu = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).int()
    return dict(block_tables=table.cuda(), cu_seqlens_kv=cu.cuda(),
                batch_id_per_k_token=ids.cuda(),
                seq_starts=torch.zeros(len(lengths), dtype=torch.int32, device="cuda"),
                total_tokens=sum(lengths)), offset


class SaveWorker:
    """Producer-event host wait -> pack stream -> copy stream -> pinned D2H.

    Uses the real ATOM fused pack kernel with simplified single-segment storage.
    Persistent staging, reuse dependency and the 28,016,640-byte save are kept.
    """
    def __init__(self, mode):
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pack = torch.cuda.Stream()
        self.copy = self.pack if mode == "single" else torch.cuda.Stream()
        self.chunk_bytes = 28016640 // 4
        self.source = torch.full((self.chunk_bytes,), RANK + 1, device="cuda", dtype=torch.uint8)
        self.stage = torch.empty_like(self.source)
        self.host = torch.empty((4, self.chunk_bytes), dtype=torch.uint8, pin_memory=True)
        self.free = None
        path = Path("/workspace/ATOM/atom/kv_transfer/offload/dense/triton_kv_staging.py")
        spec = importlib.util.spec_from_file_location("m3_staging", path)
        self.staging = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.staging
        spec.loader.exec_module(self.staging)
        self.meta = self.staging.prepare_chunk_major_groups(
            [self.source], [self.chunk_bytes // 2], [([2], [0, 1])], self.source.device
        )
        # JIT warmup before overlap; no editable install or full ATOM import.
        self.staging.fused_pack_chunk_major_prepared(self.meta, 0, self.stage)

    def save(self, producer, step):
        torch.cuda.set_device(RANK)
        log("save_wait_producer", step=step)
        producer.synchronize()
        log("save_producer_ready", step=step)
        for i in range(4):
            with torch.cuda.stream(self.pack):
                if self.free is not None:
                    self.pack.wait_event(self.free)
                self.staging.fused_pack_chunk_major_prepared(self.meta, 0, self.stage)
                ready = torch.cuda.Event()
                ready.record()
            with torch.cuda.stream(self.copy):
                self.copy.wait_event(ready)
                self.host[i].copy_(self.stage, non_blocking=True)
                self.free = torch.cuda.Event()
                self.free.record()
        self.copy.synchronize()
        # CPU check does not enqueue extra GPU work or change stream dependencies.
        assert bool((self.host[:, [0, self.chunk_bytes - 1]] == RANK + 1).all())
        log("save_complete", step=step, bytes=28016640)


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=True)
    torch.set_num_threads(1)
    torch.cuda.set_device(RANK)
    props = torch.cuda.get_device_properties(RANK)
    log("environment", torch=torch.__version__, hip=torch.version.hip,
        rccl=dist.is_nccl_available() and torch.cuda.nccl.version(),
        gpu=props.name, total_gib=props.total_memory / GIB,
        mode=args.mode, resident_gib=args.resident_gib,
        nccl_socket_ifname=os.environ.get("NCCL_SOCKET_IFNAME"),
        nccl_socket_family=os.environ.get("NCCL_SOCKET_FAMILY"))
    if args.resident_gib and props.total_memory < (args.resident_gib + 30) * GIB:
        raise RuntimeError("Insufficient HBM for the explicit resident-memory case")
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    assert dist.get_world_size() == 4, "Reproduction requires TP4"
    tp = dist.new_group(list(range(4)), backend="nccl", timeout=timedelta(seconds=60))
    gather = load_gather(out)
    metas, num_blocks = [], 0
    for done, new in BATCHES:
        meta, pages = metadata(done, new)
        metas.append(meta)
        num_blocks = max(num_blocks, pages)
    # Distinct physical pages, FP8 SHUFFLE, per-token dequant, BF16 dense output.
    dtype = torch.float8_e4m3fn
    shape = (num_blocks, HEADS, DIM // 16, PAGE, 16)
    k_cache = torch.ones(shape, dtype=torch.bfloat16, device="cuda").to(dtype)
    v_cache = (-torch.ones(shape, dtype=torch.bfloat16, device="cuda")).to(dtype)
    scales = torch.ones((num_blocks, HEADS, PAGE), device="cuda")
    worker = SaveWorker(args.mode) if args.mode != "none" else None
    filler = []
    if args.resident_gib:
        # Match resident allocation, not the misleading hipMemGetInfo free metric.
        remaining = int(args.resident_gib * GIB) - torch.cuda.memory_allocated()
        while remaining > 0:
            n = min(remaining, GIB)
            filler.append(torch.zeros(n, device="cuda", dtype=torch.uint8))
            remaining -= n
    torch.cuda.synchronize()
    log("resident_ready", allocated_gib=torch.cuda.memory_allocated() / GIB,
        reserved_gib=torch.cuda.memory_reserved() / GIB)
    # Compile once, out of the timed overlap experiment, with the same strides.
    small = metadata([0], [128])[0]
    small_out = torch.empty((128, HEADS, DIM), dtype=torch.bfloat16, device="cuda")
    gather(k_cache, v_cache, small_out, torch.empty_like(small_out),
           k_scales=scales, v_scales=scales, dequant=True,
           kv_cache_layout="SHUFFLE", **small)
    torch.cuda.synchronize()
    dist.barrier(group=tp, device_ids=[RANK])
    log("warmup_complete")
    copy_stream = torch.cuda.Stream()
    previous_copy, pending_save = None, None
    tokens = [torch.full((len(b[0]),), -1, dtype=torch.int64, device="cuda") for b in BATCHES]
    host_tokens = [torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in tokens]
    reduce_tensor = torch.ones(1024, device="cuda")
    start = time.monotonic()
    for step in range(args.cycles * 2):
        idx = step % 2
        # Offload is submitted at the next forward boundary on a host worker.
        producer = torch.cuda.Event()
        producer.record()
        if worker:
            if pending_save is not None:
                pending_save.result(timeout=75)
            pending_save = worker.pool.submit(worker.save, producer, step)
        log("broadcast_enqueue", step=step, numel=tokens[idx].numel(),
            total_kv=metas[idx]["total_tokens"])
        reduce_tensor.fill_(RANK + 1)
        dist.all_reduce(reduce_tensor)  # model-collective stub, separate PG
        if RANK == 0:
            tokens[idx].fill_(step)
        dist.broadcast(tokens[idx], src=0, group=tp)
        forward_done = torch.cuda.Event()
        forward_done.record()
        # Consume the previous output, then defer the current sampled-ID copy.
        if previous_copy is not None:
            previous_copy.synchronize()
            assert bool((host_tokens[1 - idx] == step - 1).all())
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(forward_done)
            host_tokens[idx].copy_(tokens[idx], non_blocking=True)
            previous_copy = torch.cuda.Event()
            previous_copy.record()
        total = metas[idx]["total_tokens"]
        key = torch.empty((total, HEADS, DIM), dtype=torch.bfloat16, device="cuda")
        value = torch.empty_like(key)
        log("gather_enqueue", step=step, total_kv=total,
            dense_kv_gib=key.numel() * 4 / GIB)
        gather(k_cache, v_cache, key, value, k_scales=scales, v_scales=scales,
               dequant=True, kv_cache_layout="SHUFFLE", **metas[idx])
        gathered = torch.cuda.Event()
        gathered.record()
        # Reuse allocator storage on this stream, as production does between steps.
        if step + 1 != args.cycles * 2:
            del key, value
    log("drain")
    previous_copy.synchronize()
    assert bool((host_tokens[idx] == step).all())
    gathered.synchronize()
    if pending_save:
        pending_save.result(timeout=75)
        worker.pool.shutdown(wait=True)
    torch.cuda.synchronize()
    # Check gather samples on both sides of the previous int32 overflow boundary.
    positions = torch.tensor([0, 794108, 1048575, 1048576, total - 1], device="cuda")
    torch.testing.assert_close(key[positions], torch.ones_like(key[positions]), rtol=0, atol=0)
    torch.testing.assert_close(value[positions], -torch.ones_like(value[positions]), rtol=0, atol=0)
    torch.testing.assert_close(reduce_tensor, torch.full_like(reduce_tensor, 10), rtol=0, atol=0)
    dist.barrier(group=tp, device_ids=[RANK])
    result = dict(rank=RANK, result="completed_without_reproducing_hang",
                  mode=args.mode, resident_gib=args.resident_gib,
                  steps=args.cycles * 2, elapsed_seconds=time.monotonic() - start,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB)
    log("complete", **{k: v for k, v in result.items() if k != "rank"})
    (out / f"result_rank{RANK}.json").write_text(json.dumps(result, indent=2) + "\n")
    dist.destroy_process_group(tp)
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["none", "two", "single"], required=True)
    parser.add_argument("--resident-gib", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=64)
    parser.add_argument("--out", required=True)
    run(parser.parse_args())
