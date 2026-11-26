# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx950 production K1 + native KV-copy mailbox backoff reproducer.

One visible MI355X, two HIP full-CU mask streams (flags0), production K1
256-CTA/512-thread noncooperative launch with synthetic nonzero FP8 weights,
real StepMailboxes, and 267822080 bytes of contiguous native KV unpack from
one actual Slot.tensor (offline registration no-op). This differs from the
historical padded-slot diagnostic. No model checkpoint,
Store service, profiler, sampling change, or synthetic mailbox replacement.

Run from the checkout with matching ROCm PyTorch/FlyDSL/AITER dependencies:
  PYTHONPATH="$PWD" timeout --signal=TERM --kill-after=5s 300s python tools/mono_mailbox_copy_repro.py
For a revert control, set ATOM_MONO_POLL_SLEEP_UNITS=1 (main's polling; see
the .md); do not monkeypatch `_spin` or free owners after a pending GPU operation.
A hang is probabilistic: one completed run is not proof that it cannot recur.
On deadline or any other failure this tool retains the in-flight owners and
waits for the external watchdog (`timeout`); a normal run releases them and
exits. Setup does a real GPU pack/unpack but no NIC registration. GPU event-
window overlap in `side` mode is not physical CU residency.
"""

import argparse
import ctypes
import gc
import hashlib
import inspect
import json
import pathlib
import sys
import time
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repeats", type=int, default=128)
parser.add_argument(
    "--mode", choices=("side", "mono-alone", "copy-alone"), default="side"
)
parser.add_argument("--deadline", type=float, default=15)
parser.add_argument(
    "--setup-only",
    action="store_true",
    help="Run real pool/codec allocation, pack/unpack and byte validation without K1",
)
args = parser.parse_args()
if not 1 <= args.repeats <= 1024 or args.deadline <= 0:
    parser.error("invalid repeats/deadline")


def require(ok, message):
    # Not assert: these calls and checks must also run under python -O.
    if not ok:
        raise RuntimeError(message)


def run_fixture(args):
    from types import SimpleNamespace

    import torch

    from atom.kv_transfer.disaggregation.types import KVTransferRegion
    from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec
    from atom.kv_transfer.offload.mooncake_store.pool import TransferSlotPool
    from atom.mono.device import sync

    def emit(value):
        print(json.dumps(value), flush=True)

    require(torch.cuda.device_count() == 1, "expose exactly one owned GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    require(
        props.multi_processor_count == 256 and "gfx950" in props.gcnArchName,
        "needs a 256-CU gfx950",
    )
    root = pathlib.Path(__file__).resolve().parents[1]
    require(
        pathlib.Path(inspect.getfile(DSV4PageSlotCodec)).resolve().is_relative_to(root),
        "atom is not imported from this checkout",
    )
    torch.manual_seed(0)
    hip = ctypes.CDLL("libamdhip64.so")
    P = ctypes.c_void_p
    U = ctypes.c_uint
    hip.hipExtStreamCreateWithCUMask.argtypes = [
        ctypes.POINTER(P),
        U,
        ctypes.POINTER(U),
    ]
    hip.hipExtStreamGetCUMask.argtypes = [P, U, ctypes.POINTER(U)]
    hip.hipStreamGetFlags.argtypes = [P, ctypes.POINTER(U)]
    hip.hipStreamDestroy.argtypes = [P]
    original_stream = torch.cuda.current_stream()
    owners = []

    def full_stream(role):
        mask = (U * 8)(*[0xFFFFFFFF] * 8)
        raw = P()
        rc = hip.hipExtStreamCreateWithCUMask(ctypes.byref(raw), 8, mask)
        require(rc == 0, f"hipExtStreamCreateWithCUMask failed: {rc}")
        actual = (U * 8)()
        rc = hip.hipExtStreamGetCUMask(raw, 8, actual)
        require(rc == 0, f"hipExtStreamGetCUMask failed: {rc}")
        flags = U()
        rc = hip.hipStreamGetFlags(raw, ctypes.byref(flags))
        require(rc == 0, f"hipStreamGetFlags failed: {rc}")
        require(list(actual) == list(mask) and flags.value == 0, "unexpected stream")
        owners.append(raw)
        emit(
            {
                "stage": "stream",
                "role": role,
                "mask": list(actual),
                "flags": flags.value,
            }
        )
        return torch.cuda.ExternalStream(raw.value, device=device)

    compute = full_stream("compute")
    restore = full_stream("restore")
    torch.cuda.set_stream(compute)
    units = [369664, 256]
    blocks = 1448
    regions = [
        torch.randint(0, 256, (blocks * u,), dtype=torch.uint8, device=device)
        for u in units
    ]
    codec = DSV4PageSlotCodec(
        [
            KVTransferRegion(x.data_ptr(), x.numel(), u, reverse_indexed=False)
            for x, u in zip(regions, units)
        ],
        [],
        num_blocks=blocks,
        num_slots=0,
        device=device,
    )
    pool = TransferSlotPool(
        device=device,
        chunk_bytes=362 * 739840,
        save_bytes=362 * 739840 + 4096,
        load_bytes=362 * 739840 + 4096,
        client=SimpleNamespace(register=lambda *a: None, unregister=lambda *a: None),
    )
    # A single window-sized Slot.tensor is the current pool's public contiguous
    # byte view. This offline probe does not pretend to have the old diagnostic
    # strided_view API or an actual NIC registration.
    slots = pool.acquire("load", 1)
    contig = slots[0].tensor
    require(contig.is_contiguous() and contig.numel() == 267822080, "bad slot")
    srcids = list(range(724))
    dstids = list(range(724, 1448))
    codec.gpu_to_chunk_major_device_buffer(contig, srcids, stream=compute)
    torch.cuda.synchronize()
    if args.setup_only:
        owner = codec.prepare_block_id_groups([dstids], device=device, stream=restore)
        codec.chunk_major_device_buffer_to_gpu_prepared(
            contig, owner, 0, stream=restore
        )
        restore.synchronize()
        require(
            all(
                torch.equal(v[: 724 * u], v[724 * u : 1448 * u])
                for v, u in zip(regions, units)
            ),
            "setup byte validation failed",
        )
        emit(
            {
                "stage": "setup_complete",
                "actual_pool": str(type(pool)),
                "actual_codec": str(type(codec)),
                "bytes_checked": 267822080,
                "prepared_groups": owner.group_count,
            }
        )
        torch.cuda.synchronize()
        pool.release(slots)
        pool.close()
        torch.cuda.set_stream(original_stream)
        return list(owners), hip
    keepers = []
    tag = 10
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
    from atom.mono.runtime.mailboxes import StepMailboxes
    from atom.mono.runtime.peer_memory import PeerBuffer
    from atom.mono.runtime.uncached import uncached_bytes

    for module in (k1, sync):
        require(
            pathlib.Path(module.__file__).resolve().is_relative_to(root),
            f"{module.__name__} is not imported from this checkout",
        )
    emit(
        {
            "stage": "code_identity",
            "mode": args.mode,
            "k1": k1.__file__,
            "sync": sync.__file__,
            # ATOM_MONO_POLL_SLEEP_UNITS as built (None: a checkout without it)
            "poll_sleep_units": getattr(sync, "POLL_SLEEP_UNITS", None),
            "sync_sha256": hashlib.sha256(
                pathlib.Path(sync.__file__).read_bytes()
            ).hexdigest(),
        }
    )
    key = k1.AttnPreBuild(tokens=6, fold=False, tp=4, feeds_index=False)
    launch = k1.build_attn_pre(key)
    emit(
        {
            "stage": "actual_native_factory",
            "launch_api": "ordinary production factory",
            "StepBegin": "ordinary actual StepMailboxes.begin_step",
        }
    )

    def tensor(n, dtype=torch.bfloat16, value=0):
        t = torch.full((n,), value, dtype=dtype, device=device)
        keepers.append(t)
        return t

    # Every K1 argument is a pointer or scalar: the buffers below, or 0 for the
    # index, peer and unused fields of this non-token-parallel K1 (ring_scale is
    # 0 for the bf16 ring, as in the runner). A new ABI field that is neither
    # fails here, not in the watchdog loop. The FP8 weights are filled with 0x38.
    zeros = {
        "pend", "post_in", "comb_in", "ring_scale", "ring_off", "aux", "iq_w",
        "iq_ws", "iw_w", "iq_out", "iqs_out", "iw_out", "sym", "peers", "rank",
        "layer", "tl",
    }  # fmt: skip
    bufs = {}
    for name, n, dtype, value in [
        ("res_in", (20480) * 6, torch.bfloat16, 1),
        ("pre_in", (4) * 6, torch.float32, 0.25),
        ("hc_fn", 24 * 20480, torch.float32, 0),
        ("hc_scale", 3, torch.float32, 1),
        ("hc_base", 24, torch.float32, 0),
        ("attn_w", 5120, torch.bfloat16, 1),
        ("wqkv", 1792 * 5120, torch.uint8, 0x38),
        ("wqkv_s", 65536, torch.float32, 1),
        ("qn_w", 1280, torch.bfloat16, 1),
        ("kvn_w", 512, torch.bfloat16, 1),
        ("wqb", 16 * 512 * 1280, torch.uint8, 0x38),
        ("wqb_s", 65536, torch.float32, 1),
        ("cos", 4096, torch.float32, 1),
        ("sin", 4096, torch.float32, 0),
        ("pos", (1) * 6, torch.int64, 0),
        ("ring", (512) * 6, torch.bfloat16, 0),
        ("ring_rows", (1) * 6, torch.int32, 0),
        ("res_out", (20480) * 6, torch.bfloat16, 0),
        ("post_out", (4) * 6, torch.float32, 0),
        ("comb_out", (16) * 6, torch.float32, 0),
        ("pre_out", (4) * 6, torch.float32, 0),
        ("q_out", (16 * 512) * 6, torch.bfloat16, 0),
        ("normed", (5120) * 6, torch.bfloat16, 0),
        ("qr_out", (1280) * 6, torch.uint8, 0),
        ("qrs_out", (40) * 6, torch.uint8, 0),
        ("scratch", k1.scratch_bytes(6), torch.uint8, 0),
    ]:
        if name == "scratch":
            tensor_value = uncached_bytes(n, device)
            keepers.append(tensor_value)
        else:
            tensor_value = tensor(n, dtype, value)
        if name == "ring_rows":
            tensor_value.copy_(torch.arange(6, dtype=dtype, device=device))
        bufs[name] = tensor_value
    vals = {name: 0 for name in k1.ABI.names}
    for name, buf in bufs.items():
        vals[name] = buf.data_ptr()
    missing = set(k1.ABI.names) - set(bufs) - zeros
    require(not missing, f"K1 ABI fields without a buffer or a zero: {sorted(missing)}")
    peer_buffer = PeerBuffer(16, None, 0, 1, device)
    mailboxes = StepMailboxes(peer_buffer, bufs["scratch"])
    emit(
        {
            "stage": "legal_step_setup",
            "step_begin": "actual StepMailboxes.begin_step",
            "local_peer_ranks": 1,
            "K1_projection_tp": 4,
        }
    )

    def model_once():
        nonlocal tag
        tag += 1
        vals["layer"] = tag
        _run_compiled(launch, *k1.ABI.pack(vals), compute)

    mailboxes.begin_step()
    model_once()
    torch.cuda.synchronize()
    checked = [bufs["q_out"], bufs["ring"]]  # actual q_out and ring KV
    baseline = [t.clone() for t in checked]
    bufs["wqb"][:4096].fill_(0)
    mailboxes.begin_step()
    model_once()
    torch.cuda.synchronize()
    require(
        not torch.equal(checked[0], baseline[0]),
        "input perturbation did not affect Q output",
    )
    bufs["wqb"].fill_(0x38)
    mailboxes.begin_step()
    model_once()
    torch.cuda.synchronize()
    require(
        all(torch.equal(a, b) for a, b in zip(checked, baseline)),
        "restored input did not reproduce the baseline",
    )
    require(torch.count_nonzero(bufs["q_out"]).item() > 0, "empty output")
    require(
        all(
            torch.isfinite(t).all().item() for t in keepers if t.dtype.is_floating_point
        ),
        "non-finite tensor",
    )
    emit(
        {
            "stage": "mono_baseline_complete",
            "real_stage": "V4.1 attn_pre K1",
            "build": str(key),
            "grid": 256,
            "threads": 512,
            "weights": "synthetic nonzero FP8 0x38; not model checkpoint",
            "scratch_bytes": k1.scratch_bytes(6),
        }
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=compute):
        mailboxes.begin_step()
        for _ in range(32):
            model_once()
    torch.cuda.synchronize()
    owner = codec.prepare_block_id_groups([dstids], device=device, stream=restore)
    torch.cuda.synchronize()
    busy_samples = 0
    overlap_samples = 0
    for repeat in range(args.repeats):
        for region, unit in zip(regions, units):
            region[724 * unit :].fill_(91)
        torch.cuda.synchronize()
        mb = torch.cuda.Event(enable_timing=True)
        me = torch.cuda.Event(enable_timing=True)
        cb = torch.cuda.Event(enable_timing=True)
        ce = torch.cuda.Event(enable_timing=True)
        model = args.mode != "copy-alone"
        copy = args.mode != "mono-alone"
        if model:
            mb.record(compute)
            for _ in range(32):
                graph.replay()
            me.record(compute)
        busy = not me.query() if model else None
        busy_samples += int(bool(busy))
        if copy:
            cb.record(restore)
            codec.chunk_major_device_buffer_to_gpu_prepared(
                contig, owner, 0, stream=restore
            )
            ce.record(restore)
        emit(
            {
                "stage": "submitted",
                "repeat": repeat,
                "model_busy_before_copy": busy,
            }
        )
        deadline = time.monotonic() + args.deadline
        while not ((not model or me.query()) and (not copy or ce.query())):
            if time.monotonic() > deadline:
                emit(
                    {
                        "stage": "pending_deadline",
                        "repeat": repeat,
                    }
                )
                while True:
                    time.sleep(1)
            time.sleep(0.05)
        event_overlap = False
        if model and copy:
            event_overlap = max(0.0, mb.elapsed_time(cb)) < min(
                mb.elapsed_time(me), mb.elapsed_time(ce)
            )
            overlap_samples += int(event_overlap)
        if model:
            require(
                all(torch.equal(a, b) for a, b in zip(checked, baseline)),
                "Q/ring changed",
            )
        if copy:
            require(
                all(
                    torch.equal(v[: 724 * u], v[724 * u : 1448 * u])
                    for v, u in zip(regions, units)
                ),
                "native KV bytes differ",
            )
        emit(
            {
                "stage": "complete_round",
                "repeat": repeat,
                "model_ms": mb.elapsed_time(me) if model else None,
                "copy_ms": cb.elapsed_time(ce) if copy else None,
                "bytes_checked": copy,
                "Q_ring_checked": model,
                "busy_precondition": busy,
                "GPU_event_window_overlap": event_overlap,
            }
        )
    if args.mode == "side" and (busy_samples == 0 or overlap_samples == 0):
        raise RuntimeError(
            "Concurrent stress precondition not reached: no busy/overlapping sample"
        )
    emit(
        {
            "stage": "stress_coverage",
            "busy_samples": busy_samples,
            "overlap_samples": overlap_samples,
            "mode": args.mode,
        }
    )
    # The current prepared owner is caller-retained; completion is the GPU
    # fence, not a diagnostic retirement method. Keep it live through the sync.
    torch.cuda.synchronize()
    pool.release(slots)
    pool.close()
    torch.cuda.set_stream(original_stream)
    graph.reset()
    return list(owners), hip


if __name__ == "__main__":
    try:
        handles, hip_owner = run_fixture(args)
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for handle in handles:
            rc = hip_owner.hipStreamDestroy(handle)
            require(rc == 0, f"hipStreamDestroy failed: {rc}")
    except BaseException:  # noqa: BLE001 - retain owners even on interrupts
        # The traceback keeps run_fixture's frames (and all in-flight owners)
        # alive. No manual cleanup or successful completion on this path.
        failure = sys.exc_info()
        traceback.print_exception(*failure)
        print(
            json.dumps(
                {
                    "stage": "failed",
                    "error_type": failure[0].__name__,
                    "error": str(failure[1]),
                }
            ),
            flush=True,
        )
        while True:
            time.sleep(1)
    print(
        json.dumps(
            {
                "stage": "complete",
            }
        ),
        flush=True,
    )
