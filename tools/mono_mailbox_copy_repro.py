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
  timeout --signal=TERM --kill-after=5s 300s python tools/mono_mailbox_copy_repro.py
Use separate fresh processes/caches at the PR head and its parent to verify a
revert; do not monkeypatch `_spin` or free owners after a pending GPU operation.
A hang is probabilistic: one completed run is not proof that it cannot recur.
On deadline this tool retains owners and waits for the external watchdog.
"""
import argparse
import ctypes
import gc
import hashlib
import inspect
import json
import pathlib
import time
import sys
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


def run_fixture(args):
    import torch
    from atom.kv_transfer.disaggregation.types import KVTransferRegion
    from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec
    from atom.kv_transfer.offload.mooncake_store.pool import TransferSlotPool
    from types import SimpleNamespace
    from atom.mono.device import sync

    def emit(value):
        print(json.dumps(value), flush=True)

    assert torch.cuda.device_count() == 1, "expose exactly one owned GPU"
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    assert props.multi_processor_count == 256 and "gfx950" in props.gcnArchName
    root = pathlib.Path(__file__).resolve().parents[1]
    assert (
        pathlib.Path(inspect.getfile(DSV4PageSlotCodec)).resolve().is_relative_to(root)
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
        assert hip.hipExtStreamCreateWithCUMask(ctypes.byref(raw), 8, mask) == 0
        actual = (U * 8)()
        assert hip.hipExtStreamGetCUMask(raw, 8, actual) == 0
        flags = U()
        assert hip.hipStreamGetFlags(raw, ctypes.byref(flags)) == 0
        assert list(actual) == list(mask) and flags.value == 0
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
    assert contig.is_contiguous() and contig.numel() == 267822080
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
        assert all(
            torch.equal(v[: 724 * u], v[724 * u : 1448 * u])
            for v, u in zip(regions, units)
        ), "setup byte validation failed"
        emit(
            {
                "stage": "setup_complete",
                "actual_pool": str(type(pool)),
                "actual_codec": str(type(codec)),
                "bytes_checked": 267822080,
                "prepared_groups": owner.group_count,
                "real_GPU_pack_unpack": True,
                "NIC_registration_tested": False,
            }
        )
        torch.cuda.synchronize()
        pool.release(slots)
        pool.close()
        torch.cuda.set_stream(original_stream)
        return list(owners), hip
    keepers = []
    tag = 10
    from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
    from atom.mono.runtime.uncached import uncached_bytes
    from atom.mono.runtime.peer_memory import PeerBuffer
    from atom.mono.runtime.mailboxes import StepMailboxes
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    assert pathlib.Path(k1.__file__).resolve().is_relative_to(root)
    assert pathlib.Path(sync.__file__).resolve().is_relative_to(root)
    emit(
        {
            "stage": "code_identity_before_K1",
            "k1": k1.__file__,
            "sync": sync.__file__,
            "sync_sha256": hashlib.sha256(
                pathlib.Path(sync.__file__).read_bytes()
            ).hexdigest(),
        }
    )
    key = k1.AttnPreBuild(tokens=6, fold=False, tp=4, feeds_index=False)
    inspect.signature(k1.build_attn_pre).bind(key)
    launch = k1.build_attn_pre(key)
    emit(
        {
            "stage": "actual_native_factory",
            "launch_api": "ordinary production factory",
            "actual_factory": str(inspect.signature(k1.build_attn_pre)),
            "StepBegin": "ordinary actual StepMailboxes.begin_step",
            "real_tensor_shim_run_compiled": True,
        }
    )

    def tensor(n, dtype=torch.bfloat16, value=0):
        t = torch.full((n,), value, dtype=dtype, device=device)
        keepers.append(t)
        return t

    # Explicit ABI fields; unused index and peer pointers are0 in this non-token-
    # parallel K1. Zero FP8 weights are valid in every shuffled block layout.
    vals = {
        name: 0
        for name in (
            "res_in",
            "pend",
            "post_in",
            "comb_in",
            "pre_in",
            "hc_fn",
            "hc_scale",
            "hc_base",
            "attn_w",
            "wqkv",
            "wqkv_s",
            "qn_w",
            "kvn_w",
            "wqb",
            "wqb_s",
            "cos",
            "sin",
            "pos",
            "ring",
            "ring_rows",
            "ring_off",
            "res_out",
            "post_out",
            "comb_out",
            "pre_out",
            "q_out",
            "normed",
            "qr_out",
            "qrs_out",
            "aux",
            "iq_w",
            "iq_ws",
            "iw_w",
            "iq_out",
            "iqs_out",
            "iw_out",
            "scratch",
            "sym",
            "peers",
            "rank",
            "layer",
            "tl",
        )
    }
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
        vals[name] = tensor_value.data_ptr()
    peer_buffer = PeerBuffer(16, None, 0, 1, device)
    mailboxes = StepMailboxes(peer_buffer, keepers[-1])
    emit(
        {
            "stage": "legal_step_setup",
            "scratch_uncached": True,
            "step_begin": "actual StepMailboxes.begin_step",
            "local_peer_ranks": 1,
            "K1_projection_tp": 4,
            "tok_par": False,
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
    checked = [keepers[21], keepers[15]]  # actual q_out and ring KV
    baseline = [t.clone() for t in checked]
    keepers[10][:4096].fill_(0)
    mailboxes.begin_step()
    model_once()
    torch.cuda.synchronize()
    assert not torch.equal(
        checked[0], baseline[0]
    ), "input perturbation did not affect Q output"
    keepers[10].fill_(0x38)
    mailboxes.begin_step()
    model_once()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(checked, baseline))
    assert torch.count_nonzero(keepers[-5]).item() > 0
    assert all(
        torch.isfinite(t).all().item() for t in keepers if t.dtype.is_floating_point
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

    assert pathlib.Path(k1.__file__).resolve().is_relative_to(root)

    emit(
        {
            "stage": "code_identity",
            "k1": k1.__file__,
            "sync": sync.__file__,
            "sync_sha256": hashlib.sha256(
                pathlib.Path(sync.__file__).read_bytes()
            ).hexdigest(),
            "mode": args.mode,
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
                "owners_retained": True,
            }
        )
        deadline = time.monotonic() + args.deadline
        while not ((not model or me.query()) and (not copy or ce.query())):
            if time.monotonic() > deadline:
                emit(
                    {
                        "stage": "pending_deadline",
                        "repeat": repeat,
                        "owners_retained": True,
                        "external_watchdog_required": True,
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
            assert all(
                torch.equal(a, b) for a, b in zip(checked, baseline)
            ), "Q/ring changed"
        if copy:
            assert all(
                torch.equal(v[: 724 * u], v[724 * u : 1448 * u])
                for v, u in zip(regions, units)
            ), "native KV bytes differ"
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
                "GPU_event_window_not_CU_residency": True,
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
            "event_window_not_physical_CU_overlap": True,
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
            assert hip_owner.hipStreamDestroy(handle) == 0
    except BaseException:
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
                    "owners_retained_in_traceback": True,
                    "external_watchdog_required": True,
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
                "normal_exit_required": True,
                "owners_released_after_completion": True,
            }
        ),
        flush=True,
    )
