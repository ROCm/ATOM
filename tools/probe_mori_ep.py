"""Round-trip MoRI's dispatch/combine kernels, one or two groups per GPU.

``tools/probe_mori_shmem.py`` showed that two MoRI symmetric heaps can share a
GPU without aliasing, so the corruption seen under rapidserve+DPA+EP is not the
memory -- it is the kernels running over it. This probe exercises those kernels
directly.

The suspected mechanism is recorded in ATOM's own source
(``mori_prepare_finalize.py:_get_dispatch_config``): mori's IntraNode
dispatch/combine use a hand-rolled grid-wide barrier
(``CrossDeviceBarrierIntraNodeKernel``) that spins until *all* ``gridDim.x``
blocks have arrived, so every block must be co-resident. ATOM caps ``block_num``
at the device CU count to guarantee that -- but the cap assumes the process owns
the whole GPU. Rapidserve puts two EP processes on every GPU, and they launch
128 (prefill) + 64 (decode) barrier-synchronised blocks against one CU pool.

The test drives dispatch then combine with identity "experts" -- dispatch's own
output is handed straight back to combine -- over routing spread across *all*
ranks, because the failure only shows up for tokens that actually cross ranks (a
repeated prompt, whose routing is degenerate, comes back correct under
rapidserve).

Correctness is judged by *comparison, not by an analytic expectation*: run one
group alone with ``--save`` to record a reference, then re-run the same seeds
under contention with ``--compare``. This deliberately avoids assuming anything
about mori's combine semantics (whether it applies top-k weights or plain-sums
the copies); the question is only whether sharing a GPU changes the answer.

Record the reference, one group alone on the GPUs::

    torchrun --nproc-per-node=8 --master-port=29720 tools/probe_mori_ep.py \\
        --group-id 0 --role prefill --save /tmp/ref_prefill

Then replay the same seeds with a second group sharing every GPU, as rapidserve
does -- prefill-shaped and decode-shaped launches at once::

    torchrun --nproc-per-node=8 --master-port=29720 tools/probe_mori_ep.py \\
        --group-id 0 --role prefill --compare /tmp/ref_prefill &
    torchrun --nproc-per-node=8 --master-port=29721 tools/probe_mori_ep.py \\
        --group-id 1 --role decode &
    wait

``--block-num`` / ``--warp-num`` override the launch geometry, so if contention
is the cause, lowering both groups until the total fits the CU pool should make
the errors go away -- that would be the fix rather than a bypass.

A hang is a result too: the grid-wide barrier failing to complete is the
deadlock that ATOM's CU cap exists to prevent. Run under ``timeout``.
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist


def _routing(num_tokens, topk, num_experts, device, seed):
    """Top-k ids spread over every rank's experts, with weights summing to 1."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.empty(num_tokens, topk, dtype=torch.int32)
    for t in range(num_tokens):
        # randperm gives distinct experts per token, spanning all ranks.
        ids[t] = torch.randperm(num_experts, generator=gen)[:topk].to(torch.int32)
    weights = torch.full((num_tokens, topk), 1.0 / topk, dtype=torch.float32)
    return ids.to(device), weights.to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-id", type=int, default=0)
    ap.add_argument("--group-name", default="mori")
    ap.add_argument(
        "--role",
        choices=("prefill", "decode"),
        default="prefill",
        help="picks ATOM's launch geometry: prefill=128 blocks/16 warps, "
        "decode=64 blocks/4 warps",
    )
    ap.add_argument("--block-num", type=int, default=None, help="override block_num")
    ap.add_argument("--warp-num", type=int, default=None, help="override warps")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--num-tokens", type=int, default=None)
    ap.add_argument("--hidden-dim", type=int, default=7168)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--experts-per-rank", type=int, default=48)
    ap.add_argument("--max-tokens-per-rank", type=int, default=2048)
    ap.add_argument("--save", default=None, help="write per-rank reference prefix")
    ap.add_argument("--compare", default=None, help="check against a --save prefix")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="heartbeat every N iters; distinguishes a slow run from the "
        "grid-wide-barrier deadlock",
    )
    ap.add_argument(
        "--start-at",
        type=float,
        default=None,
        help="epoch seconds to begin the kernel loop; give both groups the "
        "same value so their loops actually overlap (mori init takes tens of "
        "seconds, so back-to-back launches never contend)",
    )
    args = ap.parse_args()

    group = args.group_id
    pe = int(os.environ["RANK"])
    pes = int(os.environ["WORLD_SIZE"])
    device = int(os.environ.get("LOCAL_RANK", pe))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="gloo")

    import mori
    from mori import shmem

    world_group = dist.distributed_c10d._get_default_group()
    torch._C._distributed_c10d._register_process_group(args.group_name, world_group)
    shmem.shmem_torch_process_group_init(args.group_name)

    # ATOM's geometry, from mori_prepare_finalize._get_dispatch_config.
    cu = torch.cuda.get_device_properties(device).multi_processor_count
    if args.role == "prefill":
        block_num, warps = min(128, cu), 16
    else:
        block_num, warps = min(64, cu), 4
    if args.block_num is not None:
        block_num = args.block_num
    if args.warp_num is not None:
        warps = args.warp_num

    num_tokens = args.num_tokens
    if num_tokens is None:
        num_tokens = 1024 if args.role == "prefill" else 16

    num_experts = pes * args.experts_per_rank
    tag = f"[{args.role} group {group} pe {pe} gpu {device}]"
    if pe == 0:
        print(
            f"{tag} cu={cu} block_num={block_num} warps={warps} "
            f"tokens={num_tokens} hidden={args.hidden_dim} topk={args.topk} "
            f"experts={num_experts}",
            flush=True,
        )

    cfg = mori.ops.EpDispatchCombineConfig(
        rank=pe,
        world_size=pes,
        data_type=torch.bfloat16,
        hidden_dim=args.hidden_dim,
        scale_dim=0,
        scale_type_size=torch.float32.itemsize,
        max_token_type_size=torch.bfloat16.itemsize,
        max_num_inp_token_per_rank=args.max_tokens_per_rank,
        num_experts_per_rank=args.experts_per_rank,
        num_experts_per_token=args.topk,
        warp_num_per_block=warps,
        block_num=block_num,
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
        gpu_per_node=pes,
        rdma_block_num=0,
    )
    op = mori.ops.EpDispatchCombineOp(cfg)

    dev = torch.device("cuda", device)
    reference = None
    if args.compare:
        reference = torch.load(f"{args.compare}_pe{pe}.pt")
        if len(reference) != args.iters:
            print(
                f"{tag} reference has {len(reference)} iters, running "
                f"{args.iters} -- rerun --save with matching --iters",
                flush=True,
            )
    captured = []
    worst = 0.0
    bad_iters = []
    # Timestamps let the caller confirm the two groups' kernel loops really
    # overlapped -- a contention test that ran back-to-back proves nothing.
    dist.barrier()
    if args.start_at is not None:
        late = time.time() - args.start_at
        if late > 0 and pe == 0:
            print(f"{tag} WARNING: ready {late:.1f}s after --start-at", flush=True)
        while time.time() < args.start_at:
            time.sleep(0.01)
    t_start = time.time()
    for it in range(args.iters):
        if args.progress_every and pe == 0 and it % args.progress_every == 0:
            print(
                f"{tag} iter {it}/{args.iters} at {time.time() - t_start:.1f}s",
                flush=True,
            )
        seed = 10_000 * (group + 1) + 100 * pe + it
        x = torch.randn(
            num_tokens,
            args.hidden_dim,
            dtype=torch.bfloat16,
            device=dev,
            generator=torch.Generator(device=dev).manual_seed(seed),
        )
        ids, weights = _routing(num_tokens, args.topk, num_experts, dev, seed)

        try:
            dispatched, disp_w, _disp_scale, disp_ids, _n = op.dispatch(
                x, weights, None, ids, block_num, warps
            )
            # Identity experts: hand dispatch's own output back to combine.
            # combine takes the *original* topk ids, as ATOM's finalize does.
            out = op.combine(dispatched, None, ids, block_num, warps)[0]
        except Exception as exc:  # noqa: BLE001 - report, do not abort the group
            print(f"{tag} iter {it}: kernel raised {type(exc).__name__}: {exc}",
                  flush=True)
            bad_iters.append(it)
            break

        got = out[:num_tokens].to(torch.float32)
        if not torch.isfinite(got).all():
            bad_iters.append(it)
        # Per-token signature: cheap to store, still sensitive to a single
        # mis-routed or dropped token copy.
        sig = got.sum(dim=1).cpu()
        captured.append(sig)
        if reference is not None and it < len(reference):
            err = (sig - reference[it]).abs().max().item()
            worst = max(worst, err)
            if err > 1e-3 and it not in bad_iters:
                bad_iters.append(it)
        del dispatched, disp_w, disp_ids, out

    torch.cuda.synchronize()
    t_end = time.time()
    if pe == 0:
        print(
            f"{tag} LOOP window {t_start:.2f} -> {t_end:.2f} "
            f"({t_end - t_start:.1f}s)",
            flush=True,
        )
    if args.save:
        torch.save(captured, f"{args.save}_pe{pe}.pt")
        print(f"{tag} saved {len(captured)} iters to {args.save}_pe{pe}.pt",
              flush=True)
    if not args.compare:
        pass
    elif bad_iters:
        print(
            f"{tag} FAIL: {len(bad_iters)}/{args.iters} iters differ from the "
            f"solo reference (worst {worst:.4g}), first at iter {bad_iters[0]}",
            flush=True,
        )
    else:
        print(
            f"{tag} ok: {args.iters}/{args.iters} iters match the solo "
            f"reference (worst {worst:.4g})",
            flush=True,
        )

    failed = torch.tensor([1 if bad_iters else 0])
    dist.all_reduce(failed)
    dist.barrier()
    n = int(failed.item())
    if pe == 0:
        verdict = f"{n}/{pes} ranks bad" if n else f"all {pes} ranks clean"
        print(f"\n=== {args.role} group {group}: {verdict} ===", flush=True)
    dist.destroy_process_group()
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
