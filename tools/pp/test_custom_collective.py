# SPDX-License-Identifier: MIT
# Standalone isolation test: does the custom (ca_comm) TP collective produce
# CORRECT + DETERMINISTIC results in the tp4 pp2 distributed layout?
#
# Sets up the exact worker distributed config (8 ranks = dp1 x pp2 x tp4) via
# init_pp_aware_dist_env, then runs get_tp_group().all_reduce / all_gather on
# known tensors. If the custom collective is misconfigured under PP (wrong IPC
# device mapping for the TP subgroup), results will be wrong or vary run-to-run.
#
# Run inside the container:
#   HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 tools/pp/test_custom_collective.py

import torch
import torch.multiprocessing as mp


def worker(rank, world_size, tp, pp):
    torch.cuda.set_device(rank)
    from atom.distributed.pp_comm import init_pp_aware_dist_env
    from aiter.dist.parallel_state import get_tp_group

    init_pp_aware_dist_env(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        global_rank=rank,
        world_size=world_size,
        distributed_init_method="tcp://127.0.0.1:29577",
        backend="nccl",
        data_parallel_size=1,
        prefill_context_model_parallel_size=1,
    )
    tpg = get_tp_group()
    rig = tpg.rank_in_group
    tp_ws = tpg.world_size
    expected = float(sum(range(1, tp_ws + 1)))  # each rank contributes (rig+1)

    results = []
    for it in range(6):
        # all-reduce: each rank fills with (rig+1); sum should be 1+2+3+4=10
        x = torch.full((8192,), float(rig + 1), device=rank, dtype=torch.bfloat16)
        y = tpg.all_reduce(x)
        torch.cuda.synchronize()
        v = float(y[0].item())
        allmatch = bool((y == expected).all().item())
        results.append((v, allmatch))

    vals = [r[0] for r in results]
    allmatch = all(r[1] for r in results)
    deterministic = len(set(vals)) == 1
    correct = abs(vals[0] - expected) < 1e-3 and allmatch
    print(
        f"[rank {rank} pp_stage={rank // tp} tp_rig={rig}] "
        f"allreduce vals={vals} expected={expected} "
        f"correct={correct} deterministic={deterministic}",
        flush=True,
    )

    # ---- custom all-gather ----
    ag_ok = ag_det = None
    try:
        ag_vals = []
        for it in range(4):
            x = torch.full((256,), float(rig + 1), device=rank, dtype=torch.bfloat16)
            g = tpg.all_gather(x, dim=0)
            torch.cuda.synchronize()
            # expect blocks of 1,2,3,4 (each rank's value) -> boundary values
            ag_vals.append(tuple(float(g[i * 256].item()) for i in range(tp_ws)))
        exp_ag = tuple(float(i + 1) for i in range(tp_ws))
        ag_ok = ag_vals[0] == exp_ag
        ag_det = len(set(ag_vals)) == 1
        print(
            f"[rank {rank} tp_rig={rig}] all_gather got={ag_vals[0]} exp={exp_ag} "
            f"correct={ag_ok} deterministic={ag_det}",
            flush=True,
        )
    except Exception as e:
        print(f"[rank {rank}] all_gather ERROR {e}", flush=True)


def main():
    world_size = 8
    tp, pp = 4, 2
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(world_size, tp, pp), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
