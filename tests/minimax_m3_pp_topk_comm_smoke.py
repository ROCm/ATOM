# SPDX-License-Identifier: MIT
"""Real AITER TP2 x PP2 transport check.

Run on four GPUs:
    torchrun --standalone --nproc-per-node=4 tests/minimax_m3_pp_topk_comm_smoke.py
"""

import os

import torch
import torch.distributed as dist


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from aiter.dist.parallel_state import (
        destroy_model_parallel,
        get_pp_group,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from aiter.ops.communication import set_custom_all_reduce

    from atom.distributed.pp_comm import (
        async_send_intermediate_tensors,
        commit_pp_send_work,
        recv_intermediate_tensors,
        send_intermediate_tensors,
    )
    from atom.model_ops.minimax_m3.pp_topk import PP_TOPK_INDICES, PP_TOPK_METADATA
    from atom.models.utils import IntermediateTensors

    # Exercise the real tensor-dict protocol without allocating custom-AR pools.
    set_custom_all_reduce(False)
    init_distributed_environment()
    assert dist.get_world_size() == 4
    initialize_model_parallel(2, 2)
    pp, tp = get_pp_group(), get_tp_group()
    for gather in ("0", "1"):
        os.environ["ATOM_PP_SEND_ALLGATHER"] = gather
        for asynchronous in (False, True):
            for carry in (True, False, True):
                hidden = torch.arange(96, device="cuda").reshape(6, 16)
                indices = (
                    torch.arange(48, dtype=torch.int32, device="cuda").reshape(1, 3, 16)
                    + 1000 * tp.rank_in_group
                )
                metadata = {
                    "source": (12, 12),
                    "signature": ("prefill", torch.bfloat16),
                }
                payload = {
                    "hidden_states": hidden.clone(),
                    "residual": hidden.clone() + 1,
                    "draft_aux_7": hidden.clone() + 2,
                }
                if carry:
                    payload.update(
                        {PP_TOPK_INDICES: indices.clone(), PP_TOPK_METADATA: metadata}
                    )
                if pp.rank_in_group == 0:
                    out = IntermediateTensors(payload)
                    if asynchronous:
                        works = async_send_intermediate_tensors(out)
                        for value in payload.values():
                            if isinstance(value, torch.Tensor):
                                value.fill_(-99)
                        commit_pp_send_work(works)
                    else:
                        send_intermediate_tensors(out)
                else:
                    received = recv_intermediate_tensors().tensors
                    torch.testing.assert_close(received["hidden_states"], hidden)
                    torch.testing.assert_close(received["residual"], hidden + 1)
                    torch.testing.assert_close(received["draft_aux_7"], hidden + 2)
                    if carry:
                        torch.testing.assert_close(received[PP_TOPK_INDICES], indices)
                        assert received[PP_TOPK_METADATA] == metadata
                    else:
                        assert PP_TOPK_INDICES not in received
                        assert PP_TOPK_METADATA not in received
                dist.barrier()
    if dist.get_rank() == 0:
        print(
            "PASS: TP2 x PP2, sync/async, all-gather off/on, optional top-k + draft aux",
            flush=True,
        )
    destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
