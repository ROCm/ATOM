# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""DSV4 pipeline-parallel communication integration."""

from __future__ import annotations

from typing import Any


def install_dsv4_pipeline_shape_exchange() -> None:
    """Teach Megatron's dynamic P2P exchange about DSV4's HC dimension."""
    import torch
    from megatron.core.pipeline_parallel import p2p_communication

    communicator = p2p_communication.P2PCommunicator
    if getattr(communicator, "_lumen_dsv4_shape_exchange", False):
        return

    original = communicator._communicate_shapes

    def _communicate_shapes(
        self: Any,
        tensor_send_next: Any,
        tensor_send_prev: Any,
        recv_prev: bool,
        recv_next: bool,
    ) -> tuple[list[int], list[int]]:
        if not getattr(self.config, "dsv4_mode", False):
            return original(
                self, tensor_send_next, tensor_send_prev, recv_prev, recv_next
            )

        # mHC keeps its stream axis across PP boundaries:
        # [sequence, batch, hc, hidden].
        shape_rank = 4
        device = torch.cuda.current_device()
        recv_prev_shape_tensor = (
            torch.empty((shape_rank,), device=device, dtype=torch.int64)
            if recv_prev
            else None
        )
        recv_next_shape_tensor = (
            torch.empty((shape_rank,), device=device, dtype=torch.int64)
            if recv_next
            else None
        )

        def make_shape_tensor(tensor: Any, direction: str) -> Any:
            if tensor is None:
                return None
            if tensor.dim() != shape_rank:
                raise RuntimeError(
                    f"DSV4 PP tensor sent {direction} must be 4-D "
                    f"[sequence, batch, hc, hidden], got {tuple(tensor.shape)}"
                )
            return torch.tensor(tensor.size(), device=device, dtype=torch.int64)

        send_prev_shape_tensor = make_shape_tensor(tensor_send_prev, "backward")
        send_next_shape_tensor = make_shape_tensor(tensor_send_next, "forward")

        if self.config.use_ring_exchange_p2p:
            torch.distributed.ring_exchange(
                tensor_send_prev=send_prev_shape_tensor,
                tensor_recv_prev=recv_prev_shape_tensor,
                tensor_send_next=send_next_shape_tensor,
                tensor_recv_next=recv_next_shape_tensor,
                group=self.pp_group,
            )
        else:
            # Keep shape metadata ordered like payload P2P. Megatron's default
            # batch_isend_irecv shape path can deadlock for pipelines with 4+
            # stages.
            reqs = p2p_communication._p2p_ops(
                tensor_send_prev=send_prev_shape_tensor,
                tensor_recv_prev=recv_prev_shape_tensor,
                tensor_send_next=send_next_shape_tensor,
                tensor_recv_next=recv_next_shape_tensor,
                group=self.pp_group,
                prev_pipeline_rank=self.prev_rank,
                next_pipeline_rank=self.next_rank,
            )
            for req in reqs.values():
                req.wait()

        recv_prev_shape = (
            recv_prev_shape_tensor.tolist()
            if recv_prev_shape_tensor is not None
            else [0] * shape_rank
        )
        recv_next_shape = (
            recv_next_shape_tensor.tolist()
            if recv_next_shape_tensor is not None
            else [0] * shape_rank
        )
        return recv_prev_shape, recv_next_shape

    communicator._communicate_shapes = _communicate_shapes
    communicator._lumen_dsv4_shape_exchange = True
