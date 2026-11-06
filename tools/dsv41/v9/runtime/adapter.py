# ruff: noqa: I001
# Keep measured import ordering/namespace; AST parity is verified by prepare.py.
"""One fused node owns graph-private scratch; eager uses original registered pool."""

import ctypes
from pathlib import Path
import torch
from formula import make_codes


class PushJoinRS:
    def __init__(self, group, rows):
        self.group, self.rows = group, rows
        self.padded = (rows + 3) // 4 * 4
        assert 4096 <= rows <= self.padded <= 16384
        self.comm = group.device_communicator.ca_comm
        assert (
            group.world_size == 4
            and not self.comm.disabled
            and self.comm.fully_connected
        )
        assert self.comm._pool["input"].max_size >= self.padded * 5120 * 2
        self.codes, _ = make_codes(self.padded, group.rank_in_group, device="cuda")
        self.graph_scratch = []
        self.lib = ctypes.CDLL(str(Path(__file__).with_name("push_join.so")))
        self.lib.dsv41_push_last_error.restype = ctypes.c_char_p
        self.lib.dsv41_push_join_rs.argtypes = [ctypes.c_int64] * 10
        self.lib.dsv41_push_join_rs.restype = ctypes.c_int

    def __call__(self, routed, shared, routes, output=None):
        assert routes in (1, 6)
        assert routed.dtype == shared.dtype == torch.bfloat16
        assert (
            tuple(shared.shape) == (self.rows, 5120)
            and routed.numel() == self.rows * routes * 5120
        )
        assert routed.is_cuda and shared.is_cuda and routed.device == shared.device
        assert routed.is_contiguous() and shared.is_contiguous()
        if output is None:
            output = shared.new_empty((self.padded // 4, 5120))
        assert (
            output.dtype == shared.dtype
            and output.shape == (self.padded // 4, 5120)
            and output.device == shared.device
            and output.is_contiguous()
        )
        if torch.cuda.is_current_stream_capturing():
            scratch = shared.new_empty((self.padded, 5120))
            assert self.comm._IS_CAPTURING and self.comm._capture_registered(scratch)
            self.graph_scratch.append(scratch)
            ptr = scratch.data_ptr()
        else:
            ptr = self.comm._pool["input"].data_ptr
        status = self.lib.dsv41_push_join_rs(
            self.comm._ptr,
            routed.data_ptr(),
            shared.data_ptr(),
            output.data_ptr(),
            ptr,
            self.codes.data_ptr(),
            self.rows,
            self.padded,
            routes,
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(self.lib.dsv41_push_last_error().decode())
        return output
