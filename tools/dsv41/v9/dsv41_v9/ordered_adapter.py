"""TP4 ordered BF16 reduce-scatter of a 1536x5120 tile over the custom-AR IPC context.

The per-chunk summation order comes from `formula.make_codes`, so the result is
bitwise reproducible; `guarded_service_v2` checks it against a fixed-order
reference before enabling it.
"""

import ctypes

import torch

from . import kernels
from .formula import make_codes


class OrderedRS:
    def __init__(self, group):
        self.group = group
        self.comm = group.device_communicator.ca_comm
        assert group.world_size == 4 and not self.comm.disabled
        assert self.comm.fully_connected
        self.codes, _ = make_codes(1536, group.rank_in_group, device="cuda")
        assert self.codes.numel() == 384 * 5120 // 128
        self.library = kernels.load("ordered_rs")
        self.call = self.library.dsv41_rs_ordered
        self.call.argtypes = [ctypes.c_int64] * 7
        self.call.restype = ctypes.c_int
        self.library.dsv41_rs_last_error.restype = ctypes.c_char_p
        self.hits = 0

    def __call__(self, value, output=None):
        assert tuple(value.shape) == (1536, 5120)
        assert value.dtype == torch.bfloat16 and value.is_contiguous()
        assert self.comm.should_custom_rs(value, 0)
        if output is None:
            output = torch.empty((384, 5120), dtype=value.dtype, device=value.device)
        assert tuple(output.shape) == (384, 5120) and output.is_contiguous()
        assert output.dtype == value.dtype and output.device == value.device
        registered = (
            self.comm._IS_CAPTURING
            and torch.cuda.is_current_stream_capturing()
            and self.comm._capture_registered(value)
        )
        pool = self.comm._pool["input"]
        status = self.call(
            self.comm._ptr,
            value.data_ptr(),
            output.data_ptr(),
            self.codes.data_ptr(),
            torch.cuda.current_stream().cuda_stream,
            0 if registered else pool.data_ptr,
            0 if registered else pool.max_size,
        )
        if status:
            raise RuntimeError(self.library.dsv41_rs_last_error().decode())
        self.hits += 1
        return output
