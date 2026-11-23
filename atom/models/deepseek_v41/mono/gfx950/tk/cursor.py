"""One-writer DEVICE cursor RMW; callers preserve the existing stream contract."""
import torch
import triton
import triton.language as tl


@triton.jit
def _cursor_device_kernel(Cursor, Slots, Anchors, Cu, STRIDE: tl.constexpr, SIZE: tl.constexpr):
    slot = tl.load(Slots).to(tl.int64)
    slot = tl.where(slot < 0, slot+SIZE, slot)
    tl.device_assert((slot >= 0) & (slot < SIZE), 'STATE slot out of range')
    anchor = tl.load(Anchors).to(tl.int64)
    start = tl.load(Cu).to(tl.int64)
    ptr = Cursor+slot*STRIDE
    old = tl.load(ptr)
    tl.store(ptr, old+anchor-start+1)


def supported(cache, step, anchors):
    return (step.scheduled_bs == 1 and cache.cursor.dtype == torch.int64
            and step.slots.dtype == anchors.dtype == step.cu_seqlens_q.dtype == torch.int32
            and cache.cursor.is_cuda and cache.cursor.device == step.slots.device
            == anchors.device == step.cu_seqlens_q.device and cache.cursor.stride(1) == 1)


def commit_device(cache, step, anchors):
    assert supported(cache, step, anchors)
    result = _cursor_device_kernel[(1,)](cache.cursor, step.slots, anchors, step.cu_seqlens_q,
        STRIDE=cache.cursor.stride(0), SIZE=cache.cursor.shape[0],num_warps=1,debug=True)
    cache.pending = None
    return result
