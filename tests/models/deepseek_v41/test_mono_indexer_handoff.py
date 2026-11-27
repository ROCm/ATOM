# SPDX-License-Identifier: MIT
"""A selection must wait for score flags beyond the first CTA-sized group."""

import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""),
    reason="gfx950 required",
)


def _build_waiter():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr.typing import Int32, Int64

    from atom.mono.device.ops import kernel_symbol, launcher, rsrc
    from atom.mono.device.sync import Mailbox, sreg
    from atom.mono.dsa_indexer.paged.score import _wait_chunks

    name = kernel_symbol("validate_v41_score_handoff")

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(flags: Int64, signal: Int64, count: Int32):
        mailbox = Mailbox(Int32(0))
        c = {"tid": fx.thread_idx.x, "poll": mailbox.poll, "isf": sreg(flags, 0, "isf")}
        _wait_chunks(c, 0, count, 0, count, 3)
        if fx.thread_idx.x == 0:
            bo.buffer_store(Int32(7), rsrc(signal), 0)

    current_stream = fx.Stream(None)

    @launcher(name)
    def launch(
        flags: Int64, signal: Int64, count: Int32, stream: fx.Stream = current_stream
    ):
        kernel(flags, signal, count).launch(
            grid=(1, 1, 1), block=(512, 1, 1), stream=stream
        )

    return launch


@pytest.mark.parametrize("count", [513, 1280])
def test_waits_for_delayed_tail_flags(count):
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    from atom.mono.runtime.uncached import uncached_bytes

    launch = _build_waiter()
    flags = (
        uncached_bytes(count * 8, torch.device("cuda", torch.cuda.current_device()))
        .view(torch.int32)
        .view(count, 2)
    )
    signal = torch.zeros(1, device="cuda", dtype=torch.int32)
    flags[:, 1].fill_(1)
    _run_compiled(
        launch, flags.data_ptr(), signal.data_ptr(), count, torch.cuda.current_stream()
    )
    torch.cuda.synchronize()
    assert signal.item() == 7
    flags.zero_()
    flags[: min(512, count - 1), 1].fill_(1)
    signal.zero_()
    torch.cuda.synchronize()
    consumer = torch.cuda.Stream()
    producer = torch.cuda.Stream()
    complete = torch.cuda.Event()
    with torch.cuda.stream(consumer):
        _run_compiled(
            launch,
            flags.data_ptr(),
            signal.data_ptr(),
            count,
            torch.cuda.current_stream(),
        )
        complete.record()
    time.sleep(0.02)
    waited = not complete.query()
    # Always release the consumer, including when checking a broken waiter.
    with torch.cuda.stream(producer):
        flags[:, 1].fill_(1)
    complete.synchronize()
    assert waited, f"selection completed before the tail of {count} flags was ready"
    assert signal.item() == 7
