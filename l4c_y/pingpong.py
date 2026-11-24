# L4-comm: xGMI latency microbench kernels (one lane of one workgroup).
# mode 0 ping-pong: role 0 stores v into the peer's flag (system-scope atomic xchg, write-through) and spins on its
#        own (local, uncached) flag; role 1 mirrors. n round trips -> one-way = time / (2 n).
# mode 1 remote atomic poll: n dependent sys-scope atomic reads (add 0) of the peer's flag word.
# mode 2 remote volatile 16-byte load chain: n dependent volatile global loads from the peer buffer (sc0 sc1).
# mode 3 local volatile 16-byte load chain on the local uncached buffer.
# mode 5/6 remote/local volatile loads, 64 KB stride over a 32 MB window (translation misses)
# mode 4 remote store + release: n x (16-byte sys store to the peer + s_waitcnt vmcnt(0)) (write ack round trip).
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from amd_kernels.common import buffer_ops
from .common import _flag_store, _flag_read, _spin


def _now():
    return fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def _vload(addr):
    p = fx.inttoptr(fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=16), addr)
    return fx.generic_load(p, dtype=fx.Int32, count=4, volatile=True)


def compile_pingpong(mode: int):
    name = f"l4c_xgmi_latency_m{mode}"

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(arg_mine: fx.Int64, arg_peer: fx.Int64, i32_role: fx.Int32, i32_n: fx.Int32, i32_base: fx.Int32,
               arg_out: fx.Int64):
        tx = fx.thread_idx.x
        if tx == 0:
            t0 = _now()
            acc = fx.Int32(0)
            for i in range(fx.Int32(0), i32_n, fx.Int32(1)):
                v = i32_base + i + 1
                if fx.const_expr(mode == 0):
                    if i32_role == 0:
                        _flag_store(arg_peer, v)
                        _spin(arg_mine, v)
                    else:
                        _spin(arg_mine, v)
                        _flag_store(arg_peer, v)
                elif fx.const_expr(mode == 1):
                    acc = acc + _flag_read(arg_peer + fx.Int64(acc & 0x40))
                elif fx.const_expr(mode == 2):
                    w = _vload(arg_peer + 256 + fx.Int64(acc & 0x40))
                    acc = acc + fx.Int32(w[0])
                elif fx.const_expr(mode == 3):
                    w = _vload(arg_mine + 256 + fx.Int64(acc & 0x40))
                    acc = acc + fx.Int32(w[0])
                elif fx.const_expr(mode == 5):
                    # remote chain, a new 64 KB-strided page each step (32 MB window): translation-miss round trip
                    w = _vload(arg_peer + 65536 + fx.Int64(((i * 64 + 1) % 512) * 65536) + fx.Int64(acc & 0x40))
                    acc = acc + fx.Int32(w[0])
                elif fx.const_expr(mode == 6):
                    w = _vload(arg_mine + 65536 + fx.Int64(((i * 64 + 1) % 512) * 65536) + fx.Int64(acc & 0x40))
                    acc = acc + fx.Int32(w[0])
                else:
                    sr = buffer_ops.create_buffer_resource_from_addr(arg_peer + 512, num_records_bytes=256)
                    buffer_ops.buffer_store(Vec.from_elements([v, v, v, v], fx.Int32), sr, 0, cache_modifier=0x11)
                    fx.rocdl.s_waitcnt(vmcnt=0)
            t1 = _now()
            orr = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=64)
            d = t1 - t0
            buffer_ops.buffer_store(Vec.from_elements([fx.Int32(d & fx.Int64(0xFFFFFFFF)), acc], fx.Int32), orr, 0)

    @flyc.jit
    def launch(arg_mine: fx.Int64, arg_peer: fx.Int64, i32_role: fx.Int32, i32_n: fx.Int32, i32_base: fx.Int32,
               arg_out: fx.Int64, stream: fx.Stream):
        kernel(arg_mine, arg_peer, i32_role, i32_n, i32_base, arg_out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch


def compile_rowread(peers: int = 3, sys_cm: bool = True):
    """One 768-thread workgroup (the AR kernels' shape) reads, per iteration, one 12 KB row (16 B per lane) from each of
    ``peers`` peer buffers (row i % 64, sc0 sc1 like the reducer's peer reads), folds them into a dependency, then
    s_barrier; n iterations timed by lane 0 with s_memrealtime. Models the reducer's 3-peer read phase."""
    name = f"l4c_rowread_p{peers}" + ("_sys" if sys_cm else "")

    @flyc.kernel(name=name, known_block_size=[768, 1, 1])
    def kernel(arg_p0: fx.Int64, arg_p1: fx.Int64, arg_p2: fx.Int64, i32_n: fx.Int32, arg_out: fx.Int64):
        tx = fx.thread_idx.x
        t0 = _now()
        acc = fx.Int32(0)
        for i in range(fx.Int32(0), i32_n, fx.Int32(1)):
            row = fx.Int64(i % 64) * 12288
            for k, base in enumerate((arg_p0, arg_p1, arg_p2)[:peers]):
                rs = buffer_ops.create_buffer_resource_from_addr(base + row, num_records_bytes=12288)
                v = Vec(buffer_ops.buffer_load(rs, tx * 4, vec_width=4, dtype=fx.Int32,
                                               cache_modifier=0x11 if sys_cm else 0))
                acc = acc + fx.Int32(v[0]) + fx.Int32(v[3])
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
        t1 = _now()
        if tx == 0:
            orr = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=64)
            d = t1 - t0
            buffer_ops.buffer_store(Vec.from_elements([fx.Int32(d & fx.Int64(0xFFFFFFFF)), acc], fx.Int32), orr, 0)

    @flyc.jit
    def launch(arg_p0: fx.Int64, arg_p1: fx.Int64, arg_p2: fx.Int64, i32_n: fx.Int32, arg_out: fx.Int64,
               stream: fx.Stream):
        kernel(arg_p0, arg_p1, arg_p2, i32_n, arg_out).launch(grid=(1, 1, 1), block=(768, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
