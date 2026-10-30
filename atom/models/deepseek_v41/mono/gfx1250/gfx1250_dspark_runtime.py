"""Verify the compiled model kernels' resource and actual HIP residency limits."""
import ctypes as C
from pathlib import Path


class with_default_stream:
    """Supply the compile-only default required by the mother's WidthBuilds."""
    def __init__(self, launcher):
        self.launcher = launcher

    @property
    def _mem_cache(self):
        return self.launcher._mem_cache

    def __call__(self, *args, stream=None):
        if stream is None:
            import flydsl.expr as fx
            stream = fx.Stream(None)
        return self.launcher(*args, stream=stream)


def check_residency(kernels):
    import torch
    from atom.mono.runtime.compile import launcher_resources, _code_object
    assert not torch.cuda.is_current_stream_capturing()
    paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if 'libamdhip64.so' in line}
    assert len(paths) == 1
    hip = C.CDLL(paths.pop())
    P = C.c_void_p
    hip.hipModuleLoadData.argtypes = [C.POINTER(P), P]
    hip.hipModuleGetFunction.argtypes = [C.POINTER(P), P, C.c_char_p]
    hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [C.POINTER(C.c_int), P, C.c_int, C.c_size_t]
    hip.hipModuleUnload.argtypes = [P]

    def check(code):
        if code:
            raise RuntimeError(f'HIP residency check returned {code}')

    records = []
    for fn, abi in kernels:
        resources = launcher_resources(fn)
        assert len(resources) == 1
        record = resources[0]
        assert all(record[k] == 0 for k in ('scratch', 'vgpr_spill', 'sgpr_spill'))
        artifacts = list(fn._mem_cache.values())
        assert len(artifacts) == 1
        binary = C.create_string_buffer(_code_object(artifacts[0]._ir_text))
        name = record['name'].removeprefix('atom::')
        symbol = f'_ZN4atom{len(name)}{name}E'
        module, function = P(), P()
        check(hip.hipModuleLoadData(C.byref(module), binary))
        try:
            check(hip.hipModuleGetFunction(C.byref(function), module, symbol.encode()))
            blocks = C.c_int()
            check(hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(C.byref(blocks), function, 256, 0))
            capacity = blocks.value * torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
            assert capacity >= 256, capacity
            records.append(dict(record, capacity=capacity, abi_args=len(abi.names)))
        finally:
            check(hip.hipModuleUnload(module))
    return records
