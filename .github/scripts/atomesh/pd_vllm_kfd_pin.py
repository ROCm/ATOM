"""One peer-memory anchor per HIP allocation, following Mooncake PR #4657.

This adapter uses the image's libibverbs/HIP ABI without replacing its Mooncake
wheel. Data MRs remain owned by Mooncake. Anchors must precede data registration
and outlive every data MR. The diagnostic checks the driver's memory accounting
before this adapter can be used by the model runner.
"""
import ctypes as C
import ctypes.util
import os


class KfdPinAnchors:
    def __init__(self, hca):
        from pd_vllm_triton_probe import mapped_gpu_libraries
        self.hip = C.CDLL(next(p for p in mapped_gpu_libraries() if 'libamdhip64' in p))
        self.hip.hipMemGetAddressRange.argtypes = [C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_void_p]
        self.hip.hipMemGetAddressRange.restype = C.c_int
        for name, args in (
            ('hipMalloc', [C.POINTER(C.c_void_p), C.c_size_t]),
            ('hipFree', [C.c_void_p]),
            ('hipMemGetInfo', [C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]),
        ):
            fn = getattr(self.hip, name)
            fn.argtypes, fn.restype = args, C.c_int
        self.verbs = C.CDLL(ctypes.util.find_library('ibverbs') or 'libibverbs.so.1', use_errno=True)
        for name, args, result in (
            ('ibv_get_device_list', [C.POINTER(C.c_int)], C.POINTER(C.c_void_p)),
            ('ibv_free_device_list', [C.POINTER(C.c_void_p)], None),
            ('ibv_get_device_name', [C.c_void_p], C.c_char_p),
            ('ibv_open_device', [C.c_void_p], C.c_void_p),
            ('ibv_close_device', [C.c_void_p], C.c_int),
            ('ibv_alloc_pd', [C.c_void_p], C.c_void_p),
            ('ibv_dealloc_pd', [C.c_void_p], C.c_int),
            ('ibv_reg_mr', [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int], C.c_void_p),
            ('ibv_dereg_mr', [C.c_void_p], C.c_int),
        ):
            fn = getattr(self.verbs, name)
            fn.argtypes, fn.restype = args, result
        self.context = self.pd = None
        self.anchors = {}
        count = C.c_int()
        devices = self.verbs.ibv_get_device_list(C.byref(count))
        if not devices:
            raise OSError(C.get_errno(), 'ibv_get_device_list')
        try:
            for i in range(count.value):
                if self.verbs.ibv_get_device_name(devices[i]).decode() == hca:
                    self.context = self.verbs.ibv_open_device(devices[i])
                    break
        finally:
            self.verbs.ibv_free_device_list(devices)
        if not self.context:
            raise OSError(C.get_errno(), f'Cannot open anchor HCA {hca}')
        self.pd = self.verbs.ibv_alloc_pd(self.context)
        if not self.pd:
            error = C.get_errno()
            self.verbs.ibv_close_device(self.context)
            self.context = None
            raise OSError(error, 'ibv_alloc_pd')

    def memory_info(self):
        free, total = C.c_size_t(), C.c_size_t()
        rc = self.hip.hipMemGetInfo(C.byref(free), C.byref(total))
        if rc:
            raise RuntimeError(f'hipMemGetInfo failed: {rc}')
        if free.value > total.value:
            raise RuntimeError('KFD free memory exceeds total memory')
        return free.value, total.value

    def probe_driver(self, engine):
        """Use a private allocation: caller buffers can be pinned already."""
        size = 64 * 1024**2
        ptr = C.c_void_p()
        rc = self.hip.hipMalloc(C.byref(ptr), size)
        if rc:
            raise RuntimeError(f'Cannot allocate private KFD probe: {rc}')
        registered = False
        try:
            before, _ = self.memory_info()
            if before < 2 * size:
                raise RuntimeError('Insufficient memory to disambiguate KFD probe')
            rc = engine.batch_register_memory([ptr.value], [size])
            if rc:
                raise RuntimeError(f'KFD probe registration failed: {rc}')
            registered = True
            after, _ = self.memory_info()
            drop = before - after
            if abs(drop) < size // 4:
                needed = False
            elif abs(drop - size) < size // 4:
                needed = True
            else:
                raise RuntimeError(f'Inconclusive KFD probe: {drop} bytes for {size}')
        finally:
            if registered:
                rc = engine.batch_unregister_memory([ptr.value])
                if rc:
                    # Keep the allocation alive if the native MR could remain.
                    raise RuntimeError(f'KFD probe unregistration failed: {rc}')
            rc = self.hip.hipFree(ptr)
            if rc:
                raise RuntimeError(f'KFD probe hipFree failed: {rc}')
        return {'needed': needed, 'probe_bytes': size, 'before_bytes': before,
                'after_bytes': after, 'drop_bytes': drop}

    def acquire(self, address, length):
        base, size = C.c_void_p(), C.c_size_t()
        rc = self.hip.hipMemGetAddressRange(C.byref(base), C.byref(size), C.c_void_p(address))
        if rc:
            raise RuntimeError(f'hipMemGetAddressRange failed: {rc}')
        if not base.value or length <= 0 or not (base.value <= address < address + length <= base.value + size.value):
            raise ValueError('Registration range is outside its HIP allocation')
        if base.value in self.anchors:
            return base.value
        # Exactly one page-sized LOCAL_WRITE MR, not one MR per data chunk/HCA.
        mr = self.verbs.ibv_reg_mr(self.pd, C.c_void_p(base.value), os.sysconf('SC_PAGE_SIZE'), 1)
        if not mr:
            raise OSError(C.get_errno(), f'KFD peer-memory anchor failed for {base.value}')
        self.anchors[base.value] = {'mr': mr, 'allocation_bytes': size.value}
        return base.value

    def close(self):
        for base, row in list(self.anchors.items()):
            rc = self.verbs.ibv_dereg_mr(row['mr'])
            if rc:
                raise OSError(rc, f'ibv_dereg_mr anchor {base}')
            del self.anchors[base]
        if self.pd:
            rc = self.verbs.ibv_dealloc_pd(self.pd)
            if rc:
                raise OSError(rc, 'ibv_dealloc_pd')
            self.pd = None
        if self.context:
            rc = self.verbs.ibv_close_device(self.context)
            if rc:
                raise OSError(rc, 'ibv_close_device')
            self.context = None


def bind_connector(group, hca, evidence):
    """Bind once at the outer connector, before Transfer and Store see the KV.

    The fixed-pool experiment retains anchors and their tensors for the worker
    lifetime. In particular, a Transfer-only unregister must not release an
    anchor while the P Store still has a data MR for the same allocation.
    Process teardown releases the experiment's native contexts and pins.
    """
    from pathlib import Path
    import json
    import torch

    if hasattr(group, '_atomesh_kfd_adapter'):
        return
    connectors = getattr(group, '_connectors', [group])
    transfer = next(c for c in connectors if c.__class__.__name__ == 'MooncakeConnector')
    engine = transfer.connector_worker.engine
    original = group.register_kv_caches
    state = {'status': 'bound'}
    group._atomesh_kfd_adapter = state
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=True)

    def record():
        path = evidence / 'kfd-pin.json'
        path.write_text(json.dumps(state, indent=2) + '\n')
        print('[kfd-pin] ' + json.dumps(state), flush=True)

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from tensors(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from tensors(child)
        else:
            raise TypeError(f'Unsupported KV cache container {type(value)}')

    def register(kv_caches):
        if state['status'] != 'bound':
            raise RuntimeError('KFD adapter requires one fixed KV pool per worker')
        pins = KfdPinAnchors(hca)
        # Retain the pin context before any registration so failures cannot
        # release anchors ahead of partially registered native data MRs.
        group._atomesh_kfd_pins = pins
        group._atomesh_kfd_cache_tensors = kv_caches
        try:
            if os.environ.get('MOONCAKE_DISABLE_HIP_DMABUF', '0') != '0':
                raise RuntimeError('KFD adapter requires the native DMA-BUF data path')
            state['driver_probe'] = pins.probe_driver(engine)
            state['hca'] = hca
            before, total = pins.memory_info()
            reserved_before = torch.cuda.memory_reserved()
            if state['driver_probe']['needed']:
                for tensor in tensors(kv_caches):
                    if not tensor.is_cuda:
                        raise ValueError('Expected device KV storage')
                    storage = tensor.untyped_storage()
                    pins.acquire(storage.data_ptr(), storage.nbytes())
            state.update(status='registering', anchors=len(pins.anchors),
                         allocation_bytes=sum(r['allocation_bytes'] for r in pins.anchors.values()),
                         before_free_bytes=before, total_bytes=total)
            record()
            result = original(kv_caches)
            after, _ = pins.memory_info()
            reserved_delta = torch.cuda.memory_reserved() - reserved_before
            unexplained = before - after - reserved_delta
            state.update(after_free_bytes=after, torch_reserved_delta=reserved_delta,
                         unexplained_drop_bytes=unexplained)
            if abs(unexplained) > 512 * 1024**2:
                raise RuntimeError(f'KFD accounting shifted by {unexplained} untracked bytes')
            state['status'] = 'verified'
            record()
            return result
        except Exception as exc:
            state.update(status='failed', error=str(exc))
            record()
            raise

    group.register_kv_caches = register
