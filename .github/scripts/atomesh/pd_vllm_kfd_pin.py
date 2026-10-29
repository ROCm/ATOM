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
