"""Small, stage-distinct GPU code loads to locate full-model HIP loader failures."""
import ctypes
import json
import traceback
from pathlib import Path


def direct_hip_load(binary, name):
    from pd_vllm_triton_probe import mapped_gpu_libraries
    path = next(p for p in mapped_gpu_libraries() if 'libamdhip64' in p)
    lib = ctypes.CDLL(path)
    lib.hipPeekAtLastError.restype = ctypes.c_int
    lib.hipModuleLoadDataEx.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                       ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    lib.hipModuleLoadDataEx.restype = ctypes.c_int
    lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    lib.hipModuleGetFunction.restype = ctypes.c_int
    lib.hipModuleUnload.argtypes = [ctypes.c_void_p]
    lib.hipModuleUnload.restype = ctypes.c_int
    lib.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    device = ctypes.c_int(-1)
    lib.hipGetDevice(ctypes.byref(device))
    result = {'library': path, 'device': device.value, 'prior_error': lib.hipPeekAtLastError()}
    blob = ctypes.create_string_buffer(binary)
    module = ctypes.c_void_p()
    rc = lib.hipModuleLoadDataEx(ctypes.byref(module), blob, 0, None, None)
    result['hipModuleLoadDataEx'] = rc
    if rc == 0:
        function = ctypes.c_void_p()
        result['hipModuleGetFunction'] = lib.hipModuleGetFunction(ctypes.byref(function), module, name.encode())
        result['hipModuleUnload'] = lib.hipModuleUnload(module)
    return result


def probe(out, stage, tag):
    import torch
    import triton
    import triton.language as tl
    from pd_vllm_triton_probe import save_kernel
    out = Path(out) / ('stage-probe-' + stage)
    out.mkdir(parents=True, exist_ok=True)
    result = {'stage': stage, 'status': 'running', 'device': torch.cuda.current_device()}
    compiled = None
    try:
        @triton.jit
        def stage_write(output, TAG: tl.constexpr):
            offsets = tl.arange(0, 128)
            tl.store(output + offsets, offsets + TAG)
        output = torch.empty(128, dtype=torch.int32, device='cuda')
        # TAG changes the generated code, so each stage loads a fresh module.
        compiled = stage_write.warmup(output, TAG=tag, grid=(1,))
        save_kernel(compiled, out / 'kernel')
        stage_write[(1,)](output, TAG=tag)
        torch.cuda.synchronize()
        assert torch.equal(output.cpu(), torch.arange(128, dtype=torch.int32) + tag)
        result['status'] = 'success'
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc())
        if compiled is not None:
            try:
                result['direct_hip'] = direct_hip_load(compiled.asm['hsaco'], compiled.metadata.name)
            except Exception:
                result['direct_hip_error'] = traceback.format_exc()
    try:
        result['free_bytes'], result['total_bytes'] = torch.cuda.mem_get_info()
    except Exception:
        pass
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('[stage-probe] ' + json.dumps(result), flush=True)
