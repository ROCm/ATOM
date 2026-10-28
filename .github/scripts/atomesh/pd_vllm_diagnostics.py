"""Read-only runtime evidence; preserve the original model/connector exception."""
import json
import os
from pathlib import Path
import traceback


def record_runtime(out, stage):
    try:
        import torch
        import triton
        from triton.runtime import driver
        from pd_vllm_triton_probe import mapped_gpu_libraries
        index = torch.cuda.current_device()
        data = {'stage': stage, 'pid': os.getpid(), 'uname': tuple(os.uname()),
                'device': index, 'device_name': torch.cuda.get_device_name(index),
                'gcnArchName': torch.cuda.get_device_properties(index).gcnArchName,
                'triton_target': vars(driver.active.get_current_target()),
                'torch_hip_version': torch.version.hip, 'triton_version': triton.__version__,
                'mapped_gpu_libraries': mapped_gpu_libraries(),
                'env': {k: os.environ.get(k) for k in ('TRITON_CACHE_DIR', 'HOME', 'XDG_CACHE_HOME',
                    'LD_LIBRARY_PATH', 'LD_PRELOAD', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
                    'HSA_OVERRIDE_GFX_VERSION', 'HSA_XNACK')}}
        module = Path('/sys/module/amdgpu/version')
        data['host_amdgpu_module_version'] = module.read_text().strip() if module.exists() else None
        out.mkdir(parents=True, exist_ok=True)
        (out / (stage + '.json')).write_text(json.dumps(data, indent=2) + '\n')
    except Exception as exc:
        print(f'[runtime-evidence] {stage}: {exc}', flush=True)


def record_failure(out, error):
    try:
        from pd_vllm_triton_probe import save_kernel
        out.mkdir(parents=True, exist_ok=True)
        (out / 'failure.txt').write_text(''.join(traceback.format_exception(error)))
        # The failing load_binary traceback retains the exact CompiledKernel,
        # including its binary even when initialization of HIP handles fails.
        tb = error.__traceback__
        saved = set()
        while tb is not None:
            frame = tb.tb_frame
            if 'triton' in frame.f_code.co_filename:
                compiled = frame.f_locals.get('self')
                if compiled is not None and id(compiled) not in saved and isinstance(getattr(compiled, 'asm', None), dict):
                    name = str(getattr(getattr(compiled, 'metadata', None), 'name', 'compiled_kernel'))
                    save_kernel(compiled, out / name)
                    from pd_vllm_stage_probe import direct_hip_load
                    (out / (name + '-hip-load.json')).write_text(json.dumps(
                        direct_hip_load(compiled.asm['hsaco'], name), indent=2) + '\n')
                    saved.add(id(compiled))
            tb = tb.tb_next
        record_runtime(out, 'at-failure')
    except Exception as exc:
        print(f'[runtime-evidence] failure capture: {exc}', flush=True)
