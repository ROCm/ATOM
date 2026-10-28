"""Prepare CPU Store memory after model load, before native vLLM KV initialization."""
import json
import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker

from pd_vllm_memory import prepare_after_load, write_json
from pd_vllm_diagnostics import record_runtime, record_failure


def mooncake_hca(local_rank, environ=os.environ):
    """Match the successful CI's physical GPU to RDMA rail mapping."""
    visible = (environ.get('ROCR_VISIBLE_DEVICES') or environ.get('HIP_VISIBLE_DEVICES')
               or environ.get('CUDA_VISIBLE_DEVICES') or '')
    devices = [d for d in visible.split(',') if d]
    physical_gpu = int(devices[local_rank]) if devices else local_rank
    return f'rdma{physical_gpu}'


class StoreMemoryWorker(Worker):
    def _diagnostic_dir(self):
        return (Path(os.environ['ATOMESH_VLLM_MEMORY_DIR']) / 'runtime-evidence'
                / f"{os.environ['ATOMESH_VLLM_ROLE']}-rank-{self.local_rank}")

    def init_device(self):
        try:
            result = super().init_device()
        except Exception as exc:
            record_failure(self._diagnostic_dir(), exc)
            raise
        record_runtime(self._diagnostic_dir(), 'after-device-init')
        return result

    def load_model(self, *, load_dummy_weights=False):
        try:
            result = super().load_model(load_dummy_weights=load_dummy_weights)
        except Exception as exc:
            record_failure(self._diagnostic_dir(), exc)
            raise
        record_runtime(self._diagnostic_dir(), 'after-model-load')
        return result

    def compile_or_warm_up_model(self):
        evidence = self._diagnostic_dir()
        record_runtime(evidence, 'before-warmup')
        try:
            result = super().compile_or_warm_up_model()
        except Exception as exc:
            record_failure(evidence, exc)
            raise
        record_runtime(evidence, 'after-warmup')
        return result

    def initialize_from_config(self, kv_cache_config):
        if self.parallel_config.tensor_parallel_size != 4:
            raise ValueError('Store memory preparation requires TP4 for both roles')
        role = os.environ['ATOMESH_VLLM_ROLE']
        out = Path(os.environ['ATOMESH_VLLM_MEMORY_DIR'])
        precision = {
            'role': role, 'local_rank': self.local_rank,
            'has_mamba_layers': kv_cache_config.has_mamba_layers,
            'has_mixed_precision_kv_cache': kv_cache_config.has_mixed_precision_kv_cache,
            'needs_kv_cache_zeroing': kv_cache_config.needs_kv_cache_zeroing,
        }
        write_json(out / f'{role}-rank-{self.local_rank}-kv-precision.json', precision)
        print(f'[kv-precision] {json.dumps(precision)}', flush=True)
        hca = mooncake_hca(self.local_rank)
        if not (Path('/sys/class/infiniband') / hca).exists():
            raise RuntimeError(f'Missing physical GPU RDMA rail: {hca}')
        os.environ['MC_TE_FILTERS'] = hca
        write_json(out / f'{role}-rank-{self.local_rank}-mooncake.json', {
            'role': role, 'local_rank': self.local_rank, 'MC_TE_FILTERS': hca,
            'MC_MAX_MR_SIZE': os.environ.get('MC_MAX_MR_SIZE'),
            'store_enabled': role == 'prefill',
        })
        print(f'[mooncake] {role} rank={self.local_rank} MC_TE_FILTERS={hca} store={role == "prefill"}', flush=True)
        prepare_after_load(
            json.loads(Path(os.environ['ATOMESH_VLLM_STORE_CONFIG_PATH']).read_text()),
            self.vllm_config.model_config.model,
            out, role, self.local_rank,
            timeout=int(os.environ.get('VLLM_ENGINE_READY_TIMEOUT_S', '7200')),
        )
        evidence = self._diagnostic_dir()
        record_runtime(evidence, 'before-kv-init')
        try:
            result = super().initialize_from_config(kv_cache_config)
        except Exception as exc:
            record_failure(evidence, exc)
            raise
        record_runtime(evidence, 'after-kv-init')
        return result
