"""Prepare CPU Store memory after model load, before native vLLM KV initialization."""
import json
import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker

from pd_vllm_memory import bind_thread_memory, cpu_numa_nodes, prepare_after_load, write_json
from pd_vllm_diagnostics import record_runtime, record_failure
from pd_vllm_stage_probe import probe


def mooncake_hca(local_rank, environ=os.environ):
    """Match the successful CI's physical GPU to RDMA rail mapping."""
    visible = (environ.get('ROCR_VISIBLE_DEVICES') or environ.get('HIP_VISIBLE_DEVICES')
               or environ.get('CUDA_VISIBLE_DEVICES') or '')
    devices = [d for d in visible.split(',') if d]
    physical_gpu = int(devices[local_rank]) if devices else local_rank
    return f'rdma{physical_gpu}'


def prefer_own_store_segment(path):
    """Place this rank's Store objects on the segment it contributes.

    The image's Mooncake 0.3.13.post1 is built without ENABLE_MULTI_PROTOCOL, so
    its HIP transport republishes every segment as protocol "hip". A PUT or GET
    whose replica sits in another process's host segment on this node then goes
    to the HIP transport, which only knows device buffers, and fails with
    TRANSFER_FAIL (job 11048). Keys carry tp_rank, so a rank only reads its own.
    """
    import vllm.v1.worker.gpu_worker as worker_module
    group = worker_module.get_kv_transfer_group()
    segments = []
    fence = os.environ.get('ATOMESH_VLLM_STORE_SAVE_FENCE', '1') == '1'
    for connector in getattr(group, '_connectors', [group]):
        worker = getattr(connector, 'connector_worker', None)
        if worker is None or not hasattr(worker, 'store_replicate_config'):
            continue
        segment = worker.store.get_hostname()
        worker.store_replicate_config.preferred_segment = segment
        segments.append(segment)
        if fence:
            fence_store_saves(worker)
    write_json(path, {'preferred_segments': segments, 'save_fence': 'device_synchronize' if fence else None})
    if not segments:
        raise RuntimeError('No Mooncake Store connector on this Prefill worker')
    print(f'[store-segment] preferred_segment={segments} save_fence={fence}', flush=True)


def fence_store_saves(worker):
    """Finish all of this device's work before a Store save records its event.

    The Store worker records a CUDA event on the current stream and its send
    thread PUTs once that event completes. Job 11054 reloaded those blocks as
    garbage while the same blocks sent to Decode were correct, so the event did
    not cover every KV write.
    """
    import torch
    original = worker.wait_for_save

    def wait_for_save(metadata):
        if any(request.can_save for request in metadata.requests):
            torch.cuda.synchronize()
        return original(metadata)

    worker.wait_for_save = wait_for_save


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
        probe(self._diagnostic_dir(), 'after-device-init', 1)
        return result

    def load_model(self, *, load_dummy_weights=False):
        try:
            result = super().load_model(load_dummy_weights=load_dummy_weights)
        except Exception as exc:
            record_failure(self._diagnostic_dir(), exc)
            raise
        record_runtime(self._diagnostic_dir(), 'after-model-load')
        probe(self._diagnostic_dir(), 'after-model-load', 2)
        return result

    def determine_available_memory(self):
        result = super().determine_available_memory()
        probe(self._diagnostic_dir(), 'after-memory-profile', 3)
        return result

    def compile_or_warm_up_model(self):
        evidence = self._diagnostic_dir()
        record_runtime(evidence, 'before-warmup')
        import vllm.v1.worker.gpu_worker as worker_module
        original_warmup = worker_module.kernel_warmup
        def observed_kernel_warmup(worker):
            result = original_warmup(worker)
            probe(evidence, 'after-aiter-warmup', 5)
            return result
        worker_module.kernel_warmup = observed_kernel_warmup
        try:
            result = super().compile_or_warm_up_model()
        except Exception as exc:
            record_failure(evidence, exc)
            raise
        finally:
            worker_module.kernel_warmup = original_warmup
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
        gate = prepare_after_load(
            json.loads(Path(os.environ['ATOMESH_VLLM_STORE_CONFIG_PATH']).read_text()),
            self.vllm_config.model_config.model,
            out, role, self.local_rank,
            timeout=int(os.environ.get('VLLM_ENGINE_READY_TIMEOUT_S', '7200')),
        )
        store_nodes = gate.get('store_nodes')
        bind_store = role == 'prefill' and store_nodes and set(store_nodes) != set(cpu_numa_nodes())
        write_json(out / f'{role}-rank-{self.local_rank}-store-numa.json', {
            'store_nodes': store_nodes, 'bound': bool(bind_store)})
        if bind_store:
            bind_thread_memory(store_nodes)
            print(f'[store-memory] {role}-{self.local_rank}: Store allocation bound to NUMA {store_nodes}',
                  flush=True)
        evidence = self._diagnostic_dir()
        record_runtime(evidence, 'before-kv-init')
        original_ensure = None
        if os.environ.get('ATOMESH_VLLM_KFD_PIN_ADAPTER') == '1':
            import vllm.v1.worker.gpu_worker as worker_module
            from pd_vllm_kfd_pin import bind_connector
            original_ensure = worker_module.ensure_kv_transfer_initialized
            def ensure_with_kfd_adapter(*args, **kwargs):
                result = original_ensure(*args, **kwargs)
                bind_connector(worker_module.get_kv_transfer_group(), hca, evidence)
                return result
            worker_module.ensure_kv_transfer_initialized = ensure_with_kfd_adapter
        try:
            result = super().initialize_from_config(kv_cache_config)
        except Exception as exc:
            record_failure(evidence, exc)
            raise
        finally:
            if original_ensure is not None:
                worker_module.ensure_kv_transfer_initialized = original_ensure
            if bind_store:
                bind_thread_memory(None)
        if role == 'prefill' and os.environ.get('ATOMESH_VLLM_STORE_OWN_SEGMENT', '1') == '1':
            prefer_own_store_segment(out / f'{role}-rank-{self.local_rank}-store-segment.json')
        record_runtime(evidence, 'after-kv-init')
        probe(evidence, 'after-kv-init', 4)
        return result
