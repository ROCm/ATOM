"""Prepare CPU Store memory after model load, before native vLLM KV initialization."""
import json
import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker

from pd_vllm_memory import prepare_after_load


class StoreMemoryWorker(Worker):
    def initialize_from_config(self, kv_cache_config):
        if self.parallel_config.tensor_parallel_size != 4:
            raise ValueError('Store memory preparation requires TP4 for both roles')
        prepare_after_load(
            json.loads(Path(os.environ['MOONCAKE_CONFIG_PATH']).read_text()),
            self.vllm_config.model_config.model,
            Path(os.environ['ATOMESH_VLLM_MEMORY_DIR']),
            os.environ['ATOMESH_VLLM_ROLE'], self.local_rank,
            timeout=int(os.environ.get('VLLM_ENGINE_READY_TIMEOUT_S', '7200')),
        )
        return super().initialize_from_config(kv_cache_config)
