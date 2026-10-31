"""C48_P_PP_MQ_CHUNKS=C: widen the RPC broadcast ring MultiprocExecutor._init_executor builds in the EngineCore from
10 chunks to C (2 RPCs per step; the engine blocks once the slowest stage CPU is 10 RPCs behind).

pd_c48_launch puts this directory on PYTHONPATH only when the variable is set: the EngineCore builds the ring before
it imports any harness module, and a custom --distributed-executor-backend is rejected for PP
(engine/arg_utils.py _check_feature_supported). Readers attach with the geometry in the ring's handle.
"""
import os
import sys

TARGET = "vllm.v1.executor.multiproc_executor"


def widen_rpc_ring(module, chunks):
    init = module.MultiprocExecutor._init_executor

    def _init_executor(self):
        base = module.MessageQueue

        class WideQueue(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **{"max_chunks": chunks, **kwargs})

        module.MessageQueue = WideQueue
        try:
            init(self)
        finally:
            module.MessageQueue = base
        print(f"[C48-PP] rpc broadcast ring {self.rpc_broadcast_mq.buffer.max_chunks} chunks", flush=True)

    module.MultiprocExecutor._init_executor = _init_executor


if os.environ.get("C48_P_PP_MQ_CHUNKS"):
    import importlib.abc
    import importlib.util

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            run = spec.loader.exec_module

            def exec_module(module):
                run(module)
                widen_rpc_ring(module, int(os.environ["C48_P_PP_MQ_CHUNKS"]))

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Hook())
