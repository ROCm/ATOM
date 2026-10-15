"""Loading-only memory evidence; removed from execution before serving starts."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading

import torch


class LoadingMemoryProbe:
    def __init__(self, worker):
        self.worker = worker
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.phase = "before_model_load"
        self.completed_layers = 0
        out = Path(os.environ["C48_DIAG_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        self.path = out / f'{os.environ["C48_DIAG_ROLE"]}-rank-{worker.rank}-memory.jsonl'
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def snapshot(self, event, **extra):
        try:
            free, total = torch.cuda.mem_get_info(self.worker.device)
            allocated = torch.cuda.memory_allocated(self.worker.device)
            reserved = torch.cuda.memory_reserved(self.worker.device)
            record = {"time": datetime.now(timezone.utc).isoformat(), "event": event,
                      "pid": os.getpid(), "rank": self.worker.rank,
                      "device": str(self.worker.device), "phase": self.phase,
                      "free": free, "total": total, "torch_allocated": allocated,
                      "torch_reserved": reserved,
                      "torch_peak_allocated": torch.cuda.max_memory_allocated(self.worker.device),
                      "outside_torch_reserved": total - free - reserved,
                      "completed_online_layers": self.completed_layers, **extra}
            if event == "start":
                snap = self.worker.init_snapshot
                record["init_snapshot"] = {key: getattr(snap, key, None) for key in
                    ("free_memory", "total_memory", "torch_peak", "non_torch_memory")}
                record["requested_memory"] = self.worker.requested_memory
            # Source bytes are logical tensor bytes; storage may alias. This is
            # an explanatory counter, not additive physical peak accounting.
            from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
            infos = list(LAYERWISE_INFO.values())
            calls = [args for info in infos for _, args in list(info.loaded_weights)]
            tensors = [value for args in calls for name, value in list(args.arguments.items())
                       if name != "param" and isinstance(value, torch.Tensor)]
            record["buffered_layers"] = sum(bool(info.loaded_weights) for info in infos)
            record["buffered_calls"] = len(calls)
            record["buffered_source_cuda_logical_bytes"] = sum(t.nbytes for t in tensors if t.is_cuda)
            record["buffered_source_cpu_logical_bytes"] = sum(t.nbytes for t in tensors if t.device.type == "cpu")
            with self.lock, self.path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception as error:
            print(f"[C48-LOADMEM] evidence error rank={self.worker.rank}: {type(error).__name__}: {error}", flush=True)

    def _poll(self):
        torch.cuda.set_device(self.worker.device)
        while not self.stop_event.wait(2):
            self.snapshot("poll")

    def __enter__(self):
        from vllm.model_executor.model_loader.reload import layerwise
        self.original = layerwise._layerwise_process

        def process(layer, info):
            self.phase = "online_quant:" + type(layer).__name__
            try:
                result = self.original(layer, info)
                self.completed_layers += 1
                return result
            finally:
                self.phase = "weight_iteration"

        layerwise._layerwise_process = process
        self.snapshot("start")
        print(f"[C48-LOADMEM] start rank={self.worker.rank} file={self.path}", flush=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        from vllm.model_executor.model_loader.reload import layerwise
        self.stop_event.set()
        self.thread.join(timeout=10)
        layerwise._layerwise_process = self.original
        self.snapshot("exception" if exc_type else "loaded", exception=None if exc_type is None else exc_type.__name__)
        print(f"[C48-LOADMEM] end rank={self.worker.rank} completed_layers={self.completed_layers}", flush=True)
