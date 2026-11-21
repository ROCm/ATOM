"""Validation-only PP2 observer, attached by the explicit start_profile utility.

No tensor values are copied to the host and no tensor references enter receipts.
All wrappers are absent before begin() and restored by end(), including failures.
Host durations are enqueue/wait observations; GPU durations come from named scopes.
"""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch
from aiter.dist import parallel_state as ps
from atom.distributed import pp_comm
from atom.model_engine import model_runner
from atom.model_ops.attention_residual import AttnRes
from atom.models.kimi_k3.model import KimiDecoderLayer, KimiLinearModel

_session = None


def tensor_info(value):
    if value is None:
        return None
    assert isinstance(value, torch.Tensor), type(value)
    storage = value.untyped_storage()
    return dict(shape=list(value.shape), stride=list(value.stride()),
                storage_offset=value.storage_offset(), dtype=str(value.dtype),
                device=str(value.device), element_size=value.element_size(),
                logical_bytes=value.numel() * value.element_size(),
                storage_bytes=storage.nbytes(), storage_ptr=storage.data_ptr())


def memory():
    return dict(allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved())


class Session:
    def __init__(self, runner):
        self.runner = runner
        self.patches = []
        self.step = 0
        self.serial = 0
        self.phase = None
        self.send_keys = []
        self.recv_keys = []
        self.recv_key = None
        self.pp = ps.get_pp_group()
        self.tp = ps.get_tp_group()
        assert self.pp.world_size == 2 and self.tp.world_size == 4
        assert runner.enforce_eager, "PP observer must never wrap compiled execution"
        self.rank = torch.distributed.get_rank()
        assert type(runner).forward is model_runner.ModelRunner.forward
        models = [m for m in runner.model.modules() if isinstance(m, KimiLinearModel)]
        assert models and all(m.do_not_compile for m in models)
        self.target_models = {id(m) for m in models}
        self.target_layers = {id(m) for m in runner.model.modules() if isinstance(m, KimiDecoderLayer)}
        self.target_sites = {id(m) for m in runner.model.modules() if isinstance(m, AttnRes)}
        directory = Path(os.environ["ATOM_3577_OBSERVER_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        self.stream = (directory / f"pp{self.pp.rank_in_group}-rank{self.rank}.jsonl").open("x")
        self.seen = set()

    def emit(self, event, **fields):
        self.seen.add(event)
        self.stream.write(json.dumps(dict(event=event, rank=self.rank,
                              pp=self.pp.rank_in_group, tp=self.tp.rank_in_group,
                              step=self.step, time_ns=time.time_ns(), **fields)) + "\n")
        self.stream.flush()

    def patch(self, owner, name, replacement):
        original = getattr(owner, name)
        self.patches.append((owner, name, original))
        setattr(owner, name, functools.wraps(original)(replacement(original)))

    def call(self, label, function, *args, **kwargs):
        self.serial += 1
        scope = f"p3577/{self.rank}/{self.step}/{self.serial}/{label}"
        start = time.perf_counter_ns()
        with torch.profiler.record_function(scope):
            result = function(*args, **kwargs)
        self.emit("scope", name=scope, host_ns=time.perf_counter_ns() - start)
        return result

    def install(self):
        def forward(original):
            def wrapped(runner, batch):
                if runner is not self.runner:
                    return original(runner, batch)
                self.step += 1
                self.emit("batch", req_ids=list(batch.req_ids),
                          cached=list(map(int, batch.num_cached_tokens)),
                          scheduled=list(map(int, batch.num_scheduled_tokens)),
                          prefill_seqs=int(batch.total_seqs_num_prefill),
                          prefill_tokens=int(batch.total_tokens_num_prefill),
                          dummy=bool(batch.is_dummy_run), memory=memory())
                return self.call("runner-forward", original, runner, batch)
            return wrapped
        self.patch(model_runner.ModelRunner, "forward", forward)

        def layer(original):
            def wrapped(layer, positions, hidden_states, block_residual=None, *args, **kwargs):
                if id(layer) not in self.target_layers:
                    return original(layer, positions, hidden_states, block_residual, *args, **kwargs)
                site = layer.self_attention_attn_res
                self.emit("layer-bank", layer_idx=site.layer_idx,
                          block_size=site.block_size, bank=tensor_info(block_residual),
                          hidden=tensor_info(hidden_states), memory=memory())
                return original(layer, positions, hidden_states, block_residual, *args, **kwargs)
            return wrapped
        self.patch(KimiDecoderLayer, "forward", layer)

        def close(original):
            def wrapped(site, prefix, bank):
                if id(site) not in self.target_sites:
                    return original(site, prefix, bank)
                closes = (site.enabled and site.block_size is not None and
                          site.layer_idx % site.block_size == 0)
                if not closes:
                    return original(site, prefix, bank)
                self.emit("close-before", layer_idx=site.layer_idx,
                          bank=tensor_info(bank), prefix=tensor_info(prefix), memory=memory())
                result = self.call("close-bank-slot", original, site, prefix, bank)
                self.emit("close-after", layer_idx=site.layer_idx,
                          bank=tensor_info(result[0]), memory=memory())
                return result
            return wrapped
        self.patch(AttnRes, "maybe_close_block", close)

        def allocate(original):
            def wrapped(model, hidden, received_blocks=None):
                if id(model) not in self.target_models:
                    return original(model, hidden, received_blocks)
                self.emit("bank-allocate-before", start_layer=model.start_layer,
                          end_layer=model.end_layer, layers=model.config.num_hidden_layers,
                          block_size=model.config.attn_res_block_size,
                          hidden=tensor_info(hidden), received=tensor_info(received_blocks),
                          memory=memory())
                bank = self.call("allocate-reconstruct-bank", original, model, hidden, received_blocks)
                self.emit("bank-allocate-after", bank=tensor_info(bank), memory=memory())
                return bank
            return wrapped
        def prefix(original):
            def wrapped(model, bank):
                if id(model) not in self.target_models:
                    return original(model, bank)
                self.emit("prefix-before", start_layer=model.start_layer,
                          end_layer=model.end_layer, block_size=model.config.attn_res_block_size,
                          bank=tensor_info(bank), memory=memory())
                result = self.call("compact-written-prefix", original, model, bank)
                self.emit("prefix-after", bank=tensor_info(result), memory=memory())
                return result
            return wrapped
        if hasattr(KimiLinearModel, "block_bank"):
            self.patch(KimiLinearModel, "block_bank", allocate)
            self.patch(KimiLinearModel, "block_bank_prefix", prefix)

        def model_forward(original):
            def wrapped(model, *args, **kwargs):
                if id(model) not in self.target_models:
                    return original(model, *args, **kwargs)
                self.emit("model-partition", start_layer=model.start_layer,
                          end_layer=model.end_layer, layers=model.config.num_hidden_layers,
                          hidden=model.config.hidden_size, block_size=model.config.attn_res_block_size)
                return original(model, *args, **kwargs)
            return wrapped
        self.patch(KimiLinearModel, "forward", model_forward)

        def send(original):
            def wrapped(intermediate, owned=()):
                tensors = {key: intermediate.tensors[key] for key in pp_comm._PP_PROXY_KEYS
                           if key in intermediate.tensors}
                self.send_keys = ["metadata-size", "metadata"] + [
                    key for key, value in tensors.items()
                    if isinstance(value, torch.Tensor) and value.numel()]
                self.emit("pp-send-logical", tensors={k: tensor_info(v) for k, v in tensors.items()},
                          owned=list(owned), allgather=bool(pp_comm.pp_send_allgather_group()))
                self.phase = "send"
                try:
                    result = self.call("pp-send", original, intermediate, owned)
                    assert not self.send_keys, self.send_keys
                    for work in result:
                        if work.work is not None:
                            work.work = WaitReceipt(self, work.work, tensor_info(work.payload))
                    return result
                finally:
                    self.phase = None
            return wrapped
        # run_model holds imported aliases; patch those exact call targets.
        self.patch(model_runner, "async_send_intermediate_tensors", send)

        def receive(original):
            def wrapped():
                self.phase = "recv"
                self.recv_keys = []
                try:
                    result = self.call("pp-receive", original)
                    assert not self.recv_keys, self.recv_keys
                    self.emit("pp-receive-logical",
                              tensors={k: tensor_info(v) for k, v in result.tensors.items()},
                              memory=memory())
                    return result
                finally:
                    self.phase = None
            return wrapped
        self.patch(model_runner, "recv_intermediate_tensors", receive)

        def recv_object(original):
            def wrapped(group, *args, **kwargs):
                result = original(group, *args, **kwargs)
                if self.phase == "recv" and group is self.pp:
                    self.recv_keys = [key for key, value in result
                                      if isinstance(value, ps.TensorMetadata)
                                      and all(int(n) > 0 for n in value.size)]
                return result
            return wrapped
        self.patch(ps.GroupCoordinator, "recv_object", recv_object)

        def wire(original, operation):
            def wrapped(tensor, *args, **kwargs):
                if self.phase is None:
                    return original(tensor, *args, **kwargs)
                if operation == "isend":
                    assert self.send_keys
                    key = self.send_keys.pop(0)
                elif self.recv_keys:
                    key = self.recv_keys.pop(0)
                    self.recv_key = key
                else:
                    key = "metadata"
                self.emit("wire-buffer", operation=operation, key=key,
                          tensor=tensor_info(tensor), memory=memory())
                return self.call(f"wire-{operation}-{key}", original, tensor, *args, **kwargs)
            return wrapped
        self.patch(torch.distributed, "isend", lambda original: wire(original, "isend"))
        self.patch(torch.distributed, "recv", lambda original: wire(original, "recv"))

        def clone(original):
            def wrapped(tensor, *args, **kwargs):
                if self.phase != "send" or not tensor.is_cuda:
                    return original(tensor, *args, **kwargs)
                assert self.send_keys
                key = self.send_keys[0]
                result = self.call(f"clone-{key}", original, tensor, *args, **kwargs)
                self.emit("clone-buffer", key=key, source=tensor_info(tensor),
                          result=tensor_info(result), memory=memory())
                return result
            return wrapped
        self.patch(torch.Tensor, "clone", clone)

        def gather(original):
            def wrapped(group, tensor, *args, **kwargs):
                if self.phase != "recv":
                    return original(group, tensor, *args, **kwargs)
                self.emit("allgather-before", key=self.recv_key, world_size=group.world_size,
                          tensor=tensor_info(tensor), memory=memory())
                result = self.call(f"allgather-{self.recv_key}", original, group, tensor, *args, **kwargs)
                self.emit("allgather-after", key=self.recv_key, tensor=tensor_info(result),
                          memory=memory())
                return result
            return wrapped
        self.patch(ps.GroupCoordinator, "all_gather", gather)

    def restore(self):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()


class WaitReceipt:
    def __init__(self, session, work, payload):
        self.session, self.work, self.payload = session, work, payload

    def wait(self, *args, **kwargs):
        result = self.session.call("pp-send-completion-wait", self.work.wait, *args, **kwargs)
        self.session.emit("send-wait-complete", payload=self.payload)
        return result

    def __getattr__(self, name):
        return getattr(self.work, name)


def begin(runner):
    global _session
    assert _session is None
    session = Session(runner)
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        sources = {}
        for module in (model_runner, pp_comm, ps, inspect.getmodule(KimiLinearModel),
                       inspect.getmodule(AttnRes)):
            path = Path(module.__file__).resolve()
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        session.emit("begin", sources=sources, memory=memory(),
                     observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     dcp=runner.config.decode_context_parallel_size,
                     runner_class=type(runner).__qualname__)
        session.install()
        _session = session
    except BaseException:
        session.restore()
        session.stream.close()
        raise


def end(runner):
    global _session
    session = _session
    assert session is not None and session.runner is runner
    try:
        # Complete the last forward's retained sends at the diagnostic boundary.
        pending = getattr(runner, "_pp_pending_send", None)
        if pending:
            model_runner.commit_pp_send_work(pending)
        torch.cuda.synchronize()
        session.emit("end", memory=memory(), events=sorted(session.seen))
    finally:
        session.restore()
        session.stream.close()
        _session = None
