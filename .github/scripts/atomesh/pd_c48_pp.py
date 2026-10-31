"""Prefill PP8 pipeline depth (CPP); every switch is off unless its C48_P_PP_* variable is set.

C48_P_PP_INFLIGHT=T: EngineCore keeps max_concurrent_batches = pp_size + 1 = 9 steps in flight, middle chunks
included, and each slot comes back only after stage 7, so stage 0 starves. The worker UVA rings are sized from the
same property, so it is patched at import in the EngineCore (pd_c48_scheduler) and in every worker (pd_c48_worker).

C48_P_PP_EARLY_RELEASE=O (O < T): a step whose requests are all non-final prefill chunks samples nothing, so the
engine schedules again while fewer than O queued steps sample and fewer than T are queued in total. Steps stay in
the one FIFO, so update_from_output still runs in launch order.

C48_P_PP_STAGE_DEPTH=D: a worker CPU otherwise runs to the engine's frontier and pre-allocates the PP hand-off of
every launched step (T=16 OOMed stage 6 in the 48 GiB pool); with D it waits for its GPU to finish step N-D first.

C48_P_PP_EAGER_RECV=1: torch makes an irecv's NCCL stream wait for the posting stream, so step N's receive waits for
this stage's forward N-1 and the upstream's next forward waits for that send: each stage runs at most ~2 steps
ahead. Posting from a side stream lets the receive complete as soon as the upstream sends.

C48_P_PP_MQ_CHUNKS (pp_ring/sitecustomize.py) widens the engine's RPC ring, where the steps the stage CPUs may not
take yet wait.
"""
import collections
import inspect
import os
import textwrap

import torch
from vllm.config import VllmConfig

INFLIGHT = int(os.environ.get("C48_P_PP_INFLIGHT", "0"))
STAGE_DEPTH = int(os.environ.get("C48_P_PP_STAGE_DEPTH", "0"))
EAGER_RECV = os.environ.get("C48_P_PP_EAGER_RECV") == "1"


def wait_stage_depth(done, depth):
    while len(done) >= depth:
        done.popleft().synchronize()


def bounded_execute(worker, execute, scheduler_output):
    if not STAGE_DEPTH or not scheduler_output.total_num_scheduled_tokens:
        return execute(scheduler_output)
    done = worker.__dict__.setdefault("_c48_pp_forwards", collections.deque())
    wait_stage_depth(done, STAGE_DEPTH)
    output = execute(scheduler_output)
    done.append(torch.cuda.Event())
    done[-1].record()
    return output


def eager_recv():
    from vllm.distributed.parallel_state import GroupCoordinator
    irecv = GroupCoordinator.irecv_tensor_dict
    streams = {}

    def eager(self, *args, **kwargs):
        compute = torch.cuda.current_stream()
        side = streams.setdefault(compute.device, torch.cuda.Stream(compute.device))
        with torch.cuda.stream(side):
            tensors, handles, postprocess = irecv(self, *args, **kwargs)
        # Allocated on the side stream but read by the compute stream's copy: keep the allocator from reusing early.
        for t in (tensors or {}).values():
            if isinstance(t, torch.Tensor) and t.is_cuda:
                t.record_stream(compute)
        return tensors, handles, postprocess

    GroupCoordinator.irecv_tensor_dict = eager


SCHEDULE_IF = """\
        if self.scheduler.has_requests():
"""
# The next step may sample, so stop scheduling at the cap; blocking then drains FIFO until a sampling step retires.
SCHEDULE_IF_CPP = """\
        if self.scheduler.has_requests() and self.c48_output_room(batch_queue):
"""


def early_release(output_cap):
    from vllm.v1.engine import core
    lines, first = inspect.getsourcelines(core.EngineCore.step_with_batch_queue)
    src = "".join(lines)
    assert src.count(SCHEDULE_IF) == 1 and "elif not batch_queue:" in src, \
        "step_with_batch_queue changed; re-check the schedule-or-block test"
    # Padding keeps tracebacks on core.py's line numbers.
    patched = "\n" * (first - 1) + textwrap.dedent(src.replace(SCHEDULE_IF, SCHEDULE_IF_CPP))
    namespace = {}
    exec(compile(patched, core.__file__, "exec"), core.__dict__, namespace)
    core.EngineCore.step_with_batch_queue = namespace["step_with_batch_queue"]
    core.EngineCore.c48_output_room = lambda self, queue: sum(so.c48_samples for _, so, _ in queue) < output_cap


def mark_sampling(scheduler):
    schedule = scheduler.schedule

    # is_prefill_chunk is this step's value only until the next schedule() advances the request.
    def marked(*args, **kwargs):
        out = schedule(*args, **kwargs)
        out.c48_samples = not out.num_scheduled_tokens or any(
            not scheduler.requests[r].is_prefill_chunk for r in out.num_scheduled_tokens)
        return out

    scheduler.schedule = marked


def enable_early_release(scheduler, output_cap):
    if not output_cap < INFLIGHT:
        raise RuntimeError("C48_P_PP_EARLY_RELEASE needs C48_P_PP_INFLIGHT (total in flight) above the output cap")
    early_release(output_cap)
    mark_sampling(scheduler)


if INFLIGHT:
    VllmConfig.max_concurrent_batches = property(lambda self: INFLIGHT)
if EAGER_RECV:
    eager_recv()
if INFLIGHT or STAGE_DEPTH or EAGER_RECV:
    print(f"[C48-PP] pid={os.getpid()} inflight={INFLIGHT} stage_depth={STAGE_DEPTH} eager_recv={EAGER_RECV}",
          flush=True)
