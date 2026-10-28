"""Host-only accounting of original scheduler post-cap/stop client packets."""

import atexit
import collections
import json
import os
from functools import wraps
from pathlib import Path


class CommitQueue:
    def __init__(self, original, counts, prefixes):
        self.original, self.counts, self.prefixes = original, counts, prefixes
        self.finished = False

    def put_nowait(self, outputs):
        for _, output in outputs:
            retained = len(output.output_tokens)
            self.finished |= output.finished
            for prefix in self.prefixes:
                self.counts[prefix + "_committed_tokens"] += retained
                self.counts[prefix + "_emitting_request_cycles"] += int(retained > 0)
        return self.original.put_nowait(outputs)

    def __getattr__(self, name):
        return getattr(self.original, name)


def install():
    from atom.model_engine.scheduler import Scheduler

    original = Scheduler.postprocess
    if getattr(original, "_commit_accounting", False):
        return
    counts = collections.Counter()
    output = Path(os.environ["DSV41_OWNER_SERVING_OUTPUT"])

    def save():
        raw = (
            json.dumps(
                {
                    "pid": os.getpid(),
                    "counts": dict(counts),
                    "scope": "Original scheduler packets after stop/max_tokens; denominator includes zero-retained request cycles; no GPU sync/profiling",
                },
                indent=2,
            )
            + "\n"
        ).encode()
        if len(raw) > 128_000:
            raise ValueError("Commit counter exceeds reservation")
        path = output / f"serving_commit_pid{os.getpid()}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(raw)
        temporary.replace(path)

    @wraps(original)
    def postprocess(self, seqs, fwd_output, stream_output_queue=None, batch=None):
        if stream_output_queue is None:
            return original(self, seqs, fwd_output, stream_output_queue, batch)
        prefill = (
            fwd_output.is_prev_prefill
            if fwd_output.is_deferred_out
            else bool(batch and batch.total_seqs_num_prefill)
        )
        prefix = "prefill" if prefill else "decode"
        prefixes = (
            (prefix, "C256_" + prefix) if len(fwd_output.req_ids) == 256 else (prefix,)
        )
        for name in prefixes:
            counts[name + "_source_request_cycles"] += len(fwd_output.req_ids)
        counts["postprocess_calls"] += 1
        queue = CommitQueue(stream_output_queue, counts, prefixes)
        result = original(self, seqs, fwd_output, queue, batch)
        if queue.finished or counts["postprocess_calls"] % 64 == 0:
            save()
        return result

    postprocess._commit_accounting = True
    Scheduler.postprocess = postprocess
    atexit.register(save)
