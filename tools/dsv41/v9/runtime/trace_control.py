# Keep measured import ordering/namespace; AST parity is verified by prepare.py.
"""Use production profiler for only three M384 owner replays on request."""

import json
import os
import time
from pathlib import Path


def install():
    from formal_counters import Counters

    old_snapshot, old_replay = Counters.snapshot, Counters.replayed

    def snapshot(self, request):
        data = old_snapshot(self, request)
        if request.get("trace") and request["phase"] == "before":
            self._trace_count = 0
        return data

    def replayed(self, *args):
        old_replay(self, *args)
        if not hasattr(self, "_trace_count") or args[0].capacity != 384:
            return
        self._trace_count += 1
        if self._trace_count == 20:
            self.serving.runner.start_profiler(
                "p2_" + os.environ.get("ATOM_DSV41_OWNER_FFN_SUPPLY", "native")
            )
        elif self._trace_count == 23:
            result = self.serving.runner.stop_profiler()
            path = (
                Path(os.environ["P0_COUNTER_DIR"])
                / f"trace_done_rank{self.serving.rank}.json"
            )
            path.write_text(
                json.dumps({"m384_replays": 3, "finished_ns": time.time_ns(), **result})
                + "\n"
            )
            del self._trace_count

    Counters.snapshot, Counters.replayed = snapshot, replayed
