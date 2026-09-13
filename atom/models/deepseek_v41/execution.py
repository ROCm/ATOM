# SPDX-License-Identifier: MIT
"""CUDAGraph execution of pure V4.1 tensor stages, separate from model math.

Each stage/bucket owns stable inputs, outputs and its graph allocation pool.
Request metadata, Engram lookup and eager expert dispatch stay outside capture.
The optional AITER expert stage uses GPU route IDs during graph replay.
No captured operation reads Python request positions or committed histories.
"""

from dataclasses import dataclass

import torch


@dataclass
class DenseGraph:
    graph: torch.cuda.CUDAGraph
    inputs: tuple
    outputs: tuple


class DenseGraphExecutor:
    def __init__(self):
        self.entries = {}
        self.replays = 0

    @staticmethod
    def _copy(inputs, args):
        for target, source in zip(inputs, args):
            if source is None:
                continue
            length = source.shape[1]
            target[:, :length].copy_(source)
            target[:, length:].zero_()

    def run(self, function, *args, bucket, capture=False):
        length = args[0].shape[1]
        if length > bucket:
            raise ValueError("Dense graph input exceeds its declared token bucket")
        key = (function, bucket)
        entry = self.entries.get(key)
        if entry is None:
            if not capture:
                return function(*args)
            inputs = tuple(
                None if x is None else x.new_empty((x.shape[0], bucket, *x.shape[2:]))
                for x in args
            )
            self._copy(inputs, args)
            # Warm lazy kernels outside the recorded region; no request state
            # is reachable by these functions. Capture happens only at startup.
            function(*inputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph,
                stream=torch.cuda.current_stream(),
                capture_error_mode="thread_local",
            ):
                outputs = function(*inputs)
            entry = DenseGraph(graph, inputs, outputs)
            self.entries[key] = entry
        else:
            self._copy(entry.inputs, args)
        entry.graph.replay()
        self.replays += 1
        return tuple(x[:, :length] for x in entry.outputs)
