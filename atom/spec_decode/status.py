# SPDX-License-Identifier: MIT
"""Convert packed [bonus, rejected] counts to independently owned output rows."""

import torch

from atom.model_engine.output_transfer import D2HQueue


class SpecStatusQueue(D2HQueue):
    def __init__(self, max_requests: int, stream: torch.cuda.Stream):
        super().__init__(2 * max_requests, stream)

    @staticmethod
    def _copy_output(array):
        # Copy each row once, in scheduler order. No intermediate whole-array copy.
        return array[1].copy(), array[0].copy()
