# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A model that can run a whole decode step itself: the target forward, sampling, verification and the next drafts."""


class StepOverrideModel:
    """Base of a mono routing wrapper whose runner serves whole decode steps.

    ``ModelRunner.run_model`` offers it each decode step (``step_override``). It returns None and the production path
    runs, or the step's served outputs: ``verify_outputs(bs)``, the rejection sampler's ``(output_token_ids [bs, k + 1],
    verdict [3, bs])``, and ``output_stream``, the stream those outputs are ready on (the output copy waits for it, not
    for whatever the model queued after them)."""

    def step_override(self, input_ids, running_bs: int, batch):
        raise NotImplementedError

    def decline_step(self, batch):
        """Drop previous served state when the executor requires normal logits."""
