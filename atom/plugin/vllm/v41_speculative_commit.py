# SPDX-License-Identifier: MIT
"""Close V4.1's tentative Engram step once vLLM has judged the draft block.

A speculative step runs the draft's whole block through the target, so the
Engram cursor it would leave behind belongs to tokens that may not survive
verification. `_prepare` therefore stages the step as *tentative*: the cursor
rows for every prefix are written to one side, and `commit_speculative_state`
picks the accepted one. Nothing in the plugin path called it -- ATOM's own
runner does, around its drafter -- which left the cursor on the full block and
put every later step out by the rejected count ("needs state at 8, found 13").

`postprocess_sampled` is the hook because it is the first place where the
rejection count exists and it runs before the drafter proposes the next block,
which is the same order ATOM's runner uses.
"""

import functools
import logging

logger = logging.getLogger("atom")


def _anchors(num_rejected, query_start_loc, count):
    """Each request's flat row of its last accepted token.

    vLLM scheduled `query_start_loc[i + 1] - query_start_loc[i]` rows for
    request `i` and rejected the last `num_rejected[i]` of them, so the last
    surviving row is what is left when both are taken off the end.
    """
    ends = query_start_loc[1 : count + 1]
    return ends - 1 - num_rejected[:count].to(ends.dtype)


def patch_v41_speculative_commit() -> int:
    """Wrap `postprocess_sampled` on every runner class. Returns the count."""
    from atom.plugin.vllm.gpu_model_runner_targets import gpu_model_runner_classes

    patched = 0
    for cls in gpu_model_runner_classes():
        original = getattr(cls, "postprocess_sampled", None)
        if original is None or getattr(original, "_atom_v41_spec_commit", False):
            continue

        @functools.wraps(original)
        def postprocess_sampled(
            self,
            idx_mapping,
            sampled_token_ids,
            num_sampled,
            num_rejected,
            query_start_loc,
            *args,
            _original=original,
            **kwargs,
        ):
            from atom.plugin.vllm.deepseek_v41_bridge import (
                v41_commit_speculative_state,
                v41_speculative_step_is_pending,
            )

            if v41_speculative_step_is_pending():
                starts = query_start_loc
                if starts is None:
                    # Optional in vLLM's signature, so take it from the batch
                    # the pass-through patch exposes rather than skipping the
                    # commit -- a step left pending is a cursor left on the
                    # unverified block, which is the bug this exists to fix.
                    from atom.plugin.vllm.req_id_passthrough_patch import (
                        get_current_input_batch,
                    )

                    batch = get_current_input_batch()
                    starts = None if batch is None else batch.query_start_loc
                if starts is None:
                    raise RuntimeError(
                        "V4.1 speculative step cannot be committed: vLLM gave "
                        "no query_start_loc and no input batch carries one"
                    )
                count = int(num_rejected.shape[0])
                v41_commit_speculative_state(_anchors(num_rejected, starts, count))
            return _original(
                self,
                idx_mapping,
                sampled_token_ids,
                num_sampled,
                num_rejected,
                query_start_loc,
                *args,
                **kwargs,
            )

        postprocess_sampled._atom_v41_spec_commit = True
        cls.postprocess_sampled = postprocess_sampled
        patched += 1

    logger.info(
        "ATOM plugin: V4.1 speculative Engram commit installed on %d runner "
        "class(es).",
        patched,
    )
    return patched
