# SPDX-License-Identifier: MIT
"""Everything DSpark speculation needs that is true only on this frontend.

Three things, and none of them exists when speculation is off: the draft needs
the target's LM head, the scheduler must not mix a prefill into a verification
step, and the tentative Engram step has to be committed once vLLM has judged
the block. They are separate hooks into separate parts of vLLM -- the model
wrapper, `Scheduler.schedule`, `postprocess_sampled` -- but they share one
lifetime and one switch, so they live together.

What they also share is how each was found. Every one of the three failed
quietly: the unshared head gave correct answers from noise drafts (rejection
sampling preserves the target distribution, so only acceptance length showed
it, at 1.04); the mixed step refused inside `begin_step`; the uncommitted
cursor sat on the unverified block and put later steps out by the rejected
count. None of them is visible in a log that is not looking for it.
"""

import functools
import logging

import torch
from torch import nn

logger = logging.getLogger("atom")

_DEEPSEEK_V41_ARCHES = ("DeepseekV41ForCausalLM",)


# ---------------------------------------------------------------------------
# The draft's LM head: a vLLM-shaped view of ATOM's own
# ---------------------------------------------------------------------------
# The DSpark draft carries no output projection: `has_own_lm_head = False` on
# vLLM's `DSparkDeepseekV4ForCausalLM`, and the DeepSeek-V4.1-Flash checkpoint
# has no `mtp.*.lm_head` tensor to load into one. vLLM therefore aliases the
# target's head into the draft -- `load_dspark_model` does it through
# `get_target_lm_head`, which looks for an attribute named `lm_head`. ATOM's
# head is named `head`, so that lookup found nothing and the draft kept the
# freshly constructed `ParallelLMHead` it was given, whose weights no
# checkpoint ever filled.
#
# Aliasing ATOM's head directly does not work either. vLLM's
# `LogitsProcessor` calls `lm_head.quant_method.apply(...)` and then
# all-gathers across TP itself, while ATOM's head has no `quant_method` and
# all-gathers inside its own `forward`. The view below supplies the former
# shape over the same weight tensor -- no copy, no second set of parameters --
# and returns the rank-local shard that vLLM expects to gather.
class _TgemmHeadMethod:
    """The `quant_method` shape `LogitsProcessor._apply_head` calls."""

    def apply(self, layer, hidden_states, bias=None):
        from aiter.tuned_gemm import tgemm

        return tgemm.mm(hidden_states, layer.weight, bias)


class V41TargetLMHeadView(nn.Module):
    """ATOM's LM head, presented the way vLLM's logits processor reads one."""

    def __init__(self, head: nn.Module):
        super().__init__()
        # Not a submodule: the weight already belongs to the ATOM head, and
        # registering it twice would have the loader's completeness check see
        # a second, never-loaded copy of the same tensor.
        object.__setattr__(self, "_atom_head", head)
        self.weight = head.weight
        self.bias = getattr(head, "bias", None)
        self.tp_size = head.tp_size
        self.tp_rank = getattr(head, "tp_rank", 0)
        self.quant_method = _TgemmHeadMethod()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, hidden_states, self.bias)


def attach_v41_target_lm_head(wrapper, model) -> bool:
    """Expose `wrapper.lm_head` for vLLM's draft-head aliasing.

    On the wrapper rather than on the ATOM model: `get_target_lm_head` reads
    the object the speculator was handed, which is the wrapper, and ATOM's own
    forward has no use for a second name for its head.
    """
    if getattr(wrapper, "lm_head", None) is not None:
        return False
    head = getattr(model, "head", None)
    if head is None or getattr(head, "weight", None) is None:
        return False
    wrapper.lm_head = V41TargetLMHeadView(head)
    logger.info(
        "ATOM plugin: sharing the DeepSeek-V4.1 LM head with the DSpark "
        "draft, which carries none of its own."
    )
    return True


# ---------------------------------------------------------------------------
# Keeping a verification step clear of prefills
# ---------------------------------------------------------------------------
# V4.1's CSA2 cache runs a verification step *tentatively*: the Engram cursor
# rows for every prefix the draft block could leave behind are staged aside,
# and the sampler's accepted prefix is committed afterwards. That staging
# holds `num_speculative_tokens + 1` candidate rows per request -- a draft
# block's width. A prefill row fits in none of it: it starts from nothing and
# is longer than a block, so `V41Cache.begin_step` refuses the mixed step
# outright. ATOM's own engine never produces the mix, because it runs prefill
# steps and decode steps separately; vLLM's scheduler does, by admitting a
# waiting request, or continuing a chunked prompt, beside the running
# requests' draft blocks.
#
# The way out is to drop that step's drafts, not to hold the prefill back.
# Holding the prefill back starves it for as long as any request keeps
# drafting, which at steady-state concurrency is indefinitely. Dropping the
# drafts costs that one step its speculation: the drafter proposes again on
# the next step, and vLLM clears `spec_token_ids` after scheduling them
# anyway.
#
# Installed from `register_model` -- the `vllm.general_plugins` hook, which
# vLLM runs in the EngineCore process that owns the scheduler. `ATOMPlatform`
# is not a usable site for this: it is resolved from inside `import vllm`, may
# never activate, and the V4.1 config hook that would carry it also runs in
# the workers, where setting `scheduler_cls` changes nothing. Measured: the
# class was selected, logged, and never instantiated.
#
# `AsyncScheduler` does not override `schedule`, so patching `Scheduler`
# covers both.
def _is_v41_speculating(scheduler) -> bool:
    vllm_config = getattr(scheduler, "vllm_config", None)
    if vllm_config is None or getattr(vllm_config, "speculative_config", None) is None:
        return False
    model_config = getattr(vllm_config, "model_config", None)
    arches = getattr(model_config, "architectures", None) or []
    return any(str(arch) in _DEEPSEEK_V41_ARCHES for arch in arches)


def _admission_is_possible(scheduler) -> bool:
    """Whether this step could admit a waiting request at all.

    A nonempty waiting queue is not proof that a prefill will be scheduled.
    vLLM stops admitting at

        num_running = len(self.running) + self.num_waiting_for_streaming_input
        if num_running >= self.max_num_active_reqs: break

    so at the cap every running request decodes and nothing can join them,
    however long the queue is. Clearing their drafts there buys nothing and
    costs that step its speculation.

    Only this one gate is read, not the token budget or per-request
    eligibility: the cap is a plain comparison on state that is already
    settled, while the rest is decided inside the scheduler as it goes. Where
    this cannot tell, it says admission is possible, which keeps the drafts
    cleared -- the side that was safe before.
    """
    cap = getattr(scheduler, "max_num_active_reqs", None)
    if cap is None:
        return True
    streaming = getattr(scheduler, "num_waiting_for_streaming_input", 0) or 0
    return len(scheduler.running) + streaming < cap


def _would_mix_prefill_with_drafts(scheduler) -> bool:
    if not any(request.spec_token_ids for request in scheduler.running):
        return False
    if scheduler.waiting and _admission_is_possible(scheduler):
        return True
    # A running request can still be mid-prompt: chunked prefill keeps it in
    # `running` while it works through the rest of its tokens, and a preempted
    # request resumes by recomputing its generated tokens as well as its
    # prompt -- against `num_prompt_tokens` that second case reads as a decode
    # for all of them, which is how a prefill reaches a step that still
    # carries drafts, where `_v41_prepare_kwargs` refuses the batch outright.
    #
    # What separates the two here is how much is left to compute, not how it
    # compares to any one length. This runs at `schedule` *entry*, where
    # `num_computed_tokens` has not yet been advanced for the step being built
    # (`scheduler.py` adds `num_scheduled_tokens` and only then evaluates its
    # own `is_prefill_chunk`). So a decode request sits exactly one token
    # short, and borrowing vLLM's expression verbatim -- as this did briefly --
    # makes the predicate true for every request on every step, which clears
    # the drafts of a batch that was purely decoding and stops DSpark
    # verifying anything at all.
    #
    # One token left is a decode row whoever it belongs to, including the last
    # chunk of a prompt; more than one is the prefill shape the refusal is
    # about.
    return any(
        (request.num_tokens + getattr(request, "num_output_placeholders", 0))
        - request.num_computed_tokens
        > 1
        for request in scheduler.running
    )


def patch_v41_speculative_scheduling() -> bool:
    """Wrap `Scheduler.schedule`. Returns whether it was installed here."""
    from vllm.v1.core.sched.scheduler import Scheduler

    original = Scheduler.schedule
    if getattr(original, "_atom_v41_no_mixed_step", False):
        return False

    @functools.wraps(original)
    def schedule(self, *args, **kwargs):
        if _is_v41_speculating(self) and _would_mix_prefill_with_drafts(self):
            self._atom_v41_drafts_dropped = (
                getattr(self, "_atom_v41_drafts_dropped", 0) + 1
            )
            if self._atom_v41_drafts_dropped in (1, 10, 100, 1000, 10000):
                logger.info(
                    "ATOM: dropped a step's DeepSeek-V4.1 draft blocks so a "
                    "prefill could run in it (%d step(s) so far).",
                    self._atom_v41_drafts_dropped,
                )
            for request in self.running:
                request.spec_token_ids = []
        return original(self, *args, **kwargs)

    schedule._atom_v41_no_mixed_step = True
    Scheduler.schedule = schedule
    logger.info(
        "ATOM plugin: DeepSeek-V4.1 verification steps will be kept clear of "
        "prefills."
    )
    return True


# ---------------------------------------------------------------------------
# Committing the tentative Engram step
# ---------------------------------------------------------------------------
# A speculative step runs the draft's whole block through the target, so the
# Engram cursor it would leave behind belongs to tokens that may not survive
# verification. `_prepare` therefore stages the step as *tentative*: the
# cursor rows for every prefix are written to one side, and
# `commit_speculative_state` picks the accepted one. Nothing in the plugin
# path called it -- ATOM's own runner does, around its drafter -- which left
# the cursor on the full block and put every later step out by the rejected
# count ("needs state at 8, found 13").
#
# `postprocess_sampled` is the hook because it is the first place where the
# rejection count exists and it runs before the drafter proposes the next
# block, which is the same order ATOM's runner uses.
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
    from atom.plugin.vllm.vllm_runtime import gpu_model_runner_classes

    patched = 0
    for cls in gpu_model_runner_classes():
        original = getattr(cls, "postprocess_sampled", None)
        if original is None or getattr(original, "_atom_v41_spec_commit", False):
            continue

        @functools.wraps(original)
        def postprocess_sampled(
            self,
            idx_mapping,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc=None,
            *args,
            _original=original,
            **kwargs,
        ):
            """Upstream's signature, exactly, plus the V4.1 commit.

            The names and the default are the interface, not a style: a
            non-last pipeline rank calls this as
            `postprocess_sampled(**get_prev_sampled_outputs(...))`, whose dict
            is `sampled_tokens`/`num_sampled`/`num_rejected`/`idx_mapping` and
            carries no `query_start_loc`. Renaming the first or requiring the
            last makes that call fail at argument binding -- before any of the
            code below runs, and for every plugin model using the V2 runner
            with pipeline parallelism, speculating or not, because this patch
            is installed globally.
            """
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
                    starts = getattr(batch, "query_start_loc", None)
                if starts is None:
                    raise RuntimeError(
                        "V4.1 speculative step cannot be committed: vLLM gave "
                        "no query_start_loc and no input batch carries one"
                    )
                count = int(num_rejected.shape[0])
                v41_commit_speculative_state(_anchors(num_rejected, starts, count))
            # Forwarded by keyword, so a caller that omitted the optional
            # argument reaches the original having omitted it too rather than
            # being given a None this wrapper invented.
            return _original(
                self,
                idx_mapping,
                sampled_tokens,
                num_sampled,
                num_rejected,
                *args,
                **(
                    {}
                    if query_start_loc is None
                    else {"query_start_loc": query_start_loc}
                ),
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
