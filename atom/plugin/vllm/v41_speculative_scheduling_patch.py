# SPDX-License-Identifier: MIT
"""Keep a DeepSeek-V4.1 verification step clear of prefills.

V4.1's CSA2 cache runs a verification step *tentatively*: the Engram cursor
rows for every prefix the draft block could leave behind are staged aside, and
the sampler's accepted prefix is committed afterwards. That staging holds
`num_speculative_tokens + 1` candidate rows per request -- a draft block's
width. A prefill row fits in none of it: it starts from nothing and is longer
than a block, so `V41Cache.begin_step` refuses the mixed step outright. ATOM's
own engine never produces the mix, because it runs prefill steps and decode
steps separately; vLLM's scheduler does, by admitting a waiting request, or
continuing a chunked prompt, beside the running requests' draft blocks.

The way out is to drop that step's drafts, not to hold the prefill back.
Holding the prefill back starves it for as long as any request keeps drafting,
which at steady-state concurrency is indefinitely. Dropping the drafts costs
that one step its speculation: the drafter proposes again on the next step,
and vLLM clears `spec_token_ids` after scheduling them anyway.

Installed from `register_model` -- the `vllm.general_plugins` hook, which vLLM
runs in the EngineCore process that owns the scheduler. `ATOMPlatform` is not
a usable site for this: it is resolved from inside `import vllm`, may never
activate, and the V4.1 config hook that would carry it also runs in the
workers, where setting `scheduler_cls` changes nothing. Measured: the class
was selected, logged, and never instantiated.

`AsyncScheduler` does not override `schedule`, so patching `Scheduler` covers
both.
"""

import functools
import logging

logger = logging.getLogger("atom")

_DEEPSEEK_V41_ARCHES = ("DeepseekV41ForCausalLM",)


def _is_v41_speculating(scheduler) -> bool:
    vllm_config = getattr(scheduler, "vllm_config", None)
    if vllm_config is None or getattr(vllm_config, "speculative_config", None) is None:
        return False
    model_config = getattr(vllm_config, "model_config", None)
    arches = getattr(model_config, "architectures", None) or []
    return any(str(arch) in _DEEPSEEK_V41_ARCHES for arch in arches)


def _would_mix_prefill_with_drafts(scheduler) -> bool:
    if not any(request.spec_token_ids for request in scheduler.running):
        return False
    if scheduler.waiting:
        return True
    # A running request can still be mid-prompt: chunked prefill keeps it in
    # `running` while it works through the rest of its tokens.
    return any(
        request.num_computed_tokens < request.num_prompt_tokens
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
