# SPDX-License-Identifier: MIT
"""A vLLM-shaped view of ATOM's V4.1 LM head, so a DSpark draft can share it.

The DSpark draft carries no output projection of its own: `has_own_lm_head =
False` on vLLM's `DSparkDeepseekV4ForCausalLM`, and the DeepSeek-V4.1-Flash
checkpoint has no `mtp.*.lm_head` tensor to load into one. vLLM therefore
aliases the target's head into the draft -- `load_dspark_model` does it
through `get_target_lm_head`, which looks for an attribute named `lm_head`.

ATOM's head is named `head`, so that lookup found nothing and the draft kept
the freshly constructed `ParallelLMHead` it was given, whose weights no
checkpoint ever filled. The drafts it produced were noise, which rejection
sampling then discarded in full: answers stayed correct and mean acceptance
length sat at 1.04 against 100 drafted tokens per second. Nothing in the logs
named the cause -- an unshared head is not an error anywhere.

Aliasing ATOM's head directly does not work either. vLLM's `LogitsProcessor`
calls `lm_head.quant_method.apply(...)` and then all-gathers across TP itself,
while ATOM's head has no `quant_method` and all-gathers inside its own
`forward`. This view supplies the former shape over the same weight tensor --
no copy, no second set of parameters -- and returns the rank-local shard that
vLLM expects to gather.
"""

import logging

import torch
from torch import nn

logger = logging.getLogger("atom")


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
