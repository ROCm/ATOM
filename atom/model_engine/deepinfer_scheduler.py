# SPDX-License-Identifier: Apache-2.0
# Derived from deepseek-ai/DeepInfer, csrc/engine/scheduler.hpp at
# 553741557a57df44a46fb3ab72b6aafeaf945337 (Apache-2.0).
# Modified for ATOM: Python implementation using the EngineCore CPU DP group.
# See licenses/DeepInfer-LICENSE and licenses/DeepInfer-NOTICE.

"""DeepInfer NORMAL-engine phase selection, independent of batch allocation.

Every DP rank calls ``begin_step`` before attempting local prefill admission,
then ``finish_step`` after admission, including ranks with empty queues. The
second vote is essential: a willing rank can still fail KV/sequence allocation.
Only a globally selected prefill resets the decode counter; a dummy decode
advances it just like a real decode. This is not a post-prefill cooldown.
"""

import logging

import torch

logger = logging.getLogger("atom")


class DeepInferDPPolicy:
    KV_PREFILL_WATERMARK = 0.9

    def __init__(
        self,
        dp_size: int,
        cpu_group=None,
        prefill_interval: int = 70,
        prefill_any_interval: int = 100,
    ):
        if dp_size < 1 or (dp_size > 1 and cpu_group is None):
            raise ValueError("DeepInfer DP scheduling requires a CPU group for DP > 1")
        if prefill_interval < 1 or prefill_any_interval < 1:
            raise ValueError("DeepInfer prefill intervals must be positive")
        self.dp_size = dp_size
        self.cpu_group = cpu_group
        self.prefill_interval = prefill_interval
        self.prefill_any_interval = prefill_any_interval
        self.decode_step = 0
        self.last_vote = "decode"
        self._in_step = False
        self._attempt_prefill = False
        self._want = False
        self._vote = torch.empty(1, dtype=torch.int32, device="cpu")

    def _min(self, value: int) -> int:
        if self.dp_size == 1:
            return value
        self._vote[0] = value
        # Do not fail open: a local fallback could disagree with a peer's
        # phase and deadlock the subsequent GPU collectives.
        torch.distributed.all_reduce(
            self._vote, op=torch.distributed.ReduceOp.MIN, group=self.cpu_group
        )
        return int(self._vote.item())

    def begin_step(self, *, has_prefill: bool, kv_usage: float) -> bool:
        """Vote on the group phase; return whether THIS rank may try prefill.

        ``has_prefill`` means pending schedulable work, not an allocation-fit
        probe. Full sequence slots and allocation failures belong to the local
        admission attempt and the second vote, as in the upstream scheduler.
        """
        if self._in_step:
            raise RuntimeError("DeepInfer begin_step called before finish_step")
        self._in_step = True
        self._attempt_prefill = False
        self._want = False
        is_any = (
            self.decode_step != 0 and self.decode_step % self.prefill_any_interval == 0
        )
        is_all = self.decode_step % self.prefill_interval == 0
        self.last_vote = "any" if is_any else "all" if is_all else "decode"
        if not is_any and not is_all:
            return False
        self._want = has_prefill and kv_usage <= self.KV_PREFILL_WATERMARK
        if is_any:
            self._attempt_prefill = self._min(0 if self._want else 1) == 0
        else:
            self._attempt_prefill = self._min(1 if self._want else 0) == 1
        return self._attempt_prefill and self._want

    def finish_step(self, *, prefill_scheduled: bool) -> bool:
        """Confirm admission and advance the common clock; return group prefill.

        A rejection-only prefill output also counts, matching DeepInfer's
        non-null ``SchedulerOutputs`` containing ``ignored_seqs``. ATOM sends
        those rejections separately and uses its dummy prefill for this phase.
        """
        if not self._in_step:
            raise RuntimeError("DeepInfer finish_step called without begin_step")
        if prefill_scheduled and not (self._attempt_prefill and self._want):
            raise RuntimeError("Prefill scheduled without DeepInfer admission")
        group_prefill = (
            self._attempt_prefill and self._min(0 if prefill_scheduled else 1) == 0
        )
        logger.debug(
            "DeepInfer DP step: decode_step=%d vote=%s want=%s "
            "local_prefill=%s group_prefill=%s",
            self.decode_step,
            self.last_vote,
            self._want,
            prefill_scheduled,
            group_prefill,
        )
        self.decode_step = 0 if group_prefill else self.decode_step + 1
        self._in_step = False
        return group_prefill
