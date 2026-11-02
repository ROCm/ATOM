# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pure Kimi K1 persistent-CTA and tagged-mailbox protocol model.

Progress is proved only under ``all_declared_ctas_resident`` plus fair CTA
scheduling.  That is a logical assumption, not hardware residency validation.
"""

from __future__ import annotations

from dataclasses import dataclass

from atom.models.kimi_k3.mono.kernels.token_tiling import (
    GFX1250_MAX_TOKENS,
)

ATTENTION_CTAS = 56
PROJECTION_TASKS = 393
PROJECTION_ROWS_PER_CTA = 2
HEADS = 12
STATE_CHUNKS = 8
SLICES_PER_GROUP = 7
SLICE_GROUPS = ATTENTION_CTAS // SLICES_PER_GROUP
NPROJ = 6288
HEAD_DIM = 128


@dataclass(frozen=True)
class ResidencyAssumption:
    all_declared_ctas_resident: bool
    fair_cta_progress: bool = True
    hardware_validated: bool = False

    def require_logical_progress_assumption(self) -> None:
        if not self.all_declared_ctas_resident or not self.fair_cta_progress:
            raise ValueError(
                "K1 progress proof requires every declared CTA resident and fair"
            )
        if self.hardware_validated:
            raise ValueError(
                "CPU protocol model cannot mark hardware residency validated"
            )


@dataclass(frozen=True)
class CtaAssignment:
    block: int
    attention_slice: int | None
    gate_token: int | None
    projection_tasks: tuple[int, ...]
    kda_tasks: tuple[int, ...]


@dataclass(frozen=True)
class ProtocolAction:
    name: str
    waits: frozenset[str] = frozenset()
    publishes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ProgressProof:
    actions_executed: int
    rounds: int
    published: frozenset[str]
    assumption: ResidencyAssumption


@dataclass
class StepResetState:
    """CPU state machine for zero/reset completion and peer epoch fencing."""

    blocks: int
    ranks: int
    epoch: int

    def __post_init__(self) -> None:
        if self.blocks <= 0 or self.ranks <= 0 or self.epoch <= 0:
            raise ValueError("blocks, ranks, and epoch must be positive")
        self.reset_ctas: set[int] = set()
        self.peer_epochs: dict[int, int] = {}

    def arrive_reset(self, block: int) -> None:
        if not 0 <= block < self.blocks:
            raise ValueError(f"block {block} outside reset grid")
        self.reset_ctas.add(block)

    def arrive_peer_epoch(self, rank: int, epoch: int) -> None:
        if not 0 <= rank < self.ranks:
            raise ValueError(f"rank {rank} outside peer set")
        if epoch < self.epoch:
            raise ValueError(f"stale peer epoch {epoch}, need {self.epoch}")
        self.peer_epochs[rank] = epoch

    @property
    def reset_complete(self) -> bool:
        return len(self.reset_ctas) == self.blocks

    @property
    def peer_fence_complete(self) -> bool:
        return all(
            self.peer_epochs.get(rank, 0) >= self.epoch for rank in range(self.ranks)
        )

    @property
    def publication_allowed(self) -> bool:
        return self.reset_complete and self.peer_fence_complete

    def mailbox_tag(self, layer: int) -> int:
        if not self.publication_allowed:
            raise RuntimeError("mailbox publication before reset/peer epoch fence")
        if layer < 0:
            raise ValueError("layer must be nonnegative")
        return layer + 1

    def tag_after_reset(self, previous_tag: int) -> int:
        """Observed mailbox tag after this step's distributed zeroing."""

        if not self.reset_complete:
            return previous_tag
        return 0


@dataclass(frozen=True)
class K1Protocol:
    """CTA roles, mailbox ownership, and progress graph for one K1 launch."""

    blocks: int
    threads: int
    tokens: int
    qlen: int
    sources: int
    kda0: int = 0

    @property
    def timeline_points(self) -> int:
        return 15

    @property
    def timeline_observational_only(self) -> bool:
        """Timeline stamps add no producer/consumer edge or progress wait."""

        return True

    def __post_init__(self) -> None:
        if self.blocks < ATTENTION_CTAS + (PROJECTION_TASKS + 1) // 2:
            raise ValueError(
                f"K1 needs at least 253 CTAs for attention/projection roles, "
                f"got {self.blocks}"
            )
        if self.threads <= 0:
            raise ValueError("threads must be positive")
        if not 1 <= self.tokens <= GFX1250_MAX_TOKENS:
            raise ValueError(f"tokens must be in [1, {GFX1250_MAX_TOKENS}]")
        if not 1 <= self.qlen <= 16 or self.tokens % self.qlen:
            raise ValueError("qlen must be in [1, 16] and divide tokens")
        if not 1 <= self.sources <= 32:
            raise ValueError("sources must be in [1, 32]")
        if not 0 <= self.kda0 < self.blocks:
            raise ValueError("kda0 must name a block in the candidate grid")

    @property
    def requests(self) -> int:
        return self.tokens // self.qlen

    @property
    def kda_task_count(self) -> int:
        return self.requests * HEADS * STATE_CHUNKS

    def first_task(self, block: int) -> int:
        return (block - self.kda0 % self.blocks + self.blocks) % self.blocks

    def kda_tasks(self, block: int) -> tuple[int, ...]:
        return tuple(range(self.first_task(block), self.kda_task_count, self.blocks))

    def projection_tasks(self, block: int) -> tuple[int, ...]:
        if block < ATTENTION_CTAS:
            return ()
        first = (block - ATTENTION_CTAS) * PROJECTION_ROWS_PER_CTA
        return tuple(
            task
            for task in range(first, first + PROJECTION_ROWS_PER_CTA)
            if task < PROJECTION_TASKS
        )

    def assignment(self, block: int) -> CtaAssignment:
        if not 0 <= block < self.blocks:
            raise ValueError(f"block {block} outside [0, {self.blocks})")
        return CtaAssignment(
            block=block,
            attention_slice=block if block < ATTENTION_CTAS else None,
            gate_token=block if block < self.tokens else None,
            projection_tasks=self.projection_tasks(block),
            kda_tasks=self.kda_tasks(block),
        )

    @property
    def assignments(self) -> tuple[CtaAssignment, ...]:
        return tuple(self.assignment(block) for block in range(self.blocks))

    @staticmethod
    def _part(slice_id: int) -> str:
        return f"part:{slice_id}"

    @staticmethod
    def _weight(token: int) -> str:
        return f"wgt:{token}"

    @staticmethod
    def _msq(slice_id: int) -> str:
        return f"msq:{slice_id}"

    @staticmethod
    def _xrdy(slice_id: int) -> str:
        return f"xrdy:{slice_id}"

    @staticmethod
    def _projection(task: int) -> str:
        return f"proj:{task}"

    @staticmethod
    def _qk(request: int, head: int, chunk: int) -> str:
        return f"qk:{request}:{head}:{chunk}"

    @staticmethod
    def _onrm(request: int, head: int, chunk: int) -> str:
        return f"onrm:{request}:{head}:{chunk}"

    @property
    def all_parts(self) -> frozenset[str]:
        return frozenset(self._part(slice_id) for slice_id in range(ATTENTION_CTAS))

    @property
    def all_weights(self) -> frozenset[str]:
        return frozenset(self._weight(token) for token in range(self.tokens))

    @property
    def all_msq(self) -> frozenset[str]:
        return frozenset(self._msq(slice_id) for slice_id in range(ATTENTION_CTAS))

    @property
    def all_xrdy(self) -> frozenset[str]:
        return frozenset(self._xrdy(slice_id) for slice_id in range(ATTENTION_CTAS))

    @property
    def all_projection(self) -> frozenset[str]:
        return frozenset(self._projection(task) for task in range(PROJECTION_TASKS))

    def _task_coordinates(self, task: int) -> tuple[int, int, int]:
        request = task // (HEADS * STATE_CHUNKS)
        head = (task // STATE_CHUNKS) % HEADS
        chunk = task % STATE_CHUNKS
        return request, head, chunk

    def action_queues(self) -> tuple[tuple[ProtocolAction, ...], ...]:
        """Per-CTA program order, split at every mailbox wait/publication."""

        queues = []
        for assignment in self.assignments:
            actions = []
            if assignment.attention_slice is not None:
                slice_id = assignment.attention_slice
                actions.append(
                    ProtocolAction(
                        f"slice:{slice_id}",
                        publishes=frozenset({self._part(slice_id)}),
                    )
                )
                if assignment.gate_token is not None:
                    token = assignment.gate_token
                    actions.append(
                        ProtocolAction(
                            f"gate:{token}",
                            waits=self.all_parts,
                            publishes=frozenset({self._weight(token)}),
                        )
                    )
                actions.append(
                    ProtocolAction(
                        f"mix-partial:{slice_id}",
                        waits=self.all_weights,
                        publishes=frozenset({self._msq(slice_id)}),
                    )
                )
                actions.append(
                    ProtocolAction(
                        f"mix-finish:{slice_id}",
                        waits=self.all_msq,
                        publishes=frozenset({self._xrdy(slice_id)}),
                    )
                )
            projection = assignment.projection_tasks
            if projection:
                actions.append(
                    ProtocolAction(
                        f"projection:{assignment.block}",
                        waits=self.all_xrdy,
                        publishes=frozenset(
                            self._projection(task) for task in projection
                        ),
                    )
                )
            for task in assignment.kda_tasks:
                request, head, chunk = self._task_coordinates(task)
                actions.append(
                    ProtocolAction(
                        f"conv:{task}",
                        waits=self.all_projection,
                        publishes=frozenset({self._qk(request, head, chunk)}),
                    )
                )
                actions.append(
                    ProtocolAction(
                        f"gates:{task}",
                        waits=self.all_projection
                        | frozenset(
                            self._qk(request, head, other)
                            for other in range(STATE_CHUNKS)
                        ),
                    )
                )
                actions.append(
                    ProtocolAction(
                        f"recur:{task}",
                        publishes=frozenset({self._onrm(request, head, chunk)}),
                    )
                )
                actions.append(
                    ProtocolAction(
                        f"norm:{task}",
                        waits=self.all_projection
                        | frozenset(
                            self._onrm(request, head, other)
                            for other in range(STATE_CHUNKS)
                        ),
                    )
                )
            queues.append(tuple(actions))
        return tuple(queues)

    def prove_progress(self, assumption: ResidencyAssumption) -> ProgressProof:
        """Run the fair all-resident CTA state machine to completion."""

        assumption.require_logical_progress_assumption()
        queues = self.action_queues()
        positions = [0] * self.blocks
        published: set[str] = set()
        actions_executed = 0
        rounds = 0
        while True:
            if all(
                positions[block] == len(queues[block]) for block in range(self.blocks)
            ):
                return ProgressProof(
                    actions_executed,
                    rounds,
                    frozenset(published),
                    assumption,
                )
            progressed = False
            rounds += 1
            for block, queue in enumerate(queues):
                position = positions[block]
                if position == len(queue):
                    continue
                action = queue[position]
                if action.waits <= published:
                    published.update(action.publishes)
                    positions[block] += 1
                    actions_executed += 1
                    progressed = True
            if not progressed:
                blocked = tuple(
                    queue[positions[block]].name
                    for block, queue in enumerate(queues)
                    if positions[block] < len(queue)
                )
                raise RuntimeError(f"K1 logical protocol deadlock: {blocked}")

    @property
    def mailbox_pairs(self) -> dict[str, int]:
        return {
            "part": ATTENTION_CTAS * self.tokens * self.sources * 2,
            "wgt": self.tokens * (self.sources + 1),
            "msq": ATTENTION_CTAS * self.tokens,
            "xrdy": ATTENTION_CTAS,
            "proj": self.tokens * NPROJ,
            "qk": self.tokens * HEADS * 2 * HEAD_DIM,
            "onrm": self.tokens * HEADS * STATE_CHUNKS,
        }

    def reset_chunk_owners(self, region: str) -> tuple[tuple[int, int], ...]:
        """``(CTA, thread)`` owner of each aligned 16-byte reset chunk."""

        try:
            pairs = self.mailbox_pairs[region]
        except KeyError as error:
            raise ValueError(f"unknown mailbox region {region!r}") from error
        chunks = (pairs * 8 + 15) // 16
        lanes = self.blocks * self.threads
        return tuple(
            ((chunk % lanes) // self.threads, (chunk % lanes) % self.threads)
            for chunk in range(chunks)
        )
