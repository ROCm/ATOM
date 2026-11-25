# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A speculative verify from the ids a head picked itself (``verify_head``):
a step laid out as ``B`` requests of ``rq`` rows (``rq - 1`` draft rows then
the bonus row), its output ids [B, rq] and spec status [3, B] (accepted
drafts, rejected drafts, the anchor row) as the rejection sampler lays them
out (``verify_kernels._verify_by_ids``), at fixed addresses a decode graph
records. A head's ids are greedy picks or, for a sampled row, draws
(``HeadDraw``): exact match on them is the sampler's sample-and-match."""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from atom.model_ops.rejection_sampler import (
    _SYNTHETIC_RNG_BASE_SEED,
    _get_synthetic_cond_rates,
)
from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.verify_kernels import _verify_by_ids


@triton.jit
def _verify_ids_kernel(
    out_ptr, spec_status_ptr, cu_ptr, input_ids_ptr, target_idx_ptr,
    bonus_idx_ptr, head_ids_ptr, cond_rates_ptr, launches_ptr, seed_base,
    num_spec_steps, SP: tl.constexpr, SYNTHETIC: tl.constexpr,
):  # fmt: skip
    """Program b, request b (``_verify_by_ids``). ``launches[0]``: the launch
    count every width shares (the same on every rank: each replays the same
    launches), the tag of the head's draws on this step's ids
    (``IdVerifiers.draw``) and, ``SYNTHETIC``, of the uniforms (seeded
    ``seed_base`` + it); the launch's last program to arrive
    (``launches[1]``) then bumps it."""
    tag = tl.load(launches_ptr)
    _verify_by_ids(
        out_ptr, spec_status_ptr, cu_ptr, input_ids_ptr, target_idx_ptr,
        bonus_idx_ptr, head_ids_ptr, cond_rates_ptr, seed_base + tag,
        tl.program_id(0), num_spec_steps, SP=SP, SYNTHETIC=SYNTHETIC,
    )  # fmt: skip
    # every program read the count before its arrival
    if tl.atomic_add(launches_ptr + 1, 1) == tl.num_programs(0) - 1:
        tl.store(launches_ptr, tag + 1)
        tl.store(launches_ptr + 1, 0)


class VerifyOutput(NamedTuple):
    """A verify step's outputs as the rejection sampler lays them out."""

    # [requests, num_spec_steps + 1]: each request's accepted drafts, then its
    # correction or bonus token, then -1
    token_ids: torch.Tensor
    # [3, requests]: accepted drafts, rejected drafts, the anchor row
    spec_status: torch.Tensor


DRAW_SEED = 0x5EED_D4A3_0F1C_E5A7


class HeadDraw(NamedTuple):
    """A verify head's draw of its rows' ids (the sampler's sample-and-match
    rows), from the step's staged sampling: row r of temperature T =
    ``temperatures[r // rq]`` > ``greedy_t`` takes one draw of softmax(logits /
    T) (Gumbel-max) -- within its top-p (``top_ps[r // rq]``) where the head
    draws it -- keyed by ``seed``, the step's ``tag`` (``launches[0]``, int64
    on the device, the same on every TP rank), the row and the id; any other
    row its greedy id."""

    temperatures: torch.Tensor
    top_ps: torch.Tensor
    rq: int
    tag: torch.Tensor
    seed: int
    greedy_t: float


class IdVerifier:
    """The verify of a step of ``requests`` requests of ``rq`` rows each from
    its head's ids, its rows' layout and outputs at fixed addresses (a
    captured graph replays them): ``launch(head_ids, input_ids)`` ->
    ``VerifyOutput``. ``cond_rates`` (forced
    acceptance: the positions' conditional rates,
    ``rejection_sampler._get_synthetic_cond_rates``): its uniforms seeded from
    ``seed_base`` and ``launches`` (int64 [2] on the device:
    ``_verify_ids_kernel``'s), the launch count every width shares; without
    them the layout alone (a verify launched elsewhere: hi's post-verify)."""

    def __init__(
        self, requests: int, rq: int, device, cond_rates=None, seed_base: int = 0,
        launches: torch.Tensor | None = None,
    ):  # fmt: skip
        i32 = {"dtype": torch.int32, "device": device}
        s = rq - 1
        self.requests, self.rq = requests, rq
        self.cond_rates, self.seed_base, self.launches = cond_rates, seed_base, launches
        self.out = torch.zeros(requests, rq, **i32)
        self.spec_status = torch.zeros(3, requests, **i32)
        self.cu = torch.arange(1, requests + 1, **i32) * s
        self.target = torch.tensor(
            [b * rq + j for b in range(requests) for j in range(s)],
            dtype=torch.long,
            device=device,
        )
        self.bonus = torch.tensor(
            [b * rq + s for b in range(requests)], dtype=torch.long, device=device
        )

    def launch(self, head_ids: torch.Tensor, input_ids: torch.Tensor) -> VerifyOutput:
        synthetic = self.cond_rates is not None
        _verify_ids_kernel[(self.requests,)](
            self.out, self.spec_status, self.cu, input_ids, self.target,
            self.bonus, head_ids, self.cond_rates if synthetic else head_ids,
            self.launches, self.seed_base, self.rq - 1,
            SP=triton.next_power_of_2(self.rq), SYNTHETIC=synthetic, num_warps=1,
        )  # fmt: skip
        return VerifyOutput(self.out, self.spec_status)


class IdVerifiers:
    """A head's verifies from its ids, an ``IdVerifier`` a width it serves
    (whole requests of ``rq`` rows), made eagerly before a graph captures it.
    ``synthetic_rates`` (forced acceptance, the speculative config's): their
    conditional rates and one launch count every width shares, seeded as the
    rejection sampler's. ``temperatures`` / ``top_ps``: the step's
    per-request sampling on the device (callables: the runner's staging
    buffers), that a head draws its rows' ids by (``draw``)."""

    def __init__(self, rq: int, device, synthetic_rates, temperatures, top_ps):
        self.rq, self.device = rq, device
        self.temperatures, self.top_ps = temperatures, top_ps
        self.cond_rates = (
            None
            if synthetic_rates is None
            else _get_synthetic_cond_rates(tuple(synthetic_rates), device)
        )
        # the launch count, then its programs' arrivals (``_verify_ids_kernel``)
        self.launches = torch.zeros(2, dtype=torch.int64, device=device)
        self.widths = {}

    @classmethod
    def from_spec(cls, spec, device, temperatures, top_ps) -> "IdVerifiers":
        """The verifies of ``spec``'s steps (``SpeculativeConfig``): a request
        of its drafts and the bonus row, its forced acceptance if any."""
        return cls(
            spec.num_speculative_tokens + 1, device,
            spec.synthetic_acceptance_rates, temperatures, top_ps,
        )  # fmt: skip

    def draw(self) -> HeadDraw:
        """What a head draws its ids by (``HeadDraw``)."""
        return HeadDraw(
            self.temperatures(), self.top_ps(), self.rq, self.launches,
            DRAW_SEED, SAMPLER_EPS,
        )  # fmt: skip

    def serves(self, rows: int) -> bool:
        return rows % self.rq == 0

    def of(self, rows: int) -> IdVerifier:
        """Width ``rows``' verifier (``serves``), made on its first call."""
        verifier = self.widths.get(rows)
        if verifier is None:
            verifier = self.widths[rows] = IdVerifier(
                rows // self.rq, self.rq, self.device, self.cond_rates,
                _SYNTHETIC_RNG_BASE_SEED, self.launches,
            )  # fmt: skip
        return verifier
