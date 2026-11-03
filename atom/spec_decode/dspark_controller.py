# SPDX-License-Identifier: MIT
"""The runner's one entry point to DSpark's online machinery.

`VerifyScheduler` owns the optional components -- the online SPS timer
(`runtime`), the online STS collector (`sts`) and the joint DP planner (`dp`).
The runner calls this controller at four points of a step and once after graph
capture; nothing in the runner reaches into those components directly.

    prepare_model  -> dp_plan(batch)            (rides ForwardMode.decide)
    forward        -> forward_begin / target_done / forward_end
    postprocess    -> observe_verdict / observe_no_verdict
    capture        -> bind_startup_sps(max_q_len)

`build_online_parts` is where those components are validated and created.
"""

from __future__ import annotations

import numpy as np


def build_online_parts(verify, runner) -> None:
    """Validate the online configuration once and attach the enabled parts to
    `verify`: the online SPS timer, the joint DP planner, the online STS
    collector and the graph-priced bucket policy. Every collective among them
    shares one `DSparkComm`."""
    config = runner.config
    dspark = config.dspark
    online = dspark.online_sps or dspark.bucket_global
    if online:
        from atom.spec_decode.dspark_runtime import validate_online_runtime

        validate_online_runtime(config)
    from atom.spec_decode.dspark_comm import DSparkComm

    comm = DSparkComm()
    if dspark.online_sps:
        from atom.spec_decode.dspark_runtime import DSparkRuntime

        verify.runtime = DSparkRuntime(runner, comm=comm)
    if online and config.parallel_config.data_parallel_size > 1:
        from atom.spec_decode.dspark_dp import DPVerifyPlanner

        verify.dp = DPVerifyPlanner(verify, comm=comm)
        if verify.runtime is not None:
            verify.runtime.dp_planner = verify.dp
    # No calibration profile supplies temperatures: fit them online.
    if (
        verify.sts_temperatures is None
        and dspark.confidence_schedule
        and dspark.online_sts
    ):
        from atom.spec_decode.online_sts import OnlineSTS

        verify.sts = OnlineSTS(
            verify, config.speculative_config.num_speculative_tokens, comm=comm
        )
    if dspark.bucket_global:
        from atom.spec_decode.dspark_bucket import BucketPolicy

        verify.bucket = BucketPolicy(dspark.ragged_graph_sizes)


class DPPlan:
    """DSpark's joint DP verification plan for one step.

    `payload` rides `ForwardMode.decide`'s one DP exchange. `apply` sees every
    rank's payload and the gathered sync; when the plan beats the per-rank
    ragged fallback it rewrites this rank's batch to its share and returns
    `(q, scheduled tokens per rank)`, else None.
    """

    def __init__(self, runner, planner, batch):
        self._runner = runner
        self._planner = planner
        self._batch = batch
        self.payload = planner.prepare(batch)
        # The scheduled layout the plan rewrites from: the per-rank ragged
        # fallback applied before the exchange shrank it.
        self._original = (
            batch.scheduled_tokens,
            batch.num_scheduled_tokens,
            batch.total_tokens_num,
            batch.total_tokens_num_decode,
        )

    def apply(self, gathered, sync):
        """The ragged rewrite reproduces the planner's per-request lengths
        exactly (between ``num_bonus + 1`` and the scheduled width), so the
        rows every peer assumed for this rank are the rows it now runs. A
        difference would hand the group two shapes, so it is fatal here
        rather than a hang inside the collectives."""
        batch = self._batch
        fallback = (
            int(np.max(sync.scheduled_tokens_per_dp_rank)),
            sync.max_seqlen_q_across_dp or batch.num_spec_step + 1,
        )
        planned = self._planner.resolve(gathered, fallback)
        if planned is None:
            return None
        q, tokens, lengths = planned
        (
            batch.scheduled_tokens,
            batch.num_scheduled_tokens,
            batch.total_tokens_num,
            batch.total_tokens_num_decode,
        ) = self._original
        self._runner._dspark_apply_ragged(
            batch, batch.total_seqs_num_decode, batch.num_spec_step + 1, lengths
        )
        rank = self._planner.comm.dp_rank
        if int(batch.total_tokens_num) != int(tokens[rank]):
            raise RuntimeError(
                f"DSpark DP plan expects {int(tokens[rank])} tokens on this "
                f"rank, the ragged layout has {batch.total_tokens_num}"
            )
        return q, tokens


class DSparkController:
    def __init__(self, runner, verify):
        self.runner = runner
        self.verify = verify
        self._timing = None

    @classmethod
    def of(cls, runner):
        """The controller for a confidence-scheduled block drafter, else None."""
        verify = getattr(getattr(runner, "drafter", None), "verify_scheduler", None)
        return None if verify is None else cls(runner, verify)

    # -- prepare_model ------------------------------------------------------
    def dp_plan(self, batch) -> DPPlan | None:
        """Built on every forward the planner exists for, whatever the local
        batch type: its payload is part of the step's DP exchange."""
        planner = self.verify.dp
        return None if planner is None else DPPlan(self.runner, planner, batch)

    # -- forward ------------------------------------------------------------
    def forward_begin(self, batch, forward_mode) -> None:
        """`prepare_model` settled the planner's group state for this step
        (`DPPlan.apply`), so the timer reads the step it is timing."""
        runtime = self.verify.runtime
        if runtime is None:
            self._timing = None
            return
        planner = self.verify.dp
        self._timing = runtime.begin(
            batch, forward_mode, None if planner is None else planner.state
        )

    def target_done(self) -> None:
        if self.verify.runtime is not None:
            self.verify.runtime.target_done(self._timing)

    def forward_end(self) -> None:
        """Target plus state commit, sampling and drafting, excluding the async
        output-copy/HTTP path."""
        if self.verify.runtime is not None:
            self.verify.runtime.finish(self._timing)
        self._timing = None

    # -- ragged layout ------------------------------------------------------
    def keeps_reserved_width(self, ragged_rows, q, full_rows, full_q) -> bool:
        """`bucket_global` on a single rank: keep the reserved tokens when the
        ragged layout would not replay a smaller graph (`keeps_reserved_width`
        in `dspark_bucket`). Under DP the joint plan settles the shape."""
        if (
            self.verify.bucket is None
            or self.runner.config.parallel_config.data_parallel_size != 1
        ):
            return False
        from atom.spec_decode.dspark_bucket import keeps_reserved_width

        return keeps_reserved_width(ragged_rows, q, full_rows, full_q)

    # -- postprocess --------------------------------------------------------
    def observe_verdict(self, prev_batch, prev_bonus_num) -> bool:
        """The previous forward's verdict on the drafts recorded the forward
        before it, before this forward drafts its own. Labels need what that
        batch actually verified (its ragged lengths, not the reservation).
        Returns whether the STS collector counted this forward."""
        sts = self.verify.sts
        if sts is None:
            return False
        if (
            prev_batch is not None
            and not prev_batch.is_dummy_run
            and not prev_batch.total_tokens_num_prefill
            and len(prev_bonus_num) == prev_batch.total_seqs_num
        ):
            n = prev_batch.total_seqs_num
            sts.observe(
                list(prev_batch.req_ids[:n]),
                prev_bonus_num,
                np.asarray(prev_batch.num_scheduled_tokens[:n]) - 1,
            )
        else:
            sts.observe()
        return True

    def observe_no_verdict(self) -> None:
        """A forward that reported no verdict still counts, so the per-forward
        tag stays aligned across TP ranks."""
        if self.verify.sts is not None:
            self.verify.sts.observe()

    # -- after graph capture ------------------------------------------------
    def bind_startup_sps(self, max_q_len: int) -> bool:
        """Time the decode token graphs now, before the first request, as
        DeepInfer's startup profile does, and bind the curve. Returns whether
        it did; the caller falls back to its own calibration otherwise.

        Every model with ragged confidence scheduling and recorded decode token
        graphs takes it; V4.1, online SPS and the graph-aware budget require it
        (no synthetic stub is admitted there). Capture already ran these graphs
        on the same synthetic metadata, so no request state exists yet for the
        replays to disturb.
        """
        from atom.utils.region_graph import TokenGraph

        runner, verify = self.runner, self.verify
        if verify.calibration_profile is not None:
            return False  # explicit measurements are never overwritten
        dspark = runner.config.dspark
        required = (
            dspark.online_sps
            or dspark.bucket_global
            or runner.config.hf_config.model_type == "deepseek_v41_text"
        )
        if not (
            dspark.ragged
            and runner.drafter.uses_confidence_schedule
            and (
                required
                or (
                    TokenGraph.captured_sizes("decode") and not dspark.disable_sps_calib
                )
            )
        ):
            return False
        from atom.spec_decode.startup_sps import startup_sps_table

        verify.bind_sps(
            startup_sps_table(
                runner,
                runner._dspark_capture_q_buckets(max_q_len),
                runner.config.max_num_seqs * max_q_len,
            )
        )
        return True
