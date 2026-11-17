# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# The admission ports TileRT 0.1.7 (MIT, Copyright (c) TileRT-team) ``DsV41Graph.begin_request`` / ``set_sampling``
# (the sampling counter of ``dsv41_e2e_begin`` / ``dsv41_e2e_set_sampling``).
"""One TP rank's ``TkRank`` serving ATOM decode steps (``TkStepModel``'s runner), what ``ModelRunner`` asks of it:

- ``run_step(input_ids, batch, forward_context)``: a new request -> its admission (``admit``: the sampling mode and
  parameters, ``admit_kernel`` at the anchor's position with the anchor token, the KV ATOM holds converted by ``kv``,
  on ATOM's stream); then one TileRT step (one graph replay) and its outputs, on TkRank's own stream (``stream``);
- ``verify_outputs(bs)``: the rejection sampler's ``(output_token_ids [1, 6] int32, -1 past the a emitted tokens;
  verdict [3, 1] int32: a - 1 accepted drafts, 6 - a rejected, anchor row a - 1)``, gathered on the device from the
  logs the step appended to (``ar_num``: a; ``ar_acc``: the a tokens), no host sync;
- ``draft_ids(bs)``: [1, 5] int64 for ATOM's next input rows. TkRank verifies the drafts its own state holds and
  reads no host input, so these only keep ATOM's bookkeeping shaped (``vtok[1:6]`` after the step).

Streams (GPU): TkRank's steps and the gather of their outputs run on ``stream``, one after another, and ATOM's stream
waits for a step's outputs only where it reads them (``verify_outputs``). ATOM's own per-step work (the next step's
inputs and layout, the commit, the output copies) runs on its stream beside TkRank's next step instead of between two
of them; none of it is a collective. The admission orders the two (``join``): ATOM's stream after TkRank's last step,
TkRank's after the admission; a production forward joins first (``TkStepModel``).

The device (``TkDevice``, or a fake in the CPU tests): ``set_sampling(params)``, ``admit(first, pos, params)``,
``import_planes(planes, n)``, ``step()``, ``logs() -> (ar_acc[0], ar_num[0])``, ``drafts() -> vtok``.
"""

from __future__ import annotations

import logging
import struct

import torch

from . import kv as KV
from .gate import ROWS, SPEC_TOKENS

logger = logging.getLogger("atom")
VOCAB = 129280
SLACK = 8  # positions a chunk keeps free at the context's end (TileRT's plan)


def _f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _i64(u: int) -> int:
    u &= (1 << 64) - 1
    return u - (1 << 64) if u >= 1 << 63 else u


def sampling_refusal(temperature: float, top_p: float, top_k: int) -> str | None:
    """TileRT's sampler takes temperature and a seed; the recorded plan is its sampling mode (5)."""
    if temperature <= 0:
        return "greedy (temperature 0: TileRT's mode-2 graphs are not recorded)"
    if top_p != 1.0 or top_k not in (-1, 0):
        return f"top_p {top_p} / top_k {top_k} (TileRT's sampler: temperature and seed only)"
    return None


def pack_params(temperature: float, seed: int) -> list[int]:
    """``samp_params``' first three int64 (TileRT ``ops/sample.pack_params``: top_p 1, top_k 0, min_p 0)."""
    return [_i64(seed), _i64(_f32(temperature) | _f32(1.0) << 32), 0]


class TkDevice:
    """``TkRank`` as the runner drives it (GPU); every op on the current stream."""

    def __init__(self, tk):
        self.tk = tk

    def set_sampling(self, params):
        self.tk.set_sampling(params)

    def admit(self, first, pos, params):
        self.tk.admit_request(first, pos, True, params)

    def import_planes(self, planes, n):
        self.tk.import_planes(planes, n)

    def step(self):
        self.tk.step()

    def logs(self):
        t = self.tk.buffers.tensors
        return t["temp:ar_acc"][0], t["temp:ar_num"][0]

    def drafts(self):
        return self.tk.buffers.tensors["temp:vtok"]


class TkStepRunner:
    """The module docstring's runner. ``device``: a ``TkDevice``; ``token_map``: the Engram compressed-token map
    (``hist`` from tail ids), or None (from ATOM's cursor)."""

    def __init__(
        self,
        device,
        rank: int = 0,
        token_map=None,
        max_seq_len: int = 1 << 20,
        stream=None,
    ):
        self.dev, self.rank, self.token_map, self.cap = (
            device,
            rank,
            token_map,
            max_seq_len,
        )
        self.held = None  # the request whose state TkRank holds: req id
        self.pos = 0  # ATOM's anchor position of the last step
        self.counter = 0  # TileRT's sampling counter (samp_params[3]): set_sampling and every admission bump it
        self._samp_mode = None
        # the last step's (output ids, verdict, drafts), on ``stream``
        self._outs = None
        # ``stream`` holds steps ATOM's stream is not ordered after
        self._unjoined = False
        # TkRank's stream (``create``: a CUDA stream), where a served step's outputs are ready (ModelRunner's
        # ``output_stream``); None: every op on the caller's stream (the CPU tests)
        self.stream = self.output_stream = stream
        self.outputs_ready = torch.cuda.Event() if stream is not None else None

    def prepare(self, rows: int) -> bool:
        return rows == ROWS

    def holds(self, req_id) -> bool:
        return self.held is not None and req_id == self.held

    def release(self) -> None:
        self.held = None

    def join(self) -> None:
        """ATOM's stream after every step TkRank was given (before anything else of ATOM's touches the GPU's state
        or runs a collective)."""
        if self.stream is not None and self._unjoined:
            torch.cuda.current_stream().wait_stream(self.stream)
            self._unjoined = False

    def precheck(self, batch, anchor_pos: int) -> str | None:
        """Why this step cannot be served (None: it can). Request-constant, so a held request never fails it."""
        why = sampling_refusal(
            float(batch.temperatures[0]), float(batch.top_ps[0]), int(batch.top_ks[0])
        )
        if why is None and anchor_pos + ROWS + SLACK > self.cap:
            why = f"position {anchor_pos} at the plan's context end ({self.cap})"
        return why

    # ---- admission (TileRT's begin_request)
    def admit(self, req_id, first: int, n: int, temperature: float, kv) -> None:
        """A request whose KV ATOM holds for positions [0, n), its anchor token ``first`` at ``n``."""
        seed = (
            int(req_id) * 0x9E3779B97F4A7C15 + 0x5EED
        ) & 0x7FFFFFFFFFFFFFFF  # ATOM has no per-request seed
        params = pack_params(temperature, seed)
        mode = (True, tuple(params[1:]))
        if (
            self._samp_mode == mode
        ):  # dsv41_e2e_begin: the parameters ride with the admission
            self.counter += 1
            self.dev.admit(first, n, params + [self.counter])
        else:  # set_sampling, then dsv41_e2e_admit
            self.counter += 1
            self.dev.set_sampling(params + [self.counter])
            self._samp_mode = mode
            self.dev.admit(first, n, None)
        planes = (
            kv
            if isinstance(kv, dict)
            else KV.request_planes(kv, n, token_map=self.token_map)
        )
        self.dev.import_planes(planes, n)
        self.held, self.pos = req_id, n
        if self.rank == 0:
            logger.info("V4.1 TileRT-path step: admitted request %s at position %d (T %.3g)", req_id, n, temperature)  # fmt: skip

    # ---- the step
    def run_step(
        self, req_id, input_ids, n: int, temperature: float, kv_source
    ) -> None:
        """One served step of ``req_id``. A request TkRank does not hold is admitted first: its anchor token
        (``input_ids[0]``: after PD, P's token) at ATOM's anchor position ``n``, its KV from ``kv_source()``
        (``kv.RequestKV``). Neither is read for a held request (the device sync of the token, the cache views).
        """
        if not self.holds(req_id):
            self.join()  # the previous request's last steps, before its state is replaced
            self.admit(req_id, int(input_ids[0]), n, temperature, kv_source())
            if self.stream is not None:
                self.stream.wait_stream(torch.cuda.current_stream())
        # ATOM's anchor position (no device read: a step that skips the verdict wait launches before the last step's
        # verdict exists; there ``n`` runs ahead by the last step's rejected drafts)
        self.pos = n
        if self.stream is None:
            self.dev.step()
            self._outs = self._outputs()
            return
        with torch.cuda.stream(self.stream):
            self.dev.step()
            self._outs = self._outputs()
            self.outputs_ready.record()
        self._unjoined = True

    def verify_outputs(self, bs: int):
        if bs != 1:
            raise ValueError(f"TileRT-path step: {bs} requests")
        if self.stream is not None:
            cur = torch.cuda.current_stream()
            cur.wait_event(self.outputs_ready)
            for t in self._outs:
                t.record_stream(cur)
        out, verdict, _ = self._outs
        return out, verdict

    def _outputs(self):
        """(output ids, verdict, drafts) of the step just enqueued: one launch on the GPU (``outputs``)."""
        if self.stream is None:
            return self._outputs_torch()
        from . import outputs

        acc, num = self.dev.logs()
        return outputs.gather(acc, num, self.dev.drafts(), ROWS, SPEC_TOKENS)

    def _outputs_torch(self):
        acc, num = self.dev.logs()
        a = num.gather(0, num[:1].long()).to(
            torch.int32
        )  # this step's accepted tokens (the log's last entry)
        r = torch.arange(ROWS, device=acc.device, dtype=torch.int64)
        idx = (acc[:1].long() - a.long() + 1 + r).clamp(0, acc.numel() - 1)
        out = torch.where(r < a, acc.gather(0, idx), torch.full_like(idx, -1)).to(
            torch.int32
        )
        verdict = torch.stack([a - 1, ROWS - a, a - 1]).to(torch.int32)
        drafts = self.dev.drafts()[1 : 1 + SPEC_TOKENS].to(torch.int64)
        return out.view(1, ROWS), verdict.view(3, 1), drafts.view(1, SPEC_TOKENS)

    def draft_ids(self, bs: int):
        return self._outs[2]

    # ---- construction on a GPU rank
    @classmethod
    def create(
        cls,
        atom_config,
        ckpt: str,
        toktab_dir: str | None = None,
        fly_cache: str | None = None,
    ):
        """This rank's TkRank (collective over the TP group): packed parameters, our FlyDSL kernels from source, the
        token tables (built on the GPU by those kernels, or ``toktab_dir``'s, a cache the build fills), warmup and both
        graphs. Acceptance is real, or the mean acceptance length ATOM is told to force
        (``--spec-decode-acceptance-length``: ``synthetic_acceptance_length``), which head_seed_fold then simulates as
        TileRT's ``TILERT_SIMULATE_ACC_LEN`` does (the same unit: the target's own token counted).
        """
        from aiter.dist.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tp_group,
        )

        from ..pd import tkworker
        from . import codeobj, host
        from .weights import ckpt as CK

        rank = get_tensor_model_parallel_rank()
        group = get_tp_group().cpu_group
        spec = getattr(atom_config, "speculative_config", None)
        forced = getattr(spec, "synthetic_acceptance_length", None)
        tk = host.TkRank(
            rank, group, accept="real" if forced is None else float(forced)
        )
        # The packers run on the CPU; inside a ModelRunner the default device is the rank's GPU.
        with torch.device("cpu"):
            consts = CK.compute_engram_consts(ckpt)
            tk.load_params(ckpt, consts)
        table = codeobj.CodeTable.from_source(fly_cache, log=logger.info)
        tkworker.load_toktab(
            tk,
            rank,
            toktab_dir,
            log=logger.info if rank == 0 else None,
            table=table,
        )
        tk.load_kernels(table, extra=(host.ADMIT_SYMBOL,))
        tk.warmup()
        tk.capture()
        logger.info(
            "V4.1 TileRT-path step: TkRank ready (rank %d, acceptance %s)",
            rank,
            "real" if forced is None else f"simulated {float(forced):.4f}",
        )
        return cls(TkDevice(tk), rank, token_map=None, stream=torch.cuda.Stream())
