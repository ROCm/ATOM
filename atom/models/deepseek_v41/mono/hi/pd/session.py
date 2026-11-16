# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports TileRT 0.1.7 (MIT, Copyright (c) TileRT-team) ``tilert/pd_vllm/profiles/dsv41_rocm_engine.py``
# (``RocmDsV41EngineAdapter``), the request side of ``tilert/models/ds_v4_1_flash/e2e_graph.py`` (``begin_request``,
# ``set_sampling``, ``chunks``), ``sampling_ref.py`` / ``ops/sample.py`` (``SamplingParams``, ``pack_params``) and the
# decode node's checks of ``tilert/pd_vllm/capabilities.py`` / ``grammar_spec.py``.
"""One request's decode on the D host, CPU side: the sampling checks, the device admission, and the loop that keeps
up to two chunks of DSpark steps in flight and streams what they accepted.

``Decoder`` (TileRT's engine adapter) drives a ``DecodeEngine``, the device as TileRT's .so exposes it to Python:

=================================  =============================================================================
``DecodeEngine``                    TileRT 0.1.7
=================================  =============================================================================
``set_sampling(params, on)``        ``DsV41Graph.set_sampling``: ``dsv41_e2e_sync``, ``sym['sample'].zero_()`` a rank,
                                    ``dsv41_e2e_set_sampling(params, on)`` (``samp_params`` <- 4 x int64; graph mode
                                    5 when ``on`` (sampling), else 2 (greedy))
``admit(first, pos, params)``       ``dsv41_e2e_begin(first, pos, params, on)`` (``params`` given: the sampling mode
                                    is unchanged) / ``dsv41_e2e_admit(first, pos, True)`` (``params`` None): one
                                    ``admit_kernel<<<64,256>>>(AdmitArgs)`` a rank, outside the graphs
``load_kv(received, buffer)``       ``Dsv41RocmProfile.convert`` + ``DsV41Graph.import_kv``: the landed wire image
                                    into every rank's caches
``step(k) -> ticket``               ``dsv41_e2e_enqueue(True, k)``: k graph replays a rank, one event a rank
``wait(ticket)``                    ``dsv41_e2e_wait(ticket)``
``read_outputs(s, k, t)``           ``T('ar_num')[0][1+s : 1+s+k]`` (each step's accepted count) and
                                    ``T('ar_acc')[0][1+t : 1+t+sum]`` (the accepted tokens), rank 0
=================================  =============================================================================

The device logs ``ar_acc`` / ``ar_num`` are ``[count, entries...]``; the admission writes ``[1, first]`` / ``[1, 1]``
(the first token, "accepted" alone), every step appends its accepted tokens / their count. ``Decoder`` keeps the read
cursors (both 1 after an admission), so the device never needs a host round trip between chunks.

Request semantics kept from TileRT (``dsv41_rocm_engine.py:65-171``): the first token (P's) counts toward
``max_tokens`` and is streamed first; a stop id (EOS 1, none with ``ignore_eos``) ends the reply and is not streamed; the
budget is ``min(max_tokens, max_seq_len - seq_len - 1)``; a chunk is at most ``ar_steps`` (8) steps, no more than the
budget, the cache room (8 positions of slack) and the device log's room need; tokens past a stop or the budget are
dropped (the device may have computed them); a cancel stops enqueueing and the chunks in flight still stream. One
deliberate difference: a zero budget streams the first token (TileRT returns it without streaming it, which its router
then reports as a truncated stream).

Engines: ``FakeEngine`` (CPU: the logs as the admission and the steps write them, scripted accept counts and tokens;
for tests and for wiring a real P and router to this host without a decode) and ``TkEngine`` (ATOM's ``tk.TkRank``).

``TkEngine`` is not usable yet. Implemented: ``read_outputs`` (rank 0's ``temp:ar_num`` / ``temp:ar_acc``) and
``wait``; ``check_sampling`` refuses greedy requests. Stubs that raise ``NotImplementedError``:

- ``admit``: ``admit_kernel``'s 176-byte ``AdmitArgs`` (``TkEngine.admit`` gives the layout a static read of its ISA
  finds; the buffers it names need a ``TKSHIM_REC=all`` recording to confirm);
- ``set_sampling``: ``temp:samp_params`` and the ``sym:sample`` zeroing on all 8 ranks, and greedy's mode-2 graphs
  (``plan/decode_launches.json`` holds mode 5 only);
- ``load_kv``: ``pd.convert`` on rank 0 and the planes into all 8 ranks' ``cache:`` buffers (one process a rank);
- ``step``: the fan-out of a chunk to the 8 rank processes (every rank must replay its graph; a rank alone hangs in
  the step's cross-rank polls);
- ``build_toktab``: the DSpark token tables (TileRT ``toktab.build_table``: its embed / mHC / q-kv / kv-extract ops
  over the vocabulary), which ``tk.alloc.Buffers.toktab`` takes ready-made.
"""

from __future__ import annotations

import logging
import random
import struct
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from . import wire as W

logger = logging.getLogger(__name__)

EOS_ID = 1  # tilert/models/ds_v4_1_flash/generator.py: the decode stop set {EOS_ID}
VOCAB = 129280
NUM_MTP = 5
AR_STEPS = 8  # decode_server.py builds the engine with ar_steps=8
LOG_SLACK = 64  # e2e_graph.stream_len(cap) = cap + 64: the ar_acc / ar_num logs' length


# ---------------------------------------------------------------------------------------------------------------
# refusals (capabilities.py, grammar_spec.py): an error_type the router propagates and its HTTP status


class RequestRefused(Exception):
    error_type = "capability_error"
    http_status = 500

    def to_payload(self) -> dict:
        return {"error": str(self), "error_type": self.error_type}


class InvalidParameter(RequestRefused):
    error_type = "invalid_parameter"
    http_status = 400


class CapabilityUnavailable(RequestRefused):
    error_type = "capability_unavailable"
    http_status = 501


class GrammarUnsupported(RequestRefused):
    error_type = "grammar_unsupported"
    http_status = 501


# validate_generation_request's tables: field -> (kind, neutral, minimum, why) / (kind, neutral, capability, why)
_EMPTY, _UNSET, _NUMBER, _FLAG, _COUNT = ("empty", "unset", "number", "flag", "count")
_STATIC_FIELDS = {
    "stop_token_ids": (
        _EMPTY,
        None,
        None,
        "the decode loop uses the model's own stop set and accepts no per-request ids",
    ),
    "min_tokens": (
        _COUNT,
        0,
        0,
        "the decode loop cannot suppress its stop set for a minimum length",
    ),
    "frequency_penalty": (
        _NUMBER,
        0.0,
        None,
        "the decode sampler implements repetition and presence penalties only",
    ),
    "min_p": (_NUMBER, 0.0, None, "the decode sampler implements top-p and top-k only"),
    "seed": (
        _UNSET,
        None,
        None,
        "the decode sampler's seed is per-process, so a per-request seed cannot make the reply reproducible",
    ),
    "logit_bias": (
        _EMPTY,
        None,
        None,
        "the decode sampler has no per-request logit bias",
    ),
    "bad_words": (_EMPTY, None, None, "the decode loop has no bad-words matcher"),
    "allowed_token_ids": (
        _EMPTY,
        None,
        None,
        "the decode sampler has no per-request allow list",
    ),
    "structured_outputs": (
        _EMPTY,
        None,
        None,
        "use response_format, which this stack translates into a decode-side grammar",
    ),
    "n": (
        _COUNT,
        1,
        1,
        "one prefill KV state is transferred per request, so the decode node produces exactly one sequence",
    ),
    "best_of": (
        _COUNT,
        1,
        1,
        "the decode node produces exactly one sequence, so there is nothing to select from",
    ),
    "use_beam_search": (
        _FLAG,
        False,
        None,
        "the decode node runs single-sequence AR/MTP decode",
    ),
    "prompt_logprobs": (
        _UNSET,
        None,
        None,
        "the prefill instance is asked for one token, so no prompt distribution is collected",
    ),
    "logprob_token_ids": (
        _EMPTY,
        None,
        None,
        "the decode export returns the top candidates, not a caller-chosen vocab subset",
    ),
    "skip_special_tokens": (
        _FLAG,
        True,
        None,
        "the reply is detokenised for the output parser, which always consumes special tokens",
    ),
}
_PROFILE_FIELDS = {
    "repetition_penalty": (
        _NUMBER,
        1.0,
        "penalties",
        "this model's decode runtime has no penalty pre-pass",
    ),
    "presence_penalty": (
        _NUMBER,
        0.0,
        "penalties",
        "this model's decode runtime has no penalty pre-pass",
    ),
    "ignore_eos": (
        _FLAG,
        False,
        "ignore_eos",
        "this model's decode loop does not clear its stop set",
    ),
}
_TYPED_FIELDS = {
    "temperature": (_NUMBER, 0.0),
    "top_p": (_NUMBER, 0.0),
    "top_k": (_COUNT, None),
}
_GREEDY_TEMPERATURE = 1e-05


def _lenient_float(field: str, raw: object) -> float:
    if isinstance(raw, bool):
        raise InvalidParameter(f"{field} must be a number, got bool")
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise InvalidParameter(f"{field} must be a number, got {raw!r}") from None


def _is_neutral(field: str, value, kind: str, neutral, minimum) -> bool:
    if value is None:
        return True
    if kind == _EMPTY:
        return not value
    if kind == _UNSET:
        return False
    if kind == _NUMBER:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidParameter(
                f"{field} must be a number, got {type(value).__name__}"
            )
        return float(value) == neutral
    if kind == _FLAG:
        if not isinstance(value, bool):
            raise InvalidParameter(
                f"{field} must be a boolean, got {type(value).__name__}"
            )
        return value is neutral
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidParameter(
            f"{field} must be an integer, got {type(value).__name__}"
        )
    if value < (minimum or 0):
        raise InvalidParameter(f"{field} must be >= {minimum or 0}, got {value}")
    return value == neutral


def validate_sampling(body: dict, capabilities: dict) -> None:
    """TileRT's ``validate_generation_request(body, capabilities)`` on the decode call's ``sampling``: typed fields,
    fields no TileRT decode implements, fields this node's ``capabilities`` (``penalties`` / ``ignore_eos``) gate.
    """
    for field, (kind, minimum) in _TYPED_FIELDS.items():
        if body.get(field) is None:
            continue
        value = _lenient_float(field, body[field])
        if kind == _NUMBER:
            if minimum is not None and value < minimum:
                raise InvalidParameter(f"{field} must be >= {minimum}, got {value}")
        elif value != int(value):
            raise InvalidParameter(f"{field} must be an integer, got {body[field]!r}")
    raw_t = body.get("temperature")
    greedy = (
        raw_t is not None and _lenient_float("temperature", raw_t) < _GREEDY_TEMPERATURE
    )
    for field, (kind, neutral, minimum, why) in _STATIC_FIELDS.items():
        if field not in body or _is_neutral(field, body[field], kind, neutral, minimum):
            continue
        if field == "seed" and greedy:
            continue
        raise CapabilityUnavailable(
            f"{field} is not supported by the TileRT decode stage: {why}. The vLLM prefill instance would apply it to "
            "the first token and the decode node would ignore it for the rest of the reply, so the request is refused "
            "instead of served incorrectly."
        )
    for field, (kind, neutral, attr, why) in _PROFILE_FIELDS.items():
        if field not in body or _is_neutral(field, body[field], kind, neutral, None):
            continue
        if capabilities.get(attr):
            continue
        raise CapabilityUnavailable(
            f"{field} (from the request) is not supported by the decode node serving this pool: {why}. It would apply "
            "to the first token only, so the request is refused instead of served incorrectly."
        )


# ---------------------------------------------------------------------------------------------------------------
# sampling (sampling_ref.py, ops/sample.py, dsv41_rocm_engine._sampling_params)


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int = 0


def check_supported(sp: SamplingParams | None) -> None:
    """The device sampler takes temperature and seed only (greedy: ``None`` or temperature <= 0)."""
    if sp is None or sp.temperature <= 0.0:
        return
    if sp.top_p != 1.0 or 0 < sp.top_k < VOCAB or sp.min_p > 0.0:
        raise ValueError(
            f"top_p != 1, top_k and min_p are not supported (got top_p={sp.top_p}, top_k={sp.top_k}, "
            f"min_p={sp.min_p}); sampling takes temperature and seed only"
        )


def sampling_params(sampling: dict) -> SamplingParams | None:
    """A decode call's ``sampling`` as the device sampler's parameters; ``None`` = greedy (temperature <= 0). No
    ``seed``: a fresh random one, as TileRT draws it."""
    t = sampling.get("temperature")
    t = 1.0 if t is None else float(t)
    if t <= 0.0:
        return None
    seed = sampling.get("seed")
    return SamplingParams(
        temperature=t,
        top_p=float(sampling.get("top_p") or 1.0),
        top_k=int(sampling.get("top_k") or 0),
        min_p=float(sampling.get("min_p") or 0.0),
        seed=int(seed) if seed is not None else random.getrandbits(63),
    )


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _i64(u: int) -> int:
    u &= (1 << 64) - 1
    return u - (1 << 64) if u >= 1 << 63 else u


def pack_params(sp: SamplingParams) -> list[int]:
    """``samp_params``' 4 x int64: seed; temperature | top_p << 32 (float32 bits); top_k | min_p << 32; 0."""
    return [
        _i64(sp.seed),
        _i64(_f32_bits(sp.temperature) | _f32_bits(sp.top_p) << 32),
        _i64(sp.top_k & 0xFFFFFFFF | _f32_bits(sp.min_p) << 32),
        0,
    ]


# ---------------------------------------------------------------------------------------------------------------
# the device interface and the decode loop


class DecodeEngine(Protocol):
    """The device, as the decode loop drives it (module docstring's table)."""

    name: str
    max_seq_len: int
    num_mtp: int
    stop_token_ids: frozenset

    def check_sampling(self, sp: SamplingParams | None) -> None: ...

    def set_sampling(self, params: list[int], on: bool) -> None: ...

    def admit(
        self, first_token: int, pos: int, params: list[int] | None = None
    ) -> None: ...

    def load_kv(self, received: W.ReceivedRequest, buffer: Any) -> None: ...

    def step(self, k: int) -> Any: ...

    def wait(self, ticket: Any) -> None: ...

    def read_outputs(
        self, acc_from: int, k: int, tok_from: int
    ) -> tuple[list[int], list[int]]: ...


class Decoder:
    """TileRT's ``RocmDsV41EngineAdapter`` over a ``DecodeEngine``: ``check_sampling`` / ``prepare_grammar`` (the
    refusals), ``begin`` + ``inject`` (the admission and the KV, in TileRT's order), ``decode`` (the loop), ``reset``.
    """

    def __init__(self, engine: DecodeEngine, ar_steps: int = AR_STEPS, ahead: int = 2):
        self.engine = engine
        self.max_seq_len = int(engine.max_seq_len)
        self.per = int(engine.num_mtp) + 1
        self.ar_steps = max(1, int(ar_steps))
        self.ahead = max(1, int(ahead))
        self.stop_ids = frozenset(int(t) for t in engine.stop_token_ids)
        self.last_stats: dict = {}
        self._seq_len = 0
        self._begin: tuple | None = None
        self._begun: int | None = None
        self._samp_mode: tuple | None = None
        self._cursor = (1, 1)
        self._t_begin_ms = 0.0

    # ---- what the node serves
    def supports_logprobs(self) -> bool:
        return False

    def supports_penalties(self) -> bool:
        return False

    def supports_ignore_eos(self) -> bool:
        return True

    def capabilities(self) -> dict:
        return {
            "penalties": self.supports_penalties(),
            "ignore_eos": self.supports_ignore_eos(),
        }

    def prepare_grammar(self, grammar_spec, enable_thinking: bool = True) -> None:
        """No grammar session: a ``grammar_spec`` is refused."""
        if grammar_spec is not None:
            raise GrammarUnsupported("constrained decoding is not supported")

    def check_sampling(self, sampling: dict | None) -> None:
        try:
            sp = sampling_params(sampling or {})
            check_supported(sp)
        except ValueError as e:
            raise CapabilityUnavailable(str(e)) from None
        self.engine.check_sampling(sp)

    # ---- the admission
    def _set_sampling(self, sp: SamplingParams | None) -> None:
        check_supported(sp)
        on = sp is not None and sp.temperature > 0.0
        params = pack_params(sp if on else SamplingParams())
        self.engine.set_sampling(params, on)
        self._samp_mode = (on, tuple(params[1:]))

    def _begin_request(self, sp: SamplingParams | None, first: int, pos: int) -> None:
        """``DsV41Graph.begin_request``: the parameters ride along with the admission while the sampling mode (all but
        the seed) is unchanged, else the mode is set first."""
        check_supported(sp)
        on = sp is not None and sp.temperature > 0.0
        params = pack_params(sp if on else SamplingParams())
        if self._samp_mode == (on, tuple(params[1:])):
            self.engine.admit(int(first), int(pos), params)
        else:
            self._set_sampling(sp)
            self.engine.admit(int(first), int(pos), None)
        self._cursor = (1, 1)

    def begin(self, first_token_id: int, sampling: dict | None) -> None:
        self._begin = (int(first_token_id), sampling or {})

    def inject(self, received: W.ReceivedRequest, buffer: Any) -> None:
        """The request's admission (``begin``'s first token and sampling) and its landed KV."""
        seq = int(received.seq_len)
        if seq <= 0 or seq >= self.max_seq_len:
            raise RuntimeError(
                f"dsv41 inject: seq_len {seq} outside (0, {self.max_seq_len})"
            )
        begin, self._begin, self._begun = self._begin, None, None
        t = time.perf_counter()
        if begin is not None:
            self._begin_request(sampling_params(begin[1]), begin[0], seq)
            self._begun = begin[0]
        self._t_begin_ms = round(1000 * (time.perf_counter() - t), 2)
        self.engine.load_kv(received, buffer)
        self._seq_len = seq

    # ---- the loop
    def decode(
        self,
        first_token_id: int,
        max_tokens: int,
        sampling: dict | None,
        on_token: Callable[[int], None] | None = None,
        cancel_event=None,
        grammar_session=None,
        top_logprobs: int | None = None,
    ) -> list[int]:
        """The reply's token ids, the first included; each is passed to ``on_token`` as it is accepted.
        ``last_stats['finish_reason']``: ``stop`` / ``length`` / ``cancelled``."""
        if grammar_session is not None:
            raise GrammarUnsupported("constrained decoding is not supported")
        if top_logprobs:
            raise NotImplementedError("logprobs are not supported")
        sampling = sampling or {}
        first = int(first_token_id)
        begun, self._begun = self._begun == first, None
        if not begun:
            self._set_sampling(sampling_params(sampling))
        stop_ids = frozenset() if sampling.get("ignore_eos") else self.stop_ids
        budget = min(int(max_tokens), self.max_seq_len - self._seq_len - 1)
        if budget <= 0:
            self.last_stats = {"finish_reason": "length"}
            if on_token:
                on_token(first)
            return [first]
        if first in stop_ids:
            self.last_stats = {"finish_reason": "stop"}
            return []
        tokens = [first]
        if on_token:
            on_token(first)
        if not begun:
            self.engine.admit(first, self._seq_len, None)
            self._cursor = (1, 1)
        self.last_stats = self._run(tokens, budget, stop_ids, on_token, cancel_event)
        self.last_stats["a_begin"] = self._t_begin_ms if begun else 0
        return tokens

    def _run(self, tokens, budget, stop_ids, on_token, cancel_event) -> dict:
        eng, per = self.engine, self.per
        n_log = self.max_seq_len + LOG_SLACK
        t_cnt, s_cnt = self._cursor
        logged, steps = 1, 0
        finish = None
        accepts: list[int] = []
        t0 = time.perf_counter()
        t_first = None

        def emit(new) -> str | None:
            for tok in new:
                tok = int(tok)
                if tok in stop_ids:
                    return "stop"
                if len(tokens) >= budget:
                    return "length"
                tokens.append(tok)
                if on_token:
                    on_token(tok)
            return "length" if len(tokens) >= budget else None

        def plan(fly_tok: int, fly_steps: int) -> int:
            if finish or (cancel_event is not None and cancel_event.is_set()):
                return 0
            room = (
                self.max_seq_len - (self._seq_len + len(tokens) + fly_tok) - 8
            ) // per
            log_room = min(
                (n_log - 2 - logged - fly_tok) // per, n_log - 2 - steps - fly_steps
            )
            want = -(-(budget - len(tokens) - fly_tok) // per)
            return max(0, min(self.ar_steps, want, room, log_room))

        fly: deque = deque()
        while True:
            while len(fly) < self.ahead:
                fs = sum(k for _, k in fly)
                k = plan(fs * per, fs)
                if k < 1:
                    break
                fly.append((eng.step(k), k))
            if not fly:
                break
            ticket, k = fly.popleft()
            eng.wait(ticket)
            acc, new = eng.read_outputs(s_cnt, k, t_cnt)
            if len(new) != sum(acc):
                raise RuntimeError(
                    f"device logs: {sum(acc)} tokens accepted, {len(new)} logged"
                )
            s_cnt += k
            t_cnt += len(new)
            steps += k
            logged += len(new)
            accepts += acc
            if t_first is None:
                t_first = time.perf_counter()
            finish = emit(new)
            if finish:
                break
        if finish is None:
            cancelled = cancel_event is not None and cancel_event.is_set()
            finish = "cancelled" if cancelled else "length"
        stats = {"finish_reason": finish, "steps": steps}
        if t_first is not None:
            stats["a_chunk1"] = round(1000 * (t_first - t0), 2)
        if accepts:
            stats["mtp_verify_calls"] = len(accepts)
            stats["mtp_accept_mean"] = round(sum(accepts) / len(accepts), 3)
        return stats

    def reset(self) -> None:
        self._seq_len = 0


# ---------------------------------------------------------------------------------------------------------------
# engines


class FakeEngine:
    """A CPU ``DecodeEngine``: the device logs as the admission and the steps write them. Step i accepts
    ``accepts[i % len]`` tokens; the reply's token at index j (the first is index 0) is ``token(first, j)``, or EOS at
    ``eos_at``. ``load_kv`` checks a landed header (``wire.parse_header``) when given a buffer. ``step_s``: seconds a
    step takes. ``calls`` records what was asked of it."""

    name = "fake"

    def __init__(
        self,
        max_seq_len: int = 1 << 20,
        num_mtp: int = NUM_MTP,
        accepts=(3, 6, 1, 4, 2, 5),
        eos_at: int | None = None,
        step_s: float = 0.0,
    ):
        self.max_seq_len, self.num_mtp = int(max_seq_len), int(num_mtp)
        self.stop_token_ids = frozenset({EOS_ID})
        self.accepts = tuple(int(a) for a in accepts)
        if not all(1 <= a <= self.num_mtp + 1 for a in self.accepts):
            raise ValueError(f"accepts {self.accepts}: each in [1, {self.num_mtp + 1}]")
        self.eos_at, self.step_s = eos_at, step_s
        self.calls: list[tuple] = []
        self.ar_acc, self.ar_num = [0], [0]
        self.first, self.next_index, self.nsteps, self.tickets = 0, 0, 0, 0
        self.params: list[int] | None = None
        self.on: bool | None = None

    @staticmethod
    def token(first: int, j: int) -> int:
        """The scripted reply's token j >= 1: never EOS."""
        return 2 + (first * 31 + j * 17) % (VOCAB - 2)

    def check_sampling(self, sp) -> None:
        pass

    def set_sampling(self, params: list[int], on: bool) -> None:
        self.calls.append(("set_sampling", on))
        self.params, self.on = list(params), on

    def admit(
        self, first_token: int, pos: int, params: list[int] | None = None
    ) -> None:
        self.calls.append(("admit", first_token, pos, params is not None))
        if params is not None:
            self.params = list(params)
        self.first, self.next_index, self.nsteps = int(first_token), 1, 0
        self.ar_acc, self.ar_num = [1, int(first_token)], [1, 1]

    def load_kv(self, received: W.ReceivedRequest, buffer) -> None:
        self.calls.append(("load_kv", received.rid, received.seq_len))
        if buffer is not None:
            h = W.parse_header(buffer[W.regions(self.max_seq_len)["hdr"].offset :])
            if h["n"] != int(received.seq_len):
                raise RuntimeError(
                    f"landed header n={h['n']} != seq_len {received.seq_len}"
                )

    def step(self, k: int) -> int:
        if self.step_s:
            time.sleep(self.step_s * k)
        for _ in range(k):
            a = self.accepts[self.nsteps % len(self.accepts)]
            for _ in range(a):
                j = self.next_index
                self.ar_acc.append(
                    EOS_ID if j == self.eos_at else self.token(self.first, j)
                )
                self.next_index += 1
            self.ar_acc[0] += a
            self.ar_num.append(a)
            self.ar_num[0] += 1
            self.nsteps += 1
        self.tickets += 1
        self.calls.append(("step", k))
        return self.tickets

    def wait(self, ticket) -> None:
        pass

    def read_outputs(
        self, acc_from: int, k: int, tok_from: int
    ) -> tuple[list[int], list[int]]:
        acc = self.ar_num[1 + acc_from : 1 + acc_from + k]
        n = sum(acc)
        return list(acc), self.ar_acc[1 + tok_from : 1 + tok_from + n]


_STUB = {
    "admit": "admit_kernel's AdmitArgs (176 B) are not built yet: the layout read from its ISA (TkEngine.admit) "
    "needs a TKSHIM_REC=all recording to confirm which temps it names (TileRT: dsv41_e2e_begin / dsv41_e2e_admit)",
    "set_sampling": "the sampling mode is not set yet: temp:samp_params and the sym:sample zeroing on all 8 ranks "
    "(TileRT: dsv41_e2e_set_sampling); greedy needs mode-2 graphs, which plan/decode_launches.json (mode 5) lacks",
    "load_kv": "the landed KV is not imported yet: pd.convert on rank 0 and the planes into the cache: buffers of all 8 "
    "rank processes (TileRT: profile.convert + DsV41Graph.import_kv)",
    "step": "the 8-rank fan-out is not built yet: every rank process must replay its graph k times (TileRT: "
    "dsv41_e2e_enqueue); one rank alone hangs in the step's cross-rank polls",
    "toktab": "the DSpark token tables are not built yet (TileRT: toktab.build_table, its embed / mHC / q-kv / "
    "kv-extract ops over the vocabulary); tk.alloc.Buffers.toktab takes tables built elsewhere",
}


class TkEngine:
    """The ``DecodeEngine`` over ATOM's TP8 ``tk.TkRank``s (``tk``: this process's, rank 0's; ``peers``: the fan-out
    to ranks 1..7). Not usable yet: the stubs of the module docstring raise ``NotImplementedError``.
    """

    name = "tk"

    def __init__(self, tk=None, peers=None):
        from ..tk import alloc

        self.tk, self.peers = tk, peers
        self.max_seq_len = alloc.MAX_SEQ_LEN
        self.num_mtp = NUM_MTP
        self.stop_token_ids = frozenset({EOS_ID})

    def check_sampling(self, sp: SamplingParams | None) -> None:
        if sp is None:
            raise CapabilityUnavailable(
                "greedy decoding (temperature 0) is not served: it runs TileRT's mode-2 graphs, and this host's plan "
                "(plan/decode_launches.json) holds the mode-5 (sampling) step only"
            )

    def set_sampling(self, params: list[int], on: bool) -> None:
        raise NotImplementedError(_STUB["set_sampling"])

    def admit(
        self, first_token: int, pos: int, params: list[int] | None = None
    ) -> None:
        """Stub. ``admit_kernel<<<64, 256>>>(AdmitArgs)`` on every rank's step stream, outside the graphs. The
        176-byte ``AdmitArgs`` as a static read of the kernel's ISA finds them (unverified on a GPU; the buffer names
        are the likely ones, F = the flag at 0x84, L = the limit at 0x80):

        - 0x00 int32* / 0x08 n: zeroed (``plain_acc``); 0x10 / 0x18: ``[F, F ? first : 0, 0...]`` (``ar_acc``);
          0x20 / 0x28: ``[F, F, 0...]`` (``ar_num``); 0x30 int32[16]: ``[0] = pos - 1``, ``[1] = first``, ``[5] = 1``
          when F, ``[3] = L`` always (``ctl``);
        - when F, workgroup 0: 0x38 / 0x40 / 0x48 tokens, positions, n: ``tok[i] = i ? 128799 : first``, ``pos[i] =
          pos + i`` for ``i < 5`` and ``pos + i < L``, else -1; 0x50 / 0x58 / 0x60 the same without ``i < 5``
          (``blk_tok`` / ``blk_pos``, ``vtok`` / ``vpos``);
        - when F, workgroups ``w < n(0x48)``: 0x68 bf16* embedding table, 0x70 bf16* rows: row ``w`` = the 4 mHC copies
          of the embedding of ``first`` (``w == 0``) or 128799, zeros past the vocabulary (``param:embed`` ->
          ``temp:X``, the first step's input);
        - 0x78 pos, 0x7c first, 0x80 L, 0x84 F (int32); 0x88 int64*: when non-null, gets the 4 x int64 at 0x90
          (``samp_params``; the parameters ride along with ``dsv41_e2e_begin``).
        """
        raise NotImplementedError(_STUB["admit"])

    def load_kv(self, received: W.ReceivedRequest, buffer) -> None:
        raise NotImplementedError(_STUB["load_kv"])

    def step(self, k: int):
        raise NotImplementedError(_STUB["step"])

    def wait(self, ticket) -> None:
        ticket.synchronize()  # a torch.cuda.Event recorded after the chunk's replays

    def read_outputs(
        self, acc_from: int, k: int, tok_from: int
    ) -> tuple[list[int], list[int]]:
        t = self.tk.buffers.tensors
        acc = t["temp:ar_num"][0, 1 + acc_from : 1 + acc_from + k].tolist()
        n = sum(acc)
        return acc, t["temp:ar_acc"][0, 1 + tok_from : 1 + tok_from + n].tolist()


def build_toktab(*args, **kwargs):
    """The DSpark token tables of a rank (``toktab:layers.0.*`` / ``toktab:mtp.0.*``). Stub."""
    raise NotImplementedError(_STUB["toktab"])
