# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Wire shapes of the plugin hot-plug command, without the engine behind them.

The API server, the engine core manager and the model runner all need to agree
on what a ``POST /plugin/hotplug`` body is and on how the per-rank reports of a
hot-plug are merged.  Each of those three modules imports an inference engine
(AITER, the model runner, FastAPI); this one imports neither, so the payload
rules and the row arithmetic can be exercised on a CPU with only a recipe.

Nothing here touches the GPU or the plugin host's state: a payload is parsed
into a :class:`HotplugRequest`, the runner resolves and applies it, and the
manager merges what the ranks report.

The same file carries the rules of the plugin **round** -- the lease a
benchmark or evaluation run takes on the running server, so that a hot-plug
arriving mid-measurement is refused with ``round-in-progress`` instead of
swapping the plan underneath it.  The lease itself is two attributes on each
engine (:class:`PluginRound` is just the read-only view of them); the
transitions and the refusal envelopes are here so they can be tested without
an engine.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atom import plugin_recipe

# The utility command the EngineCore handler registers, and the internal
# endpoint that sends it.  Shared constants so the three planes cannot drift.
HOTPLUG_CMD = "hotplug_plugin"
HOTPLUG_ENDPOINT = "/plugin/hotplug"
# A capture of a handful of decode buckets is seconds, not minutes; the
# default leaves room for a slow first kernel build without waiting on a
# wedged rank forever.  Overridable per request ("timeout", seconds).
DEFAULT_HOTPLUG_TIMEOUT_S = 300.0
# Dev-mode code reload (atom/plugin_host.reload_plugins) is requested with
# "reload": true on the same hot-plug body.  `code_digest` is optional: when a
# caller sends the digest it computed (a code watcher did), every rank must
# agree with it before any module is dropped.  The two reason codes travel to
# the HTTP JSON like the round reasons, so a watcher can tell "this server is
# not a dev server" from "my digest is stale, re-read and retry".
RELOAD_UNSUPPORTED_REASON = "reload-unsupported"
RELOAD_DIGEST_MISMATCH_REASON = "reload-digest-mismatch"
_SHA256_DIGEST = re.compile(r"[0-9a-fA-F]{64}")

# The plugin-round lease: a benchmark/evaluation run holds the plan, so a
# hot-plug that arrives mid-run must not swap it.  Begin/end are synchronous
# utility commands, one round is active per engine, and neither command pauses
# generation or waits for anything -- the busy loop only writes two fields.
PLUGIN_ROUND_ENDPOINT = "/plugin/round"
PLUGIN_STATUS_ENDPOINT = "/plugin/status"
BEGIN_ROUND_CMD = "begin_plugin_round"
END_ROUND_CMD = "end_plugin_round"
# The reasons a round command, or a hot-plug during a round, is refused with.
# They travel in the utility reply's "reason" field up to the HTTP JSON, so a
# client can tell "wait for the round" from "the id is wrong".
ROUND_IN_PROGRESS_REASON = "round-in-progress"
ROUND_MISMATCH_REASON = "round-mismatch"
NO_ACTIVE_ROUND_REASON = "no-active-round"
ROUND_STATE_MISMATCH_REASON = "round-state-mismatch"
# A begin/end is a state write on every engine, not a capture: one round trip
# is enough.  The body's optional "timeout" overrides it per request.
DEFAULT_PLUGIN_ROUND_TIMEOUT_S = 30.0


class HotplugPayloadError(ValueError):
    """A request the endpoint can refuse with its ``reason`` and a HTTP 400."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


class RoundPayloadError(HotplugPayloadError):
    """A ``POST /plugin/round`` body the endpoint refuses with a HTTP 400."""


class PluginRefusal(RuntimeError):
    """A state refusal carrying the reason code an endpoint answers with.

    ``reason`` is the machine-readable code (``round-in-progress``,
    ``round-mismatch``, ...) that becomes ``error.reason`` in the JSON
    envelope; :func:`refusal_status` maps every instance to a HTTP 409.
    """

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


@dataclass(frozen=True)
class HotplugRequest:
    """What the endpoint validated: a plan source plus what to do with it.

    ``source`` is a ``plugin_recipe.RecipeSource`` (plain, picklable data) and
    is what travels to every TP rank; each rank resolves it to the same Plan
    (the ``plan_id`` checks that).  ``rows`` are padded ``running_tokens`` of
    the decode request graphs to capture -- empty means "derive them from the
    plan" (the runner does that, since only it knows the verify width).

    ``reload`` asks a dev-mode server to re-import the plugin code under its
    ``--plugin-dev-root`` trees before installing the plan; ``code_digest`` is
    the optional fingerprint the caller computed over that code (normalized to
    ``sha256:<64 lowercase hex>``), which every rank must match.
    """

    source: Any
    arm: str
    rows: tuple[int, ...]
    capture: bool
    timeout: float
    reload: bool = False
    code_digest: str | None = None


@dataclass(frozen=True)
class PluginRoundRequest:
    """A validated ``POST /plugin/round`` body: which end of the lease to move.

    ``round_id`` is the benchmark run's own name; it is what a later hot-plug
    refusal reports and what ``end`` must match.  ``timeout`` bounds the
    synchronous round trip only, not the round itself: a round lasts until it
    is ended.
    """

    action: str
    round_id: str
    timeout: float


@dataclass(frozen=True)
class PluginRound:
    """The round an engine holds, as read off its two state attributes."""

    round_id: str
    started_at: float


def _canonical(document: Any) -> str:
    """The canonical JSON text a recipe is carried as.

    Byte-identical to ``atom.plugin_recipe``'s own canonical form, because a
    ``RecipeSource`` compares the plan it resolves to against the ``plan_id``
    computed when it was built.
    """
    return json.dumps(document, sort_keys=True, separators=(",", ":"), default=str)


def _documents_of(items: Iterable[Any]) -> tuple[str, ...]:
    """Recipe documents given as JSON text, as mappings, or as file paths."""
    documents: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            documents.extend(_documents_of([_canonical(item)]))
        elif isinstance(item, str) and item.strip():
            text = item.strip()
            try:
                documents.append(_canonical(json.loads(text)))
                continue
            except ValueError:
                pass
            path = Path(text)
            if path.is_dir():
                files = sorted(path.glob("*.json"))
            elif path.is_file():
                files = [path]
            else:
                raise HotplugPayloadError(
                    "recipe-unreadable",
                    f"recipe item is neither JSON text nor an existing path: "
                    f"{text[:120]!r}",
                )
            for file in files:
                try:
                    documents.append(
                        _canonical(json.loads(file.read_text(encoding="utf-8")))
                    )
                except (OSError, ValueError) as error:
                    raise HotplugPayloadError(
                        "recipe-unreadable", f"{file}: {error}"
                    ) from error
        else:
            raise HotplugPayloadError(
                "payload-schema",
                "recipe must be a list of JSON strings, recipe objects or paths",
            )
    return tuple(documents)


def _plan_of(documents: tuple[str, ...], enable: tuple[str, ...]):
    """Recipes plus ``enable`` -> Plan, through the recipe module's own rules.

    ``atom.plugin_recipe._plan_of`` is the one function that both applies the
    ``--plugin-enable`` narrowing and computes the plan id a ``RecipeSource``
    checks itself against; re-deriving either here would let the two drift.
    Falls back to the public pieces if the helper ever moves.
    """
    plan_of = getattr(plugin_recipe, "_plan_of", None)
    if callable(plan_of):
        return plan_of(documents, enable)
    plan = (
        plugin_recipe.resolve([json.loads(d) for d in documents]) if documents else None
    )
    names = plugin_recipe.shorthand(enable)
    if plan is None or names is None:
        return names if plan is None else plan
    arms = dict(plan.arms)
    arms["plugin"] = arms["identity"] = names.providers
    return plugin_recipe.replace(
        plan,
        arms=plugin_recipe.MappingProxyType(arms),
        plan_id=plugin_recipe._digest([plan.plan_id, "enable", names.providers]),
    )


def source_from_body(
    recipe: Any = None,
    enable: Any = None,
    arm: str = "plugin",
    model_arch: str | None = None,
):
    """The ``RecipeSource`` a hot-plug body names, validated now.

    Resolving once here refuses a bad recipe at the endpoint, before any rank
    is asked to change its plan; the source (not the plan) travels so each rank
    resolves it again and the ``plan_id`` catches a recipe that resolves
    differently in a worker.
    """
    if recipe is None:
        recipe = []
    if isinstance(recipe, (str, Mapping)):
        recipe = [recipe]
    if not isinstance(recipe, (list, tuple)):
        raise HotplugPayloadError("payload-schema", "recipe must be a list")

    if enable is None:
        enable = []
    if isinstance(enable, str):
        enable = [n.strip() for n in enable.split(",") if n.strip()]
    if not isinstance(enable, (list, tuple)) or not all(
        isinstance(n, str) and n for n in enable
    ):
        raise HotplugPayloadError(
            "payload-schema", "enable must be a list of non-empty names"
        )
    names = tuple(dict.fromkeys(enable))
    if not recipe and not names:
        raise HotplugPayloadError(
            "no-recipe",
            "a hot-plug needs a recipe and/or an enable list; use arm=native "
            "with a recipe to switch an installed plan to the native arm",
        )

    documents = _documents_of(recipe)
    plan = _plan_of(documents, names)
    if plan is None:
        raise HotplugPayloadError("no-recipe", "the recipe/enable list is empty")
    if model_arch is not None and plan.model_arch != model_arch:
        raise HotplugPayloadError(
            "recipe-model-mismatch",
            f"recipe is for {plan.model_arch!r}, this server serves {model_arch!r}",
        )
    if not isinstance(arm, str) or not arm:
        raise HotplugPayloadError("payload-schema", "arm must be a non-empty string")
    if arm not in plan.arms:
        raise HotplugPayloadError(
            "unknown-arm",
            f"{arm!r} not in {sorted(plan.arms)}; a hot-plug switches arms of a "
            f"recipe, it cannot add one",
        )
    return plugin_recipe.RecipeSource(
        documents=documents,
        enable=names,
        plan_id=plan.plan_id,
        empty=plan.empty,
    )


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HotplugPayloadError(
            "rows-invalid", f"{where} must be a positive integer, got {value!r}"
        )
    return value


def checked_code_digest(value: Any) -> str | None:
    """A dev-mode code digest as a request may carry it.

    Accepts ``sha256:<64 hex>`` or a bare 64-hex string and normalizes to
    ``sha256:<lowercase hex>``, the form ``plugin_host.plugin_code_digest``
    returns and the ranks compare against.  None stays None (the caller is
    saying "reload to whatever is on disk"; a watcher should always send one).
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise HotplugPayloadError(
            "payload-schema",
            f"code_digest must be a string, got {type(value).__name__}",
        )
    text = value.strip()
    if text.lower().startswith("sha256:"):
        text = text[len("sha256:") :]
    if not _SHA256_DIGEST.fullmatch(text):
        raise HotplugPayloadError(
            "code-digest-invalid",
            "code_digest must be a sha256 hex digest (64 hex digits, "
            f"optionally 'sha256:'-prefixed), got {value[:80]!r}",
        )
    return "sha256:" + text.lower()


def parse_hotplug_payload(
    body: Mapping[str, Any], *, model_arch: str | None = None
) -> HotplugRequest:
    """Validate a ``POST /plugin/hotplug`` body; raise ``HotplugPayloadError``.

    Unknown keys are ignored on purpose: the internal caller and the server
    version are updated separately, and a stray field is not a reason to refuse
    a plan switch once every field that matters has been checked.
    """
    if not isinstance(body, Mapping):
        raise HotplugPayloadError("payload-schema", "body must be a JSON object")

    source = source_from_body(
        body.get("recipe"),
        body.get("enable"),
        body.get("arm", "plugin"),
        model_arch=model_arch,
    )

    raw_rows = body.get("rows", [])
    if raw_rows is None:
        raw_rows = []
    if not isinstance(raw_rows, (list, tuple)):
        raise HotplugPayloadError("payload-schema", "rows must be a list")
    rows = tuple(sorted({_positive_int(row, "rows[]") for row in raw_rows}))

    capture = body.get("capture", True)
    if not isinstance(capture, bool):
        raise HotplugPayloadError(
            "payload-schema", f"capture must be true or false, got {capture!r}"
        )

    timeout = body.get("timeout", DEFAULT_HOTPLUG_TIMEOUT_S)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise HotplugPayloadError("payload-schema", "timeout must be a number")
    if timeout <= 0:
        raise HotplugPayloadError("payload-schema", "timeout must be positive")

    reload_requested = body.get("reload", False)
    if not isinstance(reload_requested, bool):
        raise HotplugPayloadError(
            "payload-schema", f"reload must be true or false, got {reload_requested!r}"
        )
    code_digest = checked_code_digest(body.get("code_digest"))

    return HotplugRequest(
        source=source,
        arm=body.get("arm", "plugin"),
        rows=rows,
        capture=capture,
        timeout=float(timeout),
        reload=reload_requested,
        code_digest=code_digest,
    )


def checked_round_id(round_id: Any) -> str:
    """A round id as the lease carries it: a non-empty string."""
    if not isinstance(round_id, str) or not round_id.strip():
        raise RoundPayloadError(
            "payload-schema", f"round_id must be a non-empty string, got {round_id!r}"
        )
    return round_id


def parse_plugin_round_payload(body: Mapping[str, Any]) -> PluginRoundRequest:
    """Validate a ``POST /plugin/round`` body; raise ``RoundPayloadError``.

    Unknown keys are ignored for the same reason as in a hot-plug body: the
    caller and the server are deployed separately, and a stray field is not a
    reason to refuse a state the caller can change.
    """
    if not isinstance(body, Mapping):
        raise RoundPayloadError("payload-schema", "body must be a JSON object")
    action = body.get("action")
    if action not in ("begin", "end"):
        raise RoundPayloadError(
            "payload-schema", f"action must be 'begin' or 'end', got {action!r}"
        )
    round_id = checked_round_id(body.get("round_id"))
    timeout = body.get("timeout", DEFAULT_PLUGIN_ROUND_TIMEOUT_S)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise RoundPayloadError(
            "payload-schema", f"timeout must be a number, got {timeout!r}"
        )
    if timeout <= 0:
        raise RoundPayloadError("payload-schema", "timeout must be positive")
    return PluginRoundRequest(
        action=str(action), round_id=round_id, timeout=float(timeout)
    )


def capture_rows(plan: Any, full_q_len: int | None = None) -> list[int]:
    """The padded rows a plan's capture tables declare, ascending.

    A plan's tables are keyed by ``"any"``, ``"eager"`` or a padded-rows int; a
    DSpark recipe set also declares the draft pass's rows (``bs * mtp_k``).
    ``full_q_len`` restricts the answer to target verify rows (``bs *
    (1 + mtp_k)``), which are the only ones a decode request graph is keyed by.
    """
    if plan is None:
        return []
    rows = sorted(
        int(key)
        for key in plan.tables
        if isinstance(key, int) and not isinstance(key, bool) and key > 0
    )
    if full_q_len:
        rows = [row for row in rows if row % int(full_q_len) == 0]
    return rows


def default_capture_rows(
    plan: Any, full_q_len: int, ladder: Iterable[int]
) -> list[int]:
    """Which rows to recapture when the request names none.

    A plan whose capture tables touch buckets this server recorded is
    targeted: recapture exactly those.  A plan that only changes the ``any`` /
    ``eager`` tables (or whose capture buckets are all outside this server's
    ladder) changes every recorded graph's decisions, so recapture the whole
    ladder.
    """
    ladder = sorted({int(bs) for bs in ladder})
    targeted = [
        row
        for row in capture_rows(plan, full_q_len)
        if row // int(full_q_len) in set(ladder)
    ]
    if targeted:
        return targeted
    return [bs * int(full_q_len) for bs in ladder]


def rows_to_bs(rows: Iterable[int], full_q_len: int) -> list[int]:
    """Padded ``running_tokens`` -> the decode batch sizes they replay.

    Rows are the *padded* token count of a decode step, so a row is a legal
    request bucket only when it divides by the verify width: ``bs =
    row // (mtp_k + 1)``.  Anything else is a draft-pass row or a recipe
    authored for another deployment, and recapturing it would record a target
    graph under a bucket the runtime never asks for.
    """
    full_q_len = int(full_q_len)
    if full_q_len < 1:
        raise HotplugPayloadError("rows-invalid", f"full_q_len {full_q_len} < 1")
    sizes: set[int] = set()
    for row in rows:
        row = _positive_int(row, "rows[]")
        if row % full_q_len:
            raise HotplugPayloadError(
                "rows-not-decodable",
                f"row {row} is not a multiple of the verify width {full_q_len} "
                f"(mtp_k + 1); only target decode rows can be recaptured",
            )
        sizes.add(row // full_q_len)
    return sorted(sizes)


def aggregate_reports(reports: Iterable[Mapping[str, Any]]) -> dict:
    """Merge the per-engine rank-0 reports of one hot-plug into the reply.

    A refusal has already been raised by the manager's synchronous broadcast
    (it inspects each engine's ``error``), so every input here is a successful
    report; the check that their plan ids agree is what makes "one plan on
    every rank" observable rather than assumed.

    A dev-mode reload's fields are lifted from the per-rank reports to the
    top level, so a watcher reads the reply without walking ``ranks``:
    ``reloaded`` (every rank re-imported the code), ``code_digest`` (the one
    digest every reloading rank computed; disagreeing ranks are refused),
    ``reload_seconds`` (the slowest rank), ``prepared`` (provider names whose
    ``prepare(device)`` ran) and ``native`` (rank 0's provider -> prepare
    result map, i.e. the native build the reload loaded).
    """
    reports = [dict(report or {}) for report in reports]
    if not reports:
        raise RuntimeError("no engine reported this hot-plug")
    plan_ids = {report.get("plan_id") for report in reports}
    if len(plan_ids) > 1:
        raise HotplugPayloadError(
            "plan-id-mismatch",
            f"the ranks resolved the request to different plans: {sorted(plan_ids)}",
        )
    captured = [report for report in reports if report.get("status") == "ok"]
    rows = sorted({int(row) for report in captured for row in report.get("rows") or ()})
    sizes = sorted(
        {int(size) for report in captured for size in report.get("sizes") or ()}
    )
    digests = {
        digest
        for report in reports
        if isinstance((digest := report.get("code_digest")), str) and digest
    }
    if len(digests) > 1:
        # Not a plan mismatch: the same plan was reloaded from different code on
        # different ranks. The runner agrees the digest across its TP group
        # before reloading, so this is a cross-engine disagreement (a DP fleet
        # is refused before this point); surface it instead of hiding it.
        raise HotplugPayloadError(
            "code-digest-mismatch",
            f"the engines reloaded different plugin code: {sorted(digests)}",
        )
    natives = [report.get("native") for report in reports if report.get("native")]
    prepared = sorted(
        {name for report in reports for name in report.get("prepared") or ()}
    )
    return {
        "status": "ok",
        "plan_id": next(iter(plan_ids)),
        "rows": rows,
        "sizes": sizes,
        "capture_seconds": max(
            (float(report.get("capture_seconds") or 0.0) for report in reports),
            default=0.0,
        ),
        "reloaded": bool(reports)
        and all(report.get("reloaded") is True for report in reports),
        "code_digest": next(iter(digests)) if digests else None,
        "reload_seconds": max(
            (float(report.get("reload_seconds") or 0.0) for report in reports),
            default=0.0,
        ),
        "prepared": prepared,
        "native": natives[0] if natives else None,
        "ranks": reports,
    }


# -- the plugin-round lease --------------------------------------------------


def active_plugin_round(engine: Any) -> PluginRound | None:
    """The round *engine* holds right now, read off its state attributes.

    ``None`` for an engine that never began one and for one whose id was
    cleared.  Only the utility handlers write those attributes, so the
    scheduler, the generation loop and the model never observe a round.
    """
    round_id = getattr(engine, "plugin_round_id", None)
    if not isinstance(round_id, str) or not round_id:
        return None
    started_at = getattr(engine, "plugin_round_started_at", None)
    if isinstance(started_at, bool) or not isinstance(started_at, (int, float)):
        started_at = 0.0
    return PluginRound(round_id=round_id, started_at=float(started_at))


def take_plugin_round(
    engine: Any, round_id: Any, *, started_at: float | None = None
) -> dict:
    """Make *round_id* the engine's one active plugin round.

    Raises ``RoundPayloadError`` for a malformed id and ``PluginRefusal``
    (``round-in-progress``) when another round already holds the plan.  That
    second refusal is what the API turns into a 409: an overlapping benchmark
    is a state conflict, not a bad request.
    """
    round_id = checked_round_id(round_id)
    active = active_plugin_round(engine)
    if active is not None:
        raise PluginRefusal(
            ROUND_IN_PROGRESS_REASON,
            f"plugin round {active.round_id!r} is already active on this engine; "
            f"end it before beginning another",
        )
    started_at = time.time() if started_at is None else float(started_at)
    engine.plugin_round_id = round_id
    engine.plugin_round_started_at = started_at
    return {"status": "ok", "round_id": round_id, "started_at": started_at}


def release_plugin_round(
    engine: Any, round_id: Any, *, ended_at: float | None = None
) -> dict:
    """Clear the engine's round when *round_id* is the one holding it.

    An end that names another round is refused (``round-mismatch``) and leaves
    the active one in place: only the run that took the lease may drop it.
    """
    round_id = checked_round_id(round_id)
    active = active_plugin_round(engine)
    if active is None:
        raise PluginRefusal(
            NO_ACTIVE_ROUND_REASON,
            "no plugin round is active on this engine; nothing to end",
        )
    if active.round_id != round_id:
        raise PluginRefusal(
            ROUND_MISMATCH_REASON,
            f"this engine runs plugin round {active.round_id!r}, not "
            f"{round_id!r}; end the active one",
        )
    ended_at = time.time() if ended_at is None else float(ended_at)
    engine.plugin_round_id = None
    engine.plugin_round_started_at = None
    return {
        "status": "ok",
        "round_id": round_id,
        "started_at": active.started_at,
        "ended_at": ended_at,
        "duration_s": max(0.0, ended_at - active.started_at),
    }


def round_in_progress_reply(cmd: str, engine: Any) -> dict | None:
    """The refusal a hot-plug answers with while a round holds the plan.

    ``None`` when no round is active, and the caller may proceed.  Building
    the reply here rather than in the handler is what lets a CPU test pin the
    exact envelope -- reason code plus the active round -- that the manager
    raises and the API answers with a 409.
    """
    active = active_plugin_round(engine)
    if active is None:
        return None
    elapsed = max(0.0, time.time() - active.started_at)
    return {
        "cmd": cmd,
        "error": (
            f"refused: plugin round {active.round_id!r} has held the plan for "
            f"{elapsed:.1f}s; end it (POST {PLUGIN_ROUND_ENDPOINT} with "
            f'{{"action": "end", "round_id": {active.round_id!r}}}) before a '
            f"hot-plug, then retry"
        ),
        "reason": ROUND_IN_PROGRESS_REASON,
        "active_round": {
            "round_id": active.round_id,
            "started_at": active.started_at,
            "elapsed_s": round(elapsed, 3),
        },
    }


def command_refusal_reason(replies: Iterable[Any]) -> str | None:
    """The one reason code every failing utility reply agrees on, if any.

    The manager raises its refusal with this code, so the HTTP layer can
    answer ``round-in-progress`` without parsing prose, and two engines that
    failed for different (or unnamed) reasons stay a generic 409.
    """
    codes = {
        reply.get("reason")
        for reply in replies or ()
        if isinstance(reply, Mapping) and reply.get("error")
    }
    codes = {code for code in codes if isinstance(code, str) and code}
    return next(iter(codes)) if len(codes) == 1 else None


def aggregate_round_replies(action: str, round_id: str, replies: Iterable[Any]) -> dict:
    """Merge the per-engine replies of one begin/end into the answer.

    Every engine must report the same round: a begin that only some engines
    took, or an end they disagree about, is a state mismatch the caller has to
    see rather than a plan that silently differs per rank.  The engines' wall
    clocks are not comparable, so the lease interval is the earliest start and
    the latest end any engine reported.
    """
    reports = [dict(reply) for reply in replies or () if isinstance(reply, Mapping)]
    if not reports:
        raise RuntimeError(f"no engine reported the plugin round {action}")
    ids = {report.get("round_id") for report in reports}
    if ids != {round_id}:
        raise PluginRefusal(
            ROUND_STATE_MISMATCH_REASON,
            f"the engines disagree about the round they {action}: "
            f"{sorted(map(str, ids))}",
        )
    starts = [
        float(report["started_at"])
        for report in reports
        if isinstance(report.get("started_at"), (int, float))
    ]
    answer = {
        "status": "ok",
        "action": action,
        "round_id": round_id,
        "started_at": min(starts) if starts else None,
        "engines": len(reports),
    }
    if action == "end":
        ends = [
            float(report["ended_at"])
            for report in reports
            if isinstance(report.get("ended_at"), (int, float))
        ]
        answer["ended_at"] = max(ends) if ends else time.time()
        answer["duration_s"] = (
            max(0.0, answer["ended_at"] - answer["started_at"])
            if answer["started_at"] is not None
            else None
        )
    return answer


def hotplug_enabled(config: Any = None) -> bool:
    """Whether this server permits a runtime plan switch.

    ``--plugin-hotplug`` (or ``ATOM_PLUGIN_HOTPLUG=1``, folded into the same
    Config field when the engine args are parsed) binds every seam at build
    time and keeps the process able to install a plan later.  The environment
    is consulted only when a Config was built without the field at all.
    """
    if bool(getattr(config, "plugin_hotplug", False)):
        return True
    from atom.utils import envs  # heavy package: only the fallback needs it

    return bool(envs.ATOM_PLUGIN_HOTPLUG)


def topology_refusal(config: Any) -> str | None:
    """Why this deployment cannot hot-plug its decode graphs, if it cannot.

    A capture is a whole-TP-group operation and this first cut orchestrates one
    engine at a time: under data parallelism the DP peers would have to pause
    and capture in lockstep (their MoE/attention collectives are captured too)
    and under pipeline parallelism the verify width differs per stage.  Both
    are follow-ups; refusing them is better than capturing a graph whose
    collective the peers are not in.
    """
    if getattr(config, "pipeline_parallel_size", 1) > 1:
        return "pipeline parallelism (pp > 1) is not supported by hot-plug yet"
    parallel = getattr(config, "parallel_config", None)
    if getattr(parallel, "data_parallel_size", 1) > 1:
        return "data parallelism (dp > 1) is not supported by hot-plug yet"
    if getattr(config, "enable_dp_attention", False):
        return "DP attention is not supported by hot-plug yet"
    if getattr(config, "enable_rapidserve", False):
        return (
            "intra-GPU prefill/decode disaggregation is not supported by "
            "hot-plug yet"
        )
    return None


def refusal_status(error: BaseException) -> int:
    """The HTTP status a hot-plug failure should answer with.

    A recipe or payload the caller can fix is a 400; anything the engine state
    refuses (hot-plug off, unsupported topology, a plan the host declines, a
    capture that failed) is a 409, and a rank that did not answer in time is a
    504.  The endpoint never answers 500 for these: a refusal is not a server
    bug, and a caller polling a fleet needs to tell the two apart.
    """
    if isinstance(error, (HotplugPayloadError, plugin_recipe.RecipeRefused)):
        return 400
    if isinstance(error, TimeoutError):
        return 504
    if isinstance(error, ValueError):
        return 400
    return 409


def refusal_reason(error: BaseException) -> str:
    """The short reason code an endpoint puts beside a refusal's message."""
    reason = getattr(error, "reason", None)
    if isinstance(reason, str) and reason:
        return reason
    if isinstance(error, TimeoutError):
        return "hotplug-timeout"
    if isinstance(error, ValueError):
        return "hotplug-refused"
    return "hotplug-failed"
