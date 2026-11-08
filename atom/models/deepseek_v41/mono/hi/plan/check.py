# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from ATOM atom/mono/plan/check.py (MIT, Copyright (C) 2026 Advanced
# Micro Devices, Inc.).
"""The hand-off contract of a traced kernel, checked at build time.

All CTAs are co-resident and each walks the program's stages in order. A poll
always makes progress if the region it waits on is written by an earlier stage,
or by its own stage when the region is an exchange (an all-reduce: a task puts
before it polls its peers). ``check`` holds a traced ``Record`` to that and to
the regions' declarations.

``prev_reads`` (``Stage.prev_reads``, by stage): regions a stage polls as the previous launch of the program wrote
them (its tag and parity slot: this launch's writes go to the other slot). Allowed when the region is declared, has
writers, and the reader is not one of them; nothing else is relaxed. (The previous launch's writes are complete when
the poll sees their tag, wherever its writers sit in the program.)
"""

from .program import HOST, Region
from .trace import ARG, REGION, Record, Space


class ContractError(Exception):
    """A traced kernel breaks its hand-off contract (it could deadlock or read garbage)."""


def check(
    record: Record,
    regions: list[Region],
    prev_reads: dict[str, tuple[str, ...]] | None = None,
) -> None:
    by_name = {r.name: r for r in regions}
    first = {}
    for i, stage in enumerate(record.stages):
        first.setdefault(stage, i)
    prev_reads = prev_reads or {}
    problems, used, put_seen, polled_before_put = [], set(), set(), set()
    for stage, names in prev_reads.items():
        for name in names:
            r = by_name.get(name)
            if r is None:
                problems.append(
                    f"{stage} reads undeclared region {name} from the previous launch"
                )
            elif not r.writers or stage in r.writers:
                problems.append(f"{stage} reads {name} from the previous launch: it needs writers ({r.writers}), "
                                f"and not the reader")  # fmt: skip
    for a in record.accesses:
        r = by_name.get(a.region)
        if r is None:
            problems.append(f"{a.kind} of undeclared region {a.region} in {a.stage}")
            continue
        used.add(a.region)
        if a.space is not r.space:
            problems.append(
                f"{a.kind} of {a.region} in {a.stage} as {a.space.value}, declared {r.space.value}"
            )
        if a.kind == "poll" and a.region in prev_reads.get(a.stage, ()):
            continue  # the previous launch's (checked above)
        if a.kind == "put":
            put_seen.add((a.stage, a.region))
            if a.stage not in r.writers:
                problems.append(
                    f"{a.region} put by {a.stage}, its writers are {r.writers}"
                )
            continue
        if a.stage in r.writers:
            if not r.exchange:
                problems.append(
                    f"{a.region} polled by its own writer {a.stage} (not an exchange)"
                )
            elif (a.stage, a.region) not in put_seen:
                polled_before_put.add((a.stage, a.region))
            continue
        for w in r.writers:
            if w == HOST:
                continue  # published by the host before the launch (block tests, shells)
            if w not in first:
                problems.append(
                    f"{a.region} polled by {a.stage}, its writer {w} never runs"
                )
            elif first[w] > first[a.stage]:
                problems.append(
                    f"{a.region} polled by {a.stage}, which runs before its writer {w}"
                )
    for stage, region in sorted(polled_before_put):
        problems.append(f"exchange {region} polled in {stage} before any put of it")
    for name in sorted(set(by_name) - used):
        r = by_name[name]
        if r.space is Space.PEER and not any(w in first for w in r.writers):
            continue  # another program's peer region declared for the layout only (pinned by ``Program.peer_first``)
        problems.append(f"declared region {name} is never accessed")
    if problems:
        raise ContractError("\n".join(problems))


# ---- the memory model (``check_memory``): plain memory, bulk + flag, peer formats
#
# Measured on MI355X: (1) within one launch a plain (cache mode 0) store is not seen by a
# CTA on another XCD (82-99 % stale words; XCD = CTA id mod 8), and a load issued before its producer's store lands
# reads the old bytes: a plain hand-off needs device-scope (sc1) stores and loads, unless writer and reader share a
# CTA or an XCD, and an ordering edge; (2) bulk + flag needs ``s_waitcnt vmcnt(0)`` after the bulk stores, before the
# flag (0.004-1 % stale without it); (3) a 16 B store to a peer can land as two 8 B halves: every 8 B half of a peer
# store must carry the tag.

SC1 = 16  # device scope (ops.CM_DEV)
SYS = 17  # system scope (ops.CM_SYS): SC0 | SC1


def _names(m) -> list[str]:
    return m.target.split("|") if m.target else []


def _plain_names(m, dense) -> list[str]:
    """The plain targets a load / store goes through: an arg's names, a dense region's; a *mixed* load (an address
    derived from roots of different kinds: a select between an arg and a region, or an arg indexed by a value loaded
    from scratch) as a load of every arg among its candidates (the conservative reading: it may address the arg).
    """
    names = _names(m)
    if m.kind == ARG:
        return names
    if m.kind == REGION:
        return [n for n in names if n in dense]
    if m.kind == "mixed" and m.op == "load":
        return [n for n, k in zip(names, m.kinds) if k == ARG]
    return []


def _via(x, rld) -> str:
    """A note for a problem whose reader loads ``x`` through mixed addresses."""
    mixed = sorted({m.target for m in rld if m.kind == "mixed"})
    if not mixed:
        return ""
    return f" [{x} loaded through a mixed address ({', '.join(mixed)}): taken as a load of {x}]"


def check_memory(record: Record, program, waive: tuple[str, ...] = ()) -> list[str]:
    """Hold a traced record (``Record.mem``) to the memory model above; raise ``ContractError`` listing every
    problem. ``program``: the planned ``Program`` (stage placement, regions, ``prev_reads``). ``waive``: plain
    targets (arg / region names) whose same-launch hand-off problems are reported but not raised (a documented
    finding pending its fix). -> the waived problems.

    **Plain targets**: args the program stores, and *dense* scratch regions (stored without a tag in any 8 B
    half: bulk + flag payloads). A *mixed* load (an address derived from roots of different kinds: a select between
    an arg and a region, or a cache arg indexed by a position polled from a channel, Flash's sparse attention) is a
    load of every arg among its candidates: the check cannot see which one it addresses. For a plain target X stored
    by stage W and loaded by another stage R:

    - W before R (a same-launch hand-off): allowed when task t of W and R run on one CTA (same base and task count);
      otherwise R must poll, before its first load of X, a tagged region put by W or by a stage W reaches through
      polls (the ordering edge), W must put such a region after its stores of X with an ``s_waitcnt vmcnt(0)`` in
      between (bulk + flag), and both sides must be device scope (sc1) unless W and R sit on the same XCDs (same task
      count, bases equal mod ``target.xcds``).
    - R before W (R reads the previous launch's X, W overwrites it in this launch): W must be reached from R
      through polls (W waits for something R put after reading), or share R's CTAs.

    A stage that both loads and stores X is taken as reading and writing its own elements (not checked), and so is
    a stage's ``Stage.later_args`` (stored for later launches only).
    **Peer stores**: system scope, every 8 B half (a 4 B store: its word) holding a word derived from the tag;
    peer loads system scope."""
    grid, xcds = program.target.grid, program.target.xcds
    if not xcds:  # a target whose memory model is not modeled (``Target.xcds``)
        return []
    stages = {s.name: s for s in program.stages}
    first = {}
    for i, stage in enumerate(record.stages):
        first.setdefault(stage, i)
    by_name = {r.name: r for r in program.regions}
    prev = {s.name: set(s.prev_reads) for s in program.stages}
    problems, waived = [], []

    # dense (untagged) scratch regions: payloads of bulk + flag hand-offs
    dense = set()
    for m in record.mem:
        if (
            m.op == "store"
            and m.kind == "region"
            and (m.tagged is None or not any(m.tagged))
        ):
            dense.update(_names(m))

    # plain targets: stores / loads per stage
    stores, loads = {}, {}
    for m in record.mem:
        if m.op not in ("load", "store") or m.stage not in stages:
            continue
        for n in _plain_names(m, dense):
            (stores if m.op == "store" else loads).setdefault(n, {}).setdefault(
                m.stage, []
            ).append(m)

    # ordering edges: V -> S when S polls (this launch) a tagged region V puts
    succ = {s: set() for s in stages}
    polls = {s: [] for s in stages}  # (seq, region)
    puts = {s: [] for s in stages}
    for a in record.accesses:
        if a.stage not in stages or a.region in dense:
            continue
        if a.kind == "put":
            puts[a.stage].append((a.seq, a.region))
            continue
        if a.region in prev[a.stage]:
            continue
        polls[a.stage].append((a.seq, a.region))
        for w in by_name[a.region].writers if a.region in by_name else ():
            if w in succ and w != a.stage:
                succ[w].add(a.stage)
    reach = {}
    for s in stages:  # stages reachable from s through ordering edges
        seen, todo = set(), list(succ[s])
        while todo:
            v = todo.pop()
            if v not in seen:
                seen.add(v)
                todo.extend(succ[v])
        reach[s] = seen

    def writers(region):
        return by_name[region].writers if region in by_name else ()

    def same_cta(a, b):
        return a.tasks == b.tasks and a.base % grid == b.base % grid

    def same_xcd(a, b):
        return a.tasks == b.tasks and (a.base - b.base) % xcds == 0

    waits = {
        s: [m.seq for m in record.mem if m.stage == s and m.op == "wait" and m.cm == 0]
        for s in stages
    }

    for s in program.stages:
        for x in s.later_args:
            if s.name not in stores.get(x, {}):
                problems.append(f"{s.name} declares later_args {x} but never stores it")
    for x in sorted(set(stores) & set(loads)):
        out = waived if x in waive else problems
        for w, wst in stores[x].items():
            if x in stages[w].later_args:
                continue  # stored for later launches only (``Stage.later_args``)
            for r, rld in loads[x].items():
                if r == w or same_cta(stages[w], stages[r]):
                    continue
                if first[w] < first[r]:
                    # regions R polls before its first load of X, put by W or a stage W reaches
                    lo = min(m.seq for m in rld)
                    via = {s for s in reach[w] if s == r or r in reach[s]}
                    guards = {
                        f
                        for seq, f in polls[r]
                        if seq < lo and any(v == w or v in reach[w] for v in writers(f))
                    }
                    if not guards:
                        out.append(f"plain {x}: written by {w} and read by {r} in the same launch with no ordering "
                                   f"(no tagged region {w} puts is polled by {r} before its load): {r} on another "
                                   f"CTA can read the previous launch's bytes{_via(x, rld)}")  # fmt: skip
                        continue
                    # W's puts that start an edge to R, after its stores of X, each drained
                    path = {
                        f
                        for _, f in puts[w]
                        if any(f in {g for _, g in polls[v]} for v in via)
                    }
                    last = max(m.seq for m in wst)
                    if not any(seq > last for seq, f in puts[w] if f in path):
                        out.append(f"plain {x}: {w} stores it after its last put of {sorted(path)}, which {r} "
                                   f"waits on: the flag can be seen before the data")  # fmt: skip
                    for seq, f in puts[w]:
                        before = [m.seq for m in wst if m.seq < seq]
                        if (
                            f in path
                            and before
                            and not any(max(before) < q < seq for q in waits[w])
                        ):
                            out.append(
                                f"plain {x}: {w} puts {f} (which orders {r}'s read) without an s_waitcnt "
                                f"vmcnt(0) after its stores of {x}: the flag can overtake the data"
                            )
                            break
                    if not same_xcd(stages[w], stages[r]):
                        bad = sorted(
                            {
                                f"{m.op} cm {m.cm} in {m.stage}"
                                for m in wst + rld
                                if not m.cm & SC1
                            }
                        )
                        if bad:
                            out.append(f"plain {x}: written by {w} (base {stages[w].base}, {stages[w].tasks} tasks) "
                                       f"and read by {r} (base {stages[r].base}, {stages[r].tasks} tasks) in the same "
                                       f"launch, possibly on another XCD, without device scope ({', '.join(bad)}): "
                                       f"a plain store is not seen across XCDs within a launch"
                                       f"{_via(x, rld)}")  # fmt: skip
                elif w not in reach[r]:
                    out.append(
                        f"plain {x}: read by {r} and overwritten later in the same launch by {w}, which does "
                        f"not wait for {r} (no ordering edge): {w} on another CTA can overwrite it first{_via(x, rld)}"
                    )
    for m in record.mem:
        if m.kind != "peer" or m.op == "wait":
            continue
        if m.cm & SYS != SYS:
            problems.append(
                f"peer {m.target}: {m.op} in {m.stage} with cache bits {m.cm}, not system scope"
            )
        if m.op == "store" and (m.tagged is None or not all(m.tagged)):
            what = (
                "its words are not visible to the check"
                if m.tagged is None
                else f"halves tagged {m.tagged}"
            )
            problems.append(f"peer {m.target}: a {m.nbytes} B store in {m.stage} without the tag in every 8 B half "
                            f"({what}): a peer store can land as two 8 B halves at different times")  # fmt: skip
    if problems:
        raise ContractError("\n".join(dict.fromkeys(problems)))
    return list(dict.fromkeys(waived))
