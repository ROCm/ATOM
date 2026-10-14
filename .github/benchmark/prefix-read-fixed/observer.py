"""81198 experiment-only READ submission observer; no imports of vLLM or MoRI.

Load this file by absolute path from a readonly mount. Explicitly register final
scheduler plans and actual session->region mappings, then install on ONE decode
worker and its private MoRIIOWrapper. This is NOT automatic scheduler integration.
One allocation per (request, transfer, rank) is supported; reuse is invalid.

Only existing poll_transfer_batch DONE results certify completion (81198 scans
Failed()/Succeeded() for every status). No observer polls, waits, RDMA calls, or
status proxies. Failure/timeout/removal without DONE never becomes success.
close() restores original instance callables before serialization; orchestrator
must close every rank after warmup and before formal timing. No file I/O on READ.
Metadata/geometry provenance and cross-process episode propagation remain caller
responsibilities; this local component is NOT GPU/dispatch ready.
"""
import argparse
from collections import Counter
from contextvars import ContextVar
import copy
import json
import threading


class Observer:
    def __init__(self, enabled=True, max_events=100000):
        self.enabled = enabled
        self.max_events = max_events
        self.events = []
        self.plans = {}
        self.sessions = {}
        self.seen = set()
        self.statuses = {}
        self.restores = []
        self.invalid = []
        self.dropped = 0
        self.closed = False
        self.lock = threading.RLock()
        self.context = ContextVar("prefix_read_context", default=None)
        self.bypass = ContextVar("prefix_read_explicit_skip", default=False)
        self.skipped = set()
        self.completion_seen = set()
        self.annotations = {}

    def _bad(self, reason):
        with self.lock:
            self.dropped += 1
            if len(self.invalid) < 32:
                self.invalid.append(str(reason))

    def _safe(self, fn, *args):
        try:
            return fn(*args)
        except Exception as exc:
            self._bad(type(exc).__name__)
            return None

    def _event(self, **event):
        with self.lock:
            if len(self.events) >= self.max_events:
                self._bad("event capacity exceeded")
                return None
            event["id"] = len(self.events)
            self.events.append(event)
            return event["id"]

    def register(self, plan, sessions):
        """Before serving: final N/E, effective pages, full D/P tables, geometry.

        sessions maps actual session objects to unique region names from plan.
        Registration is explicit, never inferred from a request substring.
        """
        p = copy.deepcopy(plan)
        key = (p["request"], p["transfer"], p["rank"])
        if key in self.plans:
            self._bad("ambiguous repeated allocation")
            return
        self.plans[key] = p
        self.sessions[key] = self._session_map(sessions)

    @staticmethod
    def _session_map(sessions):
        # Runtime triples are layer scoped: attention layers may share one MR
        # and session. Legacy unit seam accepts the simple session->region map.
        entries = ((None, s, name) for s, name in sessions.items()) if isinstance(sessions, dict) else sessions
        result = {}
        for layer, session, region in entries:
            key = (layer, id(session))
            if key in result:
                raise ValueError("ambiguous session within one layer")
            result[key] = (session, region)
        return result

    def attach(self, key, plan, sessions):
        """Complete deferred geometry after the ORIGINAL lazy session lookup."""
        with self.lock:
            if key not in self.plans or self.sessions[key]:
                raise ValueError("missing/duplicate late session attachment")
            if any(plan[field] != self.plans[key][field]
                   for field in ("request", "transfer", "episode", "rank", "full_local", "producer", "prompt", "external")):
                raise ValueError("late plan identity changed")
            self.plans[key] = copy.deepcopy(plan)
            self.sessions[key] = self._session_map(sessions)

    def _replace(self, obj, name, replacement):
        had = name in vars(obj)
        saved = vars(obj).get(name)
        self.restores.append((obj, name, had, saved))
        setattr(obj, name, replacement)

    def install(self, worker):
        """Patch only instance boundaries with pinned 81198 signatures."""
        if not self.enabled:
            return
        if self.restores or self.closed:
            raise ValueError("one installation/window only")
        wrapper = worker.moriio_wrapper
        read_req = worker._read_blocks_for_req
        post = worker._post_read_with_backoff
        read = wrapper.read_remote_data
        poll = wrapper.poll_transfer_batch

        def observed_req(req_id, meta):
            if not self.enabled:
                return read_req(req_id, meta)
            key = (req_id, meta.transfer_id, worker.tp_rank)
            if key in self.skipped:
                token = self.bypass.set(True)
                try:
                    return read_req(req_id, meta)
                finally:
                    self.bypass.reset(token)
            token = None
            try:
                key = (req_id, meta.transfer_id, worker.tp_rank)
                with self.lock:
                    if key not in self.plans or key in self.seen:
                        self._bad("missing plan or repeated allocation")
                    else:
                        self.seen.add(key)
                        token = self.context.set((key, None))
                        self._event(type="begin", key=list(key))
            except Exception as exc:
                self._bad(type(exc).__name__)
            try:
                result = read_req(req_id, meta)
            except BaseException:
                self._safe(self._event_exception)
                raise
            else:
                ctx = self.context.get()
                if ctx is not None:
                    self._safe(lambda: self._event(type="end", key=list(ctx[0])))
                return result
            finally:
                if token is not None:
                    self.context.reset(token)

        def observed_post(session, sizes, local_offsets, remote_offsets,
                          request_id, layer_name, deadline):
            if not self.enabled or self.bypass.get():
                return post(session, sizes, local_offsets, remote_offsets,
                            request_id, layer_name, deadline)
            ctx = self.context.get()
            token = None
            if ctx is not None:
                token = self.context.set((ctx[0], layer_name))
            try:
                return post(session, sizes, local_offsets, remote_offsets,
                            request_id, layer_name, deadline)
            finally:
                if token is not None:
                    self.context.reset(token)

        def observed_read(transfer_size_byte, local_offset=0, remote_offset=0,
                          session=None):
            if not self.enabled or self.bypass.get():
                return read(transfer_size_byte, local_offset, remote_offset, session)
            # Snapshot BEFORE handing mutable vectors to transport, in memory only.
            snapshot = self._safe(lambda: (list(transfer_size_byte),
                                          list(local_offset), list(remote_offset)))
            try:
                status = read(transfer_size_byte, local_offset, remote_offset, session)
            except BaseException:
                self._safe(self._event_exception)
                raise
            self._safe(self._record_read, status, session, snapshot)
            return status

        def observed_poll(transfer_statuses):
            if not self.enabled:
                return poll(transfer_statuses)
            # Real caller supplies a list; do not consume arbitrary iterators.
            snapshot = self._safe(lambda: tuple(transfer_statuses)
                                  if isinstance(transfer_statuses, (list, tuple))
                                  else (_ for _ in ()).throw(ValueError("status iterable")))
            result = poll(transfer_statuses)
            self._safe(self._record_poll, snapshot, result)
            return result

        self._replace(worker, "_read_blocks_for_req", observed_req)
        self._replace(worker, "_post_read_with_backoff", observed_post)
        self._replace(wrapper, "read_remote_data", observed_read)
        self._replace(wrapper, "poll_transfer_batch", observed_poll)

    def _event_exception(self):
        self._bad("serving call raised; observation incomplete")

    def _record_read(self, status, session, snapshot):
        ctx = self.context.get()
        if ctx is None or snapshot is None:
            self._bad("READ without episode/range context")
            return
        key, layer = ctx
        sizes, local, remote = snapshot
        regions = self.sessions[key]
        entry = regions.get((layer, id(session)), regions.get((None, id(session))))
        if entry is None or entry[0] is not session:
            raise ValueError("READ session lacks authoritative layer region")
        region = entry[1]
        failed = bool(status.Failed())
        with self.lock:
            op = self._event(type="submit", key=list(key), layer=layer,
                             region=region, sizes=sizes, local=local, remote=remote,
                             state="Failed" if failed else "Accepted",
                             acceptance_basis="81198 synchronous return: not Failed")
            if not failed and op is not None:
                if id(status) in self.statuses:
                    self._bad("status reused across submissions")
                self.statuses[id(status)] = (status, op)

    def _record_poll(self, statuses, result):
        state = result.name
        if state not in ("PENDING", "DONE", "FAILED"):
            self._bad("unsupported poll result")
            return
        if state == "PENDING":
            return
        with self.lock:
            for status in statuses:
                entry = self.statuses.get(id(status))
                if entry is not None:
                    marker = (entry[1], state)
                    if marker in self.completion_seen:
                        continue
                    self.completion_seen.add(marker)
                    # FAILED is batch failure, NOT a per-operation Failed assertion.
                    self._event(type="completion", operation=entry[1],
                                state="Succeeded" if state == "DONE" else "BatchFailed",
                                basis="existing poll_transfer_batch result")

    def close(self, path=None):
        """Restore serving first. Persistence failures invalidate returned evidence."""
        self.enabled = False
        for obj, name, had, saved in reversed(self.restores):
            if had:
                setattr(obj, name, saved)
            else:
                delattr(obj, name)
        self.restores.clear()
        self.closed = True
        evidence = dict(plans=list(self.plans.values()), events=self.events,
                        dropped=self.dropped, invalid=list(self.invalid), closed=True,
                        **copy.deepcopy(self.annotations))
        if path is not None:
            try:
                with open(path, "x", encoding="utf-8") as stream:
                    json.dump(evidence, stream, separators=(",", ":"))
                    stream.write("\n")
            except Exception as exc:
                self._bad("persistence: " + type(exc).__name__)
                evidence.update(dropped=self.dropped, invalid=list(self.invalid))
        return evidence


def legacy_attention(local, producer, max_tail, external):
    """Offline 81198 old HMA selector; NEVER submits or computes transport time."""
    if external == 0:
        return [], []
    d, p = list(local), list(producer)
    if len(d) > len(p):
        if len(d) - len(p) > max_tail:
            raise ValueError("old selector tail overflow")
        d = d[:len(p)]
    elif len(d) < len(p):
        p = p[-len(d):] if d else []
    return d, p


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid nonnegative integer")
    return value


def _canonical(ranges):
    """Canonical disjoint affine byte intervals; no per-byte materialization."""
    out = []
    for local, remote, size in sorted(ranges):
        _integer(local)
        _integer(remote)
        _integer(size, 1)
        if out and local < out[-1][0] + out[-1][2]:
            raise ValueError("duplicate/overlapping READ destination")
        if (out and local == out[-1][0] + out[-1][2]
                and remote == out[-1][1] + out[-1][2]):
            out[-1][2] += size
        else:
            out.append([local, remote, size])
    return out


def expected(plan):
    """Small supported layout: full position-indexed attention, strided regions.

    KDA regions use explicit full local/producer slot groups of equal length.
    Scratch/tail-aligned KDA, heterogeneous TP and omitted geometry are unsupported.
    Segments are runtime per-slot (local delta, remote delta, logical bytes), not
    RDMA wire bytes. An integration must derive them independently of submission.
    """
    if plan["layout"] != "homogeneous-strided-v1":
        raise ValueError("unsupported layout")
    peer = plan["peer"]
    if (not peer["engine"] or _integer(peer["tp_size"], 1) != len(plan["ranks"])
            or _integer(peer["tp_rank"]) >= peer["tp_size"]
            or _integer(peer["dp_rank"]) != 0):
        raise ValueError("unsupported peer routing/heterogeneous TP")
    n, e = _integer(plan["prompt"], 1), _integer(plan["external"])
    if e > n - 1 or plan["role"] != "D":
        raise ValueError("unsupported final interval/role")
    local, producer = plan["full_local"], plan["producer"]
    if len(local) != len(producer) or not local:
        raise ValueError("group mismatch")
    result, kinds = {}, {}
    layers = set()
    for layer in plan["layers"]:
        name, kind, group = layer["name"], layer["kind"], layer["group"]
        if name in layers or kind not in ("attention", "kda"):
            raise ValueError("unsupported/duplicate layer")
        layers.add(name)
        _integer(group)
        d, p = local[group], producer[group]
        for table in (d, p):
            if len(set(table)) != len(table):
                raise ValueError("aliased block table")
            for block in table:
                _integer(block)
        if kind == "attention":
            b = _integer(layer["page_tokens"], 1)
            end = (n - 1 + b - 1) // b
            if len(p) != end or len(d) < (n + b - 1) // b:
                raise ValueError("not a full position-indexed page table")
            start = (n - 1 - e) // b if e else end
            pairs = list(zip(d[start:end], p[start:end], strict=True))
        else:
            if len(d) != len(p) or not d:
                raise ValueError("unsupported KDA slot alignment")
            pairs = list(zip(d, p, strict=True))
        if not layer["regions"]:
            raise ValueError("missing regions")
        for region in layer["regions"]:
            key = (name, region["name"])
            if key in result:
                raise ValueError("duplicate region")
            ls = _integer(region["local_stride"], 1)
            rs = _integer(region["remote_stride"], 1)
            lb = _integer(region["local_base"])
            rb = _integer(region["remote_base"])
            segments = region["segments"]
            if not segments:
                raise ValueError("missing segment geometry")
            for dl, dr, size in segments:
                _integer(dl)
                _integer(dr)
                _integer(size, 1)
                if dl + size > ls or dr + size > rs:
                    raise ValueError("segment exceeds stride")
            result[key] = _canonical([(lb + x * ls + dl, rb + y * rs + dr, size)
                                      for x, y in pairs for dl, dr, size in segments])
            kinds[key] = kind
    if not layers or {layer["group"] for layer in plan["layers"]} != set(range(len(local))):
        raise ValueError("missing layer/group inventory")
    return result, kinds


def combine(evidences):
    """Offline merge rank files without confusing rank-local operation IDs."""
    merged = dict(plans=[], events=[], dropped=0, invalid=[], closed=True)
    for source in evidences:
        part = copy.deepcopy(source)
        events = part["events"]
        if [ev["id"] for ev in events] != list(range(len(events))):
            merged["invalid"].append("rank file has truncated/duplicate sequence")
        offset = len(merged["events"])
        for ev in events:
            ev["id"] += offset
            if ev["type"] == "completion":
                ev["operation"] += offset
        merged["events"].extend(events)
        merged["plans"].extend(part["plans"])
        merged["dropped"] += part["dropped"]
        merged["invalid"].extend(part["invalid"])
        merged["closed"] &= part["closed"]
    return merged


def check(evidence, expected_ranks=None):
    """Fail closed on missing rank, plan, zero-op end, ranges, drops or completion."""
    errors, incomplete = [], []
    logical = {"attention": 0, "kda": 0}
    try:
        if not evidence["closed"] or evidence["dropped"] or evidence["invalid"]:
            raise ValueError("unclosed/dropped/invalid observation")
        plans, episodes = {}, {}
        for p in evidence["plans"]:
            key = (p["request"], p["transfer"], p["rank"])
            episode = (p["request"], p["transfer"], p["episode"], p["role"])
            if key in plans or not all(isinstance(x, str) and x for x in episode):
                raise ValueError("ambiguous plan/episode")
            ranks = expected_ranks if expected_ranks is not None else p["ranks"]
            if not ranks or len(set(ranks)) != len(ranks) or p["rank"] not in ranks:
                raise ValueError("invalid rank inventory")
            if set(p["ranks"]) != set(ranks):
                raise ValueError("rank contract mismatch")
            plans[key] = p
            episodes.setdefault(episode, (set(), set(ranks)))[0].add(p["rank"])
        if not plans or any(got != wanted for got, wanted in episodes.values()):
            raise ValueError("missing rank or empty evidence")
        events = evidence["events"]
        ids = [ev["id"] for ev in events]
        if ids != list(range(len(events))):
            raise ValueError("missing/duplicate event sequence")
        completion = {}
        for ev in events:
            if ev["type"] == "completion":
                op = ev["operation"]
                if (type(op) is not int or not 0 <= op < ev["id"]
                        or events[op]["type"] != "submit"
                        or events[op]["state"] != "Accepted"
                        or ev["basis"] != "existing poll_transfer_batch result"):
                    raise ValueError("orphan/invalid completion")
                completion.setdefault(op, set()).add(ev["state"])
            elif ev["type"] not in ("begin", "end", "submit"):
                raise ValueError("unknown event")
            elif tuple(ev["key"]) not in plans:
                raise ValueError("orphan event")
        for key, p in plans.items():
            wanted, kinds = expected(p)
            rows = [ev for ev in events if tuple(ev.get("key", ())) == key]
            if Counter(ev["type"] for ev in rows)["begin"] != 1 or sum(
                    ev["type"] == "end" for ev in rows) != 1:
                raise ValueError("missing begin/end or duplicate episode")
            if rows[0]["type"] != "begin" or rows[-1]["type"] != "end":
                raise ValueError("submission outside episode")
            actual = {region: [] for region in wanted}
            for ev in rows:
                if ev["type"] != "submit":
                    continue
                region = (ev["layer"], ev["region"])
                if region not in wanted or ev["state"] not in ("Accepted", "Failed"):
                    raise ValueError("unknown READ region/state")
                ranges = list(zip(ev["local"], ev["remote"], ev["sizes"], strict=True))
                _canonical(ranges)
                if ev["state"] == "Accepted":
                    actual[region].extend(ranges)
                    logical[kinds[region]] += sum(ev["sizes"])
                    if completion.get(ev["id"]) != {"Succeeded"}:
                        incomplete.append(f"operation {ev['id']} lacks successful DONE")
            for region, ranges in actual.items():
                if _canonical(ranges) != wanted[region]:
                    errors.append(f"{key}/{region}: cached, missing or wrong-source READ")
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        errors.append(str(exc))
    return dict(submission="FAIL" if errors else "PASS",
                completion="BLOCKED" if errors or incomplete else "PASS",
                errors=errors, incomplete=incomplete, logical_bytes=logical,
                byte_semantics="logical submitted payload, not wire",
                ready=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="+", help="one JSON file per decode rank")
    parser.add_argument("--ranks", required=True, help="external rank inventory: 0,1,...,7")
    args = parser.parse_args()
    try:
        evidence = []
        for path in args.evidence:
            with open(path, encoding="utf-8") as stream:
                evidence.append(json.load(stream))
        result = check(combine(evidence), [int(x) for x in args.ranks.split(",")])
    except Exception as exc:
        result = dict(submission="FAIL", completion="BLOCKED", errors=[str(exc)], ready=False)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["completion"] == "PASS" else 1)
