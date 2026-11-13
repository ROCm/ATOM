# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-rank prefill backlog for `share_cache` routing, as the CoreManager sees it.

A rank reports (seqs ingested, prefill tokens left) every few milliseconds. The
ledger adds what that report cannot know yet: the charges of seqs routed in a
dispatch but not sent (`unsent`), and of seqs sent after the report's ingest
count (`unseen`, in send order). It also carries the successor debt: when a
prefix producer leaves before it prefilled, the waiter that inherits its cold
prefix is charged what it loses, and the rank is sent the corrected credit
until a report acknowledges it.

No locking and no I/O: the CoreManager holds its load lock around every call and
sends what `take_credit_updates` returns after releasing it.
"""

from __future__ import annotations

from collections import deque


class BacklogLedger:
    def __init__(self, n_ranks: int) -> None:
        self.n_ranks = n_ranks
        self.report: list[tuple[int, int] | None] = [None] * n_ranks
        self.sent = [0] * n_ranks
        self.reset()

    def reset(self) -> None:
        """Forget every pending charge. The send counters keep running: they
        count what the engine ingested, which a reset does not undo."""
        n = self.n_ranks
        # Per rank, the seqs sent after the rank's last report, in send order:
        # [seq id or None once released, charge].
        self.unseen: list[deque] = [deque() for _ in range(n)]
        self.unseen_tokens = [0] * n
        # seq id -> (rank, charge, seq): charged in a dispatch, not sent yet.
        # Routing the next seq of the same batch has to see them.
        self.unsent: dict = {}
        self.unsent_tokens = [0] * n
        self.prompt_tokens: dict = {}
        # seq id -> (rank, 1-based send ordinal on that rank).
        self.ordinals: dict = {}
        # Successor debt: sid -> [(credit, extra tokens)]; the corrected credit
        # to send (sid -> (rank, credit)) and the one last sent (sid -> credit).
        self.debts: dict = {}
        self.credit_updates: dict = {}
        self.credit_sent: dict = {}
        # Debt of seqs the rank already ingested, which its report counts only
        # once the matching credit update is acknowledged.
        self.reported_credit_tokens = [0] * n

    def _popped(self, rank: int) -> int:
        """Seqs sent to `rank` that its report counts."""
        return self.sent[rank] - len(self.unseen[rank])

    # -- dispatch ---------------------------------------------------------------

    def note_charged(self, seq, rank: int, charge: int) -> None:
        """`seq` was charged to `rank` in a dispatch and is not sent yet."""
        self.prompt_tokens[seq.id] = int(seq.num_prompt_tokens)
        self.unsent[seq.id] = (rank, charge, seq)
        self.unsent_tokens[rank] += charge

    def drop_unsent(self, seq_id) -> int:
        """Forget a charged-but-unsent seq (sent, rolled back or released);
        returns its charge. Idempotent."""
        entry = self.unsent.pop(seq_id, None)
        if entry is None:
            return 0
        rank, charge, _ = entry
        self.unsent_tokens[rank] -= charge
        return charge

    def note_sent(self, rank: int, seqs, seq_load: dict) -> None:
        """Seqs just sent to `rank`, in send (= ingest) order. `seq_load` is
        the CoreManager's live charges: a seq absent from it was released
        already."""
        q = self.unseen[rank]
        for offset, seq in enumerate(seqs):
            load = seq_load.get(seq.id)
            if load is not None:
                self.ordinals[seq.id] = (rank, self.sent[rank] + offset + 1)
            if seq.id in self.unsent:
                charge = self.drop_unsent(seq.id)
            else:
                charge = load[2] if load is not None else 0
            q.append([seq.id if load is not None else None, charge])
            self.unseen_tokens[rank] += charge
        self.sent[rank] += len(seqs)
        self._pop_seen(rank)

    # -- rank reports -----------------------------------------------------------

    def on_report(self, rank: int, ingested: int, left: int, acks) -> None:
        """A rank's report: what it ingested, what it has left to prefill, and
        the credit updates it applied."""
        self.ack(rank, acks)
        self.report[rank] = (ingested, left)
        self._pop_seen(rank)

    def _pop_seen(self, rank: int) -> None:
        """Drop the sent seqs the rank's last report already counts."""
        report = self.report[rank]
        if report is None:
            return
        target = min(int(report[0]), self.sent[rank])
        q = self.unseen[rank]
        while self._popped(rank) < target and q:
            sid, charge = q.popleft()
            self.unseen_tokens[rank] -= charge
            self.reported_credit_tokens[rank] += sum(
                extra for _, extra in self.debts.get(sid, ())
            )

    def ack(self, rank: int, updates) -> None:
        for sid, credit in updates:
            sent = self.ordinals.get(sid)
            debts = self.debts.get(sid, [])
            last_sent = self.credit_sent.get(sid)
            if (
                sent is None
                or sent[0] != rank
                or last_sent is None
                or credit < last_sent
                or not any(value == credit for value, _ in debts)
            ):
                continue
            remaining = [(value, extra) for value, extra in debts if value < credit]
            acknowledged = sum(extra for value, extra in debts if value >= credit)
            if sent[1] <= self._popped(rank):
                self.reported_credit_tokens[rank] -= acknowledged
            if remaining:
                self.debts[sid] = remaining
            else:
                self.debts.pop(sid, None)
            if self.credit_updates.get(sid) == (rank, credit):
                del self.credit_updates[sid]
                self.credit_sent.pop(sid, None)

    # -- successor debt -----------------------------------------------------------

    def add_successor_debt(self, sid, rank: int, extra: int, charge: int) -> None:
        """`sid`, on `rank`, loses `extra` tokens of credit: its charge is now
        `charge`."""
        credit = max(0, self.prompt_tokens[sid] - charge)
        self.credit_updates[sid] = (rank, credit)
        self.debts.setdefault(sid, []).append((credit, extra))
        if sid in self.unsent:
            _, previous, seq = self.unsent[sid]
            seq.dp_est_cached_tokens = min(seq.dp_est_cached_tokens, credit)
            self.unsent[sid] = (rank, previous + extra, seq)
            self.unsent_tokens[rank] += extra
        elif sid in self.ordinals:
            ordinal = self.ordinals[sid][1]
            popped = self._popped(rank)
            if ordinal > popped:
                self.unseen[rank][ordinal - popped - 1][1] += extra
                self.unseen_tokens[rank] += extra
            else:
                self.reported_credit_tokens[rank] += extra

    def forget(self, sid) -> None:
        """`sid` finished or was released."""
        sent = self.ordinals.pop(sid, None)
        debts = self.debts.pop(sid, ())
        if sent is not None and sent[1] <= self._popped(sent[0]):
            self.reported_credit_tokens[sent[0]] -= sum(extra for _, extra in debts)
        self.credit_updates.pop(sid, None)
        self.credit_sent.pop(sid, None)
        self.prompt_tokens.pop(sid, None)

    def take_credit_updates(self) -> list[tuple[int, list]]:
        """The corrected credits to send, per rank, for seqs already sent; they
        count as sent until `unsend`."""
        by_rank: dict[int, list] = {}
        for sid, (rank, credit) in self.credit_updates.items():
            if sid in self.ordinals and self.credit_sent.get(sid) != credit:
                by_rank.setdefault(rank, []).append((sid, credit))
        for updates in by_rank.values():
            self.credit_sent.update(updates)
        return list(by_rank.items())

    def unsend(self, updates) -> None:
        """A send of `updates` failed: offer them again at the next take."""
        for sid, credit in updates:
            if self.credit_sent.get(sid) == credit:
                del self.credit_sent[sid]

    # -- routing ------------------------------------------------------------------

    def tokens(self, rank: int, fallback: int) -> int:
        """Prefill left on `rank`: its report plus the charges the report does
        not cover. A rank that has not reported yet uses `fallback`."""
        report = self.report[rank]
        if report is None:
            return fallback
        return (
            int(report[1])
            + self.unseen_tokens[rank]
            + self.unsent_tokens[rank]
            + self.reported_credit_tokens[rank]
        )
