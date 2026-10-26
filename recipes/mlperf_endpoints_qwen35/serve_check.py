# The ATOM server with a KV-block ownership check after every schedule: a physical block in the
# block tables of two live sequences (running or just scheduled) is logged as BLOCKCHECK.
# Run with prefix caching off, where no block may be shared.
import logging
import runpy
import sys

if (
    __name__ == "__main__"
):  # as the real entry point: metrics before anything imports prometheus_client
    from atom.metrics.prometheus import initialize_metrics

    initialize_metrics()

import atom.model_engine.scheduler as S

_orig = S.Scheduler.schedule
_log = logging.getLogger("atom")


_armed = [False]


def schedule(self):
    out = _orig(self)
    if not _armed[0]:
        _armed[0] = True
        _log.warning("BLOCKCHECK armed in this process")
    try:
        seqs = {id(s): s for s in self.running}
        if (
            out is not None
            and isinstance(out, tuple)
            and len(out) > 1
            and isinstance(out[1], dict)
        ):
            seqs.update({id(s): s for s in out[1].values()})
        owner, dup, ex = {}, 0, None
        for k, s in seqs.items():
            for b in list(s.block_table):
                if b in owner and owner[b] != k:
                    dup += 1
                    ex = ex or (
                        int(b),
                        getattr(seqs[owner[b]], "id", "?"),
                        getattr(s, "id", "?"),
                    )
                owner[b] = k
        if dup:
            _log.warning(
                "BLOCKCHECK: %d shared blocks among %d live seqs, e.g. block %s in seqs %s and %s",
                dup,
                len(seqs),
                *ex,
            )
    except Exception as e:  # noqa: BLE001 -- never break the server
        _log.warning("BLOCKCHECK error: %r", e)
    return out


S.Scheduler.schedule = schedule

# decode slot mapping vs the position: slot must be block_table[pos // bs] * bs + pos % bs
import numpy as np

import atom.model_ops.attentions.aiter_attention as AA

_orig_pd = AA.AiterAttentionMetadataBuilder.prepare_decode
_slot_steps = [0, 0]


def prepare_decode(self, batch, running_bs, running_tokens, max_seqlen_q):
    out = _orig_pd(self, batch, running_bs, running_tokens, max_seqlen_q)
    if _slot_steps[0] == 0:
        _log.warning("SLOTCHECK armed in this process")
    try:
        if max_seqlen_q == 1 and not batch.is_dummy_run:
            var = self.model_runner.forward_vars
            n = batch.total_tokens_num_decode
            bsz = self.model_runner.block_size
            slots = np.array(var["slot_mapping"].np[:n])
            ctx = np.asarray(batch.context_lens, dtype=np.int64)[:n]
            pos = ctx - 1
            bad = []
            for i, (bt, p) in enumerate(zip(batch.block_tables, pos)):
                exp = (
                    int(bt[p // bsz]) * bsz + int(p % bsz)
                    if p // bsz < len(bt)
                    else -999
                )
                if exp != int(slots[i]):
                    bad.append(
                        (
                            i,
                            int(p),
                            int(slots[i]),
                            exp,
                            len(bt),
                            int(batch.last_block_num_tokens[i]),
                        )
                    )
            _slot_steps[0] += 1
            if bad:
                _slot_steps[1] += 1
                _log.warning(
                    "SLOTCHECK step %d: %d/%d decode slots wrong, e.g. (row, pos, slot, expected, blocks, last_block_num) %s",
                    _slot_steps[0],
                    len(bad),
                    n,
                    bad[:3],
                )
    except Exception as e:  # noqa: BLE001
        _log.warning("SLOTCHECK error: %r", e)
    return out


AA.AiterAttentionMetadataBuilder.prepare_decode = prepare_decode

# prefill slot mapping: every scheduled token's slot vs its row's block table
import atom.model_ops.attentions.backends as BK

_orig_wps = BK.CommonAttentionBuilder._write_prefill_slots
_pf = [0, 0]


def _write_prefill_slots(
    self, batch, scheduled_bs, positions, seqlens_q, cached_lens, context_lens
):
    _orig_wps(
        self, batch, scheduled_bs, positions, seqlens_q, cached_lens, context_lens
    )
    try:
        if _pf[0] == 0:
            _log.warning("PREFILLCHECK armed in this process")
        _pf[0] += 1
        bsz = self.model_runner.block_size
        slots = np.array(
            self.model_runner.forward_vars["slot_mapping"].np[: positions.shape[0]]
        )
        if not batch.block_tables:
            return
        bad, k = [], 0
        for r in range(scheduled_bs):
            bt = batch.block_tables[r]
            for j in range(int(seqlens_q[r])):
                p = int(positions[k])
                exp = int(bt[p // bsz]) * bsz + p % bsz if p // bsz < len(bt) else -999
                if exp != int(slots[k]):
                    bad.append((r, j, p, int(slots[k]), exp))
                k += 1
        if bad:
            _pf[1] += 1
            _log.warning(
                "PREFILLCHECK step %d: %d/%d prefill slots wrong in %d seqs, e.g. (row, j, pos, slot, expected) %s",
                _pf[0],
                len(bad),
                k,
                len({b[0] for b in bad}),
                bad[:3],
            )
    except Exception as e:  # noqa: BLE001
        _log.warning("PREFILLCHECK error: %r", e)


BK.CommonAttentionBuilder._write_prefill_slots = _write_prefill_slots

if __name__ == "__main__":
    sys.argv = ["atom.entrypoints.openai_server"] + sys.argv[1:]
    runpy.run_module("atom.entrypoints.openai_server", run_name="__main__")
