# Prefill policy while decodes run: ATOM_SCHED_DECODE_PREFILL_CAP and
# ATOM_SCHED_PREFILL_DUTY (atom/model_engine/scheduler.py, prefill_duty.py).
from collections import deque
from unittest import mock

from conftest import MockConfig

from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus, SequenceType


def _sched(**env):
    with mock.patch.dict("os.environ", {k: str(v) for k, v in env.items()}):
        return Scheduler(
            MockConfig(
                max_num_seqs=4,
                max_num_batched_tokens=512,
                max_model_len=4096,
                num_kvcache_blocks=400,
                enable_chunked_prefill=True,
            )
        )


def _decoding(seq_factory, tokens):
    seq = seq_factory(tokens)
    seq.status = SequenceStatus.RUNNING
    seq.type = SequenceType.DECODE
    return seq


def test_cap_applies_only_while_decoding(seq_factory):
    sched = _sched(ATOM_SCHED_DECODE_PREFILL_CAP=128)
    assert sched._decode_prefill_cap == 128
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    # No decode running: the full 512-token budget.
    assert batch.total_tokens_num_prefill == 400

    sched = _sched(ATOM_SCHED_DECODE_PREFILL_CAP=128)
    sched.running = deque([_decoding(seq_factory, [1, 2, 3, 4])])
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    assert batch.total_tokens_num_prefill == 128
    # The budget is restored after the tick.
    assert sched.max_num_batched_tokens == 512


def test_cap_off_by_default(seq_factory):
    sched = _sched()
    assert not sched._decode_prefill_cap
    assert sched._prefill_duty is None
    sched.running = deque([_decoding(seq_factory, [1, 2, 3, 4])])
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    assert batch.total_tokens_num_prefill == 400


def test_duty_hold_schedules_decode_only(seq_factory):
    sched = _sched(ATOM_SCHED_PREFILL_DUTY=0.5)
    assert sched._prefill_duty is not None
    dec = _decoding(seq_factory, [1, 2, 3, 4])
    sched.running = deque([dec])
    sched.add(seq_factory(list(range(64))))
    with mock.patch.object(sched._prefill_duty, "hold", return_value=True) as hold:
        batch, _ = sched.schedule()
    hold.assert_called_once()
    assert batch.total_seqs_num_prefill == 0
    assert batch.total_seqs_num_decode == 1
    assert len(sched.waiting) == 1
    with mock.patch.object(sched._prefill_duty, "hold", return_value=False):
        batch, _ = sched.schedule()
    assert batch.total_seqs_num_prefill == 1


def test_idle_cap_applies_only_without_decodes(seq_factory):
    # p1/prefix-cache: ATOM_P1_SCHED_IDLE_PREFILL_CAP is the idle twin.
    sched = _sched(ATOM_P1_SCHED_IDLE_PREFILL_CAP=256)
    assert sched._idle_prefill_cap == 256 and not sched._decode_prefill_cap
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    assert batch.total_tokens_num_prefill == 256
    assert sched.max_num_batched_tokens == 512

    sched = _sched(ATOM_P1_SCHED_IDLE_PREFILL_CAP=256)
    sched.running = deque([_decoding(seq_factory, [1, 2, 3, 4])])
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    # decodes running and no decode cap: the full budget, as before
    assert batch.total_tokens_num_prefill == 400


def test_idle_and_decode_caps_together(seq_factory):
    env = dict(ATOM_P1_SCHED_IDLE_PREFILL_CAP=256, ATOM_SCHED_DECODE_PREFILL_CAP=128)
    sched = _sched(**env)
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    assert batch.total_tokens_num_prefill == 256
    sched = _sched(**env)
    sched.running = deque([_decoding(seq_factory, [1, 2, 3, 4])])
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    assert batch.total_tokens_num_prefill == 128
    assert sched.max_num_batched_tokens == 512


def test_idle_cap_off_by_default_and_clamped(seq_factory):
    assert _sched()._idle_prefill_cap == 0
    assert _sched(ATOM_P1_SCHED_IDLE_PREFILL_CAP=4096)._idle_prefill_cap == 512
