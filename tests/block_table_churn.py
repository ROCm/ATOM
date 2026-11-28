# SPDX-License-Identifier: MIT
"""Scheduler churn on a running batch of block tables, shared by the GPU and
the CPU row-plan tests (`test_block_table_row_plan.py`,
`test_block_table_row_plan_cpu.py`)."""

import numpy as np

from atom.model_engine.sequence import new_block_table


def ids(rng, n, page_limit):
    """``n`` random page ids below ``page_limit``."""
    return rng.integers(0, page_limit, size=n).tolist()


def decode_churn(running, rng, pyrng, step, *, columns, max_running, page_limit):
    """One decode step's changes to ``running``, in place: page appends,
    speculative rollback, a preemption that remaps a page, finishes (later
    rows shift up), admissions (now and then a row of exactly ``columns``
    pages) and a reorder."""
    for t in running:  # decode growth
        if pyrng.random() < 0.15 and len(t) < columns:
            t.append(int(rng.integers(0, page_limit)))
    if pyrng.random() < 0.05:  # speculative rollback / truncation
        t = pyrng.choice(running)
        if len(t) > 1:
            del t[len(t) - pyrng.randint(1, min(3, len(t) - 1)) :]
    if pyrng.random() < 0.03:  # preemption: a table remapped in place
        t = pyrng.choice(running)
        t[0] = int(rng.integers(0, page_limit))
    for _ in range(np.random.default_rng(step).poisson(0.6)):  # finishes
        if len(running) > 1:
            running.pop(pyrng.randrange(len(running)))
    while len(running) < max_running and pyrng.random() < 0.5:  # admissions
        n = columns if pyrng.random() < 0.02 else int(rng.integers(1, columns // 4))
        new = new_block_table(ids(rng, n, page_limit))
        running.insert(pyrng.randrange(len(running) + 1), new)
    if pyrng.random() < 0.05:  # a reorder
        pyrng.shuffle(running)
