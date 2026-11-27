# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""HOST's decode-step pieces nl1..nl4 behind runtime switches, for one-server A/B arms.

Module attributes, default on (the production behaviour), no environment variables. ``ENABLED`` turns every piece off
at once; each piece also has its own switch. With a piece off, the step runs the base's (rF13b's) launches for it:
the same kernels, launches and copies. A harness flips them per arm
(e.g. wire_dev ``WIRE_SW_nl=atom.model_engine.host_nl:ENABLED:True:False``).

Where each switch is read, so a flip takes effect there:
- at the decode graph's recording and its eager warm-up: ``LM_HEAD_IN_GRAPH``, ``HIDDEN_FROM_GRAPH``, ``EMBED_FULL_TABLE``
  (the embedding op runs inside the recorded forward), ``PLAN_NO_REFRESH`` (the attention op's explicit-plan refresh);
- at every step, on the host: ``NO_DECODE_PLAN``, ``DEFER_D2H``, ``VERIFY_NEXT_IDS``, ``VERIFY_GREEDY_NO_NOISE``.
"""

ENABLED = True
# nl1
LM_HEAD_IN_GRAPH = True  # the LM head + logits all-gather recorded in the decode graph (off: rF13b's LM-head graph after it)
HIDDEN_FROM_GRAPH = True  # the step's hidden states are the final norm's output (off: copied into forward_vars["outputs"])
NO_DECODE_PLAN = True  # no kv_indices / FlyDSL plan built in prepare when nothing reads them (off: built every step)
EMBED_FULL_TABLE = True  # each rank looks its rows up in the whole table, no entry all-reduce (off: shard lookup + AR)
# nl2
DEFER_D2H = True  # the step's D2H copies wait for the next step's staging event (off: rF13b's forward-done event)
VERIFY_NEXT_IDS = True  # the verify launch writes each request's next id (off: rF13b's gather after it)
# nl3
PLAN_NO_REFRESH = True  # sparse layers' plan_free PA plans are not refreshed every step (off: refreshed every step)
# nl4
VERIFY_GREEDY_NO_NOISE = True  # the verify launch draws noise only on sampling rows (off: on every row, rF13b's)


def on(piece: str) -> bool:
    """Whether ``piece`` (a switch name above) is in effect: ``ENABLED`` and its own switch."""
    return bool(ENABLED and globals()[piece])

# Eager steps that took a piece's off path (counted per call, cumulative): rF13b's decode plan built in prepare,
# rF13b's forward-done event recorded for the step's D2H copies, verify launches drawing noise on every row, steps
# that gathered the next ids after the verify launch.
STATS = {"nl_off_decode_plan": 0, "nl_off_d2h_event": 0, "nl_off_verify_noise": 0, "nl_off_next_gather": 0}
