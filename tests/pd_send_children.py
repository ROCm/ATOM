# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Send processes `test_pd_send_process.py` spawns in place of `child_main`."""

import os
import time

BOUND, DONE, FATAL, STOP = "bound", "done", "fatal", "stop"
MARKER = "PD_SEND_TEST_MARKER"


def silent(spec, from_stage, to_stage):
    time.sleep(600)


def bound_then_exit(spec, from_stage, to_stage):
    to_stage.send((BOUND, "engine"))
    os._exit(3)


def bound_then_fatal(spec, from_stage, to_stage):
    to_stage.send((BOUND, "engine"))
    to_stage.send((FATAL, "write listener exited: OSError('gone')"))
    time.sleep(600)


def bound_then_done(spec, from_stage, to_stage):
    to_stage.send((BOUND, "engine"))
    to_stage.send((DONE, ["xfer-a"]))
    while from_stage.recv()[0] != STOP:
        pass


def echo(spec, from_stage, to_stage):
    to_stage.send((BOUND, "engine"))
    while True:
        message = from_stage.recv()
        to_stage.send(("echo", message))
        if message[0] == STOP:
            return


def _block(spec):
    with open(os.environ[MARKER], "w") as f:
        f.write(str(os.getpid()))
    time.sleep(600)


def blocked_in_setup(spec, from_stage, to_stage):
    """The real `child_main`, held where it would map the stage's pool."""
    import torch

    from atom.kv_transfer.disaggregation.mooncake import send_process

    torch.cuda.set_device = lambda device: None
    send_process._open_views = _block
    send_process.child_main(spec, from_stage, to_stage)


def stage_with_blocked_child(spec):
    """A stage whose send process never binds."""
    from atom.kv_transfer.disaggregation.mooncake import send_process

    send_process.child_main = blocked_in_setup
    send_process.StageLink(spec)
