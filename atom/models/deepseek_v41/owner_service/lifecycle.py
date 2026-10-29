"""Release owner and native captures before native communicator teardown."""

import functools
import gc


def install():
    import torch
    import torch.distributed as dist
    from aiter.dist.parallel_state import get_tp_group

    from atom.model_engine.model_runner import ModelRunner
    from atom.utils.graph_holders import release_registered_graphs

    if getattr(ModelRunner, "_owner_graph_exit_installed", False):
        return
    original = ModelRunner.exit

    @functools.wraps(original)
    def exit_runner(self):
        if not self.still_running:
            return original(self)
        group = get_tp_group()
        torch.cuda.synchronize()
        release_registered_graphs()
        gc.collect()
        dist.barrier(group=group.cpu_group)
        return original(self)

    ModelRunner.exit = exit_runner
    ModelRunner._owner_graph_exit_installed = True
