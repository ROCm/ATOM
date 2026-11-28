# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""R-Fork (``--load-format remote_instance``) for ATOM models: receiver
preparation, and seed-side cleanup of send groups that refused receivers left.

SGLang's R-Fork receiver builds the model, then copies the seed's parameters
raw (one RCCL broadcast per tensor, in ``named_parameters()`` order) without
calling ``model.load_weights()``. ATOM's post-load pass runs inside
``load_weights``, so the receiver lacks the state it creates (e.g. the
hyper-connection ``fused_w_cat``/``fused_rank``, MoE prepare/finalize) and
fails at the first forward. A layer the pass reshapes (padded outputs) would
also stop lining up with the seed's tensor.

Binds to ``RemoteInstanceModelLoader.load_model_from_remote_instance_by_nccl
(self, model, client, model_config, device_config)``, unchanged from SGLang
0.5.20 (ATOM's pinned image) to 0.5.22. ATOM models opt in with
``remote_instance_load_supported`` (Qwen4Exp: PTPC-FP8 checkpoints, the
validated ones); others are refused. Only the ``nccl`` backend is covered.
The seed only sends target weights, so a draft model must load from disk:
``--speculative-draft-load-format auto``.
"""

from __future__ import annotations

import functools
import logging
import socket
import time

import torch
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.model_executor.model_runner_components.weight_exporter import (
    WeightExporter,
)
from sglang.srt.model_loader.loader import (
    DefaultModelLoader,
    RemoteInstanceModelLoader,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from torch import nn
from torch.distributed.constants import default_pg_timeout

logger = logging.getLogger("atom")

# SGLang builds R-Fork send groups with the default timeout; a receiver gives up
# on its group after this long.
_GROUP_TIMEOUT_S = default_pg_timeout.total_seconds()
# (id(exporter), group name) -> creation time. WeightExporter has __slots__.
_send_group_born: dict[tuple[int, str], float] = {}


def _is_atom_model(model: nn.Module) -> bool:
    return type(model).__module__.startswith("atom.")


def _check_supported(model: nn.Module) -> None:
    # Opt-in: a model whose post-load pass derives state from weight values
    # (e.g. DeepSeek MLA w_kc/w_vc) would compute it from the placeholders.
    if not getattr(model, "remote_instance_load_supported", False):
        raise NotImplementedError(
            f"R-Fork (--load-format remote_instance) is not supported for ATOM "
            f"model {type(model).__name__} with this checkpoint; load it from "
            f"disk instead (for a speculative draft: "
            f"--speculative-draft-load-format auto)"
        )


def _params_moved_out(model: nn.Module, numel_before: dict[str, int]) -> list[str]:
    """Parameters whose weights the post-load pass moved where the copy can't
    reach: deleted and re-bound as plain tensors (MXFP4 Triton MoE) or emptied
    (MegaMoE). ``layer.x = None`` drops the parameter on the seed too."""
    params_after = dict(model.named_parameters())
    moved = []
    for name, numel in numel_before.items():
        if name in params_after:
            if numel and params_after[name].numel() == 0:
                moved.append(name)
            continue
        owner, _, attr = name.rpartition(".")
        if model.get_submodule(owner)._parameters.get(attr, moved) is not None:
            moved.append(name)
    return moved


@torch.no_grad()
def prepare_atom_model_for_receive(
    model: nn.Module, model_config, device_config
) -> None:
    """Replay a disk load's post-load pass, minus the checkpoint read.

    Same as ``DefaultModelLoader.load_model`` after ``load_weights``: ATOM's
    per-module step (``run_module_post_load_hooks``, under the process-global
    SGLang framework/config that building the model set, as on a disk load),
    then SGLang's ``postprocess_weights``, under the model dtype. Runs on
    ``--load_dummy`` placeholders (finite, positive scales), which hooks such
    as the FP8 PLE scale check accept. Not idempotent, so it must not run again
    after the copy.
    """
    # Heavy (fused MoE, custom ops): kept out of plugin registration.
    from atom.model_loader.loader import (
        initialize_dummy_weights,
        run_module_post_load_hooks,
    )

    numel_before = {}
    for name, param in model.named_parameters():
        if param.is_meta:
            raise NotImplementedError(f"R-Fork cannot receive into meta param {name}")
        numel_before[name] = param.numel()
    initialize_dummy_weights(model, "xavier")

    with set_default_torch_dtype(model_config.dtype):
        for module in model.modules():
            run_module_post_load_hooks(module)
        DefaultModelLoader.postprocess_weights(
            model, torch.device(device_config.device)
        )

    moved = _params_moved_out(model, numel_before)
    if moved:
        raise NotImplementedError(
            f"R-Fork copies parameters only, but post-load processing moved "
            f"{len(moved)} of them (e.g. {moved[0]}) out of the parameter list; "
            f"load this model from disk instead"
        )
    _check_supported(model)  # again: the pass may quantize online (MXFP4)


def _release_seed(loader, client, device_config) -> None:
    """Join and drop the seed's send group, as the copy would.

    SGLang asks the seed to build the group before the model exists, and the
    seed's scheduler blocks until every receiver rank joins (up to the 30 min
    process-group timeout). A refused rank joins and drops it so the seed
    serves again at once; no weights are sent. The seed keeps its side of the
    group until a later request with the same name, or any later request once
    the group is older than that timeout (``_wrap_seed_init_group``).
    """
    success, message = client.build_group(
        gpu_id=device_config.gpu_id,
        tp_rank=loader.load_config.tp_rank,
        instance_ip=socket.gethostbyname(socket.gethostname()),
    )
    if success:
        torch.distributed.distributed_c10d.destroy_process_group(
            client._model_update_group
        )
    else:
        logger.warning("R-Fork: could not release the seed: %s", message)


def _all_ranks_ok(ok: bool) -> bool:
    """Whether every receiver TP rank prepared successfully.

    Each rank asks the seed for its own send group, and the seed broadcasts on
    all of them; one rank that bails out while the others go on would hang the
    seed. So the ranks agree first, and all of them release the seed together.
    """
    group = get_tp_group()
    if group.world_size == 1:
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    torch.distributed.all_reduce(
        flag, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    return bool(flag.item())


def _wrap_nccl_receive(original):
    @functools.wraps(original)
    def load_by_nccl(self, model, client, model_config, device_config):
        if _is_atom_model(model):
            start = time.perf_counter()
            # Any failure here (a refusal, OOM, a hook on placeholders) on any
            # rank would leave the seed waiting; build_group never raises.
            # Every rank calls _all_ranks_ok exactly once on either path.
            try:
                _check_supported(model)
                prepare_atom_model_for_receive(model, model_config, device_config)
            except Exception:
                _all_ranks_ok(False)
                _release_seed(self, client, device_config)
                raise
            if not _all_ranks_ok(True):
                _release_seed(self, client, device_config)
                raise RuntimeError(
                    "R-Fork: another receiver rank failed to prepare; see its log"
                )
            logger.info(
                "R-Fork: prepared %s for the seed's weight layout in %.2f s",
                type(model).__name__,
                time.perf_counter() - start,
            )
        return original(self, model, client, model_config, device_config)

    load_by_nccl._atom_rfork = True
    return load_by_nccl


def _wrap_seed_init_group(original):
    """Seed side: drop send groups refused receivers left behind.

    The seed only drops a send group after sending weights, which never happens
    for a refused receiver. On each new request, drop the group of the same name
    (so a retry does not fail on a duplicate name) and any group older than the
    process-group timeout, which its receiver has given up on by then. A younger
    group of another receiver may still be waiting for its send, so it stays.
    """

    @functools.wraps(original)
    def init_group(
        self, master_address, ports, group_rank, world_size, group_name, *a, **kw
    ):
        name = f"{group_name}_{ports.split(',')[self.tp_rank]}_{self.tp_rank}"
        groups = self._weights_send_group
        for key in [k for k in _send_group_born if k[0] == id(self)]:
            if key[1] not in groups:  # sent and dropped by SGLang
                del _send_group_born[key]
        now = time.monotonic()
        for key in list(groups):
            born = _send_group_born.get((id(self), key), now)
            if key == name or now - born > _GROUP_TIMEOUT_S:
                _send_group_born.pop((id(self), key), None)
                logger.info("R-Fork: dropping the stale send group %s", key)
                torch.distributed.distributed_c10d.destroy_process_group(
                    groups.pop(key)
                )
        result = original(
            self, master_address, ports, group_rank, world_size, group_name, *a, **kw
        )
        if name in groups:
            _send_group_born[(id(self), name)] = time.monotonic()
        return result

    return init_group


def apply_remote_instance_patch() -> None:
    original = RemoteInstanceModelLoader.load_model_from_remote_instance_by_nccl
    if getattr(original, "_atom_rfork", False):
        return
    RemoteInstanceModelLoader.load_model_from_remote_instance_by_nccl = (
        _wrap_nccl_receive(original)
    )
    WeightExporter.init_weights_send_group_for_remote_instance = _wrap_seed_init_group(
        WeightExporter.init_weights_send_group_for_remote_instance
    )
