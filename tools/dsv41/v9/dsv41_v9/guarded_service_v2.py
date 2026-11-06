"""Guarded service dispatcher for the owner/prefill FP8 combination.

`install` patches the owner serving path; candidate imports are deferred, so a
disabled install (`DSV41_COMBINED_GUARDED` unset) mutates nothing.

- Owner 384-row steps: Mega-MHC supply (`new_supply`), the minimal-wait stage-1
  G1 (`variants`) and the ordered reduce-scatter (`ordered_adapter`).
- Prefill FFN on layers 2-18 with 4096-16384 rows: explicit owned FFN
  (`explicit_ffn`).

Anything else falls through to the native implementation.
"""

import collections
import functools
import importlib
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

ACTIVE = threading.local()
STATE = None
INSTALLED = False
PREPARE = None
PREFILL_ROWS = range(4096, 16385)
PREFILL_LAYERS = range(2, 19)


def install():
    global STATE, INSTALLED, PREPARE
    if os.environ.get("DSV41_COMBINED_GUARDED", "0") != "1":
        return {"installed": False, "reason": "default_off"}
    if INSTALLED:
        raise RuntimeError("Install exactly once per service worker")
    import aiter.ops.flydsl.moe_kernels as kernels
    import torch
    import torch.distributed as dist
    from aiter.jit.utils.torch_guard import torch_compile_guard

    from atom.models.deepseek_v41.owner import row_sharded_mhc as seam
    from atom.models.deepseek_v41.owner.service.owner_serving import OwnerServing
    from atom.utils.forward_context import get_forward_context

    from .explicit_ffn import explicit_owned_ffn
    from .new_supply import supplied_forward as agv_forward
    from .ordered_adapter import OrderedRS
    from .variants import minimal_wait

    pf = importlib.import_module("atom.models.deepseek_v41.owner.prefill_ffn")
    supply = importlib.import_module(
        "atom.models.deepseek_v41.owner.service.owner_ffn_supply"
    )
    STATE = {
        "counts": collections.Counter(),
        "prefill_bindings": collections.Counter(),
        "owner_bindings": [],
        "rs": None,
        "ready": False,
    }
    original_init, original_target = OwnerServing.__init__, OwnerServing.target
    original_rs, original_g1 = seam.v41_mhc_reduce_scatter, kernels.flydsl_moe_stage1
    original_supply, original_prefill = (
        supply.supplied_forward,
        pf.prequantized_owned_ffn,
    )

    def prepare(group):
        if STATE["ready"]:
            assert STATE["group"] is group
            return snapshot()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Communication preparation must precede graph capture")
        adapter = OrderedRS(group)
        generator = torch.Generator(device="cuda").manual_seed(
            61931 + group.rank_in_group
        )
        checks = []
        for scale in (1.0, 0.00390625):
            value = (
                torch.randn((1536, 5120), generator=generator, device="cuda") * scale
            ).to(torch.bfloat16)
            reference, result = original_rs(value), adapter(value)
            torch.cuda.synchronize()
            checks.append(
                torch.equal(reference.view(torch.uint8), result.view(torch.uint8))
            )
        agreement = [None] * group.world_size
        dist.all_gather_object(agreement, all(checks), group=group.cpu_group)
        if not all(agreement):
            raise RuntimeError(
                f"Ordered RS differs from the native RS bitwise on ranks {agreement}"
            )
        STATE.update(rs=adapter, ready=True, group=group)
        report = os.environ.get("DSV41_COMBINED_REPORT")
        if report:
            out = Path(report)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"rank{group.rank_in_group}_startup.json").write_text(
                json.dumps(
                    {"rs_cases_equal": checks, "all_rank_agreement": agreement},
                    indent=2,
                )
                + "\n"
            )

        return snapshot()

    PREPARE = prepare

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if os.environ.get("ATOM_DSV41_OWNER_FFN_SUPPLY") != "mhc":
            raise ValueError(
                "DSV41_COMBINED_GUARDED requires ATOM_DSV41_OWNER_FFN_SUPPLY=mhc"
            )
        prepare(self.group)
        self._combined_guarded_state = STATE

    @functools.wraps(original_target)
    def target(self, input_ids, metadata, packet_rows):
        previous = getattr(ACTIVE, "owner384", False)
        active = metadata.step.width == 384
        if active and not STATE["ready"]:
            raise RuntimeError("Owner adapter is not ready")
        before = dict(STATE["counts"])
        ACTIVE.owner384 = active
        try:
            result = original_target(self, input_ids, metadata, packet_rows)
            if torch.cuda.is_current_stream_capturing():
                delta = {
                    key: STATE["counts"][key] - before.get(key, 0)
                    for key in ("owner_agv", "owner_wait0", "owner_rs")
                }
                STATE["owner_bindings"].append(
                    {"capacity": metadata.step.width, "dispatches": delta}
                )
            return result
        finally:
            ACTIVE.owner384 = previous

    def supplied(moe, normed, config, group, reduce_scatter):
        if getattr(ACTIVE, "owner384", False) and STATE["ready"]:
            STATE["counts"]["owner_agv"] += 1
            return agv_forward(moe, normed, config, group, reduce_scatter)
        STATE["counts"]["owner_supply_native_fallback"] += 1
        return original_supply(moe, normed, config, group, reduce_scatter)

    def stage1(**kwargs):
        if getattr(ACTIVE, "owner384", False):
            if (
                tuple(kwargs["a"].shape) == (1536, 5120)
                and kwargs["tile_m"] == 64
                and kwargs["tile_n"] == 128
                and kwargs["tile_k"] == 256
            ):
                STATE["counts"]["owner_wait0"] += 1
                return minimal_wait(**kwargs)
            STATE["counts"]["owner_g1_native_fallback"] += 1
        return original_g1(**kwargs)

    @torch_compile_guard(mutates_args=[], gen_fake=seam._scatter_fake)
    def guarded_owner_rs_v2(value: torch.Tensor) -> torch.Tensor:
        if (
            getattr(ACTIVE, "owner384", False)
            and STATE["ready"]
            and tuple(value.shape) == (1536, 5120)
            and value.dtype == torch.bfloat16
            and value.is_contiguous()
        ):
            STATE["counts"]["owner_rs"] += 1
            return STATE["rs"](value)
        return original_rs(value)

    def prefill(moe, local, rows):
        context = get_forward_context()
        layer_id = getattr(moe.experts, "layer_id", None)
        eligible = (
            getattr(ACTIVE, "prefill_enabled", True)
            and STATE["ready"]
            and rows in PREFILL_ROWS
            and layer_id in PREFILL_LAYERS
            and (moe.experts.tp_size, moe.experts.ep_size, moe.experts.dp_size)
            == (4, 1, 1)
            and not getattr(moe, "prefill_route_packet", False)
            and pf.supported(moe, local, rows, context.attn_metadata)
        )
        if not eligible:
            STATE["counts"]["prefill_native_fallback"] += 1
            return original_prefill(moe, local, rows)
        STATE["counts"]["prefill_explicit"] += 1
        STATE["prefill_bindings"][
            (rows, layer_id, bool(torch.cuda.is_current_stream_capturing()))
        ] += 1
        return explicit_owned_ffn(moe, local, rows)

    OwnerServing.__init__, OwnerServing.target = initialize, target
    kernels.flydsl_moe_stage1 = stage1
    seam.v41_mhc_reduce_scatter = guarded_owner_rs_v2
    supply.supplied_forward, pf.prequantized_owned_ffn = supplied, prefill
    INSTALLED = True
    return {"installed": True}


def snapshot():
    if STATE is None:
        return {"installed": False}
    return {
        "installed": INSTALLED,
        "ready": STATE["ready"],
        "host_dispatch_counts": dict(STATE["counts"]),
        "owner_graph_bindings": list(STATE["owner_bindings"]),
        "prefill_host_bindings": [
            {"rows": k[0], "layer": k[1], "capture": k[2], "calls": n}
            for k, n in STATE["prefill_bindings"].items()
        ],
        # Python eager/capture dispatch only; graph replays are not counted here.
    }


def prepare_communication(group):
    if not INSTALLED or PREPARE is None:
        raise RuntimeError("Install before communication preparation")
    return PREPARE(group)


@contextmanager
def selected_prefill(enabled):
    before = getattr(ACTIVE, "prefill_enabled", True)
    ACTIVE.prefill_enabled = enabled
    try:
        yield
    finally:
        ACTIVE.prefill_enabled = before
