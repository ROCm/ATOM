# ruff: noqa: I001
# Keep measured import ordering/namespace; AST parity is verified by prepare.py.
"""Default-off service dispatcher for the qualified owner/prefill combination.

This source is a prototype until real module/Inductor/FULL and service probes pass.
All candidate imports are deferred, leaving a disabled install without mutations.
"""

import collections
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading

ACTIVE = threading.local()
STATE = None
INSTALLED = False
PREPARE = None


def read_qualification(path):
    value = json.loads(Path(path).read_text())
    if value["status"] != "full17_FFN_225shape_numerical_pass":
        raise ValueError("Complete shape qualification required before service install")
    if value["layers"] != list(range(2, 19)) or len(set(value["real_rows"])) != 225:
        raise ValueError("Qualification scope mismatch")
    for name, expected in value["evidence_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Qualification evidence changed: {name}")
    return value


def install():
    global STATE, INSTALLED, PREPARE
    if os.environ.get("DSV41_COMBINED_GUARDED", "0") != "1":
        return {"installed": False, "reason": "default_off"}
    if INSTALLED:
        raise RuntimeError("Install exactly once per service worker")
    qualification = read_qualification(os.environ["DSV41_COMBINED_QUALIFICATION"])
    source = Path(os.environ["DSV41_COMBINED_FROZEN_SOURCE"]).resolve()
    for name, expected in qualification["candidate_source_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Candidate source/binary changed: {name}")
    sys.path.insert(0, str(source))
    import torch
    import torch.distributed as dist
    from aiter.dist.parallel_state import get_tp_group  # noqa: F401
    from aiter.jit.utils.torch_guard import torch_compile_guard
    import aiter.ops.flydsl.moe_kernels as kernels
    from variants import minimal_wait
    from ordered_adapter import OrderedRS
    from explicit_ffn import explicit_owned_ffn
    from new_supply import supplied_forward as agv_forward
    from atom.models.deepseek_v41.owner_service.owner_serving import OwnerServing
    from atom.model_ops.deepseek_v41 import row_sharded_mhc as seam
    from atom.utils.forward_context import get_forward_context

    pf = importlib.import_module("atom.models.deepseek_v41.prefill_ffn")
    supply = importlib.import_module(
        "atom.models.deepseek_v41.owner_service.owner_ffn_supply"
    )
    STATE = {
        "qualified_rows": frozenset(qualification["real_rows"]),
        "qualified_layers": frozenset(qualification["layers"]),
        "counts": collections.Counter(),
        "prefill_bindings": collections.Counter(),
        "owner_bindings": [],
        "rs": None,
        "ready": False,
        "qualification_sha256": hashlib.sha256(
            Path(os.environ["DSV41_COMBINED_QUALIFICATION"]).read_bytes()
        ).hexdigest(),
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
        dist.all_gather_object(
            agreement,
            {"rs_ok": all(checks), "qualification": STATE["qualification_sha256"]},
            group=group.cpu_group,
        )
        if not all(
            v["rs_ok"] and v["qualification"] == STATE["qualification_sha256"]
            for v in agreement
        ):
            raise RuntimeError(
                "All-rank RS calibration or qualification agreement failed"
            )
        STATE.update(rs=adapter, ready=True)
        STATE["group"] = group
        out = Path(os.environ["DSV41_COMBINED_REPORT"])
        out.mkdir(parents=True, exist_ok=True)
        (out / f"rank{group.rank_in_group}_startup.json").write_text(
            json.dumps(
                {
                    "rs_cases_equal": checks,
                    "all_rank_agreement": agreement,
                    "qualified_prefill_shapes": len(STATE["qualified_rows"]),
                    "scope": "Startup communication/qualification agreement only; not service or task accuracy acceptance",
                },
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
                "Candidate arm must explicitly select owner Mega-MHC supply"
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
        qualified = (
            getattr(ACTIVE, "prefill_enabled", True)
            and STATE["ready"]
            and rows in STATE["qualified_rows"]
            and layer_id in STATE["qualified_layers"]
            and (moe.experts.tp_size, moe.experts.ep_size, moe.experts.dp_size)
            == (4, 1, 1)
            and not getattr(moe, "prefill_route_packet", False)
            and pf.supported(moe, local, rows, context.attn_metadata)
        )
        if not qualified:
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
    return {
        "installed": True,
        "qualified_rows": len(STATE["qualified_rows"]),
        "service_acceptance": "not_run",
    }


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
        "qualification_sha256": STATE["qualification_sha256"],
        "counter_scope": "Python eager/capture dispatch; graph replay backend work requires binding with actual replay counters",
    }


def prepare_communication(group):
    if not INSTALLED or PREPARE is None:
        raise RuntimeError("Install before communication preparation")
    return PREPARE(group)


from contextlib import contextmanager


@contextmanager
def selected_prefill(enabled):
    before = getattr(ACTIVE, "prefill_enabled", True)
    ACTIVE.prefill_enabled = enabled
    try:
        yield
    finally:
        ACTIVE.prefill_enabled = before
