"""Prepare-bound C1/q4 compact task front with the exact author M16 G1/G2.

Experimental author-tree integration. No service diagnostics in this module's
hot path. It owns static weight references, graph-stable buffers and explicit
raw ABIs; Native state and the Quark layer loop keep their existing ownership.
"""

import hashlib
import importlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import GateMode, get_2stage_cfgs, get_padded_M
from aiter.ops.flydsl.kernels.mxfp4_gemm1 import gemm1_grid
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import kas_per_chunk_dw_for
from aiter.ops.flydsl.kernels.mxmoe_dispatcher import (
    _active_m_blocks_upper_bound,
    get_g2,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.flydsl.mxfp4_gemm1_kernels import _get_compiled_mxfp4_gemm1_port
from aiter.ops.flydsl.mxfp4_kname import (
    _parse_mxfp4_g1_kname,
    native_scale_layout_for,
    parse_flydsl_v2_gemm2_kernel,
)

import atom.models.qwen4_exp_mono.chain.moe_front_integrated_kernel as front
from atom.model_ops.triton_topk_softmax import triton_topk_softmax
from atom.models.qwen4_exp_mono.chain.moe_front_integrated_contract import (
    BM,
    CAPACITY,
    EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    ROWS,
    TOPK,
    G1Static,
    G2Static,
)
from atom.mono.runtime.compile import launcher_resources

LOGGER = logging.getLogger("atom")
EXPECTED_G1 = "flydsl_mxmoe_g1_a4w4_16x256x256_f16in_hpf_nt_xcd4"
EXPECTED_G2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16"
# Private raw ABI access is scoped to these frozen author implementations.
SOURCE_HASHES = {
    "aiter.fused_moe": "dfe1f65b957656a58581651e3da9c4d0b94ca7ca65d7764551accc9bf6274541",
    "aiter.ops.flydsl.mxfp4_gemm1_kernels": "4acda5aaec8e0dbe5abaa0a51f6b17f07e54adfc1f2ab9741eb4e49a47bbd79d",
    "aiter.ops.flydsl.kernels.mxmoe_dispatcher": "5b51a36a47be834249ef370d20b857144880bf44f674aa72045ea1734ffebf28",
    "aiter.ops.flydsl.mxfp4_kname": "b00efd78767055957efbfaab24ba338171250476b10bdb3aad434c231190e13c",
}


class Unsupported(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise Unsupported(reason)


def check_sources():
    actual = {}
    for name, expected in SOURCE_HASHES.items():
        path = Path(importlib.import_module(name).__file__)
        actual[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual[name] == expected, f"author raw ABI source changed: {name}")
    require(len(actual) == 4, "missing frozen ABI source identities")
    return actual


def validate_layer(ex, device):
    qm = ex.quant_method
    require(
        ex.top_k == 10
        and ex.num_fused_shared_experts == 1
        and ex.shared_expert_scoring_func == "sigmoid"
        and ex.scoring_func == "softmax"
        and ex.renormalize
        and not ex.use_grouped_topk
        and ex.custom_routing_function is None
        and ex.e_score_correction_bias is None
        and ex.routed_scaling_factor == 1.0
        and not ex.apply_router_weight_on_input
        and ex.expert_map is None
        and ex.expert_mask is None
        and ex.activation == ActivationType.Silu
        and ex.w13_bias is None
        and ex.w2_bias is None
        and getattr(ex, "swiglu_limit", 0.0) in (None, 0.0)
        and qm.quant_type == QuantType.per_1x32
        and qm.quant_dtype == dtypes.fp4x2
        and not qm.fp8_activations
        and not qm.is_guinterleave
        and qm.hidden_pad == qm.intermediate_pad == 0
        and qm.fused_experts is None
        and getattr(ex, "w13_input_scale", None) is None
        and getattr(ex, "w2_input_scale", None) is None,
        "unsupported routing, quantization or expert ownership",
    )
    tensors = (ex.w13_weight, ex.w2_weight, ex.w13_weight_scale, ex.w2_weight_scale)
    shapes = ((513, 1280, 1280), (513, 2560, 320), (656640, 80), (1313280, 24))
    for i, (tensor, shape) in enumerate(zip(tensors, shapes)):
        require(
            tuple(tensor.shape) == shape
            and tensor.is_contiguous()
            and tensor.device == device,
            f"weight {i}: shape/stride/device",
        )
        require(
            (
                tensor.dtype == dtypes.fp4x2
                if i < 2
                else tensor.dtype in (torch.uint8, dtypes.fp8_e8m0)
            ),
            f"weight {i}: dtype",
        )
    require(
        all(getattr(t, "is_shuffled", False) for t in tensors[:2]), "unshuffled weight"
    )
    require(
        getattr(ex.w2_weight, "inter_real", INTERMEDIATE) in (None, INTERMEDIATE),
        "padded G2 intermediate",
    )
    return tensors


def resolve_config(ex):
    """Same resolver inputs as the observed author fused_moe full-row call."""
    cfg = get_2stage_cfgs(
        get_padded_M(ROWS),
        HIDDEN,
        INTERMEDIATE,
        EXPERTS,
        TOPK,
        torch.bfloat16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        GateMode.SEPARATED,
        is_ep=False,
        has_stage1_bias=False,
        has_stage2_bias=False,
        situ_beta=1.0,
        situ_linear_beta=1.0,
        swiglu_limit=getattr(ex, "swiglu_limit", 0.0),
        opus_weights_shuffled=True,
        gate_up_interleave=getattr(ex.w13_weight, "gate_up_interleave", None),
        input_dtype=torch.bfloat16,
        has_num_local_tokens=False,
    )
    require(
        cfg.block_m == BM
        and cfg.output_aux == "opus"
        and not cfg.prequant
        and not cfg.run_1stage
        and cfg.fuse_quant == "fp4"
        and cfg.stage1.func.__name__ == "_mxfp4_a4w4_stage1_fw"
        and cfg.stage2.func.__name__ == "_mxfp4_a4w4_stage2_fw"
        and cfg.stage1.keywords == {"kernelName1": EXPECTED_G1, "interleave": False}
        and cfg.stage2.keywords == {"kernelName2": EXPECTED_G2},
        f"Native selected an unsupported M16 configuration: {cfg}",
    )
    return cfg


@dataclass(frozen=True)
class BoundExpert:
    owner: object
    tensors: tuple
    byte_views: tuple
    cfg: object
    g1: G1Static
    g2: G2Static


class PreparedChain:
    def __init__(self, layers, device):
        self.sources = check_sources()
        require(len(layers) == 48, "expected 48 Qwen target layers")
        owners = tuple(layer.mlp.experts for layer in layers)
        tensors = tuple(validate_layer(ex, device) for ex in owners)
        configs = tuple(resolve_config(ex) for ex in owners)
        p1 = _parse_mxfp4_g1_kname(EXPECTED_G1)
        p2 = parse_flydsl_v2_gemm2_kernel(EXPECTED_G2)
        scale_layout = native_scale_layout_for(BM, p1["out_dtype"])
        require(
            scale_layout
            and p1["inline_quant"]
            and p1["act"] == "silu"
            and p2["epilog"] == "atomic"
            and p2["sort_block_m"] == BM,
            "compiled arithmetic/layout contract changed",
        )
        self.front = front.Workspace.allocate(device)
        self.topk_ids = torch.empty((ROWS, TOPK), device=device, dtype=torch.int32)
        self.topk_ids[:, 10] = 512
        self.topk_routed_ids = self.topk_ids[:, :10]
        self.topk_weights = torch.empty(
            (ROWS, TOPK), device=device, dtype=torch.float32
        )
        inter_scale_cols = INTERMEDIATE // 32
        scale_bytes = max(
            CAPACITY * max(64, inter_scale_cols * 2),
            ((CAPACITY + BM - 1) // BM) * kas_per_chunk_dw_for(INTERMEDIATE) * 4,
        )
        scale_rows = (
            ((scale_bytes + inter_scale_cols - 1) // inter_scale_cols + 31) // 32 * 32
        )
        self.intermediate = torch.empty(
            (CAPACITY, INTERMEDIATE // 2), device=device, dtype=torch.uint8
        )
        self.intermediate_scale = torch.empty(
            (scale_rows, inter_scale_cols), device=device, dtype=torch.uint8
        )
        self.unused_a_quant = torch.empty(
            (ROWS, HIDDEN // 2), device=device, dtype=torch.uint8
        )
        self.unused_a_scale = torch.empty(1, device=device, dtype=torch.uint8)
        self.g1_launcher = _get_compiled_mxfp4_gemm1_port(
            BM,
            p1["use_nt"],
            p1["inline_quant"],
            p1["prefetch_hidden"],
            HIDDEN,
            INTERMEDIATE,
            EXPERTS,
            p1["BN"],
            p1["BK"],
            interleave=False,
            xcd_swizzle=p1["xcd_swizzle"],
            a_dtype=p1["a_dtype"],
            out_dtype=p1["out_dtype"],
            act=p1["act"],
            situ_beta=1.0,
            situ_linear_beta=1.0,
            swiglu_limit=float("inf"),
            enable_bias=False,
            native_scale_layout=scale_layout,
            num_waves=p1.get("num_waves", 4),
            k_wave=p1.get("k_wave", 1),
        )
        self.g2_launcher = get_g2(
            p2["tile_m"],
            p2["tile_n"],
            p2["tile_k"],
            p2["use_nt"],
            8192,
            p2["epilog"],
            INTERMEDIATE,
            p2["a_dtype"],
            b_dtype=p2["b_dtype"],
            topk=TOPK,
            SBM=BM,
            persist=False,
            cu_num=0,
            out_dtype="bf16",
            g2_bf16_lds=p2["bf16_lds"],
            g2_spart=p2["spart"],
            enable_bias=False,
            g2_prefetch_ids=False,
            use_valid_token_count=False,
        )
        b = self.front
        grid1 = gemm1_grid(
            ROWS, BM, NE=EXPERTS, TOPK=TOPK, INTER=INTERMEDIATE, BN=p1["BN"]
        )
        max_m_blocks = (CAPACITY + BM - 1) // BM
        grid2 = min(
            max_m_blocks, _active_m_blocks_upper_bound(ROWS, TOPK, EXPERTS, BM, BM)
        )
        require(grid1 == 220 and grid2 == 44, "Native grid contract changed")
        self.layers = tuple(
            BoundExpert(
                ex,
                ts,
                tuple(t.view(torch.uint8) for t in ts),
                cfg,
                G1Static(
                    a_quant=self.unused_a_quant.data_ptr(),
                    a_scale=self.unused_a_scale.data_ptr(),
                    weight=ts[0].data_ptr(),
                    weight_scale=ts[2].data_ptr(),
                    expert_ids=b.experts.data_ptr(),
                    valid=b.valid.data_ptr(),
                    m_indices=b.m_indices.data_ptr(),
                    rows=ROWS,
                    grid=grid1,
                    intermediate=self.intermediate.data_ptr(),
                    intermediate_scale=self.intermediate_scale.data_ptr(),
                ),
                G2Static(
                    intermediate=self.intermediate.data_ptr(),
                    intermediate_scale=self.intermediate_scale.data_ptr(),
                    weight=ts[1].data_ptr(),
                    weight_scale=ts[3].data_ptr(),
                    expert_ids=b.experts.data_ptr(),
                    valid=b.valid.data_ptr(),
                    sorted_ids=b.sorted_ids.data_ptr(),
                    sorted_weights=b.sorted_weights.data_ptr(),
                    unused_bias=b.out.data_ptr(),
                    rows=ROWS,
                    max_m_blocks=max_m_blocks,
                    grid_blocks=grid2,
                    intermediate_width=INTERMEDIATE,
                    hidden_width=HIDDEN,
                    output=b.out.data_ptr(),
                    unused_output_scale=b.out.data_ptr(),
                ),
            )
            for ex, ts, cfg in zip(owners, tensors, configs)
        )
        # Compile and materialize exact Native compiled callables with isolated
        # stateless zero inputs. No model KV/conv/replay/HC state is touched.
        x = torch.zeros((ROWS, HIDDEN), device=device, dtype=torch.bfloat16)
        logits = torch.zeros((ROWS, EXPERTS), device=device, dtype=torch.bfloat16)
        self.prepare_front(logits)
        stream = torch.cuda.current_stream()
        _run_compiled(self.g1_launcher, *self.layers[0].g1.pack(x.data_ptr(), stream))
        _run_compiled(self.g2_launcher, *self.layers[0].g2.pack(stream))
        self.g1_compiled = self.g1_launcher._cf
        self.g2_compiled = self.g2_launcher._cf
        self.resources = {
            "g1": launcher_resources(self.g1_launcher),
            "g2": launcher_resources(self.g2_launcher),
        }
        require(all(self.resources.values()), "missing compiled resource evidence")
        require(
            all(
                not v["scratch"]
                and not v.get("vgpr_spill", 0)
                and not v.get("sgpr_spill", 0)
                for group in self.resources.values()
                for v in group
            ),
            "compiled expert spills",
        )
        self.metadata = {
            "rows": ROWS,
            "layers": 48,
            "g1": EXPECTED_G1,
            "g2": EXPECTED_G2,
            "g1_grid": grid1,
            "g2_grid_blocks": grid2,
            "g2_hidden_max": 8192,
            "num_local_tokens": None,
            "use_valid_token_count": False,
            "intermediate_shape": list(self.intermediate.shape),
            "intermediate_scale_shape": list(self.intermediate_scale.shape),
            "sources": self.sources,
            "resources": self.resources,
            "g2_environment": {
                name: os.environ.get(name, default)
                for name, default in (
                    ("MXFP4_G2_BHOIST", "1"),
                    ("MXFP4_G2_ASCALE_PF", "1"),
                    ("MXFP4_G2_SPART", "402"),
                    ("MXFP4_G2_BF16_LDS", "1"),
                )
            },
        }

    def prepare_front(self, logits):
        triton_topk_softmax(self.topk_weights, self.topk_routed_ids, logits, 1, None)
        return self.front.prepare(logits, ids=self.topk_ids, weights=self.topk_weights)

    def forward_layer(self, layer_id, mixed, logits):
        self.prepare_front(logits)
        layer = self.layers[layer_id]
        stream = torch.cuda.current_stream()
        self.g1_compiled(*layer.g1.pack(mixed.data_ptr(), stream))
        self.g2_compiled(*layer.g2.pack(stream))
        return self.front.out


def prepare_layers(layers, tp, device):
    if tp != 1:
        return None
    try:
        result = PreparedChain(layers, device)
    except Unsupported as error:
        LOGGER.warning("Qwen compact M16 not admitted: %s", error)
        return None
    LOGGER.info(
        "Qwen compact M16 prepared: 48 layers, q4 real rows4, G1=%s G2=%s valid=False",
        EXPECTED_G1,
        EXPECTED_G2,
    )
    return result
