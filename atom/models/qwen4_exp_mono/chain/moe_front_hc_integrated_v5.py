"""Isolated C1/q4 HC/router/top-k/tasks -> direct Native G1 -> G2 adapter.

Prepare freezes owners and ABIs; the existing Quark loop owns pending/PLE and
all Native attention/state seams. HC publishes the actual mixed input and task
table consumed by G1/G2. No wrapper recomputes routing between the phases.
"""

import logging

import torch

import atom.models.qwen4_exp_mono.chain.moe_front_hc_v5_device as hc_device
from atom.models.qwen4_exp_mono.chain.moe_front_hc_integrated_v5_contract import (
    HcStatic,
)
from atom.models.qwen4_exp_mono.chain.moe_front_hc_v5_contract import (
    FLAGS_BYTES,
    HC_TASKS,
)
from atom.models.qwen4_exp_mono.chain.moe_front_integrated import (
    PreparedChain,
    Unsupported,
    require,
)
from atom.mono.plan.layout import DIAG_WORDS
from atom.mono.runtime.compile import compile_kernels, launcher_resources

LOGGER = logging.getLogger("atom")


def validate_hc(bound_weights, built, device):
    require(len(bound_weights) == 48, "expected 48 HC bindings")
    require(
        built.mtp is not None
        and built.mtp.plan.rows == 4
        and built.mtp.plan.query_length == 4
        and not built.native_hc_down
        and built.router_splits == 1
        and built.hc_router is not None,
        "requires four-part C1 HC/router",
    )
    for name, shape, dtype in (
        ("mixed", (4, 2560), torch.bfloat16),
        ("raw", (4, 4), torch.bfloat16),
        ("logits", (4, 513), torch.bfloat16),
        ("hc_payload", (4 * 2560,), torch.int32),
    ):
        value = getattr(built, name)
        require(
            value is not None
            and tuple(value.shape) == shape
            and value.dtype == dtype
            and value.device == device
            and value.is_contiguous(),
            f"HC output binding: {name}",
        )
    for item in bound_weights:
        for value, shape in (
            (item.mlp_hc.up, (10240, 320)),
            (item.router, (513, 2560)),
        ):
            require(
                tuple(value.shape) == shape
                and value.dtype == torch.bfloat16
                and value.device == device
                and value.is_contiguous(),
                "HC/router weight layout",
            )


class PreparedHcChain(PreparedChain):
    def __init__(self, layers, bound_weights, built, device):
        validate_hc(bound_weights, built, device)
        # Inherited prepare warms only stateless expert phases on independent
        # zero inputs. No model HC payload/cache/replay state is touched.
        super().__init__(layers, device)
        self.hc_kernel = hc_device.build(False)
        # No HC execution: ready=epoch1 cannot leak into fresh payload=epoch0.
        compile_kernels([(self.hc_kernel, hc_device.ABI)])
        resources = launcher_resources(self.hc_kernel)
        require(
            resources
            and all(
                not x["scratch"]
                and not x.get("vgpr_spill", 0)
                and not x.get("sgpr_spill", 0)
                and x["vgpr"] <= 128
                and x["lds"] <= 65536
                for x in resources
            ),
            "HC front resources",
        )
        self.flags = torch.zeros(FLAGS_BYTES // 4, device=device, dtype=torch.int32)
        self.diag = torch.zeros(HC_TASKS * DIAG_WORDS, device=device, dtype=torch.int32)
        self.mixed, self.raw, self.logits, self.payload = (
            built.mixed,
            built.raw,
            built.logits,
            built.hc_payload,
        )
        self.hc_owners = tuple(bound_weights)
        b = self.front
        self.hc_layers = tuple(
            HcStatic(
                item.mlp_hc.up.data_ptr(),
                self.raw.data_ptr(),
                item.router.data_ptr(),
                self.logits.data_ptr(),
                self.mixed.data_ptr(),
                self.payload.data_ptr(),
                self.topk_ids.data_ptr(),
                self.topk_weights.data_ptr(),
                b.sorted_ids.data_ptr(),
                b.sorted_weights.data_ptr(),
                b.experts.data_ptr(),
                b.valid.data_ptr(),
                b.m_indices.data_ptr(),
                b.reverse.data_ptr(),
                b.out.data_ptr(),
                self.flags.data_ptr(),
                self.diag.data_ptr(),
            )
            for item in bound_weights
        )
        self.metadata.update(
            hc_source=hc_device.SOURCE,
            hc_resources=resources,
            hc_phase_threads=512,
            hc_phase_ctas=160,
            hc_down_parts=4,
            hc_debug=False,
            hc_warmup="compile-only; no HC payload execution",
            phases=["HC/router/topk/tasks/clear", "direct_G1", "direct_G2"],
            runtime_front="FlyDSL; no Native topk or Triton task launch",
        )

    def prepare_layer(self, layer_id, part, normed):
        self.hc_kernel(
            *self.hc_layers[layer_id].pack(
                part.data_ptr(), normed.data_ptr(), normed.stride(0)
            ),
            stream=torch.cuda.current_stream(),
        )

    def forward_layer(self, layer_id, mixed, logits):
        # HC phase completion publishes actual mixed/tasks/output clear.
        layer = self.layers[layer_id]
        stream = torch.cuda.current_stream()
        self.g1_compiled(*layer.g1.pack(mixed.data_ptr(), stream))
        self.g2_compiled(*layer.g2.pack(stream))
        return self.front.out


def prepare_layers(layers, bound_weights, built, tp, device):
    if tp != 1:
        return None
    try:
        result = PreparedHcChain(layers, bound_weights, built, device)
    except Unsupported as error:
        LOGGER.warning("Qwen HC compact M16 not admitted: %s", error)
        return None
    LOGGER.info(
        "Qwen HC compact M16 prepared: 48 layers, C1 q4; HC front -> direct G1/G2"
    )
    return result
