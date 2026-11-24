# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MoE entry's sequence-parallel input: ``prepare_mxfp4_moe`` quantizes and routes a local bf16 shard; the caller
concatenates the row-major payloads of every rank in rank order (padding rows included) and builds the matching input
type from them. Both formats quantize straight from the bf16 input; the compact one carries the FP4 scales and an exact
correction mask instead of the FP4 codes.
"""
from dataclasses import dataclass

import torch

from amd_kernels.warm_up import entry


@dataclass(frozen=True)
class MoeInput:
    """Quantized operands and routing, no bf16 activations or expert weights.

    ``tensors()`` returns the gather payload in constructor order, without copies. The scales are e8m0 bytes per 32
    values, the routing weights include the routed scaling factor; the shared payloads are the shared expert's MXFP8
    input.
    """
    routed_codes: torch.Tensor
    routed_scales: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    shared_codes: torch.Tensor | None = None
    shared_scales: torch.Tensor | None = None

    def tensors(self):
        return (self.routed_codes, self.routed_scales, self.topk_weights, self.topk_ids, self.shared_codes,
                self.shared_scales)

    def compact(self):
        """The 7328 B/token payload (packing costs a launch); release this object's FP4 codes after packing."""
        from .sidecar_pack import pack_mxfp4_sidecar
        if self.shared_codes is None or self.shared_scales is None:
            raise ValueError("compact input requires shared MXFP8 operands")
        if not isinstance(self.routed_codes,torch.Tensor) or self.routed_codes.ndim!=2:
            raise ValueError("expected routed codes [M,3072]")
        n=self.routed_codes.shape[0]
        compact=MoeInputCompact(self.routed_scales,self.topk_weights,self.topk_ids,
                                self.shared_codes,self.shared_scales,None)
        compact._validate(check_mask=False)
        if (self.routed_codes.shape!=(n,3072) or self.routed_codes.dtype!=torch.uint8
                or not self.routed_codes.is_contiguous()
                or self.routed_codes.device!=self.shared_codes.device):
            raise ValueError("expected contiguous uint8 routed codes [M,3072]")
        mask=pack_mxfp4_sidecar(self.routed_codes,self.routed_scales,
                               self.shared_codes,self.shared_scales)
        return MoeInputCompact(*compact.tensors()[:-1],mask)

    def validate(self, w):
        from .chain import H, TOPK, variant_for
        if not isinstance(self.routed_codes, torch.Tensor) or self.routed_codes.ndim != 2:
            raise ValueError("routed_codes must be row-major [M,3072]")
        n = self.routed_codes.shape[0]
        variant_for(n)
        fields = ((self.routed_codes, (n, H // 2), torch.uint8),
                  (self.routed_scales, (n, H // 32), torch.uint8),
                  (self.topk_weights, (n, TOPK), torch.float32),
                  (self.topk_ids, (n, TOPK), torch.int32),
                  (self.shared_codes, (n, H), torch.uint8),
                  (self.shared_scales, (n, H // 32), torch.uint8))
        device = w.router_weight.device
        for tensor, shape, dtype in fields:
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != dtype
                    or not tensor.is_contiguous() or tensor.device != device):
                raise ValueError(f"expected contiguous {dtype} {shape} on {device}")


@dataclass(frozen=True)
class MoeInputCompact:
    """The lossless gather payload, in ``tensors()`` order, all row-major.

    The FP4 scales and one correction bit per value rebuild the direct FP4 codes from the shared FP8 ones (signed zero
    and the quantization kept): 2304 B/token less than ``MoeInput`` with a shared expert, at one launch to pack and one
    to rebuild.
    """
    routed_scales: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    shared_codes: torch.Tensor
    shared_scales: torch.Tensor
    correction_mask: torch.Tensor

    def tensors(self):
        return (self.routed_scales,self.topk_weights,self.topk_ids,self.shared_codes,
                self.shared_scales,self.correction_mask)

    def _validate(self, *, check_mask=True):
        from .chain import variant_for
        if not isinstance(self.shared_codes,torch.Tensor) or self.shared_codes.ndim!=2:
            raise ValueError("expected shared codes [M,6144]")
        n=self.shared_codes.shape[0]
        if self.shared_codes.device.type!="cuda":
            raise ValueError("compact input requires CUDA operands")
        variant_for(n)
        specs=(((n,192),torch.uint8),((n,4),torch.float32),((n,4),torch.int32),
               ((n,6144),torch.uint8),((n,192),torch.uint8),((n,768),torch.uint8))
        for t,(shape,dtype) in zip(self.tensors() if check_mask else self.tensors()[:-1],specs):
            if (not isinstance(t,torch.Tensor) or t.shape!=shape or t.dtype!=dtype
                    or not t.is_contiguous() or t.device!=self.shared_codes.device):
                raise ValueError(f"expected contiguous {dtype} {shape} on {self.shared_codes.device}")

    def expand(self):
        """Reconstruct FP4 codes bit for bit; keep routing, scales and shared codes."""
        from .sidecar_unpack import unpack_mxfp4_sidecar
        self._validate()
        q=unpack_mxfp4_sidecar(self.correction_mask,self.routed_scales,
                              self.shared_codes,self.shared_scales)
        return MoeInput(q,self.routed_scales,self.topk_weights,self.topk_ids,
                        self.shared_codes,self.shared_scales)


def prepare_mxfp4_moe(x, w):
    """One local bf16 [M, 6144] shard prepared before the caller's all-gather.

    The router MFMAs read bf16, the routed experts MXFP4 and the shared expert MXFP8, with the unsharded entry's
    quantizers at every M; both scales are row-major, even if M is not a multiple of 32. Returns only tensor payloads;
    the caller owns the gather and the fused tail after it.
    """
    from .chain import H, ROUTER_QUANT_FROM, _check_weights, _route_parts, _route_quant_small, variant_for
    if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != H
            or x.dtype != torch.bfloat16 or not x.is_contiguous()):
        raise ValueError("expected contiguous BF16 [M,6144]")
    n = x.shape[0]
    variant_for(n)
    _check_weights(w)
    if n < ROUTER_QUANT_FROM:
        (q, s), (q8, s8), wt, ids = _route_quant_small(x, w)
    else:
        from .router_mxfp4_mxfp8_quant_row_scales import router_mxfp4_mxfp8_quant_row_scales
        (q, s), (q8, s8), parts = router_mxfp4_mxfp8_quant_row_scales(x, w.router_weight_lines)
        wt, ids = _route_parts(parts, w)
    return MoeInput(q, s.view(torch.uint8), wt, ids, q8, s8)


def _prepare_mode(x, w):
    """What a preparation fixes about its kernels besides the token count (``amd_kernels.warm_up``)."""
    return w.shared_weights is not None, w.router_weight_lines is not None


def _prepare_sizes(mode):
    from .chain import WARM_UP_TOKENS
    return WARM_UP_TOKENS


def _prepare_inputs(mode, n, x, w):
    """One warm-up call at ``n`` tokens (zeros: the dispatch depends on shapes only)."""
    from .chain import H
    return (torch.zeros((n, H), device=w.router_weight.device, dtype=torch.bfloat16), w), {}


prepare_mxfp4_moe = entry("moe/mxfp4_h6144_i768_e128_top4:prepare", mode=_prepare_mode, sizes=_prepare_sizes,
                          inputs=_prepare_inputs)(prepare_mxfp4_moe)


def _mxfp4_moe_prepared(x, w, *, return_partials):
    from .chain import mxfp4_moe_experts, variant_for
    x.validate(w)
    q4 = (x.routed_codes, x.routed_scales)
    from .shared_expert import shared_expert, shared_expert_parallel, shared_parallel_for
    n = x.routed_codes.shape[0]
    sw = w.shared_weights
    q8 = (x.shared_codes, x.shared_scales)
    # to 512 rows the shared tiles lead the routed small-tile launches (the row-major scales of the folded tier)
    folded = variant_for(n) == "a4w4_small"
    parallel = shared_parallel_for(n) and not folded
    current = torch.cuda.current_stream(x.routed_codes.device)
    if folded:
        from .experts_a4w4_small import a4w4_experts
        parts, shared = a4w4_experts(q4, w.w13, w.w2, w.a4w4_scales, x.topk_weights, x.topk_ids, shared=(*q8, sw))
    else:
        if parallel:
            shared = shared_expert_parallel(x.shared_codes, sw, current, x_quant=q8)
        else:
            shared = shared_expert(x.shared_codes, sw, x_quant=q8)
        parts = mxfp4_moe_experts(q4, x.topk_weights, x.topk_ids, w)
        if parallel:
            current.wait_stream(sw.stream)
            shared.record_stream(current)
    if return_partials:
        return parts, shared
    from .common.reduce_slots_shared import reduce_slots_shared
    return reduce_slots_shared(parts, shared)


def gather_mxfp4_moe(local, comm):
    """A prepared shard gathered losslessly (compact) for the same public MoE entry.

    Args:
        local: ``prepare_mxfp4_moe``'s explicit result with shared operands; every rank supplies the same positive row
            count, zero padding included.
        comm: a warmed ``AllGatherComm`` on the four-rank TP group (it owns the IPC staging; this helper owns its
            result buffers).
    """
    if not isinstance(local, MoeInput):
        raise ValueError("expected an explicit prepared MoeInput")
    if local.shared_codes is None or local.shared_scales is None:
        raise ValueError("lossless mixed-format gather requires shared MXFP8 operands")
    compact = local.compact()
    rows = local.routed_codes.shape[0]
    if rows > 8192:
        raise ValueError("prepared TP4 gather supports at most 8192 rows per rank")
    outs = [torch.empty((4*rows,*t.shape[1:]),dtype=t.dtype,device=t.device)
            for t in compact.tensors()]
    comm.all_gather(outs,compact.tensors())
    return MoeInputCompact(*outs)
