"""DCP project-before-merge (PBM) for the MLA decode combine, vLLM 3966924d, --dcp-comm-backend a2a.

Stock (per MLA layer, H = DCP-gathered heads, h = H / n local heads, L = kv_lora_rank, V = v_head_dim):
  forward_mqa -> partial latent [B, H, L] + lse [B, H] -> pack [n, B, h, L+2] -> a2a -> LSE combine [B, h, L]
  -> _v_up_proj (W_UV of the h local heads) -> [B, h*V].
PBM (ATOM's DCP combine): W_UV of all H heads first -> [B, H, V] -> pack [n, B, h, V+2] -> a2a -> LSE combine
written straight into _v_up_proj's output. The merge is a per-(token, head) weighted sum and W_UV is per-head
linear, so they commute. The a2a carries 130 instead of 514 values per head.

The patched combine returns the 4-D a2a receive buffer instead of [B, h, L]; the patched _v_up_proj recognises it
by ndim and runs the unpack-combine into `out`, so the merged result is written once, with no extra copy. Nothing
reads the combine output between the two calls (mla_attention.MLAAttention.forward_impl and
kimi_k3.nvidia.mla.MultiHeadLatentAttention._attention, PCP off); pd_c48_worker pins both files.
"""
import functools
import os
import sys

import torch
import torch.distributed as dist

from vllm.v1.attention.ops import dcp as dcp_ops

_logged = set()
_patched = []


def install(layer) -> bool:
    """Gather W_UV over DCP and route this layer's combine through PBM. Call after process_weights_after_loading."""
    mgr = getattr(layer, "dcp_manager", None)
    if (
        mgr is None
        or not mgr.use_a2a
        or getattr(layer, "use_pcp", False)  # finalize_mla_pcp_decode runs between combine and _v_up_proj
        or getattr(layer, "W_UV", None) is None
        or getattr(layer, "is_aiter_triton_fp8_bmm_enabled", False)
        or getattr(layer, "is_aiter_triton_fp4_bmm_enabled", False)
    ):
        return False
    # Gathered head order is DCP-rank-major, the same order as the Q all-gather.
    layer.W_UV_dcp = mgr.group.all_gather(layer.W_UV.contiguous(), dim=0)  # [H, L, V]
    mgr.combine = functools.partial(pbm_send, layer)
    if type(layer) not in _logged:
        _logged.add(type(layer))
        print(f"[pbm-patch] pid {os.getpid()}: {type(layer).__name__} {getattr(layer, 'layer_name', '?')} "
              f"W_UV_dcp {tuple(layer.W_UV_dcp.shape)} {layer.W_UV_dcp.dtype}", file=sys.stderr, flush=True)
    return True


def pbm_send(layer, partial_out, partial_lse, *, seq_lens=None, query_start_loc=None):
    """W_UV on every gathered head, pack with LSE (+ empty-shard mask), one a2a. Returns the receive buffer."""
    group = layer.dcp_manager.group
    n = group.world_size
    B, H, _ = partial_out.shape
    w = layer.W_UV_dcp
    assert H == w.shape[0] and H % n == 0, (H, tuple(w.shape), n)
    V = w.shape[-1]
    v = partial_out.new_empty((B, H, V))
    torch.bmm(partial_out.transpose(0, 1), w, out=v.transpose(0, 1))
    pack = dcp_ops._dcp_a2a_lse_pack_dim(v.dtype)
    send, recv = dcp_ops._dcp_a2a_send_recv_buffers((n, B, H // n, V + pack), v.device, v.dtype)
    dcp_ops._dcp_a2a_pack_send(
        v, partial_lse, send, n, H // n, V, pack, seq_lens=seq_lens, query_start_loc=query_start_loc
    )
    dist.all_to_all_single(recv.view(-1), send.view(-1), group=group.device_group)
    return recv


def unpack_into(recv, out, is_lse_base_on_e):
    """LSE combine of the PBM receive buffer [n, B, h, V+pack] into out [B, h*V]."""
    n, B, h, d = recv.shape
    V = d - dcp_ops._dcp_a2a_lse_pack_dim(recv.dtype)
    o = out.view(-1, h, V)
    assert o.shape[0] == B, (tuple(o.shape), tuple(recv.shape))
    dcp_ops._dcp_a2a_unpack_combine_kernel[(B, h, 1)](
        recv, o, o,
        *recv.stride(), *o.stride(), 0, 0,
        N=n, HEAD_DIM=V, IS_BASE_E=is_lse_base_on_e, RETURN_LSE=False,
        LSE_PACK_DIM=d - V,
    )


def patch_class(cls) -> None:
    """Wrap cls.process_weights_after_loading (install PBM) and cls._v_up_proj (consume the PBM buffer)."""
    if cls in _patched:
        return
    orig_pwal = cls.process_weights_after_loading
    orig_vup = cls._v_up_proj

    @functools.wraps(orig_pwal)
    def process_weights_after_loading(self, *args, **kwargs):
        ret = orig_pwal(self, *args, **kwargs)
        install(self)
        return ret

    @functools.wraps(orig_vup)
    def _v_up_proj(self, x, out):
        if x.dim() == 4:
            return unpack_into(x, out, self.impl.lse_base_on_e)
        return orig_vup(self, x, out)

    cls.process_weights_after_loading = process_weights_after_loading
    cls._v_up_proj = _v_up_proj
    _patched.append(cls)


def installed(model) -> dict:
    """Count PBM layers; raise if a DCP a2a MLA layer of a patched class kept the stock combine."""
    layers = [m for m in (model.modules() if model is not None else ())
              if isinstance(m, tuple(_patched)) and getattr(getattr(m, "dcp_manager", None), "use_a2a", False)]
    stock = [getattr(m, "layer_name", type(m).__name__) for m in layers if getattr(m, "W_UV_dcp", None) is None]
    if stock:
        raise RuntimeError(f"DCP PBM did not install on {len(stock)} layers: {stock[:4]}")
    return {"layers": len(layers), "heads": sorted({int(m.W_UV_dcp.shape[0]) for m in layers}),
            "dtype": sorted({str(m.W_UV_dcp.dtype) for m in layers})}
