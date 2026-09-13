"""
TurboQuant compiled kernels — torch.compile optimized versions.

These use torch.compile for automatic kernel fusion on ROCm.
For maximum performance, use Baybridge/FlyDSL kernels (Phase 2b).
"""

import torch


@torch.compile(mode="reduce-overhead", fullgraph=True)
def compress_keys_compiled(
    K: torch.Tensor,
    PiT: torch.Tensor,
    Pi: torch.Tensor,
    ST: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused key compression: normalize → rotate → quantize → QJL."""
    K_f = K.float()

    # Normalize
    vec_norms = torch.norm(K_f, dim=-1, keepdim=True)
    K_normed = K_f / (vec_norms + 1e-8)

    # Rotate
    rotated = K_normed @ PiT.float()

    # Quantize (nearest centroid)
    diffs = rotated.unsqueeze(-1) - centroids.float()
    indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

    # Dequantize + un-rotate + rescale
    y_hat = centroids[indices.long()]
    k_mse = (y_hat @ Pi.float()) * vec_norms

    # QJL
    residual = K_f - k_mse
    residual_norms = torch.norm(residual, dim=-1)
    projected = residual @ ST.float()
    signs = torch.sign(projected).to(torch.int8)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)

    return indices, k_mse.half(), signs, vec_norms.squeeze(-1).half(), residual_norms.half()


@torch.compile(mode="reduce-overhead", fullgraph=True)
def compress_values_compiled(
    V: torch.Tensor,
    PiT: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused value compression: normalize → rotate → quantize."""
    V_f = V.float()
    vec_norms = torch.norm(V_f, dim=-1, keepdim=True)
    V_normed = V_f / (vec_norms + 1e-8)
    rotated = V_normed @ PiT.float()
    diffs = rotated.unsqueeze(-1) - centroids.float()
    indices = diffs.abs().argmin(dim=-1).to(torch.uint8)
    return indices, vec_norms.squeeze(-1).half()


@torch.compile(mode="reduce-overhead", fullgraph=True)
def decompress_values_compiled(
    indices: torch.Tensor,
    vec_norms: torch.Tensor,
    centroids: torch.Tensor,
    Pi: torch.Tensor,
) -> torch.Tensor:
    """Fused value decompression: lookup → un-rotate → rescale."""
    y_hat = centroids[indices.long()]
    reconstructed = (y_hat @ Pi.float()) * vec_norms.float().unsqueeze(-1)
    return reconstructed.half()


@torch.compile(mode="reduce-overhead", fullgraph=True)
def attention_scores_compiled(
    Q: torch.Tensor,
    k_mse: torch.Tensor,
    qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
    S: torch.Tensor,
    scale: float,
    correction_scale: float,
) -> torch.Tensor:
    """Fused asymmetric attention scoring with QJL correction."""
    Q_f = Q.float()

    # Term 1: MSE score
    term1 = Q_f @ k_mse.float().T

    # Term 2: QJL correction
    q_proj = Q_f @ S.T.float()
    qjl_ip = q_proj @ qjl_signs.float().T
    term2 = correction_scale * qjl_ip * residual_norms.float().unsqueeze(0)

    return (term1 + term2) * scale
