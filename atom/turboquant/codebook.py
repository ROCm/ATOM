"""
Lloyd-Max optimal scalar quantizer for Gaussian-distributed coordinates.

After random rotation by orthogonal matrix Pi, each coordinate of a
unit-normalized vector is approximately N(0, 1/sqrt(d)).
Lloyd-Max finds MSE-optimal centroids and decision boundaries for this
distribution at a given bit width.
"""

import math
import torch
from scipy import integrate


def _gaussian_pdf(x: float, sigma: float) -> float:
    """Unnormalized Gaussian PDF (normalization cancels in Lloyd-Max)."""
    return math.exp(-0.5 * (x / sigma) ** 2)


def solve_lloyd_max(
    d: int,
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute MSE-optimal centroids and boundaries for N(0, 1/sqrt(d)).

    Args:
        d: Dimension (128 for standard attention heads)
        bits: Quantization bit width (2 for keys, 3 for values)
        max_iter: Maximum Lloyd-Max iterations
        tol: Convergence tolerance

    Returns:
        centroids: (2^bits,) sorted float32 tensor
        boundaries: (2^bits - 1,) sorted float32 tensor
    """
    n_levels = 1 << bits
    sigma = 1.0 / math.sqrt(d)
    def pdf(x):
        return _gaussian_pdf(x, sigma)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [lo * 3] + boundaries + [hi * 3]

        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])

        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            break
        centroids = new_centroids

    boundaries = [
        (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
    ]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


class LloydMaxCodebook:
    """Pre-computed Lloyd-Max codebook for a given dimension and bit width."""

    def __init__(self, d: int, bits: int):
        if bits not in {1, 2, 3, 4}:
            raise ValueError(f"Unsupported bit width: {bits}")
        self.d = d
        self.bits = bits
        self.n_levels = 1 << bits
        self.centroids, self.boundaries = solve_lloyd_max(d, bits)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize to nearest centroid index. Returns uint8."""
        diffs = x.unsqueeze(-1) - self.centroids.to(x.device)
        return diffs.abs().argmin(dim=-1).to(torch.uint8)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Lookup centroid values from indices."""
        return self.centroids.to(indices.device)[indices.long()]

    def to(self, device):
        """Move codebook to device."""
        self.centroids = self.centroids.to(device)
        self.boundaries = self.boundaries.to(device)
        return self
