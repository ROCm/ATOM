"""
TurboQuantEngine — KV cache compression with asymmetric attention.

Implements the full pipeline:
  1. Compress keys: rotate → 2-bit Lloyd-Max + 1-bit QJL signs
  2. Compress values: rotate → 3-bit Lloyd-Max
  3. Attention: score(q,k) ≈ <q, k_mse> + QJL_correction
  4. Decompress values (standalone or fused in attention)
"""

import math
from dataclasses import dataclass


import torch
import torch.nn.functional as F

from .codebook import LloydMaxCodebook
from .bitpack import pack_2bit, unpack_2bit, pack_3bit, unpack_3bit, pack_1bit, unpack_1bit
from .constants import HEAD_DIM, DEFAULT_TOTAL_BITS, DEFAULT_SEED


@dataclass
class CompressedKeys:
    """Compressed key cache for one attention head."""
    indices: torch.Tensor      # (seq_k, head_dim) uint8
    k_mse: torch.Tensor        # (seq_k, head_dim) fp16 — reconstructed keys
    qjl_signs: torch.Tensor    # (seq_k, head_dim) int8 — QJL correction signs
    vec_norms: torch.Tensor    # (seq_k,) fp16 — original L2 norms
    residual_norms: torch.Tensor  # (seq_k,) fp16 — residual L2 norms


@dataclass
class CompressedValues:
    """Compressed value cache for one attention head."""
    indices: torch.Tensor    # (seq_v, head_dim) uint8
    vec_norms: torch.Tensor  # (seq_v,) fp16 — original L2 norms


@dataclass
class PackedKeys:
    """Bit-packed compressed key cache — minimal memory footprint."""
    packed_indices: torch.Tensor   # (seq_k, head_dim//4) uint8 — 2-bit packed
    packed_signs: torch.Tensor     # (seq_k, head_dim//8) uint8 — 1-bit packed
    vec_norms: torch.Tensor        # (seq_k,) fp16
    residual_norms: torch.Tensor   # (seq_k,) fp16
    # Total: 32 + 16 + 2 + 2 = 52 bytes/vector (vs 256 bytes unpacked)


@dataclass
class PackedValues:
    """Bit-packed compressed value cache — minimal memory footprint."""
    packed_indices: torch.Tensor  # (seq_v, head_dim*3//8) uint8 — 3-bit packed
    vec_norms: torch.Tensor       # (seq_v,) fp16
    # Total: 48 + 2 = 50 bytes/vector (vs 130 bytes unpacked)


def _generate_rotation_matrix(d: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Haar-distributed random orthogonal matrix via QR decomposition."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    return (Q * diag_sign.unsqueeze(0)).to(device)


def _generate_qjl_matrix(d: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Random Gaussian projection matrix for QJL bias correction."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 10000)
    return torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32).to(device)


class TurboQuantEngine:
    """
    KV cache compressor using Lloyd-Max quantization + QJL correction.

    Usage:
        engine = TurboQuantEngine(head_dim=128, device="cuda")
        ck = engine.compress_keys(K)      # K: (seq_k, 128) fp16
        cv = engine.compress_values(V)    # V: (seq_v, 128) fp16
        out = engine.fused_attention(Q, ck, cv)  # Q: (seq_q, 128) fp16
    """

    def __init__(
        self,
        head_dim: int = HEAD_DIM,
        total_bits: int = DEFAULT_TOTAL_BITS,
        seed: int = DEFAULT_SEED,
        device: str = "cpu",
    ):
        self.head_dim = head_dim
        self.total_bits = total_bits
        self.mse_bits_keys = max(total_bits - 1, 1)  # 2 bits for keys
        self.mse_bits_values = total_bits              # 3 bits for values
        self.device = device

        # Pre-compute rotation and projection matrices
        self.Pi = _generate_rotation_matrix(head_dim, seed, device)
        self.PiT = self.Pi.T.contiguous()
        self.S = _generate_qjl_matrix(head_dim, seed, device)
        self.ST = self.S.T.contiguous()

        # Build codebooks
        self.key_codebook = LloydMaxCodebook(head_dim, self.mse_bits_keys)
        self.val_codebook = LloydMaxCodebook(head_dim, self.mse_bits_values)
        self.key_codebook.to(device)
        self.val_codebook.to(device)

        # Scaling constants
        self.scale = 1.0 / math.sqrt(head_dim)
        self.correction_scale = math.sqrt(math.pi / 2) / head_dim

    def to(self, device: str) -> "TurboQuantEngine":
        """Move engine to device."""
        self.device = device
        self.Pi = self.Pi.to(device)
        self.PiT = self.PiT.to(device)
        self.S = self.S.to(device)
        self.ST = self.ST.to(device)
        self.key_codebook.to(device)
        self.val_codebook.to(device)
        return self

    @torch.no_grad()
    def compress_keys(self, K: torch.Tensor) -> CompressedKeys:
        """
        Compress key cache: 2-bit Lloyd-Max + 1-bit QJL signs.

        Args:
            K: (seq_k, head_dim) tensor in any precision

        Returns:
            CompressedKeys with indices, k_mse, qjl_signs, norms
        """
        K_f = K.float()

        # Step 1: Normalize to unit vectors
        vec_norms = torch.norm(K_f, dim=-1, keepdim=True)
        K_normed = K_f / (vec_norms + 1e-8)

        # Step 2: Rotate into coded space
        rotated = K_normed @ self.PiT.float()

        # Step 3: Lloyd-Max quantize
        indices = self.key_codebook.quantize(rotated)

        # Step 4: Reconstruct (dequantize + un-rotate + rescale)
        y_hat = self.key_codebook.dequantize(indices)
        k_mse = (y_hat @ self.Pi.float()) * vec_norms

        # Step 5: QJL correction
        residual = K_f - k_mse
        residual_norms = torch.norm(residual, dim=-1)
        projected = residual @ self.ST.float()
        signs = torch.sign(projected).to(torch.int8)
        signs[signs == 0] = 1

        return CompressedKeys(
            indices=indices,
            k_mse=k_mse.half(),
            qjl_signs=signs,
            vec_norms=vec_norms.squeeze(-1).half(),
            residual_norms=residual_norms.half(),
        )

    @torch.no_grad()
    def compress_values(self, V: torch.Tensor) -> CompressedValues:
        """
        Compress value cache: 3-bit Lloyd-Max (no QJL needed).

        Args:
            V: (seq_v, head_dim) tensor in any precision

        Returns:
            CompressedValues with indices and norms
        """
        V_f = V.float()

        vec_norms = torch.norm(V_f, dim=-1, keepdim=True)
        V_normed = V_f / (vec_norms + 1e-8)
        rotated = V_normed @ self.PiT.float()
        indices = self.val_codebook.quantize(rotated)

        return CompressedValues(
            indices=indices,
            vec_norms=vec_norms.squeeze(-1).half(),
        )

    @torch.no_grad()
    def decompress_values(self, cv: CompressedValues) -> torch.Tensor:
        """
        Decompress value cache back to fp16.

        Args:
            cv: CompressedValues from compress_values()

        Returns:
            (seq_v, head_dim) fp16 tensor
        """
        y_hat = self.val_codebook.dequantize(cv.indices)
        reconstructed = (y_hat @ self.Pi.float()) * cv.vec_norms.float().unsqueeze(-1)
        return reconstructed.half()

    @torch.no_grad()
    def attention_scores(
        self, Q: torch.Tensor, ck: CompressedKeys
    ) -> torch.Tensor:
        """
        Compute asymmetric attention scores with QJL correction.

        score(q, k) ≈ <q, k_mse> + ||r|| * sqrt(pi/2)/d * <S·q, signs>

        Args:
            Q: (seq_q, head_dim) query tensor
            ck: CompressedKeys from compress_keys()

        Returns:
            (seq_q, seq_k) raw attention scores (before softmax)
        """
        Q_f = Q.float()
        k_mse = ck.k_mse.float()
        signs = ck.qjl_signs.float()
        r_norms = ck.residual_norms.float()

        # Term 1: MSE reconstruction score
        term1 = Q_f @ k_mse.T

        # Term 2: QJL bias correction
        q_proj = Q_f @ self.S.T.float()
        qjl_ip = q_proj @ signs.T
        term2 = self.correction_scale * qjl_ip * r_norms.unsqueeze(0)

        return (term1 + term2) * self.scale

    @torch.no_grad()
    def fused_attention(
        self,
        Q: torch.Tensor,
        ck: CompressedKeys,
        cv: CompressedValues,
    ) -> torch.Tensor:
        """
        Full attention: scores → softmax → weighted sum of decompressed V.

        Args:
            Q: (seq_q, head_dim) query tensor
            ck: CompressedKeys
            cv: CompressedValues

        Returns:
            (seq_q, head_dim) attention output in fp16
        """
        # Compute scores
        scores = self.attention_scores(Q, ck)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Decompress values and compute weighted sum
        V_recon = self.decompress_values(cv)
        output = attn_weights.float() @ V_recon.float()

        return output.half()

    @torch.no_grad()
    def pack_keys(self, ck: CompressedKeys) -> PackedKeys:
        """Bit-pack compressed keys for minimal memory storage."""
        return PackedKeys(
            packed_indices=pack_2bit(ck.indices),
            packed_signs=pack_1bit(ck.qjl_signs),
            vec_norms=ck.vec_norms,
            residual_norms=ck.residual_norms,
        )

    @torch.no_grad()
    def unpack_keys(self, pk: PackedKeys) -> CompressedKeys:
        """Unpack bit-packed keys for attention computation."""
        indices = unpack_2bit(pk.packed_indices, self.head_dim)
        signs = unpack_1bit(pk.packed_signs, self.head_dim)

        # Reconstruct k_mse from indices
        y_hat = self.key_codebook.dequantize(indices)
        k_mse = (y_hat @ self.Pi.float()) * pk.vec_norms.float().unsqueeze(-1)

        return CompressedKeys(
            indices=indices,
            k_mse=k_mse.half(),
            qjl_signs=signs,
            vec_norms=pk.vec_norms,
            residual_norms=pk.residual_norms,
        )

    @torch.no_grad()
    def pack_values(self, cv: CompressedValues) -> PackedValues:
        """Bit-pack compressed values for minimal memory storage."""
        return PackedValues(
            packed_indices=pack_3bit(cv.indices),
            vec_norms=cv.vec_norms,
        )

    @torch.no_grad()
    def unpack_values(self, pv: PackedValues) -> CompressedValues:
        """Unpack bit-packed values for decompression."""
        return CompressedValues(
            indices=unpack_3bit(pv.packed_indices, self.head_dim),
            vec_norms=pv.vec_norms,
        )

    @torch.no_grad()
    def compress_keys_packed(self, K: torch.Tensor) -> PackedKeys:
        """Compress + bit-pack keys in one call."""
        return self.pack_keys(self.compress_keys(K))

    @torch.no_grad()
    def compress_values_packed(self, V: torch.Tensor) -> PackedValues:
        """Compress + bit-pack values in one call."""
        return self.pack_values(self.compress_values(V))

    @torch.no_grad()
    def fused_attention_packed(
        self, Q: torch.Tensor, pk: PackedKeys, pv: PackedValues
    ) -> torch.Tensor:
        """Full attention from bit-packed KV cache."""
        ck = self.unpack_keys(pk)
        cv = self.unpack_values(pv)
        return self.fused_attention(Q, ck, cv)

    def memory_footprint(self, seq_len: int) -> dict:
        """Calculate packed vs unpacked vs uncompressed memory usage."""
        from .bitpack import packed_memory_footprint
        packed = packed_memory_footprint(seq_len, self.head_dim)

        d = self.head_dim
        # Unpacked: indices (uint8) + signs (int8) + norms
        unpacked_key = seq_len * d + seq_len * d + seq_len * 2 + seq_len * 2
        unpacked_val = seq_len * d + seq_len * 2
        uncomp = seq_len * d * 2 * 2

        return {
            "packed_key_bytes": packed["key_bytes"],
            "packed_val_bytes": packed["val_bytes"],
            "packed_total_bytes": packed["total_bytes"],
            "packed_ratio": packed["compression_ratio"],
            "packed_bytes_per_token": packed["total_bytes_per_token"],
            "unpacked_total_bytes": unpacked_key + unpacked_val,
            "unpacked_ratio": uncomp / (unpacked_key + unpacked_val),
            "uncompressed_bytes": uncomp,
        }
