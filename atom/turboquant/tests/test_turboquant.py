"""Tests for TurboQuant KV cache compression."""

import pytest
import torch
import math

from turboquant import TurboQuantEngine, LloydMaxCodebook, solve_lloyd_max
from turboquant.bitpack import (
    pack_2bit, unpack_2bit, pack_3bit, unpack_3bit,
    pack_1bit, unpack_1bit, packed_memory_footprint,
)


class TestCodebook:
    def test_solve_lloyd_max_2bit(self):
        centroids, boundaries = solve_lloyd_max(128, 2)
        assert centroids.shape == (4,)
        assert boundaries.shape == (3,)
        # Centroids should be sorted
        assert torch.all(centroids[:-1] <= centroids[1:])
        # Boundaries between centroids
        for i in range(3):
            assert centroids[i] < boundaries[i] < centroids[i + 1]

    def test_solve_lloyd_max_3bit(self):
        centroids, boundaries = solve_lloyd_max(128, 3)
        assert centroids.shape == (8,)
        assert boundaries.shape == (7,)
        assert torch.all(centroids[:-1] <= centroids[1:])

    def test_codebook_quantize_dequantize(self):
        cb = LloydMaxCodebook(128, 2)
        sigma = 1.0 / math.sqrt(128)
        x = torch.randn(100, 128) * sigma
        indices = cb.quantize(x)
        assert indices.dtype == torch.uint8
        assert indices.max() <= 3
        recon = cb.dequantize(indices)
        assert recon.shape == x.shape
        # Reconstruction should be close to original
        mse = ((x - recon) ** 2).mean()
        assert mse < sigma ** 2  # Better than zero-quantization


class TestEngine:
    @pytest.fixture
    def engine(self):
        return TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")

    @pytest.fixture
    def random_kv(self):
        torch.manual_seed(42)
        seq_len = 64
        K = torch.randn(seq_len, 128)
        V = torch.randn(seq_len, 128)
        Q = torch.randn(8, 128)
        return Q, K, V

    def test_compress_keys_shapes(self, engine, random_kv):
        Q, K, V = random_kv
        ck = engine.compress_keys(K)
        assert ck.indices.shape == (64, 128)
        assert ck.indices.dtype == torch.uint8
        assert ck.k_mse.shape == (64, 128)
        assert ck.k_mse.dtype == torch.float16
        assert ck.qjl_signs.shape == (64, 128)
        assert ck.qjl_signs.dtype == torch.int8
        assert ck.vec_norms.shape == (64,)
        assert ck.residual_norms.shape == (64,)

    def test_compress_values_shapes(self, engine, random_kv):
        Q, K, V = random_kv
        cv = engine.compress_values(V)
        assert cv.indices.shape == (64, 128)
        assert cv.indices.dtype == torch.uint8
        assert cv.vec_norms.shape == (64,)

    def test_key_reconstruction_quality(self, engine, random_kv):
        Q, K, V = random_kv
        ck = engine.compress_keys(K)
        K_f = K.float()
        k_mse_f = ck.k_mse.float()
        cos_sim = torch.nn.functional.cosine_similarity(K_f, k_mse_f, dim=-1)
        mean_cos = cos_sim.mean().item()
        # Random data has high effective dim → lower quality than real KV cache
        assert mean_cos > 0.70, f"Key cosine similarity too low: {mean_cos}"

    def test_value_roundtrip_quality(self, engine, random_kv):
        Q, K, V = random_kv
        cv = engine.compress_values(V)
        V_recon = engine.decompress_values(cv)
        cos_sim = torch.nn.functional.cosine_similarity(
            V.float(), V_recon.float(), dim=-1
        )
        mean_cos = cos_sim.mean().item()
        assert mean_cos > 0.80, f"Value cosine similarity too low: {mean_cos}"

    def test_attention_scores_shape(self, engine, random_kv):
        Q, K, V = random_kv
        ck = engine.compress_keys(K)
        scores = engine.attention_scores(Q, ck)
        assert scores.shape == (8, 64)

    def test_attention_scores_vs_exact(self, engine, random_kv):
        Q, K, V = random_kv
        ck = engine.compress_keys(K)
        approx_scores = engine.attention_scores(Q, ck)
        exact_scores = (Q.float() @ K.float().T) * engine.scale
        # Correlation should be high
        cos_sim = torch.nn.functional.cosine_similarity(
            approx_scores.flatten().unsqueeze(0),
            exact_scores.flatten().unsqueeze(0),
        )
        assert cos_sim.item() > 0.70, f"Score correlation too low: {cos_sim.item()}"

    def test_fused_attention_shape(self, engine, random_kv):
        Q, K, V = random_kv
        ck = engine.compress_keys(K)
        cv = engine.compress_values(V)
        out = engine.fused_attention(Q, ck, cv)
        assert out.shape == (8, 128)
        assert out.dtype == torch.float16

    def test_fused_attention_vs_exact(self, engine, random_kv):
        Q, K, V = random_kv
        ck = engine.compress_keys(K)
        cv = engine.compress_values(V)
        approx_out = engine.fused_attention(Q, ck, cv)
        # Exact attention
        exact_scores = (Q.float() @ K.float().T) * engine.scale
        exact_weights = torch.softmax(exact_scores, dim=-1)
        exact_out = exact_weights @ V.float()
        cos_sim = torch.nn.functional.cosine_similarity(
            approx_out.float(), exact_out.float(), dim=-1
        )
        mean_cos = cos_sim.mean().item()
        assert mean_cos > 0.60, f"Attention output cosine similarity too low: {mean_cos}"

    def test_memory_footprint(self, engine):
        stats = engine.memory_footprint(1024)
        assert stats["packed_ratio"] > 4.5  # ~5.02× with bit-packing
        assert stats["packed_total_bytes"] < stats["uncompressed_bytes"]

    def test_2bit_mode(self, random_kv):
        engine = TurboQuantEngine(head_dim=128, total_bits=2)
        Q, K, V = random_kv
        ck = engine.compress_keys(K)
        cv = engine.compress_values(V)
        out = engine.fused_attention(Q, ck, cv)
        assert out.shape == (8, 128)
        # 2-bit is lower quality but should still work
        assert ck.indices.max() <= 1  # 1-bit keys in 2-bit total mode

    def test_qjl_signs_binary(self, engine, random_kv):
        Q, K, V = random_kv
        ck = engine.compress_keys(K)
        unique_signs = torch.unique(ck.qjl_signs)
        # Signs should be -1 or 1 only
        assert all(s in [-1, 1] for s in unique_signs.tolist())


class TestBitPacking:
    def test_pack_unpack_2bit(self):
        indices = torch.randint(0, 4, (32, 128), dtype=torch.uint8)
        packed = pack_2bit(indices)
        assert packed.shape == (32, 32)  # 128 / 4 = 32
        unpacked = unpack_2bit(packed, 128)
        assert torch.equal(indices, unpacked)

    def test_pack_unpack_3bit(self):
        indices = torch.randint(0, 8, (32, 128), dtype=torch.uint8)
        packed = pack_3bit(indices)
        assert packed.shape == (32, 48)  # 128 * 3 / 8 = 48
        unpacked = unpack_3bit(packed, 128)
        assert torch.equal(indices, unpacked)

    def test_pack_unpack_1bit(self):
        signs = torch.where(torch.randn(32, 128) > 0,
                            torch.ones(1, dtype=torch.int8),
                            -torch.ones(1, dtype=torch.int8))
        packed = pack_1bit(signs)
        assert packed.shape == (32, 16)  # 128 / 8 = 16
        unpacked = unpack_1bit(packed, 128)
        assert torch.equal(signs, unpacked)

    def test_packed_memory_5x(self):
        stats = packed_memory_footprint(1024, 128)
        assert stats["compression_ratio"] > 4.5  # Should be ~5.02×
        assert stats["total_bytes_per_token"] < 110  # < 110 bytes/token

    def test_engine_packed_roundtrip(self):
        engine = TurboQuantEngine(head_dim=128, total_bits=3)
        K = torch.randn(64, 128)
        V = torch.randn(64, 128)
        Q = torch.randn(4, 128)

        # Compress + pack
        pk = engine.compress_keys_packed(K)
        pv = engine.compress_values_packed(V)

        # Check packed sizes
        assert pk.packed_indices.shape == (64, 32)   # 2-bit packed
        assert pk.packed_signs.shape == (64, 16)     # 1-bit packed
        assert pv.packed_indices.shape == (64, 48)   # 3-bit packed

        # Attention from packed
        out = engine.fused_attention_packed(Q, pk, pv)
        assert out.shape == (4, 128)

        # Compare with unpacked attention
        ck = engine.compress_keys(K)
        cv = engine.compress_values(V)
        out_unpacked = engine.fused_attention(Q, ck, cv)

        # Results should be identical (packing is lossless)
        assert torch.allclose(out.float(), out_unpacked.float(), atol=1e-3)


class TestGPU:
    """GPU tests — skip if no GPU available."""

    @pytest.fixture
    def gpu_engine(self):
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
        return TurboQuantEngine(head_dim=128, total_bits=3, device="cuda")

    def test_gpu_compress_decompress(self, gpu_engine):
        K = torch.randn(256, 128, device="cuda")
        V = torch.randn(256, 128, device="cuda")
        Q = torch.randn(16, 128, device="cuda")

        ck = gpu_engine.compress_keys(K)
        cv = gpu_engine.compress_values(V)
        out = gpu_engine.fused_attention(Q, ck, cv)

        assert out.device.type == "cuda"
        assert out.shape == (16, 128)

    def test_gpu_large_sequence(self, gpu_engine):
        K = torch.randn(4096, 128, device="cuda")
        V = torch.randn(4096, 128, device="cuda")
        Q = torch.randn(1, 128, device="cuda")

        ck = gpu_engine.compress_keys(K)
        cv = gpu_engine.compress_values(V)
        out = gpu_engine.fused_attention(Q, ck, cv)
        assert out.shape == (1, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
