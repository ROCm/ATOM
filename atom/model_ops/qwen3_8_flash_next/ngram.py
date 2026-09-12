"""Qwen3.8-Flash-Next PLE n-gram memory: hashed lookup into a 320M-row embedding table.

Port of `qwen3_8_flash_next/nvidia/ple_layer.py:Qwen3_8FlashNextNGramEmbedding`.

Every token is hashed into `(ngram_size - 1) * heads_per_ngram` rows -- one per
(n-gram order, head) pair -- and the retrieved vectors are concatenated back to
`ple_embed_dim`. For Qwen3.8-Flash-Next that is 16 heads x 160 dims = 2560, over a table
of 320,001,536 rows: 102.4 GB in BF16, 28% of the checkpoint.

Each head owns a disjoint slice of the table whose length is the n-th prime
after `ngram_vocab_size_base - 1`, so a hashed id modulo that prime stays
inside the head's slice. The primes and the per-n-gram multipliers are DERIVED
from the config (seed + layer id) rather than read from the checkpoint -- so a
derivation that drifts from the one the weights were trained with would send
every lookup to the wrong row while raising nothing. The checkpoint ships its
own copies, and `_verify_derived` checks ours against them at load: they are
registered as parameters (not buffers) purely so the loader hands them over.
"""

import torch
from torch import nn

from atom.model_ops.embed_head import VocabParallelEmbedding

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


def splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def is_prime_64(value: int) -> bool:
    """Deterministic Miller-Rabin, exact for every 64-bit input."""
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


class Qwen3_8FlashNextNGramEmbedding(nn.Module):
    """Hashed n-gram lookup producing `[tokens, ple_embed_dim]`."""

    def __init__(
        self,
        config,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        max_num_reqs: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        eos = config.eos_token_id
        self.eos_token_id = int(eos[0] if isinstance(eos, (list, tuple)) else eos)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        # Per-n-gram odd multipliers, derived from (seed, ple layer id).
        max_multiplier = ((1 << 63) - 1) // self.unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = int(getattr(config, "seed", 1234)) + (
            _PLE_LAYER_PRIME * ple_dense_layer_id
        )
        multipliers = [
            2 * (splitmix64(base_seed + _SPLITMIX_GAMMA * (index + 1)) % half_bound) + 1
            for index in range(self.ngram_size)
        ]
        self.layer_multipliers = self._derived_constant(multipliers)

        # Disjoint prime-sized slice per head.
        base = int(config.ngram_vocab_size_base)
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(self.ngram_heads):
            global_head = ple_dense_layer_id * self.ngram_heads + local_head
            size = nth_prime_after(base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        self.ngram_heads_vocab_sizes = self._derived_constant(sizes)
        self.ngram_heads_offsets = self._derived_constant(offsets)

        divisor = int(config.make_ngram_vocab_size_divisible_by)
        self.table_rows = ((offset + divisor - 1) // divisor) * divisor
        self.ngram_embedding = VocabParallelEmbedding(
            self.table_rows, self.head_dim, prefix=f"{prefix}.ngram_embedding"
        )
        # The 128 checkpoint shards are slices of the PADDED table, so the
        # shard width divides it exactly.
        self.checkpoint_shard_rows = (
            self.table_rows + self.split_ngram_parts - 1
        ) // self.split_ngram_parts
        self.ngram_embedding.weight.weight_loader = self._embedding_shard_loader

        self.register_buffer(
            "positions_buffer",
            torch.arange(max_total_tokens, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "padded_buffer",
            torch.full(
                (max_num_reqs, max_total_tokens),
                self.eos_token_id,
                dtype=torch.int64,
            ),
            persistent=False,
        )

    @staticmethod
    def _derived_constant(values: list[int]) -> nn.Parameter:
        """A derived int64 constant the checkpoint is allowed to confirm.

        Registered as a (gradient-free) parameter rather than a buffer so the
        weight loader sees it and `_verify_derived` runs; nothing here is
        learned.
        """
        tensor = torch.tensor(values, dtype=torch.long)
        param = nn.Parameter(tensor, requires_grad=False)
        param.weight_loader = Qwen3_8FlashNextNGramEmbedding._verify_derived
        return param

    @staticmethod
    def _verify_derived(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Fail loudly if our derivation disagrees with the checkpoint."""
        expected = param.data.to(device=loaded_weight.device)
        if expected.shape != loaded_weight.shape or not torch.equal(
            expected, loaded_weight.to(expected.dtype)
        ):
            raise ValueError(
                "Qwen3.8-Flash-Next PLE hash constants do not match the checkpoint: "
                f"derived {expected.tolist()[:4]}..., "
                f"checkpoint {loaded_weight.tolist()[:4]}.... Every n-gram "
                "lookup would read the wrong table row."
            )

    def _embedding_shard_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_index: int = 0
    ) -> None:
        """Copy one checkpoint shard's overlap with this rank's vocab range."""
        embedding = self.ngram_embedding
        checkpoint_start = shard_index * self.checkpoint_shard_rows
        tp_start = embedding.vocab_start_idx
        tp_end = embedding.vocab_end_idx
        overlap_start = max(checkpoint_start, tp_start)
        overlap_end = min(checkpoint_start + loaded_weight.shape[0], tp_end)
        if overlap_start >= overlap_end:
            return
        rows = overlap_end - overlap_start
        source = loaded_weight.narrow(0, overlap_start - checkpoint_start, rows)
        target = param.data.narrow(0, overlap_start - tp_start, rows)
        target.copy_(source.to(device=target.device, dtype=target.dtype))

    @staticmethod
    def _shift_precompute(
        tokens: torch.Tensor, eos_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Positions plus each token's distance from the last EOS before it.

        n-grams must not reach across a document boundary, so the shift below
        is only valid while `position_in_segment >= shift`.
        """
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        max_columns: int | None = None,
    ) -> torch.Tensor:
        """Hash every token into one row id per (n-gram order, head).

        Split out of `forward` so it can be diffed against the reference
        without materializing the 102 GB table.

        `max_columns` is the longest per-request run in this batch. The
        intermediate tensors are `[reqs, columns, heads]`, so leaving it at the
        full workspace width makes a decode step allocate hundreds of MB to
        hold one token per request; pass the real width when the caller knows
        it (the metadata builder does).
        """
        input_ids = input_ids.reshape(-1).long()
        query_start_loc = query_start_loc.long()
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]
        if num_tokens > self.positions_buffer.numel():
            raise ValueError(
                f"PLE received {num_tokens} tokens, but its workspace supports "
                f"at most {self.positions_buffer.numel()}"
            )
        if num_reqs > self.padded_buffer.shape[0]:
            raise ValueError(
                f"PLE received {num_reqs} requests, but its workspace supports "
                f"at most {self.padded_buffer.shape[0]}"
            )

        if max_columns is None:
            max_columns = self.padded_buffer.shape[1]
        max_columns = max(1, min(int(max_columns), self.padded_buffer.shape[1]))

        positions = self.positions_buffer[:num_tokens]
        packed = self.padded_buffer[:num_reqs, :max_columns]
        packed.fill_(self.eos_token_id)
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(
            0, packed.shape[1] - 1
        )
        packed[request_indices, columns] = input_ids
        ngram_context = ngram_context[:num_reqs].to(
            device=input_ids.device, dtype=torch.long
        )

        # Prepend the tail of the previous chunk so n-grams survive chunking.
        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(
            context, self.eos_token_id
        )
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(
                    context, positions_2d, position_in_segment, shift, self.eos_token_id
                )
            )

        adjusted_columns = columns + self.ngram_size - 1
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted[index] * self.layer_multipliers[index]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        return torch.cat(id_blocks, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        max_columns: int | None = None,
    ) -> torch.Tensor:
        ngram_ids = self.compute_ngram_ids(
            input_ids, query_start_loc, ngram_context, max_columns
        )
        # Look the heads up as one flat batch: the TP path returns `[N, dim]`
        # for an N-element index whatever its shape, so a 2D index would come
        # back already flattened on one path and 3D on the other.
        rows = self.ngram_embedding(ngram_ids.reshape(-1))
        return rows.reshape(ngram_ids.shape[0], self.embedding_dim)
