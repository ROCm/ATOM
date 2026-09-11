"""Device-side Engram op and the module that attaches it to a model.

`EngramOp` is the per-layer device compute; `EngramModules` builds one op per
engram layer, loads their weights, and hands back the `EngramHost` (defined in
`engram.py`) that feeds them -- the runner drives that host each step.

Structure is taken from the checkpoint's own tensors and from the V4.1 tech
report section 2.4.2, which states two deliberate departures from the original
Engram design: the short causal convolution is omitted, and the embedding tables
are trained with momentum + Sinkhorn balancing rather than Adam. Only the first
matters here -- there is no ShortConv, and no weights for one in the checkpoint.

Per engram layer the checkpoint carries six tensors:

    embed.weight   [num_rows, 256]        fp8   the table, ~98 GB, stays on host
    embed.scale    [num_rows, 8]          e8m0  one scale per 32 values of a row
    wkv.weight     [(hc_mult+1)*H, 6144]  fp8   key projections then value
    wkv.scale      [800, 192]             e8m0  32x32 blocks
    k_weight       [hc_mult, H]           bf16  RMSNorm gain, key side
    q_weight       [hc_mult, H]           bf16  RMSNorm gain, query side

The embedding half never appears here: the lookup happens on the host and the
result arrives as `embeddings`, 6144 wide (3 n-gram orders x 8 heads x 256, or
equivalently the report's "2048 per order").
"""

from __future__ import annotations

import json
import logging
import math
import os

import torch
from torch import nn

from atom.model_ops.engram import (
    CompressedTokenizer,
    EngramConfig,
    EngramHost,
    EngramPrefetcher,
    HostEmbeddingTable,
    NgramHashMapping,
)

logger = logging.getLogger(__name__)


def _rms_norm(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with a plain multiplicative gain.

    The tech report notes that the largest trained RMSNorm weight magnitude in
    this model is about 1, which is what these tensors look like -- so the gain
    is used directly rather than as (1 + gain).
    """
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)).to(x.dtype) * gain


class EngramOp(nn.Module):
    """One Engram module: gate a host-supplied memory read into the residual.

    `forward` takes the embeddings rather than token ids. The table is ~98 GB and
    lives on the host; a layer that indexed it inline would stall the step it
    belongs to, which is the entire problem this module exists to avoid.
    """

    def __init__(
        self,
        layer_id: int,
        hidden_size: int = 5120,
        engram_hidden_size: int = 6144,
        hc_mult: int = 4,
        norm_eps: float = 1e-20,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.engram_hidden_size = engram_hidden_size
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps

        # One fused projection, laid out as the checkpoint stores it: the
        # hc_mult key projections first, the single shared value projection last.
        self.wkv = nn.Linear(
            engram_hidden_size, (hc_mult + 1) * hidden_size, bias=False
        )
        self.k_weight = nn.Parameter(torch.ones(hc_mult, hidden_size))
        self.q_weight = nn.Parameter(torch.ones(hc_mult, hidden_size))

    @property
    def key_rows(self) -> int:
        return self.hc_mult * self.hidden_size

    def forward(
        self, hidden_states: torch.Tensor, embeddings: torch.Tensor
    ) -> torch.Tensor:
        """[N, hc_mult, H] + [N, 6144] -> [N, hc_mult, H].

        The token dimension is flat, matching the rest of ATOM: a model's
        residual stream is [num_tokens, hc, dim], not [batch, seq, ...]. Leading
        dimensions are otherwise free, so a [B, T, hc, H] caller also works.

        Context-aware gating: each branch scores its own key against its own
        slice of the residual, and the shared value is admitted in proportion to
        that score.
        """
        if embeddings.shape[-1] != self.engram_hidden_size:
            raise ValueError(
                f"engram embeddings are {embeddings.shape[-1]} wide, expected "
                f"{self.engram_hidden_size}"
            )
        if hidden_states.shape[-2] != self.hc_mult:
            raise ValueError(
                f"hidden states carry {hidden_states.shape[-2]} branches, "
                f"expected hc_mult={self.hc_mult}"
            )
        if hidden_states.shape[:-2] != embeddings.shape[:-1]:
            raise ValueError(
                f"hidden states cover {tuple(hidden_states.shape[:-2])} tokens "
                f"but embeddings cover {tuple(embeddings.shape[:-1])}"
            )
        lead = hidden_states.shape[:-2]

        kv = self.wkv(embeddings)
        keys = kv[..., : self.key_rows].view(*lead, self.hc_mult, self.hidden_size)
        value = kv[..., self.key_rows :]

        key = _rms_norm(keys, self.k_weight, self.norm_eps)
        query = _rms_norm(hidden_states, self.q_weight, self.norm_eps)
        gate = (key * query).sum(dim=-1) / math.sqrt(self.hidden_size)
        # Signed square root before the sigmoid: keeps the gate responsive for
        # small scores without letting large ones saturate it.
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        return gate.sigmoid().unsqueeze(-1) * value.unsqueeze(-2)

    @torch.no_grad()
    def load_checkpoint_weights(
        self,
        wkv: torch.Tensor,
        k_weight: torch.Tensor,
        q_weight: torch.Tensor,
        wkv_scale: torch.Tensor | None = None,
        block: int = 32,
    ) -> None:
        """Load the six-tensor layout, dequantizing the fp8 wkv if needed.

        Shapes are checked rather than reshaped into submission: a silently
        transposed or mis-split wkv produces plausible numbers and a wrong model.
        """
        expected = ((self.hc_mult + 1) * self.hidden_size, self.engram_hidden_size)
        if tuple(wkv.shape) != expected:
            raise ValueError(f"wkv is {tuple(wkv.shape)}, expected {expected}")
        for name, tensor in (("k_weight", k_weight), ("q_weight", q_weight)):
            if tuple(tensor.shape) != (self.hc_mult, self.hidden_size):
                raise ValueError(
                    f"{name} is {tuple(tensor.shape)}, expected "
                    f"{(self.hc_mult, self.hidden_size)}"
                )
        if wkv_scale is not None:
            rows, cols = wkv.shape
            if tuple(wkv_scale.shape) != (rows // block, cols // block):
                raise ValueError(
                    f"wkv scale is {tuple(wkv_scale.shape)}, expected "
                    f"{(rows // block, cols // block)} for {block}x{block} blocks"
                )
            wkv = (
                wkv.float().reshape(rows // block, block, cols // block, block)
                * wkv_scale.float().reshape(rows // block, 1, cols // block, 1)
            ).reshape(rows, cols)
        self.wkv.weight.copy_(wkv.to(self.wkv.weight.dtype))
        self.k_weight.copy_(k_weight.to(self.k_weight.dtype))
        self.q_weight.copy_(q_weight.to(self.q_weight.dtype))


# ---------------------------------------------------------------------------
# Model attachment: build the ops and the EngramHost that feeds them (was
# engram_module.py).
# ---------------------------------------------------------------------------

_EMBED = "layers.{}.engram.embed.weight"
_EMBED_SCALE = "layers.{}.engram.embed.scale"
_WKV = "layers.{}.engram.wkv.weight"
_WKV_SCALE = "layers.{}.engram.wkv.scale"
_K_WEIGHT = "layers.{}.engram.k_weight"
_Q_WEIGHT = "layers.{}.engram.q_weight"


class EngramModules(nn.Module):
    """The engram layers of one model, plus the host state they read from."""

    def __init__(
        self,
        config: EngramConfig,
        hash_mapping: NgramHashMapping,
        ops: dict[int, EngramOp],
        tables: dict[int, HostEmbeddingTable],
        handles: list = (),
    ):
        super().__init__()
        self.config = config
        self.hash_mapping = hash_mapping
        self.ops = nn.ModuleDict({str(k): v for k, v in ops.items()})
        self._tables = tables
        # The tables are memory-mapped views into the checkpoint. Dropping these
        # handles unmaps them, so they are held for the model's lifetime.
        self._handles = list(handles)

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return self.config.layer_ids

    def __contains__(self, layer_id: int) -> bool:
        return str(layer_id) in self.ops

    def __getitem__(self, layer_id: int) -> EngramOp:
        return self.ops[str(layer_id)]

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str,
        hf_config: dict | None = None,
        tokenizer=None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> EngramModules | None:
        """Build from a checkpoint directory. None when the model has no engram.

        The embedding tables are memory-mapped, never materialized: they are
        about 98 GB each here, and a host that can hold them in page cache still
        cannot afford to copy them into the process.
        """
        if hf_config is None:
            with open(os.path.join(model_path, "config.json")) as fh:
                hf_config = json.load(fh)
        text_config = hf_config.get("text_config", hf_config)
        config = EngramConfig.from_hf(text_config)
        if config is None:
            return None

        hidden_size = int(text_config["hidden_size"])
        hc_mult = int(text_config.get("hc_mult", 4))
        norm_eps = float(text_config.get("rms_norm_eps", 1e-6))
        engram_hidden = config.num_hash_heads * config.head_dim

        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
        hash_mapping = NgramHashMapping(
            config,
            CompressedTokenizer(tokenizer, expected_size=config.compressed_vocab_size),
        )

        from safetensors import safe_open

        with open(os.path.join(model_path, "model.safetensors.index.json")) as fh:
            weight_map = json.load(fh)["weight_map"]

        ops: dict[int, EngramOp] = {}
        tables: dict[int, HostEmbeddingTable] = {}
        handles = []
        for layer_id, num_rows in zip(config.layer_ids, config.num_embeddings):
            shard = weight_map[_EMBED.format(layer_id)]
            handle = safe_open(os.path.join(model_path, shard), framework="pt")
            handles.append(handle)
            tables[layer_id] = HostEmbeddingTable(
                handle.get_tensor(_EMBED.format(layer_id)),
                num_rows=num_rows,
                head_dim=config.head_dim,
                scale=handle.get_tensor(_EMBED_SCALE.format(layer_id)),
            )
            op = EngramOp(
                layer_id,
                hidden_size=hidden_size,
                engram_hidden_size=engram_hidden,
                hc_mult=hc_mult,
                norm_eps=norm_eps,
            ).to(dtype)
            op.load_checkpoint_weights(
                handle.get_tensor(_WKV.format(layer_id)),
                handle.get_tensor(_K_WEIGHT.format(layer_id)),
                handle.get_tensor(_Q_WEIGHT.format(layer_id)),
                wkv_scale=handle.get_tensor(_WKV_SCALE.format(layer_id)),
            )
            ops[layer_id] = op

        logger.info(
            "engram: %d modules on layers %s, %d hash heads, tables mapped from %s",
            len(ops),
            list(config.layer_ids),
            config.num_hash_heads,
            model_path,
        )
        return cls(config, hash_mapping, ops, tables, handles)

    def build_engram_host(
        self,
        device: torch.device,
        max_num_tokens: int,
        dtype: torch.dtype | None = None,
    ) -> EngramHost:
        """The contract ModelRunner looks for by name.

        The staging buffers default to the dtype the engram layers actually
        compute in, so the embeddings arrive ready to feed `wkv` -- staging in
        float32 against bf16 weights is a dtype error at the first matmul.
        """
        if dtype is None:
            dtype = next(self.ops.parameters()).dtype
        return EngramHost(
            EngramPrefetcher(self.hash_mapping, self._tables),
            max_num_tokens=max_num_tokens,
            num_hash_heads=self.config.num_hash_heads,
            head_dim=self.config.head_dim,
            device=device,
            dtype=dtype,
        )
