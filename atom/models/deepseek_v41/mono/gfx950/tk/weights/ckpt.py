# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports TileRT 0.1.7 (MIT, Copyright (c) TileRT-team) ``tilert/models/ds_v4_1_flash/weights.py``, with the parts of
# ``model_args.py`` and ``engram_tables.py`` it uses.
"""DeepSeek-V4.1-Flash's HF checkpoint read into the per-rank *logical* tensors of the TP8 decode ("D") weights, on
the CPU: what the packers of ``pack`` take (``params`` drives both).

- ``CkptIndex``: every tensor of the ``*.safetensors`` shards by name (offset, dtype, shape); reads a tensor, or a row
  range of it, with ``pread`` (no mmap, no whole-shard load).
- ``load_layer_all`` / ``load_mtp_all`` / ``load_global_all``: a scope's tensors per rank, TileRT's slicing kept:
  attention's wq_b / wo_a rows, wo_b columns and attn_sink by head (8 heads a rank), the shared expert's w1 / w3 rows
  and w2 columns (288 of 2304 a rank), each routed expert's w1 / w3 rows and w2 columns the same, head and the DSpark
  markov head by vocab rows (16160 a rank), Engram's wkv by output rows (3200 a rank); the rest replicated.
- Engram's tables (layers 1 and 14): rank r holds the rows of its three hash columns (``engram_row_range``, the
  ``hashcol`` layout TileRT's HF loader runs) or 1/8 of the rows (``engram_official_row_range``); the hash constants
  (primes, offsets, multipliers, the compressed token map) are ``compute_engram_consts``'s.

Gammas come back float32 here; ``params`` keeps a bf16 checkpoint's own bf16 tensor, as TileRT's HF loader does (the
packers store gammas as bf16 when that is lossless, so the bytes are the same either way).
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

NUM_PES = 8
DIM = 5120
N_HEADS = 64
HEAD_DIM = 512
HEADS_LOCAL = N_HEADS // NUM_PES
WQB_ROWS = HEADS_LOCAL * HEAD_DIM
O_LORA = 1024
VOCAB = 129280
VOCAB_LOCAL = VOCAB // NUM_PES
INTER = 2304
INTER_LOCAL = INTER // NUM_PES
BLK = 32
ENGRAM_COLS = 24
ENGRAM_COLS_LOCAL = ENGRAM_COLS // NUM_PES
ENGRAM_HEAD_DIM = 256
ENGRAM_WKV_ROWS = DIM * 5 // NUM_PES
ENGRAM_LAYOUTS = ("hashcol", "official")

_COMPRESS_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0)


@dataclass
class ModelArgs:
    """The fields of TileRT's ``ModelArgsDsV41Flash`` the weight pipeline, the RoPE tables and the checkpoint check
    read (TP8 only)."""

    tp: int = NUM_PES
    max_seq_len: int = 65536
    vocab_size: int = 129280
    dim: int = 5120
    n_layers: int = 40
    n_mtp_layers: int = 3
    hc_mult: int = 4
    n_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1280
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: tuple[int, ...] = _COMPRESS_RATIOS
    kv_source_layers: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layers: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    index_topk: int = 512
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    moe_inter_dim: int = 2304
    n_routed_experts: int = 384
    n_activated_experts: int = 6
    route_scale: float = 1.5
    engram_layer_ids: tuple[int, ...] = (1, 14)
    engram_num_embeddings: tuple[int, ...] = (384006168, 384016682)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 99092
    dspark_block_size: int = 5
    dspark_n_routed_experts: int = 128

    def __post_init__(self) -> None:
        assert self.tp == NUM_PES, f"tp must be {NUM_PES}: {self.tp}"

    @classmethod
    def from_ckpt(cls, ckpt_dir: str, **overrides) -> ModelArgs:
        """The defaults, after checking the checkpoint's ``config.json`` (``text_config``) agrees with them."""
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            cfg = json.load(f)["text_config"]
        args = cls(**overrides)
        checks = {
            "vocab_size": args.vocab_size,
            "hidden_size": args.dim,
            "moe_intermediate_size": args.moe_inter_dim,
            "num_hidden_layers": args.n_layers,
            "num_attention_heads": args.n_heads,
            "head_dim": args.head_dim,
            "qk_rope_head_dim": args.rope_head_dim,
            "q_lora_rank": args.q_lora_rank,
            "o_lora_rank": args.o_lora_rank,
            "o_groups": args.o_groups,
            "n_routed_experts": args.n_routed_experts,
            "num_experts_per_tok": args.n_activated_experts,
            "routed_scaling_factor": args.route_scale,
            "sliding_window": args.window_size,
            "index_topk": args.index_topk,
            "hc_mult": args.hc_mult,
            "num_nextn_predict_layers": args.n_mtp_layers,
            "dspark_block_size": args.dspark_block_size,
        }
        bad = {k: (cfg.get(k), v) for k, v in checks.items() if cfg.get(k) != v}
        assert not bad, f"checkpoint config differs from ModelArgs: {bad}"
        assert tuple(cfg["compress_ratios"]) == args.compress_ratios
        assert tuple(cfg["kv_source_layer_ids"]) == args.kv_source_layers
        assert tuple(cfg["index_source_layer_ids"]) == args.index_source_layers
        return args


# ---------------------------------------------------------------------------------------------------------------
# the checkpoint index

_ST_DTYPE = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
}
_IO_THREADS = 64
_CHUNK = 256 << 20


@dataclass(frozen=True)
class TensorInfo:
    path: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    @property
    def row_bytes(self) -> int:
        return self.nbytes // self.shape[0]


class CkptIndex:
    """The tensors of a directory's ``*.safetensors`` files, read on demand into CPU tensors."""

    def __init__(self, ckpt_dir: str) -> None:
        self.ckpt_dir = ckpt_dir
        self.tensors: dict[str, TensorInfo] = {}
        self._fds: dict[str, int] = {}
        self._lock = threading.Lock()
        files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors"))
        if not files:
            raise FileNotFoundError(f"no *.safetensors under {ckpt_dir}")
        for fname in files:
            path = os.path.join(ckpt_dir, fname)
            fd = os.open(path, os.O_RDONLY)
            n = int.from_bytes(os.pread(fd, 8, 0), "little")
            hdr = json.loads(os.pread(fd, n, 8))
            os.close(fd)
            base = 8 + n
            for k, e in hdr.items():
                if k == "__metadata__":
                    continue
                if k in self.tensors:
                    raise RuntimeError(f"{k} in both {self.tensors[k].path} and {path}")
                dt = _ST_DTYPE.get(e["dtype"])
                if dt is None:
                    raise RuntimeError(f"{k}: unsupported dtype {e['dtype']}")
                lo, hi = e["data_offsets"]
                self.tensors[k] = TensorInfo(
                    path, dt, tuple(e["shape"]), base + lo, hi - lo
                )

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def info(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"{name} not in checkpoint {self.ckpt_dir}") from None

    def _fd(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            with self._lock:
                fd = self._fds.get(path)
                if fd is None:
                    fd = os.open(path, os.O_RDONLY)
                    self._fds[path] = fd
        return fd

    def _pread_into(self, path: str, offset: int, out: torch.Tensor) -> None:
        assert out.is_contiguous() and out.device.type == "cpu"
        mv = memoryview(out.view(torch.uint8).reshape(-1).numpy())
        fd = self._fd(path)
        pos, n = 0, len(mv)
        while pos < n:
            want = min(_CHUNK, n - pos)
            got = os.preadv(fd, [mv[pos : pos + want]], offset + pos)
            if got <= 0:
                raise OSError(f"{path}: short read at {offset + pos}")
            pos += got

    def read(self, name: str, out: torch.Tensor | None = None) -> torch.Tensor:
        ti = self.info(name)
        if out is None:
            out = torch.empty(ti.shape, dtype=ti.dtype)
        assert out.dtype == ti.dtype and tuple(out.shape) == ti.shape, (name, out.shape)
        self._pread_into(ti.path, ti.offset, out)
        return out

    def read_rows(
        self, name: str, r0: int, r1: int, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        ti = self.info(name)
        assert 0 <= r0 < r1 <= ti.shape[0], (name, r0, r1, ti.shape)
        shape = (r1 - r0, *ti.shape[1:])
        if out is None:
            out = torch.empty(shape, dtype=ti.dtype)
        assert out.dtype == ti.dtype and tuple(out.shape) == shape, (
            name,
            out.shape,
            shape,
        )
        self._pread_into(ti.path, ti.offset + r0 * ti.row_bytes, out)
        return out

    def close(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()


def is_hf_checkpoint(path: str) -> bool:
    """An HF checkpoint directory (safetensors index and config, not a TileRT-converted one)."""
    return (
        os.path.isfile(os.path.join(path, "model.safetensors.index.json"))
        and os.path.isfile(os.path.join(path, "config.json"))
        and not os.path.isfile(os.path.join(path, "tilert_meta.json"))
    )


_INDEX_CACHE: dict[str, CkptIndex] = {}


def get_index(ckpt: str | CkptIndex) -> CkptIndex:
    if isinstance(ckpt, CkptIndex):
        return ckpt
    key = os.path.realpath(ckpt)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = CkptIndex(ckpt)
    return _INDEX_CACHE[key]


def _ranks(ranks: Iterable[int] | None) -> list[int]:
    rs = list(range(NUM_PES)) if ranks is None else [int(r) for r in ranks]
    assert rs and all(0 <= r < NUM_PES for r in rs) and len(set(rs)) == len(rs), rs
    return rs


def _gamma(idx: CkptIndex, name: str) -> torch.Tensor:
    return idx.read(name).float()


def _rows(
    idx: CkptIndex, name: str, ranks: Sequence[int], per: int
) -> list[torch.Tensor]:
    """Rows ``[per * r, per * (r + 1))`` of ``name`` for each rank."""
    ti = idx.info(name)
    assert ti.shape[0] == per * NUM_PES, (name, ti.shape, per)
    return [idx.read_rows(name, per * r, per * (r + 1)) for r in ranks]


def _cols(
    idx: CkptIndex, name: str, ranks: Sequence[int], per: int
) -> list[torch.Tensor]:
    """Columns ``[per * r, per * (r + 1))`` of ``name`` (last dim) for each rank, contiguous."""
    ti = idx.info(name)
    assert ti.shape[-1] == per * NUM_PES, (name, ti.shape, per)
    full = idx.read(name)
    return [full[..., per * r : per * (r + 1)].contiguous() for r in ranks]


def _pmap(fn, items) -> None:
    items = list(items)
    if len(items) <= 4:
        for a in items:
            fn(*a)
        return
    with ThreadPoolExecutor(min(_IO_THREADS, len(items))) as ex:
        list(ex.map(lambda a: fn(*a), items))


# ---------------------------------------------------------------------------------------------------------------
# Engram's hash constants (``engram_tables.py``)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def hash_primes(
    n_layers: int, max_ngram_size: int, n_heads: int, vocab_size: int
) -> np.ndarray:
    """int64 [n_layers, max_ngram_size - 1, n_heads]: the consecutive primes from ``vocab_size`` on (each hash
    column's table rows)."""
    count = n_layers * (max_ngram_size - 1) * n_heads
    out, n = [], vocab_size - 1
    while len(out) < count:
        n += 1
        if _is_prime(n):
            out.append(n)
    return np.array(out, dtype=np.int64).reshape(n_layers, max_ngram_size - 1, n_heads)


def hash_multipliers(
    layer_ids, max_ngram_size: int, compressed_vocab: int
) -> np.ndarray:
    """int64 [len(layer_ids), max_ngram_size]: odd multipliers from ``default_rng(10007 * layer)``."""
    bound = max(1, np.iinfo(np.int64).max // compressed_vocab // 2)
    rows = [
        np.random.default_rng(10007 * layer).integers(
            0, bound, size=max_ngram_size, dtype=np.int64
        )
        for layer in layer_ids
    ]
    return np.stack(rows) * 2 + 1


def compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """(token id -> compressed id, compressed vocab): tokens that decode to the same normalized text share an id."""
    from tokenizers import Regex, normalizers

    keep = ""
    norm = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex("[ \\t\\r\\n]+"), " "),
            normalizers.Replace(Regex("^ $"), keep),
            normalizers.Strip(),
            normalizers.Replace(keep, " "),
        ]
    )
    rust = tokenizer.backend_tokenizer
    ids: dict[str, int] = {}
    table = []
    for t in range(len(tokenizer)):
        text = rust.decode([t], skip_special_tokens=False)
        key = rust.id_to_token(t) if "�" in text else norm.normalize_str(text) or text
        table.append(ids.setdefault(key, len(ids)))
    return table, len(ids)


def engram_hash_consts(args: ModelArgs | None = None, vocab: int | None = None) -> dict:
    """The tokenizer-free part of ``compute_engram_consts``: primes, each hash column's first table row
    (``offsets``) and end (``col_row_end``), the multipliers (``vocab``: the compressed vocab, default the
    config's)."""
    args = args or ModelArgs()
    vocab = args.engram_compressed_vocab_size if vocab is None else vocab
    layer_ids = tuple(args.engram_layer_ids)
    primes = torch.from_numpy(
        hash_primes(
            len(layer_ids),
            args.engram_max_ngram_size,
            args.engram_n_heads,
            args.engram_vocab_size,
        )
    )
    flat = primes.flatten(1)
    offs = torch.cat([torch.zeros_like(flat[:, :1]), flat[:, :-1].cumsum(1)], 1)
    mult = torch.from_numpy(
        hash_multipliers(layer_ids, args.engram_max_ngram_size, vocab)
    )
    ends = offs + flat
    for li, n in enumerate(args.engram_num_embeddings):
        assert int(ends[li, -1]) <= n, (li, int(ends[li, -1]), n)
    return {
        "layer_ids": torch.tensor(layer_ids, dtype=torch.int32),
        "primes": primes,
        "offsets": offs,
        "col_row_end": ends,
        "multipliers": mult.to(torch.int64),
        "num_embeddings": torch.tensor(args.engram_num_embeddings, dtype=torch.int64),
        "max_ngram_size": args.engram_max_ngram_size,
        "n_heads": args.engram_n_heads,
        "head_dim": args.engram_head_dim,
    }


def compute_engram_consts(tokenizer_dir: str, args: ModelArgs | None = None) -> dict:
    """TileRT's ``engram_consts``: ``engram_hash_consts`` plus the compressed token map of ``tokenizer_dir``'s
    tokenizer (needs ``transformers``)."""
    from transformers import AutoTokenizer

    args = args or ModelArgs()
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    token_map, vocab = compressed_token_map(tok)
    assert vocab == args.engram_compressed_vocab_size, (
        vocab,
        args.engram_compressed_vocab_size,
    )
    hc = engram_hash_consts(args, vocab)
    return {
        "layer_ids": hc["layer_ids"],
        "token_map": torch.tensor(token_map, dtype=torch.int32),
        "compressed_vocab_size": vocab,
        "pad_id": int(token_map[args.engram_pad_id]),
        "pad_token_raw": args.engram_pad_id,
        **{k: hc[k] for k in ("primes", "offsets", "col_row_end", "multipliers")},
        **{
            k: hc[k]
            for k in ("num_embeddings", "max_ngram_size", "n_heads", "head_dim")
        },
    }


def load_engram_consts(path: str) -> dict:
    return torch.load(path, weights_only=True)


def _engram_slot(layer: int, args: ModelArgs) -> int:
    return list(args.engram_layer_ids).index(layer)


def engram_row_range(consts: dict, layer: int, rank: int) -> tuple[int, int]:
    """``hashcol``: rank ``rank``'s table rows ``[lo, hi)`` of ``layer`` (its 3 of the 24 hash columns)."""
    slot = [int(x) for x in consts["layer_ids"]].index(layer)
    c0 = ENGRAM_COLS_LOCAL * rank
    lo = int(consts["offsets"][slot, c0])
    hi = int(consts["col_row_end"][slot, c0 + ENGRAM_COLS_LOCAL - 1])
    return lo, hi


def engram_official_row_range(num_embeddings: int, rank: int) -> tuple[int, int, int]:
    """``official``: rank ``rank``'s rows ``[lo, hi)`` of an even 1/8 split, and the (padded) shard size."""
    shard = (num_embeddings + NUM_PES - 1) // NUM_PES
    lo = rank * shard
    return lo, min(lo + shard, num_embeddings), shard


# ---------------------------------------------------------------------------------------------------------------
# a scope's logical tensors per rank


def _engram(
    idx: CkptIndex,
    p: str,
    layer: int,
    ranks: Sequence[int],
    out: list[dict],
    consts: dict | None,
    layout: str,
    args: ModelArgs,
    tables: bool = True,
) -> None:
    assert layout in ENGRAM_LAYOUTS, layout
    e = p + "engram."
    q = idx.read(e + "q_weight").float()
    k = idx.read(e + "k_weight").float()
    e_qk = q * k
    tname, sname = e + "embed.weight", e + "embed.scale"
    n = idx.info(tname).shape[0]
    assert n == args.engram_num_embeddings[_engram_slot(layer, args)], (n, layer)
    if layout == "hashcol":
        if consts is None:
            raise ValueError("engram_layout='hashcol' needs the engram consts")
        wkv = _rows(idx, e + "wkv.weight", ranks, ENGRAM_WKV_ROWS)
        wkv_s = _rows(idx, e + "wkv.scale", ranks, ENGRAM_WKV_ROWS // BLK)
        for i, r in enumerate(ranks):
            lo, hi = engram_row_range(consts, layer, r)
            if tables:
                out[i]["e_table"] = idx.read_rows(tname, lo, hi)
                out[i]["e_table_s"] = idx.read_rows(sname, lo, hi)
            out[i]["e_row_lo"] = torch.tensor([lo, hi], dtype=torch.int64)
            out[i]["e_wkv"], out[i]["e_wkv_s"] = wkv[i], wkv_s[i]
    else:  # (as TileRT: this layout reads the tables whatever ``tables`` says)
        wkv, wkv_s = idx.read(e + "wkv.weight"), idx.read(e + "wkv.scale")
        for i, r in enumerate(ranks):
            lo, hi, shard = engram_official_row_range(n, r)
            t = torch.zeros((shard, ENGRAM_HEAD_DIM), dtype=torch.float8_e4m3fn)
            s = torch.full(
                (shard, ENGRAM_HEAD_DIM // BLK), 127, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
            idx.read_rows(tname, lo, hi, out=t[: hi - lo])
            idx.read_rows(sname, lo, hi, out=s[: hi - lo])
            out[i]["e_table"], out[i]["e_table_s"] = t, s
            out[i]["e_row_lo"] = torch.tensor([lo, lo + shard], dtype=torch.int64)
            out[i]["e_wkv"], out[i]["e_wkv_s"] = wkv, wkv_s
    for d in out:
        d["e_qk"] = e_qk


def _hc(idx: CkptIndex, p: str, out: list[dict]) -> None:
    for half in ("attn", "ffn"):
        for kind in ("fn", "base", "scale"):
            t = idx.read(f"{p}hc_{half}_{kind}")
            assert t.dtype == torch.float32, (p, half, kind, t.dtype)
            for d in out:
                d[f"hc_{half}_{kind}"] = t


def _attn(idx: CkptIndex, p: str, ranks: Sequence[int], out: list[dict]) -> None:
    a = p + "attn."
    repl = {
        "attn_norm": _gamma(idx, p + "attn_norm.weight"),
        "q_norm": _gamma(idx, a + "q_norm.weight"),
        "kv_norm": _gamma(idx, a + "kv_norm.weight"),
        "wq_a": idx.read(a + "wq_a.weight"),
        "wq_a_s": idx.read(a + "wq_a.scale"),
        "wkv": idx.read(a + "wkv.weight"),
        "wkv_s": idx.read(a + "wkv.scale"),
    }
    wq_b = _rows(idx, a + "wq_b.weight", ranks, WQB_ROWS)
    wq_b_s = _rows(idx, a + "wq_b.scale", ranks, WQB_ROWS // BLK)
    wo_a = _rows(idx, a + "wo_a.weight", ranks, O_LORA)
    wo_a_s = _rows(idx, a + "wo_a.scale", ranks, O_LORA // BLK)
    wo_b = _cols(idx, a + "wo_b.weight", ranks, O_LORA)
    wo_b_s = _cols(idx, a + "wo_b.scale", ranks, O_LORA // BLK)
    sink = idx.read(a + "attn_sink")
    for i, r in enumerate(ranks):
        d = out[i]
        d.update(repl)
        d["wq_b"], d["wq_b_s"] = wq_b[i], wq_b_s[i]
        d["wo_a"], d["wo_a_s"] = wo_a[i], wo_a_s[i]
        d["wo_b"], d["wo_b_s"] = wo_b[i], wo_b_s[i]
        d["attn_sink"] = sink[HEADS_LOCAL * r : HEADS_LOCAL * (r + 1)].contiguous()
    c = a + "compressor."
    if c + "wkv.weight" in idx:
        repl = {
            "c_wkv": idx.read(c + "wkv.weight"),
            "c_norm": _gamma(idx, c + "norm.weight"),
        }
        if c + "wgate.weight" in idx:
            repl["c_wgate"] = idx.read(c + "wgate.weight")
        for d in out:
            d.update(repl)
    ix = a + "indexer."
    if ix + "wq_b.weight" in idx:
        repl = {
            "i_wq_b": idx.read(ix + "wq_b.weight"),
            "i_wq_b_s": idx.read(ix + "wq_b.scale"),
            "i_wproj": idx.read(ix + "weights_proj.weight"),
        }
        if ix + "wk.weight" in idx:
            repl["i_wk"] = idx.read(ix + "wk.weight")
            repl["i_k_norm"] = _gamma(idx, ix + "k_norm.weight")
        for d in out:
            d.update(repl)


def _moe(
    idx: CkptIndex,
    p: str,
    ranks: Sequence[int],
    out: list[dict],
    experts: Sequence[int] | None,
) -> None:
    f = p + "ffn."
    repl = {
        "ffn_norm": _gamma(idx, p + "ffn_norm.weight"),
        "gate_w": idx.read(f + "gate.weight"),
        "gate_b": idx.read(f + "gate.bias").float(),
    }
    s = f + "shared_experts."
    sw1 = _rows(idx, s + "w1.weight", ranks, INTER_LOCAL)
    sw1_s = _rows(idx, s + "w1.scale", ranks, INTER_LOCAL // BLK)
    sw3 = _rows(idx, s + "w3.weight", ranks, INTER_LOCAL)
    sw3_s = _rows(idx, s + "w3.scale", ranks, INTER_LOCAL // BLK)
    sw2 = _cols(idx, s + "w2.weight", ranks, INTER_LOCAL)
    sw2_s = _cols(idx, s + "w2.scale", ranks, INTER_LOCAL // BLK)
    for i in range(len(ranks)):
        d = out[i]
        d.update(repl)
        d["sw1"], d["sw1_s"], d["sw3"], d["sw3_s"] = sw1[i], sw1_s[i], sw3[i], sw3_s[i]
        d["sw2"], d["sw2_s"] = sw2[i], sw2_s[i]
    n_exp = repl["gate_w"].shape[0]
    ids = list(range(n_exp)) if experts is None else [int(e) for e in experts]
    assert all(0 <= e < n_exp for e in ids), ids
    if not ids:
        return
    e0 = f + "experts.0."
    # key: (checkpoint name, rows a rank or None, columns a rank or None); w1 / w3 by rows, w2 by columns
    src = {
        "w1": ("w1.weight", INTER_LOCAL, None),
        "w1_s": ("w1.scale", INTER_LOCAL, None),
        "w3": ("w3.weight", INTER_LOCAL, None),
        "w3_s": ("w3.scale", INTER_LOCAL, None),
        "w2": ("w2.weight", None, INTER_LOCAL // 2),
        "w2_s": ("w2.scale", None, INTER_LOCAL // BLK),
    }
    bufs = {}
    for key, (name, rows, cols) in src.items():
        ti = idx.info(e0 + name)
        assert ti.dtype in (torch.int8, torch.float8_e8m0fnu), (name, ti.dtype)
        shape = list(ti.shape)
        if rows is not None:
            shape[0] = rows
        if cols is not None:
            shape[1] = cols
        bufs[key] = [torch.empty((len(ids), *shape), dtype=ti.dtype) for _ in ranks]

    def job(j: int, key: str) -> None:
        name, rows, cols = src[key]
        full = f"{f}experts.{ids[j]}.{name}"
        dst = bufs[key]
        if rows is not None:
            for i, r in enumerate(ranks):
                idx.read_rows(full, rows * r, rows * (r + 1), out=dst[i][j])
        else:
            t = idx.read(full)
            for i, r in enumerate(ranks):
                dst[i][j].copy_(t[:, cols * r : cols * (r + 1)])

    _pmap(job, [(j, k) for j in range(len(ids)) for k in src])
    for i in range(len(ranks)):
        for key, v in bufs.items():
            out[i][key] = v[i].view(torch.uint8) if v[i].dtype == torch.int8 else v[i]
    if experts is not None:
        for d in out:
            d["expert_ids"] = torch.tensor(ids, dtype=torch.int32)


def load_layer_all(
    ckpt: str | CkptIndex,
    layer: int,
    ranks: Iterable[int] | None = None,
    *,
    experts: Sequence[int] | None = None,
    engram_consts: dict | str | None = None,
    engram_layout: str = "hashcol",
    args: ModelArgs | None = None,
    engram_tables: bool = True,
) -> list[dict[str, torch.Tensor]]:
    """``layers.<layer>``'s logical tensors for each rank of ``ranks`` (default all 8). ``experts``: the routed experts
    to load (None: all, without ``expert_ids``). ``engram_tables=False`` leaves the ``hashcol`` Engram tables out
    (``params`` reads them apart, they are 12 GB a rank)."""
    idx = get_index(ckpt)
    args = args or ModelArgs()
    rs = _ranks(ranks)
    assert 0 <= layer < args.n_layers, layer
    out: list[dict] = [{} for _ in rs]
    p = f"layers.{layer}."
    _hc(idx, p, out)
    _attn(idx, p, rs, out)
    _moe(idx, p, rs, out, experts)
    if layer in args.engram_layer_ids:
        if isinstance(engram_consts, str):
            engram_consts = load_engram_consts(engram_consts)
        _engram(
            idx, p, layer, rs, out, engram_consts, engram_layout, args, engram_tables
        )
    return out


def load_mtp_all(
    ckpt: str | CkptIndex,
    m: int,
    ranks: Iterable[int] | None = None,
    *,
    experts: Sequence[int] | None = None,
    args: ModelArgs | None = None,
) -> list[dict[str, torch.Tensor]]:
    """``mtp.<m>``'s (the DSpark draft layers') logical tensors for each rank: a layer's, plus ``main_proj`` (draft
    layer 0) and the markov head (the last)."""
    idx = get_index(ckpt)
    args = args or ModelArgs()
    rs = _ranks(ranks)
    assert 0 <= m < args.n_mtp_layers, m
    out: list[dict] = [{} for _ in rs]
    p = f"mtp.{m}."
    _hc(idx, p, out)
    _attn(idx, p, rs, out)
    _moe(idx, p, rs, out, experts)
    if p + "main_proj.weight" in idx:
        repl = {
            "main_proj": idx.read(p + "main_proj.weight"),
            "main_proj_s": idx.read(p + "main_proj.scale"),
            "main_norm": _gamma(idx, p + "main_norm.weight"),
        }
        for d in out:
            d.update(repl)
    if p + "markov_head.embed.weight" in idx:
        emb = idx.read(p + "markov_head.embed.weight")
        mh = _rows(idx, p + "markov_head.head.weight", rs, VOCAB_LOCAL)
        g = _gamma(idx, p + "norm.weight")
        for i, d in enumerate(out):
            d["mtp_norm"], d["markov_embed"], d["markov_head"] = g, emb, mh[i]
    return out


def load_global_all(
    ckpt: str | CkptIndex,
    ranks: Iterable[int] | None = None,
    *,
    with_embed: bool = True,
) -> list[dict[str, torch.Tensor]]:
    """The final norm, head's vocab rows and (``with_embed``) the full embedding for each rank."""
    idx = get_index(ckpt)
    rs = _ranks(ranks)
    out: list[dict] = [{} for _ in rs]
    g = _gamma(idx, "norm.weight")
    head = _rows(idx, "head.weight", rs, VOCAB_LOCAL)
    emb = idx.read("embed.weight") if with_embed else None
    for i, d in enumerate(out):
        d["norm"], d["head"] = g, head[i]
        if emb is not None:
            d["embed"] = emb
    return out
