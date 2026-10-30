"""Complete native attention windows over explicitly constructed legal history.

The checkpoint weights are real. Hidden states and historical cache records
are constructed, not a captured service trajectory. Layer9 consumes layer8's
actual selection on the same step; its hidden state is a separate test input.
"""

from types import SimpleNamespace

import torch
from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.deepseek_v41.metadata import (
    RequestSpan,
    visible_buffer_name,
)
from atom.model_ops.attentions.deepseek_v41.score_planner import ScorePlanner
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace
from atom.models.deepseek_v41.config import AttentionMode
from atom.models.deepseek_v41.owner_service.owner_cost_attention_chain import (
    forward_owner,
)
from atom.models.deepseek_v41.owner_service.owner_serving_kernels import sum_original_tp
from atom.utils import CpuGpuBuffer
from atom.utils import forward_context as fc
from binding import bind_forward
from candidate import attend_fp8_head_shards


class Fixture:
    def __init__(self, config, windows, rank, rows=384):
        self.config, self.windows, self.rank, self.rows = config, windows, rank, rows
        hf = config.hf_config
        topology = windows.topology[: hf.num_hidden_layers]
        self.geometry = geo = V41PoolGeometry(
            len(windows.topology),
            tuple(
                (s.layer_id, s.ratio) for s in topology if s.mode == AttentionMode.FULL
            ),
            config.kv_cache_block_size,
            hf.sliding_window,
            hf.head_dim,
            hf.index_head_dim,
            hf.engram_max_ngram_size - 1,
            kv_dtype="fp8",
            speculative_tokens=5,
            layer_ratios=tuple(sorted({s.ratio for s in topology})),
            index_topk=hf.index_topk,
            index_block_rows=hf.candidate_block_size,
            index_fp4=True,
        )
        assert geo.index_block_rows == 8 and not geo.packed
        lengths = [6] * 64 if rows == 384 else [4] * 29 + [3] * 35
        self.valid_rows = rows - int(rows == 221 and rank == 3)
        if self.valid_rows != rows:
            lengths[-1] -= 1
        assert sum(lengths) == self.valid_rows
        starts = [8191 + i % 6 for i in range(64)]
        columns = (
            max(s + n for s, n in zip(starts, lengths)) + geo.block_size - 1
        ) // geo.block_size
        pages = 64 * columns
        workspace = ScoreWorkspace(
            geo,
            rows,
            columns,
            "cuda",
            pages=pages,
            candidate_blocks=hf.candidate_topk_blocks,
        )
        self.cache = cache = PagedAttentionCache(
            geo, pages, 256, "cuda", max_tokens=rows, workspace=workspace
        )
        ratios = tuple(ratio for ratio, _ in geo.compress_ratios)
        planner = ScorePlanner(workspace, ratios, rows)
        shapes = {
            "positions": (rows,),
            "cu_seqlens_q": (65,),
            "batch_id_per_q_token": (rows,),
            "block_tables": (64, columns),
            **{visible_buffer_name(ratio): (rows,) for ratio in ratios},
        }
        buffers = {
            name: CpuGpuBuffer(
                *shape,
                dtype=torch.int64 if name == "positions" else torch.int32,
                device="cuda",
                pin_memory=True,
            )
            for name, shape in shapes.items()
        }
        buffers.update(planner.buffers("cuda", publication_group=None))
        spans, offset = [], 0
        for i, length in enumerate(lengths):
            slot = ((rank * 64 + i) * 13) % 256
            spans.append(RequestSpan(rank * 64 + i, starts[i], offset, length, slot))
            cache.cursor[slot, 0] = starts[i]
            offset += length
        tables = [tuple(range(i * columns, (i + 1) * columns)) for i in range(64)]
        self.step = cache.begin_step(
            spans,
            block_tables=tables,
            tentative=True,
            buffers=buffers,
            running_bs=64,
            running_tokens=rows,
            max_q_len=6,
            planners=(planner,),
        )
        cache.prepare_state(self.step)

        # Only records read by these two windows need historical content.
        # Every FP8 code is finite; every scale is a finite E8M0 exponent.
        def seed_fp8(target):
            values = target[..., :512]
            code = torch.randint(0, 127, values.shape, dtype=torch.uint8, device="cuda")
            sign = torch.randint(0, 2, values.shape, dtype=torch.uint8, device="cuda")
            values.copy_(code + sign * 128)
            target[..., 512:].fill_(120)

        seed_fp8(cache.pages.view("main_8")[0])
        for layer in (8, 9):
            seed_fp8(cache.state.view("window")[layer])
        cache.index_planes[8][0].random_(0, 256)
        cache.index_planes[8][1].fill_(124)
        for state in cache.compress_state(8):
            state.normal_(0, 0.2)
        self.initial_cache = cache.backing.clone()
        with torch.device("cuda"):
            scaling = hf.rope_scaling
            self.rope = RotaryEmbedding(
                hf.qk_rope_head_dim,
                max(starts) + 7,
                base=hf.compress_rope_theta,
                original_length=scaling["original_max_position_embeddings"],
                factor=scaling["factor"],
                beta_fast=scaling["beta_fast"],
                beta_slow=scaling["beta_slow"],
            )
        self.inputs = {
            layer: SinglePassHCState.from_embeddings(
                torch.empty(
                    1, rows, hf.hidden_size, device="cuda", dtype=torch.bfloat16
                ),
                hf.hc_mult,
            )
            for layer in (8, 9)
        }
        self.update(0)
        self.native = forward_owner
        self.candidate = bind_forward(forward_owner, attend_fp8_head_shards)
        self.metadata = SimpleNamespace(cache=cache, step=self.step, image_mask=None)
        fc._forward_context = fc.ForwardContext(
            no_compile_layers=dict(config.compilation_config.static_forward_context),
            attn_metadata=self.metadata,
        )
        self.buffers, self.planner = buffers, planner

    def update(self, iteration):
        for state in self.inputs.values():
            state.residual.normal_(0, 0.2 + iteration * 0.1)
            state.residual[:, self.valid_rows :].zero_()

    def restore_cache(self):
        self.cache.backing.copy_(self.initial_cache)

    def run(self, layer, variant, *, boundary=None):
        """Layer8 owns begin_forward. Layer9 reads that same forward's selection."""
        if layer == 8:
            self.step.begin_forward()
        elif 8 not in self.step.selected:
            raise ValueError("REUSE must consume the current layer8 selection")
        block = self.windows.layers[str(layer)]
        original = self.native if variant == "native" else self.candidate
        if boundary is not None:
            # Built outside capture by the probe and handed in explicitly.
            original, mark = boundary
            mark("window_start")
        normed, residual, pre, post, comb = block.prepare_attention(
            self.inputs[layer], None, None
        )
        output, audit = original(
            block.attn,
            normed,
            self.cache,
            self.step,
            self.rope,
            self.windows.heads[layer],
            self.rank,
            sum_original_tp,
        )
        result = block.prepare_ffn(output, residual, pre, post, comb)
        if boundary is not None:
            mark("window_end")
        values = result[0] if isinstance(result[0], tuple) else (result[0],)
        return {
            "output": output,
            **{f"ffn_input_{i}": x for i, x in enumerate(values)},
            "residual": result[1],
            "pre": result[2],
            "post": result[3],
            "comb": result[4],
            **{k: v for k, v in audit.items() if isinstance(v, torch.Tensor)},
            **{f"csr_{i}": x for i, x in enumerate(audit["CSR"])},
        }

    def describe(self):
        return {
            "rows": self.rows,
            "global_capacity": 4 * self.rows,
            "global_valid_rows": 1536 if self.rows == 384 else 883,
            "local_valid_rows": self.valid_rows,
            "requests": 64,
            "positions": [span.position for span in self.step.requests],
            "lengths": [span.length for span in self.step.requests],
            "block_size": self.geometry.block_size,
            "cache_bytes": self.cache.backing.numel(),
            "history": "constructed finite FP8/FP4 records; not a service hidden trajectory",
            "windows": "layer8 FULL and layer9 REUSE; independent constructed residual inputs",
        }

    def close(self):
        # No decode prefix is committed by this isolated attention fixture.
        # The whole private cache is discarded after every graph is reset.
        self.cache.pending = None
        fc._forward_context = None
