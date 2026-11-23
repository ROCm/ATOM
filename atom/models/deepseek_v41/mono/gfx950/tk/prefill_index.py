# SPDX-License-Identifier: MIT
"""TP-replicated FP8 prefill indexer: deterministic head, owned query rows.

Installed only by the admitted TK backend. Key storage, query quantization,
selection and live-step publication remain owned by the native indexer.
"""

import torch
from aiter.dist.parallel_state import get_tp_group

from atom.model_ops.indexer.rows import RowSplit


class PrefillIndexTP:
    def __init__(self, *, split_queries=True):
        import aiter.tuned_gemm as tg

        group = get_tp_group()
        assert group.world_size == 8
        self.rank = group.rank
        self.world = group.world_size
        self.split_queries = split_queries
        self._head = tg._get_flydsl_gemm_kernels().flydsl_hgemm

    @staticmethod
    def canonical(cache, step):
        return step.is_prefill and not step.decode and not cache.geometry.index_fp4

    def project_weights(self, hidden, module):
        assert hidden.shape[-1] == 5120 and hidden.dtype == torch.bfloat16
        assert module.bias is None and tuple(module.weight.shape) == (32, 5120)
        assert module.weight.dtype == torch.bfloat16 and str(module.quant_type) == "QuantType.No"
        result = self._head(
            hidden.reshape(-1, 5120), module.weight,
            block_m=32, block_n=32, block_k=64, stages=9,
            split_k=1, m_waves=2, n_waves=2, k_waves=1,
            group_m=0, k_stagger=0, b_non_temporal=False,
            policy="ft", out_dtype=torch.bfloat16,
        )
        return result.reshape(*hidden.shape[:-1], 32)

    def rows(self, layer, cache, step, count):
        if not self.split_queries or not self.canonical(cache, step):
            return None
        if layer not in (2, 8, 14) or count < 2048:
            return None
        return RowSplit.of(count, self.rank, self.world, group="tp")


def install_prefill_index(model):
    from atom.models.deepseek_v41.attention import Indexer

    strategy = PrefillIndexTP()
    installed = 0
    for module in model.modules():
        if isinstance(module, Indexer):
            assert module.heads == 32 and module.head_dim == 128
            module.prefill_index_tp = strategy
            installed += 1
    assert installed > 0
    return strategy
