from types import SimpleNamespace

import numpy as np
import torch
from test_gdn_draft_kv_slots import builder

from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder
from atom.utils import CpuGpuBuffer
from atom.utils.block_tables import block_table_state


def test_uncached_prefill_publishes_reused_draft_block_table():
    names = ["cu_seqlens_k", "slot_mapping", "context_lens", "block_tables"]
    var = {
        name: CpuGpuBuffer(4, dtype=torch.int32, device="cpu", pin_memory=False)
        for name in names + ["cu_seqlens_q"]
    }
    var["block_tables"] = CpuGpuBuffer(
        4, 8, dtype=torch.int32, device="cpu", pin_memory=False
    )

    class Group:
        def __init__(self):
            self.counts = [None] * len(names)
            self.indices = {name: i for i, name in enumerate(names)}

        def set_count(self, buf, count):
            self.counts[self.indices["block_tables"]] = count

        def publish(self, counts):
            for name, count in zip(names, counts):
                if count is not None:
                    var[name].copy_to_gpu(count)

    b = object.__new__(GDNAttentionMetadataBuilder)
    b.model_runner = SimpleNamespace(
        forward_vars=var,
        num_spec_tokens=3,
        use_mrope=True,
        h2d_groups={"prefill": Group()},
    )
    for pages in ([47, 62], [103, 109]):
        block_table_state(var["block_tables"]).prepare([pages])
        metadata = b._upload_prefill_mirrors(1, 2, 2, False, np.array([0], np.int32))
        assert metadata["block_tables"][0, :2].tolist() == pages
        assert var["block_tables"].gpu[0, :2].tolist() == pages


def test_rejected_suffix_is_removed_before_draft_slot_and_plan(monkeypatch):
    b, var = builder(monkeypatch)
    observed = []
    b.refresh_flydsl_plan = lambda ctx: observed.append(ctx.clone())
    var["kv_indptr"].gpu.fill_(0)  # stale prefill metadata
    result = b.prepare_mtp_decode(
        3,
        4,
        80,
        torch.zeros(4, dtype=torch.int32),
        num_reject_tokens=torch.tensor([3, 0, 1]),
    )
    assert var["context_lens"].gpu.tolist() == [32, 17, 79, 0]
    assert var["kv_indptr"].gpu.tolist() == [0, 2, 4, 9, 9]
    assert result["slot_mapping"].tolist() == [
        701 * 16 + 15,
        905 * 16,
        45 * 16 + 14,
        -1,
    ]
    assert observed[0].tolist() == [32, 17, 79, 0]
    assert var["kv_indices"].gpu[:9].tolist() == [
        700,
        701,
        300,
        905,
        41,
        42,
        43,
        44,
        45,
    ]


def test_next_draft_step_refreshes_page_count_without_rejecting_twice(monkeypatch):
    b, var = builder(monkeypatch)
    var["context_lens"].gpu.copy_(torch.tensor([32, 16, 64, 0]))
    b.prepare_mtp_decode(3, 1, 64, torch.zeros(4, dtype=torch.int32))
    assert var["kv_indptr"].gpu.tolist() == [0, 2, 3, 7, 7]
    var["context_lens"].gpu[:3] += 1
    result = b.prepare_mtp_decode(3, 1, 65, torch.zeros(4, dtype=torch.int32))
    assert var["kv_indptr"].gpu.tolist() == [0, 3, 5, 10, 10]
    assert result["slot_mapping"].tolist() == [702 * 16, 905 * 16, 45 * 16, -1]
