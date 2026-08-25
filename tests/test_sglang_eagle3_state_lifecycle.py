import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_bridge_with_fake_modules(monkeypatch, eagle_cls, future_map_cls):
    minimax_mod = types.ModuleType("atom.plugin.sglang.models.minimax_m3")
    minimax_mod.SGLangATOMMiniMaxM3Attention = object
    monkeypatch.setitem(
        sys.modules, "atom.plugin.sglang.models.minimax_m3", minimax_mod
    )

    overlap_mod = types.ModuleType("sglang.srt.managers.overlap_utils")
    overlap_mod.FutureMap = future_map_cls
    overlap_mod.RelayPayload = type(
        "RelayPayload",
        (),
        {
            "from_draft_input": classmethod(
                lambda cls, draft_input: types.SimpleNamespace()
            )
        },
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.overlap_utils", overlap_mod)

    eagle_mod = types.ModuleType("sglang.srt.speculative.eagle_info")
    eagle_mod.EagleDraftInput = eagle_cls
    monkeypatch.setitem(sys.modules, "sglang.srt.speculative.eagle_info", eagle_mod)

    bridge_path = (
        Path(__file__).resolve().parents[1]
        / "atom"
        / "plugin"
        / "sglang"
        / "eagle3_llama_bridge.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_eagle3_llama_bridge_under_fake_sglang", bridge_path
    )
    bridge = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(bridge)
    return bridge


def test_v0517_filter_batch_reorders_plugin_state_for_same_length_permutation(
    monkeypatch,
):
    class FutureMap:
        def stash(self, future_indices, payload):
            return None

        def _resolve_spec_extras(self, batch):
            return None

    class EagleDraftInput:
        def __init__(self):
            self.future_indices = None
            self.topk_index = torch.arange(3).view(3, 1)

        def filter_batch(self, new_indices, new_indices_cpu=None):
            self.topk_index = self.topk_index[new_indices]

        def merge_batch(self, spec_info):
            self.topk_index = torch.cat([self.topk_index, spec_info.topk_index])

    bridge = _load_bridge_with_fake_modules(monkeypatch, EagleDraftInput, FutureMap)
    bridge._patch_sglang_eagle3_state_lifecycle()

    draft_input = EagleDraftInput()
    draft_input._atom_sglang_eagle3_num_reject_tokens = torch.tensor(
        [10, 20, 30], dtype=torch.int32
    )
    draft_input.filter_batch(torch.tensor([2, 0, 1]), new_indices_cpu=[2, 0, 1])

    assert draft_input.topk_index.squeeze(-1).tolist() == [2, 0, 1]
    assert draft_input._atom_sglang_eagle3_num_reject_tokens.tolist() == [30, 10, 20]
