"""Check the PP8-P (TP1) -> TP8/DCP8-D Mooncake overlay's KDA state transfer against the image's own helpers."""

import asyncio
import importlib.util
import os
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest import mock

os.environ.setdefault("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
OVERLAY = Path(__file__).resolve().parents[3] / "benchmark/c48-model-gap/n3966924d/pp-mooncake/mooncake_connector.py"
try:
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import derive_mamba_conv_split
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec

    _spec = importlib.util.spec_from_file_location("c48_pp_mooncake_connector", OVERLAY)
    mc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(mc)
except ImportError:
    mc = None

LAYER = "language_model.model.layers.1.self_attn"
CH, HEADS, ROWS = 12288, 96, 7  # q/k/v channels at TP1, KDA heads, conv kernel 4 - 1 + 4 speculative rows
P_BASE, D_BASE, P_PAGE, D_PAGE = 1 << 40, 1 << 44, 3_686_400, 516_096


def kda_spec(tp, block_size):
    return MambaSpec(block_size=block_size, shapes=((3 * CH // tp, ROWS), (HEADS // tp, 128, 128)),
                     dtypes=(torch.bfloat16, torch.bfloat16), mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                     mamba_cache_mode="align", num_speculative_blocks=4)


def sizes(tp):
    split = derive_mamba_conv_split(kda_spec(tp, 1), tp)
    return [*split.proj_bytes, split.ssm_sizes[1]]


def producer():
    spec = kda_spec(1, 6400)
    p = NS(use_mla=True, tp_size=1, dcp_size=1, pcp_size=1, interleave_size=1, opaque_packed_storages=set(),
           kv_cache_config=NS(transfer_groups=[NS(kv_cache_spec=spec)]), _layer_specs={LAYER: spec},
           mamba_state_sizes={LAYER: sizes(1)}, mla_block_sizes={})
    for name in ("_validate_dcp_transfer", "_build_dcp_request_params", "_build_transfer_params"):
        setattr(p, name, MethodType(getattr(mc.MooncakeConnectorWorker, name), p))
    return p


def pull(rank, d_blocks, num_tokens):
    return mc.MooncakeXferMetadata(
        remote_hostname="d", remote_port=1, remote_tp_size=8, remote_tp_rank=rank,
        req_blocks={"d": ("x", [d_blocks])}, kv_caches_base_addr=[D_BASE], block_lens=[D_PAGE],
        kv_block_lens=[D_PAGE], remote_dcp_size=8, remote_dcp_rank=rank, remote_group_block_sizes=[896],
        req_num_tokens={"d": num_tokens}, mamba_state_sizes={LAYER: sizes(8)}, has_opaque_packed_regions=False)


def build(p_row, d_blocks, num_tokens, rank=3):
    region = lambda base, page: mc.TransferRegion(LAYER, 1, base, page, page, 0)
    ready = [("d", mc.SendBlockMeta("p", "x", [p_row], asyncio.Event()))]
    return asyncio.run(producer()._build_transfer_params(ready, pull(rank, d_blocks, num_tokens),
                                                         [region(P_BASE, P_PAGE)], [region(D_BASE, D_PAGE)],
                                                         stage_mla=True))


@unittest.skipIf(mc is None, "needs the vLLM image")
class AsymmetricKdaStateTest(unittest.TestCase):
    def test_conv_state_splits_per_projection(self):
        self.assertEqual(sizes(1), [CH * ROWS * 2] * 3 + [HEADS * 128 * 128 * 2])
        self.assertEqual(sizes(8), [CH // 8 * ROWS * 2] * 3 + [HEADS // 8 * 128 * 128 * 2])

    def test_final_states_and_rank_slices(self):
        # 13,000 tokens: P's final state is column cdiv(13000, 6400) - 1 = 2; D lists [CoW, final, 4 speculative].
        rank, row = 3, [0, 0, 77, 78, 79, 80, 81]
        src, dst, lengths, err_reqs, err = build(row, [5, 6, 7, 8, 9, 10], 13_000, rank)
        self.assertEqual((err_reqs, err), ([], None))
        p0, d0 = P_BASE + 77 * P_PAGE, D_BASE + 6 * D_PAGE
        conv_row = ROWS * 2
        # Rank r owns channels r*1536.. of each of q, k, v and heads 12r..12r+11 of the recurrent state.
        want_src = [p0 + (s * CH + rank * CH // 8) * conv_row for s in range(3)]
        want_src.append(p0 + 3 * CH * conv_row + rank * HEADS // 8 * 128 * 128 * 2)
        want_dst = [d0 + s * CH // 8 * conv_row for s in range(3)] + [d0 + 3 * CH // 8 * conv_row]
        self.assertEqual((src, dst, lengths), (want_src, want_dst, sizes(8)))

    def test_destination_without_cow(self):
        src, dst, _, err_reqs, _ = build([0, 0, 77, 78, 79, 80, 81], [6, 7, 8, 9, 10], 12_801, rank=0)
        self.assertEqual((err_reqs, src[0], dst[0]), ([], P_BASE + 77 * P_PAGE, D_BASE + 6 * D_PAGE))

    def test_missing_final_state_fails_the_request(self):
        for row, d_blocks, num_tokens in (([0, 0, 0, 78, 79, 80, 81], [6, 7, 8, 9, 10], 13_000),
                                          ([0, 0, 77, 78, 79, 80, 81], [6, 7, 8, 9], 13_000),
                                          ([0, 77], [6, 7, 8, 9, 10], 13_000)):
            src, _, _, err_reqs, err = build(row, d_blocks, num_tokens)
            self.assertEqual((src, err_reqs, err), ([], ["d"], "Asymmetric DCP lacks the final Mamba state"))


@unittest.skipIf(mc is None, "needs the vLLM image")
class FailedPullReleaseTest(unittest.TestCase):
    def test_failed_pulls_release_producer_blocks(self):
        # Two D ranks pull; both fail. P must free the request then, not at the 480 s abort timeout.
        import msgspec
        send = mc.SendBlockMeta("p-req", "x", [[1]], asyncio.Event())
        send.ready.set()
        p = NS(transfer_topo=NS(handshake_target_ranks=lambda tp: [0, 1], total_num_kv_heads=1), tp_rank=0, pp_size=1,
               opaque_packed_storages=set(), tp_size=1, use_mla=True, kv_cache_config=NS(has_mamba_layers=True),
               reqs_need_send={"x": send}, dcp_staging_bytes=0, dcp_size=1, finished_sending_reqs=set(),
               _encoder=msgspec.msgpack.Encoder(), _producer_cache_is_replicated=lambda: False,
               _prepare_transfer_regions=lambda meta: ([], [], None),
               _build_transfer_params=mock.AsyncMock(return_value=([], [], [], ["d"], "boom")))
        p.resolve_need_send = MethodType(mc.MooncakeConnectorWorker.resolve_need_send, p)
        sock = NS(send_multipart=mock.AsyncMock())
        for rank, released in ((0, set()), (1, {"p-req"})):
            meta = mc.MooncakeXferMetadata(remote_hostname="d", remote_port=1, remote_tp_size=2, remote_tp_rank=rank,
                                           req_blocks={"d": ("x", [[2]])}, kv_caches_base_addr=[], block_lens=[],
                                           kv_block_lens=[])
            asyncio.run(mc.MooncakeConnectorWorker.send_kv_to_decode(p, b"id", sock, meta))
            self.assertEqual(p.finished_sending_reqs, released)
        self.assertEqual((p.reqs_need_send, send.sending), ({}, 0))


if __name__ == "__main__":
    unittest.main()
