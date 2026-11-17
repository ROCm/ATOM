"""moriio over the UALink fabric (kv_transfer_config backend "fabric").

A fabric import maps the whole allocation behind each registered range, so the
connector registers each tensor once instead of in 2 GiB RDMA chunks, and the
runner allocates the KV pool and per-request state from a fabric MemPool only
for moriio over the fabric. Needs the connector's imports (aiter, torch).
"""

from __future__ import annotations

import contextlib
import types
import unittest
from unittest import mock


def import_connector():
    try:
        from atom.kv_transfer.disaggregation.moriio import moriio_connector
    except Exception:
        return None
    return moriio_connector.MoRIIOConnector


class RegistrationChunksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connector = import_connector()
        if cls.connector is None:
            raise unittest.SkipTest("moriio_connector needs aiter and torch")
        import torch

        cls.torch = torch

    def chunks(self, backend, tensor, block_size_in_dim0):
        conn = object.__new__(self.connector)
        conn.backend = backend
        return conn._registration_chunks(tensor, block_size_in_dim0)

    def test_fabric_registers_a_tensor_as_one_range(self):
        from atom.kv_transfer.disaggregation import utils

        # MLA layout: 6 blocks of 4 token rows of 8 bytes, 32 bytes a block.
        kv = self.torch.zeros((24, 8), dtype=self.torch.uint8)
        with mock.patch.object(utils, "MAX_RDMA_CHUNK_BYTES", 64):
            rdma, rdma_bpc = self.chunks("rdma", kv, 4)
            fabric, fabric_bpc = self.chunks("fabric", kv, 4)
        base = kv.data_ptr()
        self.assertEqual(rdma, [(base, 64), (base + 64, 64), (base + 128, 64)])
        self.assertEqual(rdma_bpc, 2)
        self.assertEqual(fabric, [(base, 192)])
        self.assertEqual(fabric_bpc, 6)


class KvTransferMemPoolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from atom.model_engine.model_runner import ModelRunner
        except Exception:
            raise unittest.SkipTest("model_runner needs aiter and torch")
        cls.runner_cls = ModelRunner

    def runner(self, kv_cfg):
        runner = object.__new__(self.runner_cls)
        runner.config = types.SimpleNamespace(kv_transfer_config=kv_cfg)
        return runner

    def test_only_moriio_over_the_fabric_allocates_from_a_fabric_pool(self):
        for cfg in (
            None,
            {},
            {"kv_connector": "moriio"},
            {"kv_connector": "moriio", "backend": "rdma"},
            {"kv_connector": "mooncake", "backend": "fabric"},
        ):
            with self.subTest(cfg=cfg):
                self.assertIsInstance(
                    self.runner(cfg)._kv_transfer_mem_pool(), contextlib.nullcontext
                )
        runner = self.runner({"kv_connector": "moriio", "backend": "Fabric"})
        with mock.patch(
            "mori.io.make_fabric_mem_pool", return_value="pool"
        ) as make, mock.patch("torch.cuda.use_mem_pool", return_value="ctx") as use:
            self.assertEqual(runner._kv_transfer_mem_pool(), "ctx")
        make.assert_called_once_with()
        use.assert_called_once_with("pool")
        # The pool outlives the context: the KV tensors are carved from it.
        self.assertEqual(runner._kv_fabric_pool, "pool")


if __name__ == "__main__":
    unittest.main()
