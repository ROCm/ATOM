"""Lifecycle checks for the diagnostic peer-memory anchor adapter."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

PATH = Path(__file__).resolve().parents[1] / '.github/scripts/atomesh/pd_vllm_kfd_pin.py'
SPEC = importlib.util.spec_from_file_location('kfd_pin', PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class KfdPinTest(unittest.TestCase):
    def make_adapter(self):
        obj = module.KfdPinAnchors.__new__(module.KfdPinAnchors)
        def address_range(base, size, address):
            base._obj.value, size._obj.value = 4096, 64 * 1024 * 1024
            return 0
        obj.hip = SimpleNamespace(hipMemGetAddressRange=address_range)
        obj.verbs = SimpleNamespace(ibv_reg_mr=Mock(return_value=99),
                                    ibv_dereg_mr=Mock(return_value=0),
                                    ibv_dealloc_pd=Mock(return_value=0),
                                    ibv_close_device=Mock(return_value=0))
        obj.pd, obj.context, obj.anchors = 10, 20, {}
        return obj

    def test_subranges_share_one_anchor_and_close_is_idempotent(self):
        obj = self.make_adapter()
        obj.acquire(4096, 8 * 1024 * 1024)
        obj.acquire(4096 + 8 * 1024 * 1024, 8 * 1024 * 1024)
        self.assertEqual(obj.verbs.ibv_reg_mr.call_count, 1)
        self.assertEqual(len(obj.anchors), 1)
        obj.close()
        obj.close()
        obj.verbs.ibv_dereg_mr.assert_called_once_with(99)
        obj.verbs.ibv_dealloc_pd.assert_called_once_with(10)
        obj.verbs.ibv_close_device.assert_called_once_with(20)

    def test_bad_range_never_registers(self):
        obj = self.make_adapter()
        for address, length in [(0, 1024), (4096, 0), (4096, 128 * 1024 * 1024)]:
            with self.assertRaises(ValueError):
                obj.acquire(address, length)
        obj.verbs.ibv_reg_mr.assert_not_called()

    def test_failed_deregistration_retains_anchor_and_context(self):
        obj = self.make_adapter()
        obj.acquire(4096, 1024)
        obj.verbs.ibv_dereg_mr.return_value = 5
        with self.assertRaises(OSError):
            obj.close()
        self.assertEqual(len(obj.anchors), 1)
        obj.verbs.ibv_dealloc_pd.assert_not_called()
        obj.verbs.ibv_close_device.assert_not_called()
        obj.verbs.ibv_dereg_mr.return_value = 0
        obj.close()
        self.assertEqual(obj.anchors, {})


if __name__ == '__main__':
    unittest.main()
