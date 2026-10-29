"""Lifecycle checks for the diagnostic peer-memory anchor adapter."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys
import tempfile
from unittest.mock import Mock, patch

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

    def test_private_probe_distinguishes_fixed_and_affected_drivers(self):
        size = 64 * 1024**2
        for drop, needed in [(0, False), (size, True)]:
            with self.subTest(drop=drop):
                obj = self.make_adapter()
                def alloc(pointer, size):
                    pointer._obj.value = 0x123000
                    return 0
                obj.hip.hipMalloc = Mock(side_effect=alloc)
                obj.hip.hipFree = Mock(return_value=0)
                obj.memory_info = Mock(side_effect=[(4 * size, 8 * size), (4 * size - drop, 8 * size)])
                engine = SimpleNamespace(batch_register_memory=Mock(return_value=0),
                                         batch_unregister_memory=Mock(return_value=0))
                self.assertEqual(obj.probe_driver(engine)['needed'], needed)
                engine.batch_register_memory.assert_called_once_with([0x123000], [size])
                engine.batch_unregister_memory.assert_called_once_with([0x123000])
                obj.hip.hipFree.assert_called_once()
                self.assertEqual(obj.anchors, {})

    def test_probe_never_frees_memory_with_a_live_data_mr(self):
        obj = self.make_adapter()
        def alloc(pointer, size):
            pointer._obj.value = 0x123000
            return 0
        obj.hip.hipMalloc = Mock(side_effect=alloc)
        obj.hip.hipFree = Mock(return_value=0)
        obj.memory_info = Mock(return_value=(256 * 1024**2, 512 * 1024**2))
        engine = SimpleNamespace(batch_register_memory=Mock(return_value=0),
                                 batch_unregister_memory=Mock(return_value=-1))
        with self.assertRaisesRegex(RuntimeError, 'unregistration failed'):
            obj.probe_driver(engine)
        obj.hip.hipFree.assert_not_called()

    def test_outer_connector_pins_before_all_native_registrations(self):
        events = []
        class Tensor:
            is_cuda = True
            def untyped_storage(self):
                return SimpleNamespace(data_ptr=lambda: 4096, nbytes=lambda: 8192)
        class MooncakeConnector:
            connector_worker = SimpleNamespace(engine=object())
        for needed in (True, False):
            with self.subTest(needed=needed), tempfile.TemporaryDirectory() as tmp:
                pins = SimpleNamespace(anchors={}, probe_driver=lambda engine: {'needed': needed},
                                       memory_info=lambda: (1024**3, 2 * 1024**3))
                def acquire(address, length):
                    events.append('pin')
                    pins.anchors[address] = {'allocation_bytes': length}
                pins.acquire = Mock(side_effect=acquire)
                def native(caches):
                    events.extend(['transfer', 'store'])
                    return 123
                group = SimpleNamespace(_connectors=[MooncakeConnector()], register_kv_caches=native)
                torch = SimpleNamespace(Tensor=Tensor, cuda=SimpleNamespace(memory_reserved=lambda: 0))
                events.clear()
                with patch.dict(sys.modules, {'torch': torch}), patch.object(module, 'KfdPinAnchors', return_value=pins):
                    module.bind_connector(group, 'rdma0', tmp)
                    self.assertEqual(group.register_kv_caches({'kv': Tensor()}), 123)
                self.assertEqual(events, (['pin'] if needed else []) + ['transfer', 'store'])
                self.assertEqual(group._atomesh_kfd_adapter['status'], 'verified')
                self.assertIs(group._atomesh_kfd_pins, pins)

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
