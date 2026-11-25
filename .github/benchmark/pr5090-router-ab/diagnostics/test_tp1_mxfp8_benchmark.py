from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tp1_mxfp8_benchmark import kernel_resources, validate_plan


class ResourceQueryTest(unittest.TestCase):
    def test_module_handle_uses_module_api_and_consumes_query_error(self):
        kernel = SimpleNamespace(
            n_regs=202,
            n_spills=0,
            function=0x123456789,
            metadata=SimpleNamespace(shared=68576, num_warps=2),
        )
        for error in (0, 98):
            calls = []

            def occupancy(blocks, function, threads, shared):
                calls.append((function.value, threads.value, shared.value))
                blocks._obj.value = 2
                return error

            hip = SimpleNamespace(
                hipModuleOccupancyMaxActiveBlocksPerMultiprocessor=occupancy,
                hipGetLastError=lambda: calls.append("error-consumed") or error,
            )
            with patch("tp1_mxfp8_benchmark.ctypes.CDLL", return_value=hip):
                result = kernel_resources(kernel)
            self.assertEqual(calls, [(kernel.function, 128, 0), "error-consumed"])
            self.assertEqual(result["max_blocks_per_cu"], None if error else 2)
            self.assertEqual(result["occupancy_query_rc"], error)

    def test_checkpoint_shape_and_launch_geometry(self):
        validate_plan()


if __name__ == "__main__":
    unittest.main()
