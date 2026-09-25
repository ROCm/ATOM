#include <pybind11/pybind11.h>
#include <torch/csrc/cuda/Event.h>

#include <cstring>
#include <limits>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  auto torch = py::module_::import("torch");
  py::object event_cls = torch.attr("cuda").attr("Event");
  py::object event_base = torch.attr("_C").attr("_CudaEventBase");
  module.def("from_ipc_handle", [event_cls, event_base](int device, py::bytes bytes) {
    std::string serialized = bytes;
    if (serialized.size() != sizeof(hipIpcEventHandle_t)) {
      throw py::value_error("Invalid HIP IPC event handle size");
    }
    if (device < 0 || device > std::numeric_limits<c10::DeviceIndex>::max()) {
      throw py::value_error("Expected a nonnegative CUDA device index");
    }
    py::object result = event_cls();
    if (!py::isinstance(result, event_base) ||
        Py_TYPE(result.ptr())->tp_basicsize < sizeof(THCPEvent)) {
      throw py::type_error("Expected the installed torch CUDA event layout");
    }
    auto* event = reinterpret_cast<THCPEvent*>(result.ptr());
    if (event->cuda_event.isCreated()) {
      throw py::value_error("Expected a new unrecorded event");
    }
    hipIpcEventHandle_t handle{};
    std::memcpy(&handle, serialized.data(), sizeof(handle));
    {
      py::gil_scoped_release release;
      event->cuda_event = at::cuda::CUDAEvent(device, &handle);
    }
    return result;
  });
}
