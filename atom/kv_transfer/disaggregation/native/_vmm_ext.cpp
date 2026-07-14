// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Minimal HIP VMM bridge for cross-process single-node (XGMI) KV sharing:
// allocate an exportable VMM buffer, export/import its POSIX fd, grant peer
// access, and copy. Reliable cross-process where legacy hipIpc is not, and does
// not require the source GPU to be in the consumer's visible set.
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

#define HIPCK(expr)                                                          \
  do {                                                                       \
    hipError_t _e = (expr);                                                  \
    if (_e != hipSuccess)                                                    \
      throw std::runtime_error(std::string(#expr) + ": " +                  \
                               hipGetErrorString(_e));                       \
  } while (0)

namespace {

struct Region {
  hipMemGenericAllocationHandle_t handle;
  void *ptr;
  size_t size;
};

std::unordered_map<int64_t, Region> g_regions;
int64_t g_next_id = 0;

// handle_type selects scale-up (POSIX fd, single-node/XGMI) vs scale-out
// (fabric handle, cross-node/IFOE on gfx1250).
hipMemAllocationProp make_prop(int device, bool fabric = false) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  prop.requestedHandleType = fabric ? hipMemHandleTypeFabric
                                    : hipMemHandleTypePosixFileDescriptor;
  return prop;
}

size_t round_up_to_granularity(size_t nbytes, int device, bool fabric = false) {
  hipMemAllocationProp prop = make_prop(device, fabric);
  size_t gran = 0;
  HIPCK(hipMemGetAllocationGranularity(&gran, &prop,
                                       hipMemAllocationGranularityMinimum));
  return ((nbytes + gran - 1) / gran) * gran;
}

// Grant `device` read/write peer access to a mapped range.
void grant_access(void *ptr, size_t size, int device) {
  hipMemAccessDesc desc{};
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = device;
  desc.flags = hipMemAccessFlagsProtReadWrite;
  HIPCK(hipMemSetAccess(ptr, size, &desc, 1));
}

Region &region(int64_t id) {
  auto it = g_regions.find(id);
  if (it == g_regions.end())
    throw std::runtime_error("unknown VMM region id " + std::to_string(id));
  return it->second;
}

} // namespace

bool vmm_supported(int device) {
  int value = 0;
  hipDeviceGetAttribute(
      &value, hipDeviceAttributeVirtualMemoryManagementSupported, device);
  return value != 0;
}

// Whether the device supports exporting VMM handles as fabric handles
// (cross-node / IFOE). False on gfx950 (scale-up only), true on gfx1250.
bool vmm_supported_fabric(int device) {
  int value = 0;
  hipDeviceGetAttribute(&value, hipDeviceAttributeHandleTypeFabricSupported,
                        device);
  return value != 0;
}

// Allocate an exportable VMM buffer on `device`, map it and grant `device`
// access. `fabric` picks the shareable-handle type (fd vs fabric). Returns an
// opaque region id.
int64_t vmm_alloc(int64_t nbytes, int device, bool fabric = false) {
  HIPCK(hipSetDevice(device));
  size_t size =
      round_up_to_granularity(static_cast<size_t>(nbytes), device, fabric);
  hipMemAllocationProp prop = make_prop(device, fabric);
  Region r{};
  r.size = size;
  HIPCK(hipMemCreate(&r.handle, size, &prop, 0));
  HIPCK(hipMemAddressReserve(&r.ptr, size, 0, 0, 0));
  HIPCK(hipMemMap(r.ptr, size, 0, r.handle, 0));
  grant_access(r.ptr, size, device);
  int64_t id = g_next_id++;
  g_regions[id] = r;
  return id;
}

// Export the region's handle as a 64-byte fabric handle (to send over TCP to a
// remote node); node-independent, unlike the POSIX fd.
pybind11::bytes vmm_export_fabric(int64_t id) {
  hipMemFabricHandle_t h{};
  HIPCK(hipMemExportToShareableHandle(&h, region(id).handle,
                                      hipMemHandleTypeFabric, 0));
  return pybind11::bytes(reinterpret_cast<const char *>(&h), sizeof(h));
}

// Import a peer's fabric handle (64 raw bytes), map it on `device` and grant
// `device` peer access. `nbytes` must match the producer's requested size.
int64_t vmm_import_fabric(pybind11::bytes handle, int64_t nbytes, int device) {
  HIPCK(hipSetDevice(device));
  size_t size =
      round_up_to_granularity(static_cast<size_t>(nbytes), device, true);
  hipMemFabricHandle_t h{};
  std::string s = handle;
  if (s.size() != sizeof(h))
    throw std::runtime_error("bad fabric handle size");
  std::memcpy(&h, s.data(), sizeof(h));
  Region r{};
  r.size = size;
  HIPCK(hipMemImportFromShareableHandle(&r.handle, &h, hipMemHandleTypeFabric));
  HIPCK(hipMemAddressReserve(&r.ptr, size, 0, 0, 0));
  HIPCK(hipMemMap(r.ptr, size, 0, r.handle, 0));
  grant_access(r.ptr, size, device);
  int64_t id = g_next_id++;
  g_regions[id] = r;
  return id;
}

// Export the region's handle as a POSIX file descriptor (to send over a UNIX
// socket via SCM_RIGHTS).
int vmm_export_fd(int64_t id) {
  int fd = -1;
  HIPCK(hipMemExportToShareableHandle(
      &fd, region(id).handle, hipMemHandleTypePosixFileDescriptor, 0));
  return fd;
}

// Import a peer's fd, map it on `device` and grant `device` peer access.
// `nbytes` must match the producer's requested size (rounded identically).
int64_t vmm_import(int fd, int64_t nbytes, int device) {
  HIPCK(hipSetDevice(device));
  size_t size = round_up_to_granularity(static_cast<size_t>(nbytes), device);
  Region r{};
  r.size = size;
  HIPCK(hipMemImportFromShareableHandle(
      &r.handle, reinterpret_cast<void *>(static_cast<intptr_t>(fd)),
      hipMemHandleTypePosixFileDescriptor));
  HIPCK(hipMemAddressReserve(&r.ptr, size, 0, 0, 0));
  HIPCK(hipMemMap(r.ptr, size, 0, r.handle, 0));
  grant_access(r.ptr, size, device);
  int64_t id = g_next_id++;
  g_regions[id] = r;
  return id;
}

// Wrap the first `nbytes` of the mapped region as a (non-owning) uint8 tensor.
int64_t vmm_ptr(int64_t id) {
  return reinterpret_cast<int64_t>(region(id).ptr);
}

torch::Tensor vmm_tensor(int64_t id, int64_t nbytes, int device) {
  auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(
      torch::kCUDA, device);
  return torch::from_blob(region(id).ptr, {nbytes}, opts);
}

// Device-to-device copy between two raw device pointers (peer-mapped ok). Used
// by the connector to gather/scatter KV blocks over XGMI (scale-up / fd path).
// NOTE: hipMemcpy is unsafe on a *fabric* mapping (it can livelock and wedge the
// GPU/vPOD on gfx1250) -- use vmm_copy_kernel for the fabric / scale-out path.
void vmm_copy(int64_t dst_ptr, int64_t src_ptr, int64_t nbytes) {
  HIPCK(hipMemcpy(reinterpret_cast<void *>(dst_ptr),
                  reinterpret_cast<void *>(src_ptr),
                  static_cast<size_t>(nbytes), hipMemcpyDeviceToDevice));
}

// Explicit copy kernel (no hipMemcpy) for copying to/from a *fabric* mapping.
// A shader-driven copy resolves the fabric peer where hipMemcpy/SDMA does not.
// Defined in _vmm_kernels.cu (device code -> compiled by hipcc, not g++).
void vmm_copy_kernel(int64_t dst_ptr, int64_t src_ptr, int64_t nbytes);

void vmm_free(int64_t id) {
  auto it = g_regions.find(id);
  if (it == g_regions.end())
    return;
  Region &r = it->second;
  hipMemUnmap(r.ptr, r.size);
  hipMemAddressFree(r.ptr, r.size);
  hipMemRelease(r.handle);
  g_regions.erase(it);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vmm_supported", &vmm_supported);
  m.def("vmm_supported_fabric", &vmm_supported_fabric);
  m.def("vmm_alloc", &vmm_alloc, pybind11::arg("nbytes"),
        pybind11::arg("device"), pybind11::arg("fabric") = false);
  m.def("vmm_export_fd", &vmm_export_fd);
  m.def("vmm_import", &vmm_import);
  m.def("vmm_export_fabric", &vmm_export_fabric);
  m.def("vmm_import_fabric", &vmm_import_fabric);
  m.def("vmm_ptr", &vmm_ptr);
  m.def("vmm_tensor", &vmm_tensor);
  m.def("vmm_copy", &vmm_copy);
  m.def("vmm_copy_kernel", &vmm_copy_kernel);
  m.def("vmm_free", &vmm_free);
}
