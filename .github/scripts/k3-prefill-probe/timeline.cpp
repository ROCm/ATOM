// Experimental single-stream completion counter. No HIP event objects.
#include <hip/hip_runtime_api.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
constexpr size_t kBytes = 4096;
constexpr uint64_t kLimit = (uint64_t{1} << 63) - 1;
static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr));
thread_local std::string last_error;

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void check(hipError_t result) {
  if (result != hipSuccess) throw std::runtime_error(hipGetErrorString(result));
}
void device_check(int expected) {
  int current = -1;
  check(hipGetDevice(&current));
  require(current == expected, "Wrong current HIP device");
}
void check_not_capturing(uintptr_t stream) {
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  check(hipStreamIsCapturing(reinterpret_cast<hipStream_t>(stream), &status));
  require(status == hipStreamCaptureStatusNone, "Counter probe does not support capture");
}
struct Mapping {
  void* host = MAP_FAILED;
  void* gpu = nullptr;
  int device = -1;
  bool owner = false;
  uintptr_t stream = 0;
  uint64_t generation = 0;
  std::mutex mutex;
  // There is deliberately no implicit unregister/unmap in the destructor.
  // Fatal paths are handled by the process-group supervisor. Explicit close
  // is legal only after the two-sided GPU drain protocol has completed.
};
template <class F> int protect(F function) {
  try {
    function();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  }
}
uint64_t observed(Mapping* mapping) {
  return __atomic_load_n(static_cast<uint64_t*>(mapping->host), __ATOMIC_ACQUIRE);
}
void valid_generation(uint64_t generation) {
  require(generation > 0 && generation <= kLimit, "Invalid counter generation");
}
}  // namespace

extern "C" {
const char* k3_error() { return last_error.c_str(); }

// FD-backed variant for the process-lifetime single-P experiment. The Python
// owner keeps its memfd open so peers can duplicate it through /proc. No named
// shared-memory object survives a process-group kill.
int k3_open_fd(int descriptor, const char* identity_tag, const char* scope,
               const char* device_uuid, int device, int owner,
               uintptr_t stream, void** output) {
  return protect([&] {
    require(std::strlen(identity_tag) == 32, "Identity tag must contain 32 bytes");
    require(std::strlen(scope) == 32, "Scope must contain 32 bytes");
    require(std::strlen(device_uuid) > 0 && std::strlen(device_uuid) < 128,
            "Invalid physical device UUID");
    require(owner == 0 || owner == 1, "Invalid mapping role");
    device_check(device);
    int supported = 0;
    check(hipDeviceGetAttribute(&supported, hipDeviceAttributeCanUseStreamWaitValue, device));
    require(supported == 1, "Device does not support stream wait values");
    auto* mapping = new Mapping;
    mapping->device = device;
    mapping->owner = owner != 0;
    mapping->stream = stream;
    bool registered = false;
    try {
      struct stat statbuf {};
      require(fstat(descriptor, &statbuf) == 0 && S_ISREG(statbuf.st_mode)
              && statbuf.st_size == kBytes, "Invalid counter descriptor size/type");
      const int seals = fcntl(descriptor, F_GET_SEALS);
      require(seals >= 0 && (seals & (F_SEAL_SHRINK | F_SEAL_GROW))
                            == (F_SEAL_SHRINK | F_SEAL_GROW), "Counter size must be sealed");
      mapping->host = mmap(nullptr, kBytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
      require(mapping->host != MAP_FAILED, "Cannot map counter descriptor");
      auto* tag = static_cast<char*>(mapping->host) + 64;
      auto* scope_tag = tag + 32;
      auto* uuid_tag = tag + 64;
      char expected_uuid[128] {};
      std::memcpy(expected_uuid, device_uuid, std::strlen(device_uuid));
      if (owner) {
        __atomic_store_n(static_cast<uint64_t*>(mapping->host), 0, __ATOMIC_RELEASE);
        std::memcpy(tag, identity_tag, 32);
        std::memcpy(scope_tag, scope, 32);
        std::memcpy(uuid_tag, expected_uuid, sizeof(expected_uuid));
      } else {
        require(std::memcmp(tag, identity_tag, 32) == 0, "Counter identity mismatch");
        require(std::memcmp(scope_tag, scope, 32) == 0, "Counter scope mismatch");
        require(std::memcmp(uuid_tag, expected_uuid, sizeof(expected_uuid)) == 0,
                "Counter physical device UUID mismatch");
      }
      check(hipHostRegister(mapping->host, kBytes, hipHostRegisterMapped));
      registered = true;
      check(hipHostGetDevicePointer(&mapping->gpu, mapping->host, 0));
      *output = mapping;
    } catch (...) {
      // A failed service registration poisons the Python backend. Do not
      // unregister here: it can synchronize other live streams on this device.
      // Keep the pinned mapping until the bounded experiment kills/exits the
      // process; the poisoned registry cannot allocate another failed channel.
      if (registered) throw;
      if (mapping->host != MAP_FAILED) munmap(mapping->host, kBytes);
      delete mapping;
      throw;
    }
  });
}

int k3_record(void* pointer, uintptr_t stream, uint64_t* generation) {
  return protect([&] {
    auto* mapping = static_cast<Mapping*>(pointer);
    std::lock_guard<std::mutex> guard(mapping->mutex);
    device_check(mapping->device);
    require(mapping->owner && stream == mapping->stream, "Counter has one fixed writer stream");
    check_not_capturing(stream);
    require(mapping->generation < kLimit, "Counter generation exhausted");
    uint64_t next = mapping->generation + 1;
    check(hipStreamWriteValue64(reinterpret_cast<hipStream_t>(stream), mapping->gpu, next, 0));
    // Publish only after successful enqueue, under the same lock as assignment.
    mapping->generation = next;
    *generation = next;
  });
}

int k3_wait(void* pointer, uintptr_t stream, uint64_t generation) {
  return protect([&] {
    auto* mapping = static_cast<Mapping*>(pointer);
    std::lock_guard<std::mutex> guard(mapping->mutex);
    device_check(mapping->device);
    valid_generation(generation);
    check_not_capturing(stream);
    check(hipStreamWaitValue64(reinterpret_cast<hipStream_t>(stream), mapping->gpu,
                              generation, hipStreamWaitValueGte, UINT64_MAX));
  });
}

int k3_query(void* pointer, uint64_t* value) {
  return protect([&] { *value = observed(static_cast<Mapping*>(pointer)); });
}

int k3_close_after_drain(void* pointer, uint64_t expected) {
  return protect([&] {
    auto* mapping = static_cast<Mapping*>(pointer);
    device_check(mapping->device);
    require(observed(mapping) >= expected, "Counter has unfinished writes at close");
    check(hipHostUnregister(mapping->host));
    require(munmap(mapping->host, kBytes) == 0, "Cannot unmap counter");
    delete mapping;
  });
}
}
