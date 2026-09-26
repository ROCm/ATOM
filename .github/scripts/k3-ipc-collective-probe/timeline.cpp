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

int k3_open(const char* path, const char* epoch, int device, int owner,
            uintptr_t stream, void** output) {
  return protect([&] {
    require(std::strlen(epoch) == 32, "Epoch must contain 32 bytes");
    device_check(device);
    int supported = 0;
    check(hipDeviceGetAttribute(&supported, hipDeviceAttributeCanUseStreamWaitValue, device));
    require(supported == 1, "Device does not support stream wait values");
    auto* mapping = new Mapping;
    mapping->device = device;
    mapping->owner = owner != 0;
    mapping->stream = stream;
    bool registered = false;
    bool created = false;
    int fd = -1;
    try {
      fd = ::open(path, O_RDWR | O_CLOEXEC | O_NOFOLLOW | (owner ? O_CREAT | O_EXCL : 0), 0600);
      require(fd >= 0, "Cannot open counter shared memory");
      created = owner != 0;
      if (owner) require(ftruncate(fd, kBytes) == 0, "Cannot size shared memory");
      struct stat statbuf {};
      require(fstat(fd, &statbuf) == 0 && S_ISREG(statbuf.st_mode) && statbuf.st_size == kBytes,
              "Invalid counter shared memory size/type");
      mapping->host = mmap(nullptr, kBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      require(mapping->host != MAP_FAILED, "Cannot map counter shared memory");
      ::close(fd);
      fd = -1;
      auto* tag = static_cast<char*>(mapping->host) + 64;
      if (owner) {
        __atomic_store_n(static_cast<uint64_t*>(mapping->host), 0, __ATOMIC_RELEASE);
        std::memcpy(tag, epoch, 32);
      } else {
        require(std::memcmp(tag, epoch, 32) == 0, "Counter epoch mismatch");
      }
      check(hipHostRegister(mapping->host, kBytes, hipHostRegisterMapped));
      registered = true;
      check(hipHostGetDevicePointer(&mapping->gpu, mapping->host, 0));
      *output = mapping;
    } catch (...) {
      if (registered) (void)hipHostUnregister(mapping->host);  // No commands submitted yet.
      if (mapping->host != MAP_FAILED) munmap(mapping->host, kBytes);
      if (fd >= 0) ::close(fd);
      if (created) unlink(path);
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
