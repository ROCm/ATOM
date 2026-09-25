#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define HIP(call) do { hipError_t rc = (call); if (rc != hipSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(rc)); std::exit(1); \
} } while (0)

__global__ void spin(unsigned long long cycles) {
  auto start = clock64();
  while (clock64() - start < cycles) {}
}

using Clock = std::chrono::steady_clock;
double ms(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

int main() {
  HIP(hipSetDevice(0));
  hipDeviceProp_t properties;
  HIP(hipGetDeviceProperties(&properties, 0));
  std::fprintf(stderr, "device=%s arch=%s clock_khz=%d\n",
               properties.name, properties.gcnArchName, properties.clockRate);
  hipStream_t event_stream, work_stream;
  HIP(hipStreamCreateWithFlags(&event_stream, hipStreamNonBlocking));
  HIP(hipStreamCreateWithFlags(&work_stream, hipStreamNonBlocking));
  auto cycles = static_cast<unsigned long long>(properties.clockRate) * 40;
  hipLaunchKernelGGL(spin, dim3(1), dim3(1), 0, work_stream, cycles);
  HIP(hipGetLastError());
  HIP(hipStreamSynchronize(work_stream));
  std::puts("mode,iteration,destroy_ms,remaining_work_ms,total_ms");
  for (int mode = 0; mode < 3; ++mode) {
    const char* names[] = {"normal_ready", "ipc_unrecorded", "ipc_ready"};
    for (int iteration = 0; iteration < 8; ++iteration) {
      hipEvent_t event;
      unsigned flags = hipEventDisableTiming;
      if (mode) flags |= hipEventInterprocess;
      HIP(hipEventCreateWithFlags(&event, flags));
      if (mode != 1) {
        HIP(hipEventRecord(event, event_stream));
        HIP(hipEventSynchronize(event));
        HIP(hipEventQuery(event));
      }
      hipLaunchKernelGGL(spin, dim3(1), dim3(1), 0, work_stream, cycles);
      HIP(hipGetLastError());
      auto begin = Clock::now();
      HIP(hipEventDestroy(event));
      auto destroyed = Clock::now();
      HIP(hipStreamSynchronize(work_stream));
      auto drained = Clock::now();
      std::printf("%s,%d,%.6f,%.6f,%.6f\n", names[mode], iteration,
                  ms(begin, destroyed), ms(destroyed, drained), ms(begin, drained));
      std::fflush(stdout);
    }
  }
  HIP(hipStreamDestroy(event_stream));
  HIP(hipStreamDestroy(work_stream));
}
