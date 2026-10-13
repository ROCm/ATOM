/* Diagnostic interposer: report HSA errors only inside the original warmup.
 * Calls the real API once, with unchanged arguments and return status.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static void *runtime;
static pthread_once_t runtime_once = PTHREAD_ONCE_INIT;

static void open_runtime(void) {
  /* Torch can load HSA with local symbol scope, outside RTLD_NEXT's scope.
   * Reuse the already loaded runtime; never load a second HSA build.
   */
  runtime = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_NOLOAD);
  if (!runtime) {
    fprintf(stderr, "[mooncake-hsa-evidence] Cannot find loaded HSA runtime: %s\n", dlerror());
    abort();
  }
}

static void *resolve(const char *name) {
  pthread_once(&runtime_once, open_runtime);
  void *fn = dlsym(runtime, name);
  if (!fn) {
    fprintf(stderr, "[mooncake-hsa-evidence] Missing HSA symbol: %s\n", name);
    abort();
  }
  return fn;
}

static void report(const char *api, hsa_status_t status, size_t bytes) {
  const char *enabled = getenv("MOONCAKE_TRACE_HSA_ERRORS");
  if (status == HSA_STATUS_SUCCESS || !enabled || enabled[0] != '1') return;
  struct timespec now;
  clock_gettime(CLOCK_REALTIME, &now);
  fprintf(stderr,
          "[mooncake-hsa-evidence] time=%ld.%09ld pid=%d api=%s status=0x%x bytes=%zu\n",
          now.tv_sec, now.tv_nsec, getpid(), api, (unsigned)status, bytes);
}

hsa_status_t hsa_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_code_object_reader_t reader, const char *options,
    hsa_loaded_code_object_t *loaded) {
  typedef hsa_status_t (*fn_t)(hsa_executable_t, hsa_agent_t,
      hsa_code_object_reader_t, const char *, hsa_loaded_code_object_t *);
  fn_t real = (fn_t)resolve("hsa_executable_load_agent_code_object");
  hsa_status_t status = real(executable, agent, reader, options, loaded);
  report("hsa_executable_load_agent_code_object", status, 0);
  return status;
}

hsa_status_t hsa_executable_freeze(hsa_executable_t executable, const char *options) {
  typedef hsa_status_t (*fn_t)(hsa_executable_t, const char *);
  fn_t real = (fn_t)resolve("hsa_executable_freeze");
  hsa_status_t status = real(executable, options);
  report("hsa_executable_freeze", status, 0);
  return status;
}

hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t pool, size_t size,
                                       uint32_t flags, void **ptr) {
  typedef hsa_status_t (*fn_t)(hsa_amd_memory_pool_t, size_t, uint32_t, void **);
  fn_t real = (fn_t)resolve("hsa_amd_memory_pool_allocate");
  hsa_status_t status = real(pool, size, flags, ptr);
  report("hsa_amd_memory_pool_allocate", status, size);
  return status;
}
