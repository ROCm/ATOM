// Diagnostic for the pinned Mooncake ROCm 0.3.14 ELF only. Never changes progress.
// Observe immediately before freeBatchID, after the caller has already timed out.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <vector>

static constexpr char kSymbol[] =
    "_ZN8mooncake14MultiTransport11freeBatchIDEmRKSt8functionIFvvEE";
static constexpr uintptr_t kSymbolRva = 0x35f930;
static constexpr size_t kTaskSize = 0x98;
using FreeFn = void* (*)(void*, void*, uint64_t, const void*);
static std::atomic<unsigned> snapshots{0};
static std::mutex output_mutex;

// Pread our own metadata, never KV contents. An invalid pointer becomes EIO,
// rather than making the diagnostic dereference an untrusted/stale address.
struct Reader {
    int fd = open("/proc/self/mem", O_RDONLY | O_CLOEXEC);
    ~Reader() { if (fd >= 0) close(fd); }
    bool read(uint64_t address, void* out, size_t length) const {
        return fd >= 0 && address &&
            pread(fd, out, length, static_cast<off_t>(address)) ==
                static_cast<ssize_t>(length);
    }
};

template <typename T> T field(const unsigned char* bytes, size_t offset) {
    T value;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

static std::string quote(const std::string& input) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : input) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c >= 32 && c < 127) out << c;
        else out << '?';
    }
    out << '"';
    return out.str();
}

static std::string transport_name(const Reader& reader, uint64_t object) {
    uintptr_t vtable = 0;
    Dl_info info{};
    if (!reader.read(object, &vtable, sizeof(vtable)) ||
        !dladdr(reinterpret_cast<void*>(vtable), &info) || !info.dli_sname)
        return "unknown";
    return info.dli_sname;
}

static void write_event(const std::string& event) {
    const char* directory = std::getenv("ATOMESH_MOONCAKE_PROBE_DIR");
    if (!directory || !*directory) return;
    std::lock_guard<std::mutex> lock(output_mutex);
    std::string path = std::string(directory) + "/native-timeout-" +
        std::to_string(getpid()) + ".jsonl";
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0) {
        std::fprintf(stderr, "[DEBUG-ppdep-timeout] Cannot open probe output\n");
        return;
    }
    std::string data = event + "\n";
    size_t done = 0;
    while (done < data.size()) {
        ssize_t n = write(fd, data.data() + done, data.size() - done);
        if (n <= 0) break;
        done += static_cast<size_t>(n);
    }
    close(fd);
}

static void snapshot(uint64_t id) {
    Reader reader;
    unsigned char batch[0x60]{};
    if (!reader.read(id, batch, sizeof(batch))) return;
    // Successful sync writes set status_cached before calling freeBatchID.
    if (batch[0x3a]) return;
    uint64_t begin = field<uint64_t>(batch, 0x10);
    uint64_t end = field<uint64_t>(batch, 0x18);
    if (field<uint64_t>(batch, 0) != id || end < begin ||
        (end - begin) % kTaskSize || (end - begin) / kTaskSize > 65536) {
        write_event("{\"event\":\"unsupported_batch_layout\"}");
        return;
    }
    size_t count = (end - begin) / kTaskSize;
    if (!count || snapshots.fetch_add(1) >= 3) return;
    uint64_t finished = field<uint64_t>(batch, 0x48);
    std::ostringstream out, task_details;
    size_t readable = 0, all_slices_done = 0, unpublished_done = 0;
    size_t pending = 0, failed = 0, omitted = 0, detailed = 0;
    bool first = true;
    for (size_t i = 0; i < count; ++i) {
        unsigned char task[kTaskSize]{};
        if (!reader.read(begin + i * kTaskSize, task, sizeof(task))) continue;
        ++readable;
        uint64_t slices = field<uint64_t>(task, 0);
        uint64_t success = field<uint64_t>(task, 8);
        uint64_t failures = field<uint64_t>(task, 16);
        bool complete = success + failures == slices;
        all_slices_done += complete;
        unpublished_done += complete && !task[24];
        pending += !complete;
        failed += failures != 0;
        // Include completed tasks if the batch counter disagrees, as well as
        // every unfinished/failed task. Bound output without hiding totals.
        if (complete && !failures && finished == count) continue;
        if (detailed++ >= 128) { ++omitted; continue; }
        if (!first) task_details << ',';
        first = false;
        task_details << "{\"task\":" << i << ",\"slices\":" << slices
            << ",\"success\":" << success << ",\"failed\":" << failures
            << ",\"completion_published\":" << unsigned(task[24])
            << ",\"completion_state\":" << field<uint64_t>(task, 32)
            << ",\"transferred_bytes\":" << field<uint64_t>(task, 40)
            << ",\"is_finished\":" << unsigned(task[48])
            << ",\"total_bytes\":" << field<uint64_t>(task, 56)
            << ",\"batch_id_matches\":"
            << (field<uint64_t>(task, 64) == id ? "true" : "false")
            << ",\"transport\":"
            << quote(transport_name(reader, field<uint64_t>(task, 72)))
            << ",\"request_count\":" << field<uint64_t>(task, 120);
        unsigned char request[72]{};
        if (reader.read(field<uint64_t>(task, 112), request, sizeof(request))) {
            task_details << ",\"request\":{\"opcode\":" << field<int>(request, 0)
                << ",\"source\":" << field<uint64_t>(request, 8)
                << ",\"target_segment\":" << field<uint64_t>(request, 16)
                << ",\"target_address\":" << field<uint64_t>(request, 24)
                << ",\"length\":" << field<uint64_t>(request, 32) << '}';
        }
        // Slice prefix offsets are independent of optional congestion fields.
        uint64_t sb = field<uint64_t>(task, 128), se = field<uint64_t>(task, 136);
        task_details << ",\"slice_sample\":[";
        bool first_slice = true;
        if (se >= sb && (se - sb) % 8 == 0 && (se - sb) / 8 <= 1048576) {
            for (size_t j = 0; j < (se - sb) / 8 && j < 32; ++j) {
                uint64_t sp = 0;
                unsigned char slice[120]{};
                if (!reader.read(sb + j * 8, &sp, 8) ||
                    !reader.read(sp, slice, sizeof(slice))) continue;
                if (!first_slice) task_details << ',';
                first_slice = false;
                task_details << "{\"index\":" << j
                    << ",\"source\":" << field<uint64_t>(slice, 0)
                    << ",\"length\":" << field<uint64_t>(slice, 8)
                    << ",\"status\":" << field<int>(slice, 104)
                    << ",\"task_matches\":"
                    << (field<uint64_t>(slice, 112) == begin+i*kTaskSize ? "true" : "false")
                    << '}';
            }
        }
        task_details << "]}";
    }
    struct timespec stamp{};
    clock_gettime(CLOCK_MONOTONIC, &stamp);
    out << "{\"event\":\"before_free_uncached_batch\",\"pid\":" << getpid()
        << ",\"tid\":" << syscall(SYS_gettid) << ",\"batch_id\":" << id
        << ",\"monotonic_ns\":" << uint64_t(stamp.tv_sec)*1000000000ULL + stamp.tv_nsec
        << ",\"task_count\":" << count << ",\"finished_task_count\":" << finished
        << ",\"counter_eligible\":" << unsigned(batch[0x50])
        << ",\"has_failure\":" << unsigned(batch[0x38])
        << ",\"active_completion_callbacks\":" << field<uint64_t>(batch, 0x58)
        << ",\"readable_tasks\":" << readable
        << ",\"all_slices_done_tasks\":" << all_slices_done
        << ",\"done_unpublished_tasks\":" << unpublished_done
        << ",\"pending_tasks\":" << pending << ",\"failed_tasks\":" << failed
        << ",\"omitted_task_details\":" << omitted
        << ",\"non_atomic_snapshot\":true,\"tasks\":[" << task_details.str() << "]}";
    write_event(out.str());
}

// SysV x86-64 nontrivial Status return uses a hidden first (sret) parameter.
// Keep the opaque result and original function's return untouched.
extern "C" void* probe_free(void*, void*, uint64_t, const void*) asm(
    "_ZN8mooncake14MultiTransport11freeBatchIDEmRKSt8functionIFvvEE");
extern "C" void* probe_free(void* result, void* self, uint64_t batch,
                             const void* callback) {
    struct Target { FreeFn function; bool compatible; };
    static const Target target = [](void* return_address) {
        Dl_info caller{};
        if (!dladdr(return_address, &caller) || !caller.dli_fname) std::abort();
        // Python imports the engine RTLD_LOCAL. Keep one reference for the
        // process lifetime, rather than taking the loader lock on every batch.
        void* handle = dlopen(caller.dli_fname, RTLD_NOW | RTLD_NOLOAD);
        auto original = reinterpret_cast<FreeFn>(handle ? dlsym(handle, kSymbol) : nullptr);
        if (!original || reinterpret_cast<void*>(original) == reinterpret_cast<void*>(&probe_free))
            std::abort();
        Dl_info implementation{};
        bool compatible = dladdr(reinterpret_cast<void*>(original), &implementation) &&
            reinterpret_cast<uintptr_t>(original) -
                reinterpret_cast<uintptr_t>(implementation.dli_fbase) == kSymbolRva;
#ifdef MOONCAKE_PROBE_FIXTURE
        compatible = true;
#endif
        write_event(std::string("{\"event\":\"hook_active\",\"engine_path\":") +
                    quote(implementation.dli_fname ? implementation.dli_fname : "unknown") +
                    ",\"layout_compatible\":" + (compatible ? "true}" : "false}"));
        return Target{original, compatible};
    }(__builtin_return_address(0));
    // The caller owns this still-live BatchDesc. The exact ELF layout is gated
    // above and additionally hash-checked by the launch wrapper. Successful
    // frees therefore cost one byte load, with no logging/syscall/extra polling.
    if (target.compatible && batch &&
        !__atomic_load_n(reinterpret_cast<unsigned char*>(batch) + 0x3a, __ATOMIC_ACQUIRE) &&
        std::getenv("ATOMESH_MOONCAKE_PROBE_DIR")) {
        try { snapshot(batch); }
        catch (...) { std::fprintf(stderr, "[DEBUG-ppdep-timeout] Snapshot failed\n"); }
    }
    return target.function(result, self, batch, callback);
}
