/* Experimental Store allocator override. Only the two configured allocation
 * sizes are changed; named/shared/fixed mappings retain libc behavior. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <linux/mempolicy.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

static const size_t thp_size = 2UL * 1024 * 1024;

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif

/* Temporary, opt-in failure-boundary evidence. Read while the VMA still exists. */
static void memory_snapshot(void *address, size_t length, const char *phase) {
    const char *enabled = getenv("ATOMESH_THP_DIAGNOSTICS");
    if (!enabled || strcmp(enabled, "1") != 0)
        return;
    fprintf(stderr, "[DEBUG-mooncake-memory] pid=%ld phase=%s range=%p-%p\n",
            (long)getpid(), phase, address, (char *)address + length);
    const char *paths[] = {
        "/proc/self/status", "/proc/self/numa_maps", "/proc/self/smaps",
        "/proc/meminfo", "/proc/buddyinfo", "/proc/vmstat",
        "/sys/devices/system/node/node0/meminfo",
        "/sys/devices/system/node/node1/meminfo",
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset.mems.effective",
        "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.events"
    };
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); ++i) {
        FILE *file = fopen(paths[i], "r");
        if (!file) {
            fprintf(stderr, "[DEBUG-mooncake-memory] path=%s errno=%d\n", paths[i], errno);
            continue;
        }
        char line[2048];
        int in_range = 0;
        while (fgets(line, sizeof(line), file)) {
            unsigned long start, end;
            int selected = 1;
            if (i == 0)
                selected = strncmp(line, "Cpus_allowed", 12) == 0 ||
                           strncmp(line, "Mems_allowed", 12) == 0;
            else if (i == 1)
                selected = sscanf(line, "%lx", &start) == 1 &&
                           start >= (uintptr_t)address &&
                           start < (uintptr_t)address + length;
            else if (i == 2) {
                if (sscanf(line, "%lx-%lx", &start, &end) == 2)
                    in_range = start < (uintptr_t)address + length &&
                               end > (uintptr_t)address;
                selected = in_range;
            } else if (i == 5)
                selected = strncmp(line, "compact_", 8) == 0 ||
                           strncmp(line, "thp_", 4) == 0 ||
                           strncmp(line, "pgscan_", 7) == 0 ||
                           strncmp(line, "pgsteal_", 8) == 0;
            if (selected)
                fprintf(stderr, "[DEBUG-mooncake-memory] %s %s", paths[i], line);
        }
        fclose(file);
    }
}

static int populate_thp(void *address, size_t length, const char *kind) {
    /* Avoid synchronous huge-page compaction on every first-touch fault.
     * Populate movable base pages before requesting the final THP backing. */
    const int advice[] = {MADV_NOHUGEPAGE, MADV_POPULATE_WRITE,
                          MADV_HUGEPAGE, MADV_COLLAPSE};
    const char *phases[] = {"base-pages", "populate", "huge-pages", "collapse"};
    for (size_t i = 0; i < sizeof(advice) / sizeof(advice[0]); ++i) {
        if (advice[i] == MADV_COLLAPSE)
            memory_snapshot(address, length, "before-collapse");
        for (int attempt = 1; ; ++attempt) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            fprintf(stderr, "[mooncake-store-thp] kind=%s phase=%s bytes=%zu attempt=%d begin\n",
                    kind, phases[i], length, attempt);
            int rc = madvise(address, length, advice[i]);
            int error = errno;
            clock_gettime(CLOCK_MONOTONIC, &end);
            double seconds = (end.tv_sec - start.tv_sec) +
                             (end.tv_nsec - start.tv_nsec) / 1e9;
            fprintf(stderr, "[mooncake-store-thp] kind=%s phase=%s bytes=%zu attempt=%d rc=%d errno=%d seconds=%.3f\n",
                    kind, phases[i], length, attempt, rc, rc == 0 ? 0 : error, seconds);
            if (rc == 0)
                break;
            FILE *memory = fopen("/proc/self/smaps_rollup", "r");
            if (memory) {
                char line[256];
                unsigned long huge_kib;
                while (fgets(line, sizeof(line), memory)) {
                    if (sscanf(line, "AnonHugePages: %lu kB", &huge_kib) == 1)
                        fprintf(stderr, "[mooncake-store-thp] failed-phase=%s process_anon_huge_kib=%lu\n",
                                phases[i], huge_kib);
                }
                fclose(memory);
            }
            /* Collapse can leave most of a large range backed by THP even
             * when it reports a transient failure. Retry only that phase. */
            if (advice[i] == MADV_COLLAPSE && attempt < 4 &&
                (error == EAGAIN || error == ENOMEM)) {
                usleep(100000);
                continue;
            }
            memory_snapshot(address, length, "failed-collapse");
            errno = error;
            return rc;
        }
    }
    return 0;
}

static int matches_size(const char *name, size_t size) {
    const char *value = getenv(name);
    if (!value || !*value)
        return 0;
    char *end;
    errno = 0;
    unsigned long long configured = strtoull(value, &end, 10);
    return errno == 0 && *end == '\0' && configured == size &&
           size > 0 && size % thp_size == 0;
}

void *aligned_alloc(size_t alignment, size_t size) {
    void *(*original)(size_t, size_t) = dlsym(RTLD_NEXT, "aligned_alloc");
    if (!original) {
        errno = ENOSYS;
        return NULL;
    }
    int selected = alignment == 64 &&
                   matches_size("ATOMESH_THP_LOCAL_BUFFER_BYTES", size);
    void *address = original(selected ? thp_size : alignment, size);
    if (selected && address) {
        int rc = madvise(address, size, MADV_HUGEPAGE);
        if (rc == 0)
            rc = populate_thp(address, size, "local-buffer");
        fprintf(stderr, "[mooncake-store-thp] local-buffer=%zu address=%p advice=%d\n",
                size, address, rc);
        if (rc != 0) {
            int error = errno;
            free(address);
            errno = error;
            return NULL;
        }
    }
    return address;
}

long mbind(void *address, unsigned long length, int mode,
           const unsigned long *nodes, unsigned long maxnode,
           unsigned int flags) {
    int selected = mode == MPOL_BIND && (uintptr_t)address % thp_size == 0 &&
                   matches_size("ATOMESH_THP_NUMA_REGION_BYTES", length);
    const char *fallback = getenv("ATOMESH_THP_NUMA_FALLBACK");
    const char *interleave = getenv("ATOMESH_THP_NUMA_INTERLEAVE");
    int effective_mode = selected && fallback && fallback[0] == '1' &&
                         fallback[1] == '\0' ? MPOL_PREFERRED_MANY : mode;
    /* The harness verifies NIC NUMA nodes [0, 1]. Equal base-page placement
     * lets collapse choose either node instead of one exhausted local node. */
    const unsigned long interleave_nodes = 3;
    if (selected && interleave && interleave[0] == '1' && interleave[1] == '\0') {
        effective_mode = MPOL_INTERLEAVE;
        nodes = &interleave_nodes;
        /* Linux subtracts one before copying the nodemask bitmap. */
        maxnode = sizeof(interleave_nodes) * 8 + 1;
    }
    long rc = syscall(SYS_mbind, address, length, effective_mode, nodes,
                      maxnode, flags);
    if (selected)
        fprintf(stderr, "[mooncake-store-thp] numa-policy=%d requested=%d bytes=%lu rc=%ld\n",
                effective_mode, mode, length, rc);
    if (rc == 0 && selected)
        rc = populate_thp(address, length, "numa-region");
    return rc;
}

static void *map_segment(void *address, size_t length, int prot, int flags,
                         int fd, off64_t offset, int use_mmap64) {
    void *(*original)(void *, size_t, int, int, int, off64_t) =
        dlsym(RTLD_NEXT, use_mmap64 ? "mmap64" : "mmap");
    if (!original) {
        errno = ENOSYS;
        return MAP_FAILED;
    }
    int selected = address == NULL && fd == -1 && offset == 0 &&
                   prot == (PROT_READ | PROT_WRITE) &&
                   (flags & (MAP_ANONYMOUS | MAP_PRIVATE)) ==
                       (MAP_ANONYMOUS | MAP_PRIVATE) &&
                   !(flags & (MAP_FIXED | MAP_FIXED_NOREPLACE | MAP_HUGETLB)) &&
                   matches_size("ATOMESH_THP_GLOBAL_SEGMENT_BYTES", length);
    if (!selected)
        return original(address, length, prot, flags, fd, offset);
    if (length > SIZE_MAX - thp_size) {
        errno = ENOMEM;
        return MAP_FAILED;
    }
    void *raw = original(NULL, length + thp_size, prot, flags, fd, offset);
    if (raw == MAP_FAILED)
        return raw;
    uintptr_t start = ((uintptr_t)raw + thp_size - 1) & ~(thp_size - 1);
    size_t prefix = start - (uintptr_t)raw;
    size_t suffix = thp_size - prefix;
    if (prefix)
        munmap(raw, prefix);
    if (suffix)
        munmap((void *)(start + length), suffix);
    int rc = madvise((void *)start, length, MADV_HUGEPAGE);
    fprintf(stderr, "[mooncake-store-thp] global-segment=%zu address=%p advice=%d\n",
            length, (void *)start, rc);
    if (rc != 0) {
        int error = errno;
        munmap((void *)start, length);
        errno = error;
        return MAP_FAILED;
    }
    return (void *)start;
}

void *mmap(void *address, size_t length, int prot, int flags, int fd,
           off_t offset) {
    return map_segment(address, length, prot, flags, fd, offset, 0);
}

void *mmap64(void *address, size_t length, int prot, int flags, int fd,
             off64_t offset) {
    return map_segment(address, length, prot, flags, fd, offset, 1);
}
