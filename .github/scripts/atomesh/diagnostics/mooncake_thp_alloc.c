/* Experimental Store allocator override. Only the two configured allocation
 * sizes are changed; named/shared/fixed mappings retain libc behavior. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>

static const size_t thp_size = 2UL * 1024 * 1024;

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
        fprintf(stderr, "[DEBUG-mooncake-thp] local-buffer=%zu address=%p advice=%d\n",
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
    fprintf(stderr, "[DEBUG-mooncake-thp] global-segment=%zu address=%p advice=%d\n",
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
