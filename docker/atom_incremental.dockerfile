ARG BASE_IMAGE="rocm/atom-dev:latest"
FROM ${BASE_IMAGE}

ARG ATOM_REPO="https://github.com/ROCm/ATOM.git"
ARG ATOM_COMMIT

# The full image installs ATOM editable from /app/ATOM.  Refreshing that
# checkout updates Python and data files without rebuilding the native stack.
# The workflow only selects this Dockerfile after it verifies that the commits
# between the base image and ATOM_COMMIT contain no native or packaging changes.
RUN set -eux; \
    test -n "${ATOM_COMMIT}"; \
    test -d /app/ATOM/.git; \
    git -C /app/ATOM remote set-url origin "${ATOM_REPO}"; \
    git -C /app/ATOM fetch --no-tags --depth=1 origin "${ATOM_COMMIT}"; \
    git -C /app/ATOM checkout --detach "${ATOM_COMMIT}"; \
    git -C /app/ATOM clean -ffd; \
    test "$(git -C /app/ATOM rev-parse HEAD)" = "${ATOM_COMMIT}"; \
    python -m pip install -e /app/ATOM --no-deps --no-build-isolation; \
    python -m compileall -q /app/ATOM/atom; \
    mkdir -p /etc/atom-build-info; \
    printf '%s\n' "${ATOM_COMMIT}" > /etc/atom-build-info/commit

LABEL org.opencontainers.image.source="${ATOM_REPO}" \
      org.opencontainers.image.revision="${ATOM_COMMIT}" \
      io.rocm.atom.build-mode="incremental"

CMD ["/bin/bash"]
