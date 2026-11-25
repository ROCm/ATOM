#!/bin/bash
# Build the Debian/Ubuntu package for ATOM's conversational deployment agent:
# package `amd-atom`, command `/usr/bin/atom`.
#
#   bash scripts/build_deb.sh [OUT_DIR]          # default: build/deb
#   VERSION=0.2.0 bash scripts/build_deb.sh      # default: from `git describe`
#   MAINTAINER="Name <email>" ...                # default: git config user
#
#   sudo apt install ./build/deb/amd-atom_<version>_all.deb
#   atom
#
# The package is architecture-independent and needs only python3 (>= 3.10):
# the agent is standard-library Python and runs ATOM through docker. It ships
# the agent's modules (the import closure of its entry points, computed here,
# not a hand-kept list), plus the recipes, docs and skills it searches when no
# ATOM image is available locally.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mkdir -p "${1:-$REPO_ROOT/build/deb}" && cd "${1:-$REPO_ROOT/build/deb}" && pwd)"
PKG=amd-atom
LIB=/usr/lib/$PKG
PY=python3

command -v dpkg-deb >/dev/null || { echo "error: dpkg-deb not found" >&2; exit 1; }

# ---- version ---------------------------------------------------------------
if [ -z "${VERSION:-}" ]; then
    DESC="$(git -C "$REPO_ROOT" describe --tags --long --match 'v[0-9]*' 2>/dev/null || true)"
    SHA="$(git -C "$REPO_ROOT" rev-parse --short=8 HEAD 2>/dev/null || echo unknown)"
    DIRTY=""
    git -C "$REPO_ROOT" diff --quiet HEAD -- 2>/dev/null || DIRTY=1
    # v0.1.6-rc0-826-gb832762c -> 0.1.6~rc0+git826.b832762c  (~ sorts before 0.1.6)
    VERSION="$("$PY" - "$DESC" "$SHA" "$DIRTY" <<'PY'
import re, sys
desc, sha, dirty = sys.argv[1:4]
m = re.fullmatch(r"v?(.+)-(\d+)-g([0-9a-f]+)", desc)
if m:
    base, n, sha = m.group(1), int(m.group(2)), m.group(3)
else:
    base, n = "0.0.0", 1
base = re.sub(r"[-_]?(rc|alpha|beta|a|b)(\d*)", r"~\1\2", base)
local = [f"git{n}.{sha}"] if n else []
if dirty:
    local.append("dirty")
print(base + ("+" + ".".join(local) if local else ""))
PY
)"
fi
if ! [[ "$VERSION" =~ ^[0-9][A-Za-z0-9.+~-]*$ ]]; then
    echo "error: '$VERSION' is not a valid Debian version" >&2
    exit 1
fi

if [ -z "${MAINTAINER:-}" ]; then
    NAME="$(git -C "$REPO_ROOT" config user.name || true)"
    EMAIL="$(git -C "$REPO_ROOT" config user.email || true)"
    [ -n "$NAME" ] && [ -n "$EMAIL" ] || {
        echo "error: set MAINTAINER=\"Name <email>\" (no git user configured)" >&2
        exit 1
    }
    MAINTAINER="$NAME <$EMAIL>"
fi

# ---- stage -----------------------------------------------------------------
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
ROOT="$STAGE/$PKG"
mkdir -p "$ROOT/DEBIAN" "$ROOT$LIB" "$ROOT/usr/bin" "$ROOT/usr/share/doc/$PKG"

# Python modules: everything the entry points import from the atom package,
# all of atom/agent, and the argparse sources the planner reads as text.
"$PY" -I - "$REPO_ROOT" "$ROOT$LIB" <<'PY'
import shutil, sys
from pathlib import Path

repo, dest = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(repo))
import atom.agent.cli, atom.agent.evals, atom.agent.mcp_server  # noqa: E401,F401
from atom.agent.planner import FLAG_SOURCES

pkg = repo / "atom"
files = {
    Path(m.__file__).resolve()
    for m in list(sys.modules.values())
    if getattr(m, "__file__", None) and Path(m.__file__).resolve().is_relative_to(pkg)
}
files |= {p.resolve() for p in (pkg / "agent").glob("*.py")}
files |= {(pkg / rel).resolve() for rel in FLAG_SOURCES}
for f in sorted(files):
    out = dest / f.relative_to(repo)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, out)
print(f"staged {len(files)} Python files")
PY
echo "__version__ = '$VERSION'" > "$ROOT$LIB/atom/_version.py"

# Knowledge base and skills (config.repo_root() finds them next to atom/).
cp -r "$REPO_ROOT/recipes" "$REPO_ROOT/docs" "$REPO_ROOT/README.md" "$ROOT$LIB/"
mkdir -p "$ROOT$LIB/.claude"
for d in skills commands; do
    [ -d "$REPO_ROOT/.claude/$d" ] && cp -r "$REPO_ROOT/.claude/$d" "$ROOT$LIB/.claude/"
done
find "$ROOT$LIB" -name __pycache__ -type d -prune -exec rm -rf {} +

# The command. -I keeps PYTHONPATH, user site-packages and the current
# directory (e.g. an ATOM checkout) from shadowing the packaged agent.
cat > "$ROOT/usr/bin/atom" <<EOF
#!/bin/sh
# atom: ATOM's conversational deployment agent (Debian package $PKG).
exec /usr/bin/python3 -I -c 'import sys; sys.path.insert(0, "$LIB"); from atom.agent.cli import main; main()' "\$@"
EOF

cat > "$ROOT/usr/share/doc/$PKG/copyright" <<EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: ATOM
Source: https://github.com/ROCm/ATOM

Files: *
Copyright: Advanced Micro Devices, Inc.
License: MIT
$(sed 's/^$/./; s/^/ /' "$REPO_ROOT/LICENSE")
EOF
printf '%s (%s) unstable; urgency=medium\n\n  * Built from %s.\n\n -- %s  %s\n' \
    "$PKG" "$VERSION" "$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo source)" \
    "$MAINTAINER" "$(date -R)" | gzip -9n > "$ROOT/usr/share/doc/$PKG/changelog.Debian.gz"

# Byte-compile at install (root can write $LIB; users cannot at run time).
cat > "$ROOT/DEBIAN/postinst" <<EOF
#!/bin/sh
set -e
if [ "\$1" = configure ]; then
    python3 -m compileall -q "$LIB" >/dev/null 2>&1 || true
fi
EOF
cat > "$ROOT/DEBIAN/prerm" <<EOF
#!/bin/sh
set -e
find "$LIB" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
EOF

find "$ROOT" -type d -exec chmod 755 {} +
find "$ROOT" -type f -exec chmod 644 {} +
chmod 755 "$ROOT/usr/bin/atom" "$ROOT/DEBIAN/postinst" "$ROOT/DEBIAN/prerm"

INSTALLED_KB="$(du -sk --exclude=DEBIAN "$ROOT" | cut -f1)"
cat > "$ROOT/DEBIAN/control" <<EOF
Package: $PKG
Version: $VERSION
Section: utils
Priority: optional
Architecture: all
Depends: python3 (>= 3.10)
Recommends: docker.io | docker-ce
Conflicts: atom
Installed-Size: $INSTALLED_KB
Maintainer: $MAINTAINER
Homepage: https://github.com/ROCm/ATOM
Description: conversational deployment agent for the ATOM inference engine
 Type \`atom\` to deploy and operate LLM servers on AMD Instinct GPUs by
 chatting. It inspects the machine, adapts ATOM's recipes, launches servers
 through docker, verifies and benchmarks them, follows ATOM's and AITER's
 skills, and explains failures -- asking before anything that changes the
 machine.
 .
 Standard-library Python only. Its LLM can be any OpenAI- or
 Anthropic-compatible endpoint, including an ATOM server.
 .
 Conflicts with the discontinued Atom editor package, which also installed
 /usr/bin/atom.
EOF
(cd "$ROOT" && find . -type f ! -path './DEBIAN/*' -printf '%P\0' | sort -z | xargs -0 md5sum) \
    > "$ROOT/DEBIAN/md5sums"

DEB="$OUT_DIR/${PKG}_${VERSION}_all.deb"
dpkg-deb --root-owner-group -Zxz --build "$ROOT" "$DEB" >/dev/null
echo "built $DEB ($(du -h "$DEB" | cut -f1), installs ${INSTALLED_KB} KiB)"
echo "install: sudo apt install $DEB"
