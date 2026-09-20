#!/usr/bin/env bash
# Build only the HIP/HSA pair; keep the SDK and other ROCm components intact.
set -euo pipefail

mode=${ROCM_ORDERING_EDGE:-auto}
rocm=${ROCM_PATH:-/opt/rocm}
output=/opt/atom-rocm-runtime
version=$(cat "$rocm/.info/version" 2>/dev/null || true)
mkdir -p "$output/patched/lib"
case "$mode" in auto|0|1) ;; *) echo "ROCM_ORDERING_EDGE must be auto, 0, or 1" >&2; exit 1 ;; esac
if [[ "$mode" == 0 ]] || { [[ "$mode" == auto ]] && [[ ! "$version" =~ ^7\.2\.4($|[-+]) ]]; }; then
    restore_stock=false
    if [[ "$version" =~ ^7\.2\.4($|[-+]) ]]; then restore_stock=true; fi
    printf '{"enabled": false, "restore_stock": %s, "reason": "disabled or base is not ROCm 7.2.4"}\n' "$restore_stock" > "$output/build-info.json"
    echo "Keeping stock ROCm runtime ($version, ROCM_ORDERING_EDGE=$mode)"
    exit 0
fi
if [[ ! "$version" =~ ^7\.2\.4($|[-+]) ]]; then
    echo "This backport requires ROCm 7.2.4; found '$version'" >&2
    exit 1
fi

repo=${ROCM_SYSTEMS_REPO:-https://github.com/ROCm/rocm-systems.git}
commit=${ROCM_RUNTIME_COMMIT:-b539bf7eebfd99ad0a69668caa1f4037034d501f}
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "ROCM_RUNTIME_COMMIT must be a full commit SHA" >&2; exit 1; }
jobs=${ROCM_RUNTIME_JOBS:-16}
apt-get update
apt-get install -y --no-install-recommends git cmake ninja-build g++ pkg-config \
    libelf-dev libdrm-dev libnuma-dev libdw-dev xxd
python3 -m pip install --no-cache-dir CppHeaderParser==2.7.4 ply==3.11

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git init -q "$work/source"
git -C "$work/source" remote add origin "$repo"
git -C "$work/source" config remote.origin.promisor true
git -C "$work/source" config remote.origin.partialclonefilter blob:none
git -C "$work/source" sparse-checkout init --cone
git -C "$work/source" sparse-checkout set projects/rocr-runtime projects/clr projects/hip shared cmake
git -C "$work/source" fetch --filter=blob:none --depth 1 origin "$commit"
git -C "$work/source" checkout -q FETCH_HEAD
test "$(git -C "$work/source" rev-parse HEAD)" = "$commit"

# The SDK ships these executables but may omit their CMake package configs.
mkdir -p "$work/cmake"
cat > "$work/cmake/ClangConfig.cmake" <<EOF
if(NOT TARGET clang)
  add_executable(clang IMPORTED GLOBAL)
  set_target_properties(clang PROPERTIES IMPORTED_LOCATION "$rocm/llvm/bin/clang")
endif()
set(Clang_PACKAGE_VERSION "rocm-image-shim")
EOF
cat > "$work/cmake/LLVMConfig.cmake" <<EOF
if(NOT TARGET llvm-objcopy)
  add_executable(llvm-objcopy IMPORTED GLOBAL)
  set_target_properties(llvm-objcopy PROPERTIES IMPORTED_LOCATION "$rocm/llvm/bin/llvm-objcopy")
endif()
set(LLVM_FOUND TRUE)
set(LLVM_PACKAGE_VERSION "rocm-image-shim")
EOF

prefix="$work/install"
cmake -S "$work/source/projects/rocr-runtime" -B "$work/rocr" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$rocm" \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DClang_DIR="$work/cmake" -DLLVM_DIR="$work/cmake"
cmake --build "$work/rocr" --parallel "$jobs"
cmake --install "$work/rocr" --strip

# CLR must see the new HSA declarations before the SDK's original headers.
mkdir -p "$work/hsa-headers"
cp -a "$prefix/include/hsa" "$work/hsa-headers/"
cmake -S "$work/source/projects/clr" -B "$work/clr" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF \
    -DHIP_COMMON_DIR="$work/source/projects/hip" -DROCM_PATH="$rocm" \
    -DCMAKE_PREFIX_PATH="$prefix;$rocm" -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DLLVM_DIR="$work/cmake" -DHIP_LLVM_ROOT="$rocm/llvm" \
    -DCMAKE_CXX_FLAGS="-I$work/hsa-headers" -DCMAKE_C_FLAGS="-I$work/hsa-headers"
cmake --build "$work/clr" --parallel "$jobs"
cmake --install "$work/clr" --strip
cp -P "$prefix"/lib/libhsa-runtime64.so* "$output/patched/lib/"
cp -P "$prefix"/lib/libamdhip64.so* "$output/patched/lib/"
printf '{"enabled": true, "commit": "%s", "base_rocm": "7.2.4"}\n' "$commit" > "$output/build-info.json"
