#!/usr/bin/env bash
# Official manylinux release build; no installed/source development binaries.
set -euo pipefail
export PATH="/opt/python/cp312-cp312/bin:$HOME/.cargo/bin:$PATH"
if ! command -v rustup >/dev/null; then
    curl --retry 3 --fail --silent --show-error https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
rustup toolchain install stable --profile minimal
rustup override set stable
rustup target add x86_64-unknown-linux-gnu
# Install the verified packaging version through PyPI wheels. A GitHub archive
# bootstrap can return an HTML gateway error before any source is compiled.
/opt/python/cp312-cp312/bin/python -m pip install --only-binary=:all: \
    'maturin==1.15.0' 'patchelf==0.17.2.4' cffi
git config --global --add safe.directory "$PWD"
rustc --version
maturin --version

krasis_phase_start() {
  KRASIS_PHASE_LABEL="$1"
  KRASIS_PHASE_START=$(date +%s)
  echo "::group::KRASIS build phase: $KRASIS_PHASE_LABEL"
}
krasis_phase_end() {
  status="${1:-0}"
  end=$(date +%s)
  echo "KRASIS_BUILD_TIMING phase=\"$KRASIS_PHASE_LABEL\" status=$status duration_s=$((end - KRASIS_PHASE_START))"
  echo "::endgroup::"
  return "$status"
}
krasis_phase_start "release setup CUDA and host compiler"
# numactl for libnuma
yum install -y numactl-devel 2>/dev/null || dnf install -y numactl-devel 2>/dev/null || apt-get install -y libnuma-dev 2>/dev/null || true
# CUDA toolkit for nvcc (decode kernels compiled to PTX at build time)
# and libcuda.so stubs (FLA sidecars link against the CUDA driver API,
# but release CI has no real GPU driver installed).
# build.rs and the sidecar builder need nvcc available on PATH.
if [ ! -f /usr/local/cuda-12.6/bin/nvcc ]; then
  yum install -y yum-utils 2>/dev/null || true
  yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo 2>/dev/null || \
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo 2>/dev/null || true
  yum install -y --nogpgcheck cuda-nvcc-12-6 cuda-cudart-devel-12-6 cuda-driver-devel-12-6 2>/dev/null || \
    dnf install -y --nogpgcheck cuda-nvcc-12-6 cuda-cudart-devel-12-6 cuda-driver-devel-12-6 2>/dev/null || true
  ls -la /usr/local/cuda-12.6/bin/nvcc 2>/dev/null && echo "nvcc installed OK" || echo "WARNING: nvcc install failed"
fi
# The native GDN wrapper needs the cuBLAS API headers. Calls use
# the already loaded runtime API; no cuBLAS binary is bundled.
yum install -y --nogpgcheck libcublas-devel-12-6 2>/dev/null || \
  dnf install -y --nogpgcheck libcublas-devel-12-6 2>/dev/null || true
test -f /usr/local/cuda-12.6/targets/x86_64-linux/include/cublas_v2.h || {
  echo "ERROR: required CUDA 12.6 cuBLAS headers are missing" >&2
  exit 1
}
if [ ! -x /usr/local/cuda-12.6/bin/nvcc ]; then
  echo "ERROR: required release nvcc not found at /usr/local/cuda-12.6/bin/nvcc" >&2
  find /usr/local -path '*/bin/nvcc' -print 2>/dev/null || true
  exit 1
fi
yum install -y gcc-toolset-14-gcc-c++ gcc-toolset-13-gcc-c++ 2>/dev/null || \
  dnf install -y gcc-toolset-14-gcc-c++ gcc-toolset-13-gcc-c++ 2>/dev/null || true
for candidate in \
  /opt/rh/gcc-toolset-13/root/usr/bin/g++ \
  /opt/rh/gcc-toolset-14/root/usr/bin/g++ \
  /usr/bin/g++; do
  if [ -x "$candidate" ]; then
    export KRASIS_NVCC_CCBIN="$candidate"
    break
  fi
done
if [ -z "${KRASIS_NVCC_CCBIN:-}" ]; then
  echo "ERROR: required release NVCC host compiler not found" >&2
  find /opt/rh -path '*/bin/g++' -print 2>/dev/null || true
  exit 1
fi
"$KRASIS_NVCC_CCBIN" --version | head -n 1
if [ ! -f /usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs/libcuda.so ] && \
   [ ! -f /usr/local/cuda-12.6/lib64/stubs/libcuda.so ]; then
  yum install -y --nogpgcheck cuda-driver-devel-12-6 2>/dev/null || \
    dnf install -y --nogpgcheck cuda-driver-devel-12-6 2>/dev/null || true
fi
find /usr/local/cuda-12.6 -path '*/stubs/libcuda.so' -print -quit | grep -q . || {
  echo "ERROR: CUDA driver stub libcuda.so not found under /usr/local/cuda-12.6" >&2
  find /usr/local/cuda-12.6 -maxdepth 5 -name 'libcuda.so*' -print 2>/dev/null || true
  exit 1
}
krasis_phase_end 0
krasis_phase_start "release Python deps for FLA"
# FLA kernel cross-compilation (no GPU needed — uses Triton cross-compile)
# CPU-only torch keeps download small (~200MB vs ~2.7GB for CUDA torch).
# FLA source files import torch at module level but compilation only uses Triton.
# Match the established Windows portable-source generation toolchain.
/opt/python/cp312-cp312/bin/pip install 'triton==3.5.1'
/opt/python/cp312-cp312/bin/pip install 'torch==2.13.0+cpu' --index-url https://download.pytorch.org/whl/cpu
krasis_phase_end 0
krasis_phase_start "release FLA sidecars"
PATH="/usr/local/cuda-12.6/bin:$PATH" KRASIS_DEV_SCRIPT=1 KRASIS_FLA_CROSS_COMPILE=1 KRASIS_FLA_REQUIRE_ALL_ARCHS=1 KRASIS_NVCC_CCBIN="$KRASIS_NVCC_CCBIN" \
  /opt/python/cp312-cp312/bin/python src/cuda/fla/compile_kernels.py \
  --output-dir /tmp/fla_build
mkdir -p python/krasis
cp /tmp/fla_build/libkrasis_fla_sm*.so python/krasis/
ls -la python/krasis/libkrasis_fla_sm*.so
krasis_phase_end 0
krasis_phase_start "release Marlin/FlashAttention sidecars"
# Marlin and FlashAttention are dlopened package sidecars, not Rust
# compile inputs. Restore a content-addressed bundle when the
# sidecar source/toolchain hash matches a prior release; otherwise
# compile once and upload the verified bundle for future releases.
for KRASIS_CACHE_ATTEMPT in 1 2 3; do
  set +e
  PATH="/usr/local/cuda-12.6/bin:$PATH" KRASIS_DEV_SCRIPT=1 KRASIS_NVCC_CCBIN="$KRASIS_NVCC_CCBIN" \
    /opt/python/cp312-cp312/bin/python scripts/build_sidecars.py restore-bundle --github
  restore_status=$?
  set -e
  if [ "$restore_status" -eq 0 ] || [ "$restore_status" -eq 2 ]; then break; fi
  if [ "$KRASIS_CACHE_ATTEMPT" -lt 3 ]; then sleep "$KRASIS_CACHE_ATTEMPT"; fi
done
if [ "$restore_status" -eq 0 ]; then
  echo "Restored verified Marlin/FlashAttention sidecar bundle."
elif [ "$restore_status" -eq 2 ]; then
  echo "No matching sidecar bundle found; compiling Marlin/FlashAttention."
  PATH="/usr/local/cuda-12.6/bin:$PATH" KRASIS_DEV_SCRIPT=1 KRASIS_NVCC_CCBIN="$KRASIS_NVCC_CCBIN" \
    /opt/python/cp312-cp312/bin/python scripts/build_sidecars.py build
  PATH="/usr/local/cuda-12.6/bin:$PATH" KRASIS_DEV_SCRIPT=1 KRASIS_NVCC_CCBIN="$KRASIS_NVCC_CCBIN" \
    /opt/python/cp312-cp312/bin/python scripts/build_sidecars.py pack-bundle
  mkdir -p target/sidecars
  PATH="/usr/local/cuda-12.6/bin:$PATH" KRASIS_DEV_SCRIPT=1 KRASIS_NVCC_CCBIN="$KRASIS_NVCC_CCBIN" \
    /opt/python/cp312-cp312/bin/python scripts/build_sidecars.py bundle-key > target/sidecars/upload_bundle_required
else
  echo "ERROR: sidecar bundle restore failed with status $restore_status" >&2
  exit "$restore_status"
fi
PATH="/usr/local/cuda-12.6/bin:$PATH" KRASIS_DEV_SCRIPT=1 KRASIS_NVCC_CCBIN="$KRASIS_NVCC_CCBIN" \
  /opt/python/cp312-cp312/bin/python scripts/build_sidecars.py verify
ls -la python/krasis/libkrasis_marlin.so python/krasis/libkrasis_flash_attn.so python/krasis/sidecar_manifest.json
krasis_phase_end 0

export PATH="/usr/local/cuda-12.6/bin:$PATH"
nvcc --version
maturin build --release --target x86_64-unknown-linux-gnu --manylinux 2_28 --out dist -i \
    /opt/python/cp310-cp310/bin/python /opt/python/cp311-cp311/bin/python \
    /opt/python/cp312-cp312/bin/python /opt/python/cp313-cp313/bin/python
maturin sdist --out dist
