#!/bin/bash
set -e

# =============================================================================
# Build script for xtensor benchmark environment
# Downloads, compiles and installs xtl, xsimd, xtensor locally
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
DEPS_DIR="${PROJECT_DIR}/deps"
INSTALL_BASE="${PROJECT_DIR}/install"

# Default options
COMPILER_SPEC="gcc"
USE_XSIMD="ON"
JOBS=$(nproc)
CLEAN=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --compiler=SPEC        Compiler specification (default: gcc)"
    echo "                         Examples: gcc, gcc-13, gcc-14, clang, clang-17, clang-18"
    echo "  --xsimd=ON|OFF         Enable/disable xsimd (default: ON)"
    echo "  --jobs=N               Number of parallel jobs (default: $(nproc))"
    echo "  --clean                Clean build directories before building"
    echo "  --list                 List existing builds"
    echo "  --help                 Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                    # gcc + xsimd"
    echo "  $0 --compiler=gcc-13 --xsimd=ON"
    echo "  $0 --compiler=clang-17 --xsimd=OFF"
    echo "  $0 --compiler=clang --xsimd=ON"
    echo ""
    echo "Build artifacts will be in:"
    echo "  deps/                          - source code (shared)"
    echo "  install/<compiler>_<xsimd>/    - installed libraries"
    echo "  build_<compiler>_<xsimd>/      - benchmark executable"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

list_builds() {
    echo "Existing builds:"
    echo ""
    for dir in "$PROJECT_DIR"/build_*/; do
        if [[ -d "$dir" ]]; then
            local name=$(basename "$dir")
            local exe="${dir}bench_xscalar"
            if [[ -x "$exe" ]]; then
                echo "  ${GREEN}✓${NC} $name"
                echo "    → $exe"
            else
                echo "  ${YELLOW}○${NC} $name (incomplete)"
            fi
        fi
    done
    echo ""
}

# Parse arguments
for arg in "$@"; do
    case $arg in
        --compiler=*)
            COMPILER_SPEC="${arg#*=}"
            ;;
        --xsimd=*)
            USE_XSIMD="${arg#*=}"
            ;;
        --jobs=*)
            JOBS="${arg#*=}"
            ;;
        --clean)
            CLEAN=1
            ;;
        --list)
            list_builds
            exit 0
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $arg"
            usage
            exit 1
            ;;
    esac
done

# Parse compiler specification (e.g., "gcc-13", "clang-17", "gcc", "clang")
if [[ "$COMPILER_SPEC" == *"-"* ]]; then
    # Has version: gcc-13 -> COMPILER_BASE=gcc, COMPILER_VERSION=13
    COMPILER_BASE="${COMPILER_SPEC%%-*}"
    COMPILER_VERSION="${COMPILER_SPEC#*-}"
else
    # No version: gcc -> COMPILER_BASE=gcc, COMPILER_VERSION=""
    COMPILER_BASE="$COMPILER_SPEC"
    COMPILER_VERSION=""
fi

# Validate compiler base
if [[ "$COMPILER_BASE" != "gcc" && "$COMPILER_BASE" != "clang" ]]; then
    log_error "Invalid compiler: $COMPILER_BASE (must be gcc or clang)"
    exit 1
fi

if [[ "$USE_XSIMD" != "ON" && "$USE_XSIMD" != "OFF" ]]; then
    log_error "Invalid xsimd option: $USE_XSIMD (must be ON or OFF)"
    exit 1
fi

# Set compiler commands
if [[ "$COMPILER_BASE" == "gcc" ]]; then
    if [[ -n "$COMPILER_VERSION" ]]; then
        export CC="gcc-${COMPILER_VERSION}"
        export CXX="g++-${COMPILER_VERSION}"
    else
        export CC="gcc"
        export CXX="g++"
    fi
else
    if [[ -n "$COMPILER_VERSION" ]]; then
        export CC="clang-${COMPILER_VERSION}"
        export CXX="clang++-${COMPILER_VERSION}"
    else
        export CC="clang"
        export CXX="clang++"
    fi
fi

# Check compiler exists
if ! command -v $CXX &> /dev/null; then
    log_error "$CXX not found. Please install it."
    log_error "On Ubuntu: sudo apt install g++-${COMPILER_VERSION} or clang-${COMPILER_VERSION}"
    exit 1
fi

# Get actual compiler version for naming
ACTUAL_VERSION=$($CXX --version | head -1 | grep -oP '\d+\.\d+\.\d+' | head -1 || echo "unknown")
MAJOR_VERSION=$(echo "$ACTUAL_VERSION" | cut -d. -f1)

# Build suffix includes compiler + major version + xsimd
if [[ "$USE_XSIMD" == "ON" ]]; then
    BUILD_SUFFIX="${COMPILER_BASE}${MAJOR_VERSION}_xsimd"
else
    BUILD_SUFFIX="${COMPILER_BASE}${MAJOR_VERSION}_noxsimd"
fi

INSTALL_DIR="${INSTALL_BASE}/${BUILD_SUFFIX}"
BUILD_DIR="${PROJECT_DIR}/build_${BUILD_SUFFIX}"

log_info "=============================================="
log_info "Configuration:"
log_info "  Compiler: $CXX (version $ACTUAL_VERSION)"
log_info "  XSIMD: $USE_XSIMD"
log_info "  Jobs: $JOBS"
log_info "  Install dir: $INSTALL_DIR"
log_info "  Build dir: $BUILD_DIR"
log_info "=============================================="
echo ""

# Create directories
mkdir -p "$DEPS_DIR"
mkdir -p "$INSTALL_DIR"

# =============================================================================
# Download dependencies
# =============================================================================

download_deps() {
    log_info "Downloading dependencies..."

    cd "$DEPS_DIR"

    # xtl
    if [[ ! -d "xtl" ]]; then
        log_info "Cloning xtl..."
        git clone --depth 1 https://github.com/xtensor-stack/xtl.git
    else
        log_info "xtl already exists, skipping clone"
    fi

    # xsimd
    if [[ ! -d "xsimd" ]]; then
        log_info "Cloning xsimd..."
        git clone --depth 1 https://github.com/xtensor-stack/xsimd.git
    else
        log_info "xsimd already exists, skipping clone"
    fi

    # xtensor
    if [[ ! -d "xtensor" ]]; then
        log_info "Cloning xtensor..."
        git clone --depth 1 https://github.com/xtensor-stack/xtensor.git
    else
        log_info "xtensor already exists, skipping clone"
    fi
}

# =============================================================================
# Build functions
# =============================================================================

build_xtl() {
    log_info "Building xtl..."

    local src_dir="${DEPS_DIR}/xtl"
    local build_dir="${src_dir}/build_${BUILD_SUFFIX}"

    if [[ $CLEAN -eq 1 ]]; then
        rm -rf "$build_dir"
    fi

    mkdir -p "$build_dir"
    cd "$build_dir"

    cmake .. \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_BUILD_TYPE=Release

    make -j$JOBS install

    log_info "xtl installed"
}

build_xsimd() {
    log_info "Building xsimd..."

    local src_dir="${DEPS_DIR}/xsimd"
    local build_dir="${src_dir}/build_${BUILD_SUFFIX}"

    if [[ $CLEAN -eq 1 ]]; then
        rm -rf "$build_dir"
    fi

    mkdir -p "$build_dir"
    cd "$build_dir"

    cmake .. \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_BUILD_TYPE=Release

    make -j$JOBS install

    log_info "xsimd installed"
}

build_xtensor() {
    log_info "Building xtensor..."

    local src_dir="${DEPS_DIR}/xtensor"
    local build_dir="${src_dir}/build_${BUILD_SUFFIX}"

    # Patch xtensor to accept xsimd 14.x (only once)
    if grep -q "set(xsimd_REQUIRED_VERSION 13" "${src_dir}/CMakeLists.txt" 2>/dev/null; then
        log_info "Patching xtensor for xsimd 14.x compatibility..."
        sed -i 's/set(xsimd_REQUIRED_VERSION 13\.[0-9]\.[0-9])/set(xsimd_REQUIRED_VERSION 14.0.0)/' "${src_dir}/CMakeLists.txt"
    fi

    if [[ $CLEAN -eq 1 ]]; then
        rm -rf "$build_dir"
    fi

    mkdir -p "$build_dir"
    cd "$build_dir"

    cmake .. \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DXTENSOR_USE_XSIMD="$USE_XSIMD"

    make -j$JOBS install

    log_info "xtensor installed"
}

build_benchmark() {
    log_info "Building benchmark..."

    if [[ $CLEAN -eq 1 ]]; then
        rm -rf "$BUILD_DIR"
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Compiler flags
    local CXX_FLAGS="-O3 -march=native -g -fno-omit-frame-pointer"

    cmake "$PROJECT_DIR" \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
        -DUSE_XSIMD="$USE_XSIMD"

    make -j$JOBS

    log_info "Benchmark built: ${BUILD_DIR}/bench_xscalar"
}

# =============================================================================
# Main
# =============================================================================

download_deps

build_xtl

if [[ "$USE_XSIMD" == "ON" ]]; then
    build_xsimd
fi

build_xtensor

build_benchmark

echo ""
log_info "=============================================="
log_info "Build complete!"
log_info "=============================================="
echo ""
log_info "Run benchmark:"
log_info "  ${BUILD_DIR}/bench_xscalar"
echo ""
log_info "List all builds:"
log_info "  $0 --list"
