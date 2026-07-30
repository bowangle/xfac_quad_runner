#!/bin/bash

# Paths
PROJECT_ROOT=$(pwd)
BUILD_DIR="$PROJECT_ROOT/build"

# Clean old build
rm -rf "$BUILD_DIR"

export CCACHE_DIR=/tmp/ccache

# Configure CMake (out-of-source build)
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native" \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

# Build everything
cmake --build "$BUILD_DIR" -j 1

cat build/CMakeCache.txt | grep CMAKE_CXX_COMPILER