# Top-level CMakeLists for the RACORN-1 / ACORN-1 benchmark artifact.
#
# This file is a slimmed-down derivative of the upstream USearch CMakeLists.
# It exposes a single use case — building `bench_acorn` (and the 3-hop variant
# `bench_acorn_3hop`) — so it intentionally drops the upstream language
# bindings, install/pkg-config plumbing, SimSIMD/JeMalloc/OpenMP options,
# and FP16/BF16 native-type probes. Only the header-only ACORN-1 / RACORN-1
# implementation in `include/usearch/` is exercised here.
#
# Build:
#   mkdir build && cd build
#   cmake -DCMAKE_BUILD_TYPE=Release ..
#   make -j bench_acorn bench_acorn_3hop
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(
    racorn-artifact
    VERSION 1.0.0
    LANGUAGES C CXX
    DESCRIPTION "RACORN-1 / ACORN-1 benchmark — supplementary artifact"
)

if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif ()

# Default build type to RelWithDebInfo (gives -O3 -g; matches upstream)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif ()

# All targets live under bench/CMakeLists.txt
add_subdirectory(bench)
