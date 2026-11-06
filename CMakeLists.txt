# Copied from: https://github.com/unum-cloud/USearch (Apache License 2.0)
# Modified by NAVER Corp.: reduced to the header-only index under include/usearch/ and the three
#   benchmark drivers under bench/, with NumKong's distance kernels (RACORN_USE_NUMKONG).
#
# The tree is the header-only index under include/usearch/ and the three drivers under bench/.
# There are no language bindings, tests or install rules. Of NumKong
# (https://github.com/ashvardanian/NumKong, Apache 2.0), only the header-only AVX2/FMA kernels for
# f32 L2 and cosine distances are used (RACORN_USE_NUMKONG). They come from the third_party/NumKong
# submodule (v7.2.0); include/numkong is a link to its headers.
#
# Build:
#   git submodule update --init      # once, unless cloned with --recursive
#   cmake -S . -B build
#   cmake --build build -j
# Produces build/bench/bench_acorn, bench_synthfilter, bench_realfilter.
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(
    racorn-artifact
    VERSION 2.0.0
    LANGUAGES CXX
    DESCRIPTION "RACORN experiment code: USearch HNSW core with ACORN-1/RACORN-1/NaviX and benchmarks"
)

if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# The drivers set their own flags (bench/CMakeLists.txt). So the build-type flags (-O3 -DNDEBUG for
# Release) are cleared, and the language mode is ISO C++17 without GNU extensions.
option(RACORN_USE_NUMKONG "NumKong's AVX2/FMA kernels for f32 L2 and cosine distances (needs AVX2, FMA and F16C)" ON)
if (RACORN_USE_NUMKONG)
    if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/numkong/spatial.h)
        message(FATAL_ERROR "NumKong not found: run `git submodule update --init third_party/NumKong`, or configure with -DRACORN_USE_NUMKONG=OFF")
    endif ()
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS -march=native)
    check_cxx_source_compiles("
        #if !(defined(__AVX2__) && defined(__FMA__) && defined(__F16C__))
        #error no AVX2, FMA or F16C
        #endif
        int main() { return 0; }" RACORN_CPU_HAS_AVX2_FMA_F16C)
    unset(CMAKE_REQUIRED_FLAGS)
    if (NOT RACORN_CPU_HAS_AVX2_FMA_F16C)
        message(FATAL_ERROR "This CPU has no AVX2, FMA or F16C for NumKong's kernels: configure with -DRACORN_USE_NUMKONG=OFF "
                            "to build with the plain loops (the latencies then differ from the reported runs)")
    endif ()
endif ()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
foreach (config RELEASE RELWITHDEBINFO MINSIZEREL DEBUG)
    set(CMAKE_CXX_FLAGS_${config} "")
endforeach ()

add_subdirectory(bench)
