# RACORN
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0
#
# Benchmark drivers. Each is a single translation unit over ../include/usearch/.
#
#   bench_acorn       — modulo / random-attribute predicates, prefilter,
#                       HNSW in-filtering, ACORN-1, RACORN-1, NaviX heuristics, pruning-free
#                       volumes, ef/br sweeps, soft-timeout ladders, stride and 3-hop ablations.
#   bench_synthfilter — per-query allowlists (uniform / alpha-correlated, dense-key
#                       bitset predicate), external allowlist files (LAION caption keywords),
#                       metadata-column predicates (--filter column), the heterogeneous
#                       keyword AND resolution predicate, multi-attribute conjunction with
#                       three prefilter variants.
#   bench_realfilter  — YFCC-10M tag-AND real filter, natural-selectivity buckets,
#                       official big-ann GT validation.
#
# Flags of every reported number:
#   g++ -std=c++17 -O3 -march=native -DRACORN_USE_NUMKONG=1 -I include <driver>.cpp -pthread
# With -DRACORN_USE_NUMKONG=OFF, the distances are scalar loops. That is the same as the g++ command
# without the define.

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

function(_add_racorn_bench target_name)
    add_executable(${target_name} ${target_name}.cpp)
    target_include_directories(${target_name} BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_compile_options(${target_name} PRIVATE -O3 -march=native)
    if (RACORN_USE_NUMKONG)
        target_compile_definitions(${target_name} PRIVATE RACORN_USE_NUMKONG=1)
    endif ()
    target_link_libraries(${target_name} PRIVATE Threads::Threads)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
        target_link_libraries(${target_name} PRIVATE stdc++fs)
    endif ()
endfunction()

_add_racorn_bench(bench_acorn)
_add_racorn_bench(bench_synthfilter)
_add_racorn_bench(bench_realfilter)
