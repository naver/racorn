# bench_acorn benchmark binaries.
#
# Two variants are built from the same source (cpp/bench_acorn.cpp):
#   - bench_acorn       : default build, 2-hop ACORN-1 / RACORN-1 / RACORN-1+
#   - bench_acorn_3hop  : compiled with USEARCH_ENABLE_3HOP=1, enabling the
#                         additional 3-hop expansion branch in
#                         include/usearch/index.hpp. Only this binary's
#                         `--3hop-threshold` option has effect; the default
#                         bench_acorn binary stays slimmer (no 3-hop buffers).
#
# Dependencies: just header-only ACORN-1 / RACORN-1 implementation under
# `../include/usearch/` and the `fp16` submodule under `../fp16/include/`.

find_package(Threads REQUIRED)

function(_add_bench_acorn target_name enable_3hop)
    add_executable(${target_name} bench_acorn.cpp)
    target_include_directories(${target_name} BEFORE PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../fp16/include)
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_compile_options(${target_name} PRIVATE -O2 -DNDEBUG)
    target_link_libraries(${target_name} PRIVATE Threads::Threads)
    # GCC < 9 ships std::filesystem in a separate libstdc++fs. GCC 9+
    # rolls it back into libstdc++ but the link line is still accepted.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
        target_link_libraries(${target_name} PRIVATE stdc++fs)
    endif ()
    # Use the vendored fp16 software emulation for half-precision conversions
    # rather than the native `_Float16` type. The native path is only stable
    # on GCC 12+; the fp16 library works back to GCC 8.
    target_compile_definitions(${target_name} PRIVATE USEARCH_USE_FP16LIB=1)
    if (enable_3hop)
        target_compile_definitions(${target_name} PRIVATE USEARCH_ENABLE_3HOP=1)
    endif ()
endfunction()

# Default build: no 3-hop (smaller per-cycle buffers, ACORN-1 / RACORN-1 / RACORN-1+ only)
_add_bench_acorn(bench_acorn FALSE)
# 3-hop variant: enables USEARCH_ENABLE_3HOP=1 (larger buffers + 3-hop expansion branch)
_add_bench_acorn(bench_acorn_3hop TRUE)
