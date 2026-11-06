# RACORN

This repository holds the implementation measured in the paper *RACORN: Robust Predicate-Agnostic
Filtered Vector Search under Extreme Selectivity*. It also holds the three benchmark drivers that
produced the paper's measurements. The implementation is a header-only HNSW index derived from
[USearch](https://github.com/unum-cloud/USearch) (Apache 2.0). Its base layer has three filtered
search paths: ACORN-1, RACORN-1, and NaviX's adaptive-local heuristic. RACORN-1 is ACORN-1 with
Adaptive Search Fallback and stride sampling. The NaviX path follows the NaviX authors' code, except
two of its rules, listed in `bench/README.md`. Pre-filtering over the passing set is implemented in
the drivers. The L2 and cosine distances of `f32` vectors use the AVX2 kernels of
[NumKong](https://github.com/ashvardanian/NumKong) (Apache 2.0).

## Paper and citation

Yoonseok Kim and Gyusik Choe. *RACORN: Robust Predicate-Agnostic Filtered Vector Search under
Extreme Selectivity.* Under submission, 2026.

```bibtex
@unpublished{kim2026racorn,
  title  = {{RACORN}: Robust Predicate-Agnostic Filtered Vector Search under Extreme Selectivity},
  author = {Kim, Yoonseok and Choe, Gyusik},
  year   = {2026},
  note   = {Under submission},
}
```

## Layout

```
include/usearch/
  index.hpp             HNSW core: construction, the upper-layer descent, the base-layer search and its dispatch
  racorn.hpp            ACORN-1 and RACORN-1: two-hop expansion, Adaptive Search Fallback, the three-hop variant
  navix.hpp             the NaviX heuristics
  racorn_util.hpp       the per-cycle two-hop dedup set and stride sampling
  index_plugins.hpp     the distance metrics; NumKong's kernels for f32 L2 and cosine when
                        RACORN_USE_NUMKONG is set
  README.md             the search options: the fields of index_search_config_t
bench/
  bench_acorn.cpp       controlled predicates; index and ground-truth construction; deadlines
  bench_synthfilter.cpp predicates on real data: correlated allowlists, metadata columns, conjunctions, caption keywords
  bench_realfilter.cpp  the YFCC tag queries of the Big ANN filtered track
  README.md             the drivers' options and the commands of every experiment in the paper
tools/                  get_datasets.sh (downloads and conversions of every dataset), the converters it
                        calls, and the LAION workload generators
third_party/NumKong/    NumKong v7.2.0, a submodule; include/numkong is a link to its headers
CMakeLists.txt  LICENSE  NOTICE
```

## Requirements

- GCC 11 or later. The reported numbers were compiled with GCC 11.5.0. CMake 3.13 or later, unless
  you build with the compiler alone.
- An x86-64 CPU with AVX2, FMA and F16C, for NumKong's kernels. The Build section says what to do
  without them.
- `numactl`, to bind the searches to one NUMA node.
- Python 3.6 or later with numpy 1.17 or later, for `tools/`. The two tools that read the LAION
  metadata also need `pyarrow`.

## Build

Clone with `--recursive`, or run `git submodule update --init` in a clone.

```bash
cmake -S . -B build
cmake --build build -j
```

This builds `build/bench/bench_acorn`, `bench_synthfilter` and `bench_realfilter`. The flags are
`-std=c++17 -O3 -march=native -DRACORN_USE_NUMKONG=1` for every `CMAKE_BUILD_TYPE`. Every reported
number was measured with these flags. Without CMake, these commands build the same binaries:

```bash
mkdir -p build/bench
for t in bench_acorn bench_synthfilter bench_realfilter; do
  g++ -std=c++17 -O3 -march=native -DRACORN_USE_NUMKONG=1 -I include bench/$t.cpp -o build/bench/$t -pthread
done
```

On a CPU without AVX2, FMA and F16C, the build stops with an error. To build the distances as plain
loops, configure with `-DRACORN_USE_NUMKONG=OFF`, or leave out the define. The latencies of the
plain loops differ from the reported ones.

## Quick start

This runs a short selectivity sweep on SIFT 1M, for HNSW, ACORN-1 and RACORN-1 at two
selectivities. It needs the SIFT 1M distribution of corpus-texmex, served over FTP only. The run
takes a few minutes with 32 threads.

```bash
export PATH=$PWD/build/bench:$PATH
mkdir -p ~/racorn-run/data/raw ~/racorn-run/data/volumes ~/racorn-run/data/gt ~/racorn-run/results
cd ~/racorn-run
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz && mv sift/sift_base.fvecs sift/sift_query.fvecs data/raw/
bench_acorn --build-index --dataset sift --n 1000000 --m 16 --ef-construction 100 --build-threads 32 \
    --data-dir data/raw --volume-dir data/volumes
bench_acorn --build-gt --predicate random --seed 12345 --dataset sift --n 1000000 --m 16 --ef-construction 100 \
    --k 100 --nq 1000 --selectivity 10%,5%,3%,1%,0.5%,0.1% --search-threads 32 \
    --data-dir data/raw --volume-dir data/volumes --gt-dir data/gt
bench_acorn --dataset sift --n 1000000 --m 16 --ef-construction 100 \
    --predicate random --seed 12345 --k 100 --nq 992 --skip-exact \
    --modes hnsw,acorn-1,racorn-1 --selectivity 1%,0.1% --ef-values 100,200,400 \
    --bridge-budget-ratios 2.0 --filter-first-threshold 0.3 --stride on \
    --runs 1 --warmup 1 --search-threads 32 \
    --data-dir data/raw --volume-dir data/volumes --gt-dir data/gt \
    --json results/quickstart.json
```

`bench/README.md` uses this index and ground truth for SIFT 1M.

## Using the library

This program builds an index over random vectors. It then searches the index with HNSW, ACORN-1 and
RACORN-1 under a predicate. About 1% of the keys pass it. `include/usearch/README.md` lists the
fields of `index_search_config_t`.

```cpp
// Filtered search with HNSW, ACORN-1 and RACORN-1 on one index.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <unordered_set>
#include <vector>

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

using namespace unum::usearch;
using index_t = index_gt<float, std::size_t, std::size_t>;
using member_t = index_t::member_cref_t;
using member_it_t = index_t::member_citerator_t;

// The index stores no vectors: it measures distances through a metric over the caller's data.
struct l2_metric_t {
    float const* data;
    std::size_t dims;
    float const* row(std::size_t key) const { return data + key * dims; }
    float l2(float const* a, float const* b) const { return metric_l2sq_gt<float, float>{}(a, b, dims); }
    float operator()(member_t const& a, member_t const& b) const { return l2(row(get_key(a)), row(get_key(b))); }
    float operator()(float const* q, member_t const& m) const { return l2(q, row(get_key(m))); }
    float operator()(member_it_t const& a, member_it_t const& b) const { return l2(row(get_key(*a)), row(get_key(*b))); }
    float operator()(float const* q, member_it_t const& m) const { return l2(q, row(get_key(*m))); }
};

int main() {
    std::size_t const n = 50000, dims = 64, k = 100, queries = 100;
    std::mt19937 rng(42);
    std::normal_distribution<float> gauss;
    std::vector<float> data(n * dims), query(queries * dims);
    for (float& x : data) x = gauss(rng);
    for (float& x : query) x = gauss(rng);
    std::vector<std::uint32_t> attr(n); // one random attribute per key
    for (auto& a : attr) a = rng();

    l2_metric_t metric{data.data(), dims};
    index_t index(index_config_t(16)); // M = 16
    index.reserve(index_limits_t(n, 1));
    index_update_config_t build;
    build.expansion = 100; // ef_construction
    for (std::size_t key = 0; key < n; ++key)
        index.add(key, metric.row(key), metric, build);

    // The predicate is any callable on a member: here about 1% of the keys pass.
    std::uint32_t const bound = std::uint32_t(0.01 * 4294967296.0);
    auto passes = [&](auto const& member) { return attr[get_key(member)] < bound; };

    // Ground truth: the k nearest of the passing keys, by a scan over them.
    std::vector<std::unordered_set<std::size_t>> truth(queries);
    for (std::size_t q = 0; q < queries; ++q) {
        std::vector<std::pair<float, std::size_t>> scan;
        for (std::size_t key = 0; key < n; ++key)
            if (attr[key] < bound) scan.emplace_back(metric.l2(query.data() + q * dims, metric.row(key)), key);
        std::size_t const top = std::min(k, scan.size());
        std::partial_sort(scan.begin(), scan.begin() + top, scan.end());
        for (std::size_t i = 0; i < top; ++i) truth[q].insert(scan[i].second);
    }

    index_search_config_t hnsw, acorn1, racorn1; // hnsw keeps the defaults: in-filtering, no probe
    hnsw.expansion = acorn1.expansion = racorn1.expansion = 100; // ef
    acorn1.filter_first_threshold = 0.3; // filter-first when at most 30% of the entry neighborhood passes
    racorn1.filter_first_threshold = 0.3;
    racorn1.bridge_budget_ratio = 2.0;   // br: Adaptive Search Fallback; 0 is ACORN-1
    char const* names[] = {"HNSW", "ACORN-1", "RACORN-1"};
    index_search_config_t const* configs[] = {&hnsw, &acorn1, &racorn1};
    for (int m = 0; m < 3; ++m) {
        std::size_t hits = 0;
        double seconds = 0;
        for (std::size_t q = 0; q < queries; ++q) {
            float const* v = query.data() + q * dims;
            auto start = std::chrono::steady_clock::now();
            auto found = index.search(v, k, metric, *configs[m], passes, dummy_prefetch_t{}, dummy_preevaluate_t{});
            seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            for (std::size_t i = 0; i < found.size(); ++i) hits += truth[q].count(get_key(found[i].member));
        }
        std::printf("%-8s recall@%zu at 1%%: %.3f, latency %.2f ms\n", names[m], k, double(hits) / double(queries * k),
                    seconds / queries * 1e3);
    }
}
```

```bash
g++ -std=c++17 -O3 -march=native -DRACORN_USE_NUMKONG=1 -I include example.cpp -o example && ./example
```

```
HNSW     recall@100 at 1%: 0.994, latency 6.48 ms
ACORN-1  recall@100 at 1%: 0.763, latency 1.10 ms
RACORN-1 recall@100 at 1%: 0.941, latency 1.48 ms
```

## Datasets

The raw vectors are external distributions, under the terms of their distributors. The drivers read
them from these places under the working root:

| Dataset | Source | Directory |
|---|---|---|
| SIFT 1M, GIST 1M | [corpus-texmex](http://corpus-texmex.irisa.fr/), `sift.tar.gz` and `gist.tar.gz` (FTP) | `data/raw` |
| SIFT 100M | the first 100M vectors of BIGANN in the Big ANN benchmark distribution, converted by `tools/prefix_to_fvecs.py` | `data/raw-sift100m` |
| Text2Image 40M | the first 40M vectors of Yandex Text-to-Image in the Big ANN benchmark distribution, cut by `tools/prefix_to_fvecs.py` | `data/raw` |
| YFCC 1M, 10M | the CLIP embeddings and tags of the Big ANN filtered track | `data/raw-yfcc10m` |
| LAION 1M, 25M | the public LAION-400M CLIP image embeddings, converted by `tools/laion_to_fvecs.py` | `data/raw-laion1m`, `data/raw-laion25m` |
| LAION metadata | the metadata shards of the same release, for the filters of Sections VI-C and VI-D | `data/laion_meta` |

[`tools/get_datasets.sh`](tools/get_datasets.sh) downloads and converts them. It takes one name per
dataset. Run it from the working root, with `ROOT` and `REPO` set as in `bench/README.md`, Setup:

```bash
cd $ROOT
bash $REPO/tools/get_datasets.sh sift gist sift100m text2image yfcc laion1m laion25m laion-meta
```

## Experiments

`bench/README.md` gives the commands of every figure and table of the paper.

## License & Attribution

```
RACORN
Copyright (c) 2026-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

- USearch and NumKong upstream attribution: [Apache License 2.0](NOTICE)
- NaviX authors' code (Kuzu), reimplemented in `include/usearch/navix.hpp`: [MIT License](NOTICE)
