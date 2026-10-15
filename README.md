# RACORN-1 / ACORN-1 Benchmark Artifact

> **Note:** A revised version of the paper and this artifact is in preparation
> and is planned for release in November 2026.

This repository is a paper supplementary artifact containing the reference
implementation and measurement harness for the RACORN-1 / ACORN-1
algorithms. The algorithm body is provided as a single header
(`include/usearch/index.hpp`); the measurement harness is a single C++
binary (`bench/bench_acorn.cpp`).

The base library is [USearch](https://github.com/unum-cloud/USearch)
(Apache 2.0). This artifact is a derivative work that adds the ACORN-1 /
RACORN-1 / RACORN-1+ search paths, 3-hop expansion, and the cluster-
correlation experiments on top of the USearch headers; it is released
under the same license. See [`LICENSE`](LICENSE) for the artifact license
and the USearch upstream attribution.

---

## Paper & Citation

**RACORN-1: Adaptive Recall-Preserving Speedup for Low-Selectivity Filtered
Vector Search.** Yoonseok Kim, Gyusik Choe. arXiv:2607.00768, 2026.
<https://arxiv.org/abs/2607.00768>

```bibtex
@misc{kim2026racorn1,
  title        = {RACORN-1: Adaptive Recall-Preserving Speedup for Low-Selectivity Filtered Vector Search},
  author       = {Kim, Yoonseok and Choe, Gyusik},
  year         = {2026},
  eprint       = {2607.00768},
  archivePrefix = {arXiv},
  primaryClass = {cs.DB},
  url          = {https://arxiv.org/abs/2607.00768},
}
```

---

## Directory Layout

```
.
├── CMakeLists.txt                  # top-level build (bench_acorn / bench_acorn_3hop)
├── README.md                       # this document
├── LICENSE                         # Apache 2.0 (artifact) + USearch upstream attribution
├── VERSION
├── fp16/                           # vendored fp16 headers (half-precision support)
├── include/usearch/
│   ├── index.hpp                   # HNSW + ACORN-1 / RACORN-1 / RACORN-1+ / 3-hop body
│   ├── index_plugins.hpp           # distance metrics (L2 / cosine / IP) + fp16 conversion
│   └── acorn_code_guide.md         # code guide: ACORN-1/RACORN-1/RACORN-1+ function reference
└── bench/
    ├── CMakeLists.txt              # bench_acorn / bench_acorn_3hop targets
    ├── bench_acorn.cpp             # benchmark body (CLI / build-gt / latency / qps)
    ├── bench_acorn.md              # build instructions + every CLI flag + experiment modes
    ├── run_correlation.md          # cluster-correlation reproduction guide
    ├── *.sh                        # experiment sweep scripts (see below)
    └── scripts/
        ├── build_clusters.py       # K-means cluster ID builder (sklearn)
        ├── build_all_clusters.sh   # 4 datasets × K=10,100 batch build
        └── README.md
```

---

## Quick Start

Two entry points are provided:
1. **Light reproduction** (`run_light.sh`) — one dataset, smoke parameters.
   Recommended starting point for reviewers to verify the pipeline builds
   and runs end-to-end (~21 min on a modern host).
2. **Paper-grade sweep** — see [§Reproducing the Experiments](#reproducing-the-experiments)
   for the full nq=992 multi-dataset measurement protocol (bench-mode
   `runs=10` for 1M datasets / `runs=3` for text2image_40m; qps-mode and
   cluster-correlation stages fix `runs=3` across all datasets).

### Option A — Light reproduction (one dataset)

```bash
# Step 1: download one dataset (sift / gist / text2image)
./download_datasets.sh sift

# Step 2: build + index + GT + clusters + 7-stage smoke sweep
./run_light.sh [OUT_DIR] [--no-numa]
```

Default dataset is `sift_1m`. Select a different one via the `DATASET`
env (only one at a time):

```bash
DATASET=gist           ./run_light.sh results/
DATASET=text2image_1m  ./run_light.sh results/
DATASET=text2image_40m ./run_light.sh results/      # needs ~32 GB
```

Valid values: `sift` (default), `gist`, `text2image_1m`,
`text2image_40m`. Defaults: **THREADS=32**, NUMA pinning **off**
(`run_light.sh` → `run_all.sh` exports `NUMACTL=""` unless you set
the env explicitly — see the **NUMA pinning policy** box further down).
`run_light.sh` exits with an explicit error if the required dataset
files are missing — see `bench/datasets.md` for the manual download
recipe if the FTP download fails in your environment.

Measured wall time (sift_1m, THREADS=32, NUMA off, scikit-learn present
→ stages 5-6 run):

| step | time |
|---|---:|
| cmake build (cold) | ~17 s |
| HNSW volumes (sift m=16 + m=32) | ~90 s |
| Ground truth (m=16, nq=1000) | ~65 s |
| Cluster files (K=10 + K=100) | ~25 s |
| Bench (run_final_numa_light.sh, stages 1-7) — **THREADS=32** | **~37 min** |
| └ stage 2 `run_qps_default.sh` (ef × selectivity sweep at per-dataset bbr) | ~16 min ← dominant |
| └ stage 3 `run_bench_alt.sh`                                  | ~11 min |
| └ stage 5 `run_correlation.sh`                                | ~5 min |
| └ stage 1 `run_bench_default.sh`                              | ~2 min |
| └ stage 6 `run_correlation_bbr.sh`                            | ~2 min |
| └ stage 4 `run_bench_3hop.sh`                                 | ~1.5 min |
| └ stage 7 `run_racorn1_ablation.sh`                           | ~30 s |
| **Total `run_light.sh`** | **~40 min** |

> When scikit-learn is **not** installed, stages 5-6 (cluster correlation)
> auto-skip and the bench portion drops to ~30 min — total `run_light.sh`
> ~32 min. Install `numpy scikit-learn scipy` to enable the cluster stages
> (see [Option B step 4](#option-b--manual-step-by-step)).

(Measured on a reviewer-class workstation: 2× Intel Xeon Silver 4114
(40 logical cores total, exposed as a single NUMA node), 128 GiB RAM,
RHEL 8.9, GCC 8.5.0, CMake 3.18.4, scikit-learn 0.24.2, `--no-numa`.
NUMA-pinned runs, faster CPUs, and other datasets — gist higher-dim,
text2image larger — will differ.)

> **For the full paper sweep** (4 datasets × nq=992, runs per stage as above): see
> [§Reproducing the Experiments](#reproducing-the-experiments). It
> drives `bench/run_final_numa.sh` directly. `run_all.sh` is the shared
> setup helper invoked by `run_light.sh`; it does not by itself
> produce paper-grade measurements.

> **NUMA pinning policy (read this once)**:
> - **Option A (`run_light.sh`)** — always NUMA **off** by default
>   (`run_all.sh` forces `NUMACTL=""`). To enable on a NUMA machine:
>   `NUMACTL="numactl --cpunodebind=0 --membind=0" ./run_light.sh ...`
> - **Direct `./bench/run_final_numa.sh` / `./bench/run_final_numa_light.sh`**
>   — NUMA **on** by default. Pass `--no-numa` (or `NUMACTL=""` env) on
>   machines without `numactl` / with a single NUMA node.
>
> Paper-grade measurements use the direct-invocation path so node-0
> pinning is active.

---

### Option B — Manual step-by-step

```bash
# 1. Download
./download_datasets.sh sift

# 2. Build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. Build HNSW index volumes + ground truth
cd bench
./build_volumes.sh      # all datasets in DATA_DIR
./build_all_gt.sh sift  # exact GT for sift_1m
cd -

# 4. Build cluster files (optional; needed for stages 5-6)
pip install --user numpy scikit-learn scipy
cd bench
DATA_DIR=../data/raw OUT_DIR=../data/clusters scripts/build_all_clusters.sh sift
cd -

# 5. Run the smoke sweep
cd bench
STAGES="1 2 3 4 7" ./run_final_numa_light.sh ../results_smoke --no-numa
```

> **Other datasets?** Substitute `gist` / `text2image_1m` /
> `text2image_40m` for `sift` in steps 1, 3, 4 and pass
> `DATASET=<name>` to step 5, e.g.
> `DATASET=gist ./run_final_numa_light.sh ../results_smoke --no-numa`.
> (`DATASET=` is a shortcut for the matching 7-col `DATASETS` row used
> by Option A's `run_light.sh`.)

---

## Recommended Environment

Build requirements are the same for both light and full mode.
The difference between the two modes is only in hardware specs for
measurement (CPU, RAM, NUMA).

### Build requirements (both modes)

| Item | Requirement |
|---|---|
| **OS** | Linux x86-64 (kernel 4.x+); macOS works but is untested |
| **Compiler** | GCC 12+ recommended (GCC 8+ should work; verified on GCC 12/13/14) |
| **CMake** | 3.11+ |
| **pthread** | standard library |

### Hardware — full sweep (paper-grade results)

The latency numbers in the paper were measured with default parameters
(`THREADS=32`, `nq=992`, NUMA-pinned; `runs=10` for 1M bench cells / `runs=3`
for text2image_40m bench cells and for all qps + correlation cells) on:

| Item | Specification |
|---|---|
| **CPU** | Dual-socket server with ≥32 physical cores per NUMA node (e.g. Intel Xeon / AMD EPYC) |
| **RAM** | ≥128 GB (≥40 GB per NUMA node; text2image_40m requires ~40 GB to load) |
| **Storage** | ≥40 GB free disk space for all four datasets |
| **numactl** | Required for NUMA-pinned runs (`numactl --cpunodebind=0 --membind=0`) |

> **These are the reference specs for default parameters.** Reviewers with
> different hardware can adjust the parameters to match their setup:
> - Fewer cores → lower `THREADS` (e.g. `THREADS=8`)
> - Less RAM → use sift_1m / gist_1m only (skip text2image_40m)
> - No NUMA → pass `--no-numa` flag (latency numbers will differ but recall is preserved)
>
> See `run_all.sh` env overrides and `bench/bench_acorn.md` for the full
> list of tunable parameters (`--nq`, `--runs`, `--search-threads`, etc.).

### Hardware — light mode (smoke / reproducibility check)

Any modern Linux workstation is sufficient.

| Item | Specification |
|---|---|
| **CPU** | Any x86-64 CPU; adjust `THREADS` to available core count |
| **RAM** | ≥4 GB (sift_1m only) |
| **Storage** | ≥2 GB free disk space (sift_1m dataset + index) |
| **numactl** | Optional. `run_light.sh` defaults to NUMA off (`NUMACTL=""`); set `NUMACTL="numactl --cpunodebind=0 --membind=0"` env to enable pinning |

Thread count defaults:
- `./run_light.sh` → `run_all.sh` → **THREADS=32**
- `./bench/run_final_numa_light.sh` directly → **THREADS=16**

Override with `THREADS=N` to match your core count.

> Recall trends and FBExact patterns match the paper regardless of hardware;
> absolute latency numbers will differ.

---

## Build

### Prerequisites

See [Recommended Environment](#recommended-environment) for compiler and
hardware details. In brief:

- GCC 12+ recommended (GCC 8+ should work)
- CMake 3.11+
- pthread
- `numactl` (for NUMA-pinned runs; pass `--no-numa` to skip)

(No submodules — `fp16` headers are vendored in-tree.)

### Building bench_acorn

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Two binaries are produced under `build/bench/`:
- `bench_acorn` — default build (2-hop ACORN-1 / RACORN-1 / RACORN-1+)
- `bench_acorn_3hop` — built with `USEARCH_ENABLE_3HOP=1`,
  enables the 3-hop expansion branch

```bash
./build/bench/bench_acorn --help
```

---

## Datasets

Four BigANN-Benchmarks standard datasets are supported:

| Dataset | N | dim | metric | Source |
|---|---:|---:|---|---|
| sift_1m  | 1,000,000 | 128 | L2 | http://corpus-texmex.irisa.fr/ (`sift_base.fvecs`, `sift_query.fvecs`) |
| gist_1m  | 1,000,000 | 960 | L2 | http://corpus-texmex.irisa.fr/ (`gist_base.fvecs`, `gist_query.fvecs`) |
| text2image_1m | 1,000,000 | 200 | cosine | https://big-ann-benchmarks.com/ (Yandex T2I subset) |
| text2image_40m | 40,000,000 | 200 | cosine | https://big-ann-benchmarks.com/ (Yandex T2I 40M subset) |

All benchmark scripts `cd` to `bench/` before running, so dataset files
must be placed under `bench/data/raw/` relative to the repository root.
Override with `--data-dir <path>` if you keep the files elsewhere:

```
bench/data/
├── raw/         # raw dataset files (.fvecs / .fbin)
├── volumes/     # HNSW indexes built by bench_acorn (.usearch)
├── gt/          # ground-truth files (.bin)
└── clusters/    # cluster ID files (.bin, produced by build_clusters.py)
```

For step-by-step download instructions (URLs, expected file sizes, sanity
checks per dataset), see [`bench/datasets.md`](bench/datasets.md).

---

## Reproducing the Experiments

### 1. Build indexes and ground truth (one-shot)

```bash
cd bench

# Index volumes (all 4 datasets × 3 (M, efc) configurations)
./build_volumes.sh

# Ground truth (modulo predicate, 21 selectivities, all dataset combos)
./build_all_gt.sh
```

Both scripts accept an optional dataset filter (substring match) and
skip combinations whose raw file is missing or whose output already
exists — so a partial setup is fine:

```bash
./build_volumes.sh sift gist           # only sift + gist
./build_all_gt.sh sift                 # only sift
GT_NQ=100 ./build_all_gt.sh sift       # smaller GT (default GT_NQ=1000)
```

Env overrides:
- `build_volumes.sh`: `BENCH`, `DATA_DIR`, `VOLUME_DIR`.
- `build_all_gt.sh`: `BENCH`, `THREADS`, `GT_NQ` (paths are fixed at `./data/{raw,volumes,gt}`).

The bench-step sub-scripts (`run_bench_default.sh`, `run_qps_default.sh`,
`run_bench_alt.sh`, `run_bench_3hop.sh`, `run_racorn1_ablation.sh`,
`run_correlation.sh`, `run_correlation_bbr.sh`) additionally honor:
`NQ` (overrides the per-row query count, default = dataset row value),
`RUNS` (overrides the per-row measurement runs).
A quick reviewer-friendly invocation that pairs cleanly:

```bash
GT_NQ=10 ./build_all_gt.sh sift               # tiny GT
NQ=10 RUNS=1 ./run_final_numa.sh OUT_DIR      # tiny sweep, same scope
```

#### Optional: cluster ID files (required only for stages 5-6)

Stages 5-6 of the main sweep (cluster correlation) need per-node /
per-query cluster IDs precomputed from raw vectors. This step requires
Python + scikit-learn; skip it if you only need modulo-selectivity
results (stages 1-4, 7).

```bash
pip install --user numpy scikit-learn scipy

cd bench
DATA_DIR=../data/raw OUT_DIR=../data/clusters \
    scripts/build_all_clusters.sh           # all 4 datasets × K=10,100
# or limit by dataset (substring match, same as the helpers above):
DATA_DIR=../data/raw OUT_DIR=../data/clusters \
    scripts/build_all_clusters.sh sift      # sift only
```

Idempotent: existing per-K output pairs and missing-raw datasets are
skipped cleanly. Cluster *GT* files are NOT built here — they are
generated inline by `run_correlation.sh` per (dataset, corr, K) tuple
with the `_cluster_<pos|neg>_kK` suffix so they never collide with the
modulo GT files. See [`bench/run_correlation.md`](bench/run_correlation.md)
for the full reproduction recipe.

### 2. Main sweep — modulo selectivity (ACORN-1 / RACORN-1 / RACORN-1+)

```bash
cd bench
./run_final_numa.sh OUT_DIR      # full sweep (4 datasets)
# or
./run_final_numa_light.sh OUT_DIR  # smoke run (sift_1m only, nq=100, runs=1)
```

> **NUMA pinning** is **ON by default** for both scripts when invoked
> directly (paper-grade measurements assume node-0 pinning). Pass
> `--no-numa` to disable on machines without `numactl` or with a single
> NUMA node. To check, run `numactl --hardware` — if it reports
> `available: 1 nodes (0)`, pin to node 0 is a no-op and `--no-numa`
> is the right choice.
>
> `./run_final_numa.sh OUT_DIR --no-numa`
>
> (Caveat: when called indirectly through `run_light.sh` /
> `run_all.sh`, `NUMACTL` is forced to the empty string upstream, so
> that path runs NUMA off regardless. Use the direct invocations above
> for paper-grade pinned runs.)

`run_final_numa.sh` runs the following 7 stages in one pass:

| Stage | Sub-script | Contents |
|--:|---|---|
| 1 | `run_bench_default.sh` | bench, default params + per-dataset bridge_budget_ratio sweep (modes: HNSW + ACORN-1 / RACORN-1 / RACORN-1+ at bbr=1.0, racorn-1 / racorn-1+ at other bbr values). |
| 2 | `run_qps_default.sh` | qps mode, default params, **single per-dataset bbr** (sift_1m=1.5, gist_1m=1.0, text2image_1m=2.5, text2image_40m=2.5). Runs fixed at 3 across all datasets (override via `RUNS` / `BBR` env). |
| 3 | `run_bench_alt.sh` | bench, alternate params (m=32 / efc=500 / efs sweep / 2hop sweep) |
| 4 | `run_bench_3hop.sh` | bench, 3-hop expansion sweep (`bench_acorn_3hop` required) |
| 5 | `run_correlation.sh` | cluster correlation (positive / negative, K=10/100). **QPS-mode only**, `--modes hnsw,acorn-1,racorn-1`, runs=3. GT build is skipped if the file already exists (`FORCE_GT=1` forces). |
| 6 | `run_correlation_bbr.sh` | cluster correlation × bridge_budget_ratio={0.25, 0.5}. **QPS-mode only**, `--modes hnsw,racorn-1`. |
| 7 | `run_racorn1_ablation.sh` | RACORN-1 ablation (current / stride_off / bbr=0) |

#### Light mode vs Full mode

`run_final_numa_light.sh` is a reduced-workload variant of `run_final_numa.sh`
designed for quick validation without dedicated hardware. The same 7 stages
run, but with a single dataset and fewer queries/runs.

| | `run_final_numa.sh` (full) | `run_final_numa_light.sh` (light) |
|---|---|---|
| **Datasets (default)** | gist_1m, sift_1m, text2image_1m, text2image_40m | sift_1m only |
| **Datasets (override)** | `DATASETS=...` to subset / extend | `DATASETS=...` env accepts the same multi-row format; `run_light.sh` adds a `DATASET=` shortcut for a single named dataset |
| **Queries (nq)** | 992 | 100 |
| **Runs per cell** | bench: 10 (3 for text2image_40m); qps + correlation: 3 fixed | 1 |
| **NUMA pinning** | on by default; `--no-numa` to disable | on by default **when invoked directly** (`./bench/run_final_numa_light.sh ...`); **off** when invoked through `run_light.sh` (`run_all.sh` upstream forces `NUMACTL=""`) |
| **Search threads** | 32 (override: `THREADS=N`) | 16 direct / 32 via run_light.sh (override: `THREADS=N`) |
| **bbr sweep range** | full per-dataset range | matches the rows actually in `DATASETS` (sift_1m by default) |
| **Cluster stages (5-6)** | all `DATASETS` × pos/neg × K=10/100 (4 datasets by default) | same — auto-skipped if cluster files absent (`STAGES="1 2 3 4 7"`) |
| **Purpose** | Paper-grade measurement | Smoke / reproducibility check |

```bash
# Full sweep (paper results, NUMA on by default)
cd bench && ./run_final_numa.sh OUT_DIR

# Full sweep without NUMA pinning
cd bench && ./run_final_numa.sh OUT_DIR --no-numa

# Smoke run (NUMA on by default, 16 threads)
cd bench && ./run_final_numa_light.sh OUT_DIR

# Smoke run without NUMA pinning
cd bench && ./run_final_numa_light.sh OUT_DIR --no-numa

# Adjust thread count (both modes)
THREADS=8 ./run_final_numa.sh OUT_DIR --no-numa
THREADS=8 ./run_final_numa_light.sh OUT_DIR --no-numa
```

> Skip cluster-correlation stages if cluster files are not yet built:
> `STAGES="1 2 3 4 7" ./run_final_numa_light.sh OUT_DIR`
>
> **Note**: only `run_final_numa_light.sh` honors the `STAGES` env. The
> full `run_final_numa.sh` always runs all 7 stages in order — to skip a
> stage there, edit the script or call the desired sub-script(s)
> directly.

### 3. Cluster correlation experiments standalone

Two types of files are needed:
- **Cluster ID files** — built once from raw vectors (requires sklearn)
- **Cluster GT files** — built automatically by `run_correlation.sh` inline
  (one GT file per dataset × correlation × K, suffix `_cluster_<pos|neg>_kK`)

```bash
# Step 1: build cluster ID files (one-time)
pip install --user numpy scikit-learn scipy
cd bench
DATA_DIR=../data/raw OUT_DIR=../data/clusters scripts/build_all_clusters.sh

# Step 2: run correlation sweep (cluster GT built inline automatically)
./run_correlation.sh OUT_DIR        # 4 datasets × 2 correlations × 2 K
./run_correlation_bbr.sh OUT_DIR    # additional bbr=0.25/0.5 sweep
```

See [`bench/run_correlation.md`](bench/run_correlation.md) for the full
reproduction recipe.

### 4. RACORN-1 ablation standalone

```bash
cd bench
./run_racorn1_ablation.sh OUT_DIR   # modulo-selectivity based (4 ds × 17 sel × 3 cells)
```

---

## Key CLI Options

|  | Option | Description |
|---|---|---|
| Search modes | `--modes <list>` | select from exact, hnsw, acorn-1, racorn-1, racorn-1+ |
| ACORN | `--2hop-threshold <F>` | 2-hop expansion threshold (default 0.3) |
| RACORN-1 | `--bridge-budget-ratio <F>` | bridge fallback budget scale (default 1.0; 0 disables) |
| RACORN-1 | `--racorn-stride-sampling on\|off` | toggle stride sampling (default on) |
| RACORN-1+ | `--fallback-threshold <F>` | exact-fallback selectivity threshold (default 0.003) |
| 3-hop | `--3hop-threshold <F>` | ACORN-1 only; 3-hop expansion threshold (requires `bench_acorn_3hop` binary; mutually exclusive with RACORN-1) |
| Cluster | `--cluster-correlation none\|positive\|negative` | cluster-predicate mode |
| Cluster | `--cluster-k <N>` | cluster K (default 100) |
| Measurement | `--qps`, `--runs`, `--warmup`, `--nq`, `--search-threads` | measurement setup |

> **Note on `--fallback-threshold` defaults.**
> The default RACORN-1+ exact-fallback threshold is anchored at `ef_search=200`:
> **0.003 (≈0.3%) for N ≤ 1M** and **0.0003 (≈0.03%) for N > 1M**. At
> `ef_search=100 / 400` the threshold scales with graph-traversal cost —
> ×0.5 (0.0015 / 0.00015) and ×2 (0.006 / 0.0006), respectively. In **latency
> mode** the per-ef threshold is passed explicitly by the sub-scripts via the
> dataset row (columns `FB`, `FB_EFS100`, `FB_EFS400`). In **QPS mode** it is
> computed inside the binary as `base × ef/200 × n_factor`, where
> `n_factor = 1.0 if N ≤ 1M else 0.1`, so a single QPS sweep covers all ef
> values without per-call dispatch.

See [`bench/bench_acorn.md`](bench/bench_acorn.md) for the full option list
and behavioural notes.

For a walkthrough of the ACORN-1 / RACORN-1 / RACORN-1+ code paths in
`index.hpp` (call graph, function reference, data structures), see
[`include/usearch/acorn_code_guide.md`](include/usearch/acorn_code_guide.md).

---

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

- USearch upstream attribution: [Apache License 2.0](NOTICE)
- `fp16/` (vendored): [MIT License](fp16/LICENSE) — Copyright (c) 2017
  Facebook Inc. / Georgia Institute of Technology / 2019 Google LLC.
  Source: https://github.com/maratyszcza/fp16
