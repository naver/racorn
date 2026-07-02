# bench_acorn

ACORN-1 filter-search performance benchmark. Compares HNSW, ACORN-1,
RACORN-1, and RACORN-1+ across a range of selectivity values (100% down
to 0.01%, 21 levels total) and reports recall / latency / distance
counters as medians.

Measured selectivity grid: 100%, 90%, 70%, 50%, 30%, 20%, 10%, 5%, 4%, 3%,
2%, 1%, 0.8%, 0.6%, 0.5%, 0.4%, 0.3%, 0.2%, 0.1%, 0.05%, 0.01%.


## Build

The same source file produces two binaries; the `USEARCH_ENABLE_3HOP`
macro controls whether the 3-hop expansion code path is compiled in and
whether buffers are sized for it.

| Binary | `USEARCH_ENABLE_3HOP` | 3-hop code path | `acorn_passed/failed_candidates` size |
|---|---|---|---|
| `bench_acorn` (default) | 0 | excluded | ~M² |
| `bench_acorn_3hop` | 1 | compiled in | ~M³ + a separate dedup set |

```bash
# from the project root
cmake --build build --target bench_acorn          # default (no 3-hop)
cmake --build build --target bench_acorn_3hop      # 3-hop enabled
cmake --build build --target bench_acorn bench_acorn_3hop  # both
```

Both binaries accept `--3hop-threshold`, but in `bench_acorn` (no 3-hop)
the flag is **not** a silent no-op — it emits a warning to stderr and is
otherwise ignored. The banner's `3hop thresh` line is shown only by
`bench_acorn_3hop`.

## Data directory layout

```
data/
  raw/       raw datasets (fvecs, fbin)
  volumes/   index volumes (.usearch)
  gt/        ground truth files (.bin)
  clusters/  cluster ID files (.bin, produced by scripts/build_clusters.py)
```

## Supported datasets

| Dataset | Format | Dim | Max N | Default metric | Queries | File paths |
|---|---|---|---|---|---|---|
| sift (default) | fvecs | 128 | 1,000,000 | l2 | 10,000 | `data/raw/sift_base.fvecs`, `data/raw/sift_query.fvecs` |
| gist | fvecs | 960 | 1,000,000 | l2 | 1,000 | `data/raw/gist_base.fvecs`, `data/raw/gist_query.fvecs` |
| text2image | fbin | 200 | 40,000,000 | cosine | 100,000 | `data/raw/text2image-40M.fbin`, `data/raw/text2image-query-100K.fbin` |

For where to obtain each file and how to lay them out, see
[`datasets.md`](datasets.md).

## Options

```
--dataset <name>       Dataset: sift, gist, text2image (default: sift)
--n <N>                Number of vectors to index (default: 1000000)
--m <N>                Graph connectivity M (default: 16)
--ef-construction <N>  Build-time exploration width (default: 100)
--ef-search <N>        Search-time exploration width (default: 200)
--k <N>                Top-K result count (default: 100)
--metric <name>        Distance metric: l2, cosine
                       (default: l2 for sift/gist, cosine for text2image)
--nq <N>               Number of benchmark queries (default: 100)
--search-threads <N>   Parallel search threads (default: 1)
--runs <N>             Measurement repetitions (default: 10)
--warmup <N>           Warmup iterations (default: 3)
--2hop-threshold <F>   ACORN-1 two-hop expansion threshold (default: 0.3)
--bridge-budget-ratio <F>  RACORN-1 bridge budget ratio (default: 1.0).
                           Scales the per-cycle bridge budget as
                           `one_hop_count * ratio`, applied to both the
                           fallback-trigger guard and the bridge count
                           cap. Applies to RACORN-1 / RACORN-1+ (ignored
                           by ACORN-1).
                           - 0.0: fallback fully disabled (RACORN-1
                                  degenerates to ACORN-1 behavior)
                           - < 1.0: conservative (fallback fires less
                                  often; fewer bridges)
                           - 1.0 (default): budget = 1-hop visit count
                           - > 1.0: aggressive (fallback fires more
                                  often; more bridges)
--fallback-threshold <F>   RACORN-1+ exact-fallback threshold
                           (default: 0.003). Whenever the observed bridge
                           selectivity (`fb_pass / fb_eval`) falls to or
                           below this value, the RACORN-1 traversal bails
                           out and the query is finished by
                           `search_exact_`. Only evaluated after the
                           warm-up condition `fb_count > top_limit/2` is
                           satisfied. Set to 0 to make RACORN-1+ behave
                           exactly like RACORN-1. In QPS mode the value
                           is scaled per ef_search as `base × ef/200`,
                           with an additional ×0.1 branch for N > 1M.
--3hop-threshold <F>   3-hop expansion threshold (default: -1, disabled).
                       Fires when `passed_2hop < connectivity_base * threshold`.
                       Mutually exclusive with RACORN-1 (else-if branch);
                       ignored if RACORN-1 is active. Only effective in
                       the `bench_acorn_3hop` binary — `bench_acorn`
                       emits a warning and ignores it.
--racorn-stride-sampling <s>
                       RACORN-1 stride sampling toggle (default: on).
                       Setting `off` disables stride down-sampling at
                       both call sites (acorn_expand_two_hop_'s 2-hop
                       passed compaction + racorn_fallback_'s bridge
                       pool). Used by the RACORN-1 ablation
                       (run_racorn1_ablation.sh).
--selectivity <list>   Restrict to specific selectivities (e.g. 10%,1%,0.1%).
                       Applies to --build-gt / latency / --qps modes
                       (ignored by --build-index).
--modes <list>         Restrict to specific modes (e.g. hnsw,acorn-1,
                       racorn-1,racorn-1+). Valid: exact, hnsw, acorn-1,
                       racorn-1, racorn-1+. Defaults to all five; QPS
                       mode rejects `exact` (only four). When specified,
                       the Baseline (no-predicate) row is skipped, and the
                       Summary table's Speedup / RecallRatio / DistRatio
                       columns are shown only if HNSW is included.
--skip-exact           Skip the Exact latency benchmark (GT for recall is
                       still computed)
--qps                  QPS mode: sweep ef_search values and report
                       recall / QPS / latency
--ef-values <list>     ef_search values for QPS mode
                       (default: 128,256,384,512)
--build-index          Build the index volume and exit
--build-gt             Build the ground-truth file for the query set and exit
--build-gt-k <N>       Per-query result count in the GT file
                       (default: same as --k)
--data-dir <path>      Directory for raw datasets (default: data/raw)
--volume-dir <path>    Directory for index volumes (default: data/volumes)
--gt-dir <path>        Directory for ground-truth files (default: data/gt)

── Cluster correlation mode (replaces the 21 modulo cells with one cluster cell) ────
--cluster-correlation <s>  Filter mode: none|positive|negative (default: none).
                           positive: target = query.cluster_id
                             → predicate `node.cluster_id == query.cluster_id`
                             (selectivity ~ 1/K, query-correlated cluster)
                           negative: target = a random cluster id that is
                             not query.cluster_id (deterministic
                             mt19937_64(seed + qi))
                             → predicate `node.cluster_id == target`
                             (selectivity ~ 1/K, cluster unrelated to the
                             query)
                           Replaces the 21 modulo cells with a single
                           cluster cell.
--cluster-k <N>            K of the cluster files (default: 100). Negative
                           mode requires K >= 2.
--cluster-dir <path>       Cluster file directory (default: data/clusters).
                           Filename pattern:
                             `cluster_<dataset>_n<N>_k<K>.bin`
                             `cluster_<dataset>_query_n<N>_k<K>.bin`
                           Generated by `bench/scripts/build_clusters.py`.
                           Cluster-mode GT files carry the suffix
                           `_cluster_<pos|neg>_kK` so they live alongside
                           the modulo GT files without colliding.
```

## Execution modes

### Default mode (full benchmark)

For every selectivity, runs all five modes — Exact, HNSW, ACORN-1,
RACORN-1, RACORN-1+.

```bash
./build/bench/bench_acorn --dataset sift --nq 1000 --search-threads 32
```

### --skip-exact

Skips the Exact latency benchmark (warmup + runs) and the Baseline
section. The GT used for recall is still computed via
`compute_gt_parallel`. Four modes per selectivity (HNSW, ACORN-1,
RACORN-1, RACORN-1+).

```bash
./build/bench/bench_acorn --dataset sift --nq 1000 --search-threads 32 --skip-exact
```

### --modes filter

Selects a subset of modes. Applies to both latency and QPS modes (build
modes ignore it). Saves measurement time by running only what's needed.

- Valid (case-insensitive): `exact`, `hnsw`, `acorn-1`, `racorn-1`,
  `racorn-1+`
- `exact` is rejected in QPS mode (QPS never runs exact)
- When `--modes` is set, the Baseline (no-predicate) row is suppressed
- The Summary's Speedup / RecallRatio / DistRatio columns are shown only
  if HNSW is in the chosen set

```bash
# RACORN-1 alone (no baseline; ratio columns left blank)
./build/bench/bench_acorn --dataset sift --modes racorn-1 --nq 1000 --search-threads 32

# HNSW + RACORN-1 (ratios shown)
./build/bench/bench_acorn --dataset sift --modes hnsw,racorn-1 --nq 1000 --search-threads 32

# Filter in QPS mode
./build/bench/bench_acorn --dataset sift --qps --modes acorn-1,racorn-1 --nq 1000 --search-threads 32
```

Invalid combinations (e.g. `--modes exact --skip-exact` leaving nothing
to run) error out with an explicit message and exit.

### --selectivity filter (detail mode)

`--selectivity` filters selectivities (not modes). Applies to latency
(default), `--qps`, and `--build-gt` (ignored by `--build-index`). In
latency mode the per-query recall distribution (avg, p50, p90, p95, p99)
is always printed for every cell. Exact is included by default;
use `--skip-exact` to drop it.

```bash
# Latency mode with Exact
./build/bench/bench_acorn --dataset sift --selectivity 10%,1%,0.1% --nq 1000 --search-threads 32

# Latency mode without Exact
./build/bench/bench_acorn --dataset sift --selectivity 10%,1%,0.1% --nq 1000 --search-threads 32 --skip-exact

# QPS mode with the filter
./build/bench/bench_acorn --dataset sift --qps --selectivity 1%,0.1% --nq 1000 --search-threads 32
```

### --3hop-threshold (3-hop expansion)

Effective only in the `bench_acorn_3hop` binary. When ACORN-1
(racorn=false) finds 2-hop expansion thin
(`passed_2hop < connectivity_base * threshold`), it expands further to
the neighbours of 2-hop-failed nodes. Mutually exclusive with RACORN-1
(racorn=true; else-if branch).

```bash
# Enable ACORN-1 + 3-hop on bench_acorn_3hop
./build/bench/bench_acorn_3hop --dataset sift --modes acorn-1 --3hop-threshold 1.0 \
    --nq 1000 --search-threads 32

# Same flag on bench_acorn → warning to stderr, then ignored
./build/bench/bench_acorn --dataset sift --modes acorn-1 --3hop-threshold 1.0 ...
# Warning: --3hop-threshold=1 ignored — this binary was built without USEARCH_ENABLE_3HOP=1.
#          Use `bench_acorn_3hop` to enable 3-hop expansion.
```

### --qps mode

Sweeps ef_search values and reports recall / latency / QPS. Exact is not
included. Runs HNSW, ACORN-1, RACORN-1, RACORN-1+ for every ef_search
value. The RACORN-1+ `--fallback-threshold` is scaled per ef_search as
`base × ef/200`, with an additional ×0.1 branch when N > 1M.

**Latency reporting**: in QPS mode the reported Latency(ms) is the wall
clock of the batch divided by per-query work
(`wall_time × search_threads / nq`), so it's comparable to the per-query
latency reported by non-QPS mode.

```bash
# Default ef_search grid (128,256,384,512)
./build/bench/bench_acorn --dataset sift --qps --nq 1000 --search-threads 32

# Explicit ef_search values
./build/bench/bench_acorn --dataset sift --qps --ef-values 100,200,400,800 --nq 1000 --search-threads 32
```

### --cluster-correlation mode

Replaces the 21-cell modulo selectivity sweep with a single cluster cell
whose predicate is built from K-means cluster IDs. Used to measure how
the **query–filter correlation** impacts the ACORN family.

Prerequisite: build the base + query cluster ID files with
`bench/scripts/build_clusters.py` (or `build_all_clusters.sh`). For cosine
datasets (text2image) the script L2-normalizes before running K-means.

```bash
# Build cluster files (sklearn required)
python bench/scripts/build_clusters.py --dataset sift --n 1000000 --k 100

# Latency mode — positive correlation, K=100 (selectivity ~1/K)
./build/bench/bench_acorn --dataset sift --nq 992 --search-threads 32 --skip-exact \
    --cluster-correlation positive --cluster-k 100

# Negative correlation — random non-self target (selectivity ~1/K,
# uncorrelated with the query)
./build/bench/bench_acorn --dataset sift --nq 992 --search-threads 32 --skip-exact \
    --cluster-correlation negative --cluster-k 100

# Combine with QPS mode
./build/bench/bench_acorn --dataset sift --qps --nq 992 \
    --cluster-correlation negative --cluster-k 100
```

Cluster-mode result headers add a per-query selectivity summary (the
cluster size distribution is preserved end-to-end):
```
Correlation : positive (cluster K=100)
Per-q sel.  : mean=1.0798%  p50=1.0489%  p90=1.4514%  p95=1.5312%  p99=2.1504%
```

The negative mode's random non-self target is chosen with a deterministic
per-query seed (`mt19937_64(seed=42 + qi)`); re-running with the same
seed and nq reproduces the same target assignments.

## Usage examples

### Default run (sift 1M)

```bash
./build/bench/bench_acorn
```

### gist 1M, K=10

```bash
./build/bench/bench_acorn --dataset gist --k 10 --ef-search 100
```

### text2image 1M

```bash
./build/bench/bench_acorn --dataset text2image --n 1000000 --nq 100 --search-threads 32
```

### Generate an index volume

Build the index upfront and persist it; subsequent runs with the same
parameters auto-load the volume.

```bash
# Build the volume
./build/bench/bench_acorn --dataset sift --n 1000000 --m 16 --ef-construction 100 --build-index

# Auto-load the volume and benchmark
./build/bench/bench_acorn --dataset sift --n 1000000 --k 100 --ef-search 200
```

Volume filename: `{dataset}_{N}_{D}_m{M}_ef{EF}.usearch`
- e.g. `data/volumes/sift_1000000_128_m16_ef100.usearch`

### Build / auto-load ground truth files

For an existing volume, dumps the exact-search result of every
selectivity to a binary file. **Benchmark modes (latency / qps)
auto-load any compatible GT file** and skip the exact recompute.

```bash
# Default: nq is pinned to the dataset's q_max (sift=10000, gist=1000,
# text2image=100000)
./build/bench/bench_acorn --dataset sift --n 1000000 --build-gt --search-threads 32

# Explicit nq → smaller file + faster build
./build/bench/bench_acorn --dataset sift --n 1000000 --build-gt --nq 1000 --search-threads 32

# 200 results per query
./build/bench/bench_acorn --dataset sift --n 1000000 --build-gt --build-gt-k 200 --search-threads 32

# Subset of selectivities (instead of all 21)
./build/bench/bench_acorn --dataset sift --n 1000000 --build-gt --selectivity 1%,0.1% --search-threads 32
```

**Filename**: `{dataset}_{N}_{D}_m{M}_ef{EF}_gt_k{K}_nq{NQ}.bin`
- e.g. `data/gt/sift_1000000_128_m16_ef100_gt_k100_nq1000.bin`

**Auto-load logic (benchmark modes)**:
1. Tries candidate nq values in order: `bench's nq` → `1000` →
   `dataset's q_max` → `100`
2. Picks the first file whose header is compatible:
   - magic = "GT01"
   - file_nq ≥ bench nq (only the first nq entries are used)
   - file_gt_k ≥ K (only the first K keys per query are used)
3. Matching is keyed by (modulus, threshold), so **a file with the same
   selectivity value but a different predicate is rejected**
4. If no compatible file exists, falls back to in-memory recomputation
5. Status messages at the start of the benchmark:
   - All matched: `Loaded GT file: ... (21 selectivities, 1000 queries, gt_k=100)` +
     `All 21 selectivities served from GT file`
   - Partial: `Computing exact ground truth per selectivity
     (... N of 21 selectivities, M loaded from file)`

### Index-volume behaviour

1. At startup, the benchmark checks `--volume-dir` for a volume file
   matching the current parameters.
2. If found, loads the volume (skips the build).
3. Otherwise builds the index from scratch.
4. The Summary line reports `Index source: loaded (...)` or
   `built from scratch`.

## Scripts

Pre-build:

| Script | Purpose |
|---|---|
| `build_volumes.sh` | Builds index volumes for every benchmark configuration. Accepts an optional dataset filter (`./build_volumes.sh sift gist`) and skips combos whose raw file is missing or whose volume already exists. `BENCH` / `DATA_DIR` / `VOLUME_DIR` env overrides honored. |
| `build_all_gt.sh` | Builds GT for every (dataset, N, M, ef_construction) combination. Same dataset filter + skip pattern as `build_volumes.sh`. Auto-builds missing volumes. `GT_NQ` env var tunes nq (default 1000). |
| `scripts/build_clusters.py` | Runs sklearn `MiniBatchKMeans` to produce cluster IDs for base + query vectors, writing binary files into `data/clusters/`. L2-normalizes inputs for cosine datasets (text2image). Output format: `CL01` header + uint32 cluster IDs. |
| `scripts/build_all_clusters.sh` | 4 datasets × `K_LIST` (default "10 100") batch build. Same dataset filter (`./build_all_clusters.sh sift`) and raw-file skip as the two helpers above; per-K output pairs already on disk are skipped (idempotent). Use `build_clusters.py --force` to rebuild a specific combo. |

Main sweep (the 7 stages called in order by `run_final_numa.sh`):

| # | Script | Purpose |
|--:|---|---|
| 1 | `run_bench_default.sh` | bench, default params (m=16/efc=100/efs=200) + per-dataset bridge_budget_ratio sweep. bbr=1.0 runs all 5 modes; other bbr values run racorn-1 / racorn-1+ only. |
| 2 | `run_qps_default.sh` | qps mode, default params, **single per-dataset bbr** (sift_1m=1.5, gist_1m=1.0, text2image_1m=2.5, text2image_40m=2.5; override with `BBR=...` env). Runs fixed at 3 across all datasets (override via `RUNS` env). bbr=1.0 runs all 4 modes (HNSW + ACORN-1 / RACORN-1 / RACORN-1+); other bbr values run racorn-1 / racorn-1+ only. |
| 3 | `run_bench_alt.sh` | bench, alternate params: m=32 / efc=500 / efs=100 / efs=400 × bbr sweep / 2hop_threshold sweep (0.1 / 0.5 / 1.0). |
| 4 | `run_bench_3hop.sh` | bench, 3-hop expansion sweep — needs the `bench_acorn_3hop` binary (USEARCH_ENABLE_3HOP=1). modes=acorn-1,racorn-1, ef_search ∈ {128, 256, 512}, HOP2=0.3 / HOP3=1.0. |
| 5 | `run_correlation.sh` | Cluster-correlation main sweep — **QPS-mode only**, 4 datasets × 2 corr (positive/negative) × 2 K (10/100) = 16 QPS invocations + GT build (skipped if the GT file already exists). `--modes hnsw,acorn-1,racorn-1`; `runs=3` (override via `RUNS` env). Env vars `K_LIST` / `CORR_LIST` / `NQ` / `DATASETS` / `FORCE_GT` allow partial / forced runs. |
| 6 | `run_correlation_bbr.sh` | Bridge-budget-ratio sweep on top of cluster correlation — **QPS-mode only**, default `BBR_LIST="0.25 0.5"`, `--modes hnsw,racorn-1` (HNSW serves as bbr-insensitive baseline). **Auto-skips GT build if the file already exists**; `FORCE_GT=1` forces a rebuild. |
| 7 | `run_racorn1_ablation.sh` | Modulo-selectivity RACORN-1 ablation. 4 datasets × 17 selectivities (30% ~ 0.01%) × 3 cells (current = acorn-1+racorn-1 / stride_off / bbr=0). Isolates the contributions of stride sampling vs bridge fallback. |

Entry point + smoke wrappers:

| Script | Purpose |
|---|---|
| `run_final_numa.sh` | **Main entry point**. Calls the 7 sub-scripts above in order. Pins worker threads + memory to a single NUMA socket via `numactl --cpunodebind=N --membind=N`. Each sub-script defines its own NUMA / THREADS / DATASETS so it is independently runnable as well. Result-file prefixes are disjoint across stages, so a single OUT_DIR is unambiguous. |
| `run_final_numa_light.sh` | Fast smoke variant of `run_final_numa.sh`. sift_1m only, nq=100, runs=1, default params, all modes. Used to verify RACORN-1+ exact-fallback firing and compare recall/latency. Override with `DATASETS=...` to use a different dataset row. |
| `run_correlation_light.sh` | sift_1m-only cluster-correlation smoke wrapper (calls `run_correlation.sh`). |

### Using build_all_gt.sh

```bash
cd bench

# All 12 combinations (4 datasets × 3 (M, ef_construction))
./build_all_gt.sh

# Specific datasets only (prefix match)
./build_all_gt.sh sift gist
./build_all_gt.sh text2image

# Override nq (default 1000)
GT_NQ=10000 ./build_all_gt.sh sift
```

Running `build_all_gt.sh` once before `run_final_numa.sh` lets every
benchmark case auto-load its GT, eliminating exact-recompute time.

## Output structure

### Default mode

Five modes per selectivity (four under `--skip-exact`):

| Mode | Description |
|---|---|
| Exact | Brute-force search (ground truth). Skipped under `--skip-exact`. |
| HNSW | Plain filtered HNSW. |
| ACORN-1 | Two-hop neighbour expansion (optionally 3-hop via `bench_acorn_3hop`). |
| RACORN-1 | ACORN-1 + stride sampling + bridge-node fallback (tune via `--bridge-budget-ratio`). |
| RACORN-1+ | RACORN-1 + adaptive exact-search switch. When the observed fallback selectivity drops at/below `--fallback-threshold` (after warm-up), the rest of the query is finished by `search_exact_`. Rows where at least one query switched are tagged `FBExact=YES`. |

All metrics are aggregated via a two-stage median (per-query median over
RUNs, then median across queries).

Summary table columns:

| Column | Description |
|---|---|
| Latency(ms) | Per-search latency. |
| Dist.Computed | Number of distance evaluations. |
| NodesVisited | Number of visited nodes. |
| Recall(avg) | Recall vs ground truth (mean across queries). |
| RecallRatio | Recall relative to HNSW. |
| Speedup | Speedup over HNSW. |
| DistRatio | Distance-computation reduction vs HNSW. |
| PredEvals | Predicate evaluation count (all modes; 0 if no predicate). HNSW evaluates only on distance-passing slots; ACORN-1 / RACORN-1 / RACORN-1+ batch-evaluate every unvisited 1-hop. |
| FBCount | Fallback firing count (RACORN-1 / RACORN-1+ only; blank for HNSW / ACORN-1). |
| FBSelec | Selectivity observed at fallback (RACORN-1 / RACORN-1+ only; cycles that did not fire use a NaN sentinel and render as blank). |
| FBExact | `YES` if at least one query in this cell switched to exact (RACORN-1+ only); blank otherwise. |

### QPS mode

| Column | Description |
|---|---|
| Selectivity | Filter selectivity. |
| Mode | HNSW / ACORN-1 / RACORN-1 / RACORN-1+. |
| EF_Search | Search-time exploration width. |
| Recall | Recall vs ground truth. |
| Latency(ms) | Per-query average latency (`wall_time × threads / nq`). |
| QPS | Throughput (wall-clock-based aggregate). |

### Progress reporting

- Ground truth load / compute:
  - Loaded from file: `Loaded GT file: ... (21 selectivities, 1000 queries, gt_k=100)`
  - Cache miss: `[1/18] Selectivity 100% | Exact: 1000 queries (1234.5 ms)`
- Search benchmark: `[3/84] Selectivity 100% | HNSW ---` (21 selectivity × 4 modes; total 84 with `--skip-exact`, 105 = 21 × 5 with Exact).
- QPS mode: `[12/336] 10.0% | ACORN-1 | ef=256 => recall=0.9850 qps=12345 lat=0.081ms` (21 sel × 4 modes × 4 ef_values).

### Log buffering

`stdout` / `stderr` are switched to line-buffered mode at process start
and are explicitly flushed after each (selectivity, mode) case completes.
This means that even if the process is killed mid-run, every cell
completed up to that point survives in the redirected log file —
`tail -f result.log` works for live monitoring.
