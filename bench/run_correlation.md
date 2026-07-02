# Cluster Correlation Experiment Guide

An experiment that measures how ACORN-family algorithms respond to
**query–filter correlation**. Unlike the original modulo predicate (a
random filter), every node carries a cluster_id, and the predicate admits
only nodes whose cluster_id equals the query's *target* cluster_id (always
the same equality test `node.cluster_id == target`).

| Mode | Target choice | Average selectivity |
|---|---|---|
| `positive` | target = query.cluster_id (the query's own cluster) | ~ 1/K (query-correlated) |
| `negative` | target = a random non-self cluster id picked deterministically per query | ~ 1/K (query-uncorrelated) |

Both modes share the same ~1/K selectivity scale; **the difference is
whether the surviving cluster is semantically close to the query
(positive) or unrelated (negative)**. The negative mode's random
non-self target is chosen with a deterministic seed
(`mt19937_64(seed + qi)`), so re-running with the same seed reproduces
the same target assignments.

By default the experiment is run with K=10 and K=100.

---

## 0. Working directory

All commands in this document assume you have changed into the `bench/`
directory at the repository root:

```bash
cd /path/to/artifact/bench    # adjust to your checkout path
```

---

## 1. Directory layout

```
bench/
├── bench_acorn              # main binary (from build/bench/bench_acorn)
├── run_correlation.sh        # main experiment sweep
├── run_correlation_light.sh  # sift_1m smoke wrapper
├── run_correlation.md        # this document
├── scripts/
│   ├── build_clusters.py     # cluster-file builder (Python · sklearn)
│   └── build_all_clusters.sh # 4 datasets × K=10,100 batch builder
└── data/
    ├── raw/                  # raw datasets (already present)
    ├── volumes/              # index volumes (already present, m=16/efc=100)
    ├── gt/                   # GT files (auto-generated)
    └── clusters/             # cluster files (generated in Step 2)
```

---

## 2. Prerequisites — Python dependencies

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user numpy scikit-learn scipy
```

Verify:
```bash
python3 -c 'import numpy, sklearn; print(numpy.__version__, sklearn.__version__)'
```

`MiniBatchKMeans` automatically uses numpy/BLAS parallelism. If
`OMP_NUM_THREADS` is not auto-detected, set it explicitly:
```bash
export OMP_NUM_THREADS=$(nproc)
```

---

## 3. Build cluster files

```bash
# (already in bench/ — see §0 above)
DATA_DIR=../data/raw OUT_DIR=../data/clusters scripts/build_all_clusters.sh
```

Output files (4 datasets × 2 K values = 8 pairs, 16 total):
- `cluster_<dataset>_n<N>_k<K>.bin` — base cluster IDs
- `cluster_<dataset>_query_n<N>_k<K>.bin` — query cluster IDs (tied to base centroids of the given N)

Idempotent: skips if both files for a (dataset, K) pair already exist.
To force regeneration:
```bash
python3 scripts/build_clusters.py --dataset sift --n 1000000 --k 100 \
    --data-dir ./data/raw --out-dir ./data/clusters --force
```

The 40M dataset uses ~32 GB of RAM and substantial CPU. Avoid running it
concurrently with other workloads.

Adding more K values (e.g. K=1000):
```bash
K_LIST="10 100 1000" DATA_DIR=../data/raw OUT_DIR=../data/clusters \
    scripts/build_all_clusters.sh
```

---

## 4. Smoke run (sift_1m)

Before the full sweep, run the lightweight smoke check to confirm the
environment and the cluster files are healthy:

```bash
# (already in bench/ — see §0 above)
./run_correlation_light.sh corr_light
```

- sift_1m only, K=10 + K=100, positive + negative
- Result logs: `corr_light/qps_corr_numa_sift_1m_*.log`,
  `corr_light/gt_corr_numa_sift_1m_*.log`
  (QPS-mode only — `run_correlation.sh` does not emit a separate latency
  pass; the per-query median latency printed inside the QPS cells is the
  latency view.)

Expected output (positive K=100, sift_1m):
```
Correlation : positive (cluster K=100)
Per-q sel.  : mean=1.08%  p50=1.05%  p90=1.45%  p95=1.53%  p99=2.15%
```
A mean near 1% (= 1/K) is normal. The negative mode should also show a
~1% scale (e.g. `mean=1.005%`).

---

## 5. Main sweep

```bash
# (already in bench/ — see §0 above)
./run_correlation.sh corr_$(date +%Y%m%d_%H%M)
```

Total work (32 invocations, QPS-mode only):

1. **GT build** — 4 datasets × 2 corr × 2 K = 16 runs
   (`bench_acorn --build-gt --cluster-correlation ...`).
   **Skipped per (dataset, corr, K) tuple if the GT file already exists**
   in `./data/gt/`. Set `FORCE_GT=1` to rebuild.
2. **QPS benchmark** — 16 runs, `--modes hnsw,acorn-1,racorn-1`,
   `--runs 3` (override via `RUNS` env), sweeps default ef_values
   (128 / 256 / 384 / 512).

NUMA node 0 is pinned by default. Override with `NUMA_NODE=1` etc. for
different topologies.

Resulting log layout (example):
```
corr_20260510_2230/
├── gt_corr_numa_sift_1m_pos_k10_*.log
├── gt_corr_numa_sift_1m_pos_k100_*.log
├── gt_corr_numa_sift_1m_neg_k10_*.log
├── gt_corr_numa_sift_1m_neg_k100_*.log
├── qps_corr_numa_sift_1m_pos_k10_*.log
├── qps_corr_numa_sift_1m_pos_k100_*.log
└── ...
```

---

## 6. Interpreting the results

The per-query selectivity statistics appear in the header of every log:
```
Correlation : positive (cluster K=100)
Per-q sel.  : mean=1.0798%  p50=1.0489%  p90=1.4514%  p95=1.5312%  p99=2.1504%
```

- **mean** — average per-query selectivity
- **p50 / p90 / p95 / p99** — percentiles of the per-query distribution
- These reflect the cluster size distribution (K-means imbalance is
  preserved end-to-end)

---

## 7. Key environment variables (`run_correlation.sh`)

| Variable | Default | Description |
|---|---|---|
| `BENCH` | `./bench_acorn` | binary path |
| `THREADS` | `32` | search threads (matches the physical core count of NUMA node 0) |
| `K_LIST` | `10 100` | cluster K values |
| `CORR_LIST` | `positive negative` | correlation modes |
| `NQ` | `992` | number of queries |
| `RUNS` | `3` | measurement runs per cell (constant; per-dataset RUNS column is not consulted) |
| `DATASETS` | (4 datasets) | dataset row override |
| `NUMA_NODE` | `0` | numactl `--cpunodebind` |
| `NUMACTL` | `numactl --cpunodebind=N --membind=N` | full numactl prefix override |

Partial-run examples.

text2image_40m only:
```bash
DATASETS='"text2image 40000000 text2image_40m 0.0003 0.00015 0.0006 992 3"' \
    ./run_correlation.sh corr_t2i40m
```

K=100 only:
```bash
K_LIST="100" ./run_correlation.sh corr_k100
```

positive only:
```bash
CORR_LIST="positive" ./run_correlation.sh corr_pos
```

Disable NUMA pinning (single-socket machines, etc.):
```bash
NUMACTL="" ./run_correlation.sh corr_nonuma
```

---

## 8. File format summary

### Cluster file (`cluster_*.bin`)
```
bytes  0..3   : magic "CL01"
bytes  4..7   : count       (uint32)
bytes  8..11  : K            (uint32)
bytes 12..15  : reserved
bytes 16..    : cluster IDs (uint32 × count)
```

### GT file (cluster mode, suffix `_cluster_<pos|neg>_kK`)
Same `GT01` format as the modulo GT files. Contains a single selectivity
block identified by the sentinel `(modulus=0, threshold=K)`, so it never
collides with modulo GT files in the same directory.
