# bench/scripts/

Helper scripts that don't need to live in the main C++ build.

## `build_clusters.py` / `build_all_clusters.sh`

K-means cluster assignment for ACORN correlation experiments. The
resulting binary files are consumed by `bench_acorn
--cluster-correlation positive|negative` (the predicate compares each
node's cluster ID against the current query's cluster ID).

### Dependencies

```bash
pip install --user numpy scikit-learn scipy
```

`build_clusters.py` and `build_all_clusters.sh` cap BLAS / OpenMP
threads at 16 by default (`OPENBLAS_NUM_THREADS=16`,
`MKL_NUM_THREADS=16`, `OMP_NUM_THREADS=16`) to avoid the OpenBLAS
per-process memory-region limit (typically ~127 regions) overflowing on
large NUMA hosts where `nproc` is 128+. Override explicitly only if
you know the limit on your build of OpenBLAS / MKL is higher:

```bash
OPENBLAS_NUM_THREADS=32 ./build_all_clusters.sh
```

### Single-K build

```bash
python build_clusters.py --dataset sift --n 1000000 --k 100
python build_clusters.py --dataset text2image --n 40000000 --k 10
```

Output (default `data/clusters/`):

```
data/clusters/cluster_<dataset>_n<N>_k<K>.bin        # base cluster IDs (depends on N)
data/clusters/cluster_<dataset>_query_n<N>_k<K>.bin  # query cluster IDs (tied to base centroids of the same N)
```

### Batch build (4 datasets × K=10/100)

```bash
./build_all_clusters.sh           # all 4 datasets, python3 in PATH
./build_all_clusters.sh sift      # dataset filter (substring match)
./build_all_clusters.sh sift gist # sift + gist
PYTHON=python3.11 ./build_all_clusters.sh
K_LIST="10 100 1000" ./build_all_clusters.sh   # add more K values
```

Skips:
- combos whose raw dataset file is missing (so a partial download is fine)
- per-K output pairs that already exist (idempotent re-runs)

Use `--force` on `build_clusters.py` directly to rebuild a single
combination.

`MiniBatchKMeans` with the default `batch_size=8192` and `n_init=3`
keeps memory bounded so the same code path scales from 1M to 40M.

### File format

```
bytes  0..3  : magic "CL01"
bytes  4..7  : count (uint32)
bytes  8..11 : K (uint32)
bytes 12..15 : reserved (zeroed)
bytes 16..   : cluster IDs (uint32 × count)
```

`bench_acorn` validates magic, count, and K when loading.
