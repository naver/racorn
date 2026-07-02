#!/usr/bin/env python3
"""Build per-node and per-query cluster IDs for ACORN correlation experiments.

Runs MiniBatchKMeans on a dataset's base vectors, then predicts cluster IDs
for both the base set and the query set. The two binary files are written
into `data/clusters/` and consumed by `bench_acorn --cluster-correlation
positive|negative` (predicate compares node and query cluster IDs).

Dependencies: numpy, scikit-learn.

Output binary format (16-byte header + uint32 array):
  bytes  0..3  : magic "CL01"
  bytes  4..7  : count (uint32)
  bytes  8..11 : K (uint32)
  bytes 12..15 : reserved (zeroed)
  bytes 16..   : cluster IDs (uint32 × count)

Filenames:
  data/clusters/cluster_<dataset>_n<N>_k<K>.bin     # base cluster IDs (depends on N)
  data/clusters/cluster_<dataset>_query_n<N>_k<K>.bin  # query cluster IDs (per-N; tied to base centroids)

Example:
    python build_clusters.py --dataset sift --n 1000000 --k 100
    python build_clusters.py --dataset text2image --n 40000000 --k 10
"""
import os as _os

# Cap BLAS / OpenMP threads BEFORE `import numpy` — without these,
# OpenBLAS / MKL spawn one worker per logical CPU and each worker
# requests its own memory regions. On large NUMA hosts (e.g. 128-CPU
# servers) this overflows OpenBLAS's per-process region limit
# (default ~127) and MiniBatchKMeans aborts with
#   "BLAS : Program is Terminated. Because you tried to allocate too
#    many memory regions"  → segfault.
# 16 is well under the limit and still saturates a single NUMA node.
# Caller can override (e.g. OPENBLAS_NUM_THREADS=32) explicitly.
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    _os.environ.setdefault(_k, "16")

import argparse
import os
import struct
import sys
import time

import numpy as np

# Heavy `sklearn` import is deferred to main() so `--help` works even
# when scikit-learn is not installed (lets reviewers preview the CLI
# before deciding to install the dependency).


def load_fvecs(path, count_max=0):
    """Load .fvecs file into (count, dim) float32 array. count_max=0 → all."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        first_dim = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
    rec_bytes = 4 + first_dim * 4
    file_bytes = os.path.getsize(path)
    n_total = file_bytes // rec_bytes
    n = n_total if count_max <= 0 else min(n_total, count_max)
    raw = np.fromfile(path, dtype=np.float32, count=n * (first_dim + 1))
    arr = raw.reshape(n, first_dim + 1)[:, 1:].copy()  # strip per-row dim prefix
    return arr, first_dim


def load_fbin(path, count_max=0):
    """Load .fbin file into (count, dim) float32 array."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        n_total = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        dim = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        n = n_total if count_max <= 0 else min(n_total, count_max)
        arr = np.fromfile(f, dtype=np.float32, count=n * dim).reshape(n, dim)
    return arr, dim


def l2_normalize(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    arr /= norms


def save_cluster_file(path, cluster_ids, K):
    cluster_ids = np.asarray(cluster_ids, dtype=np.uint32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"CL01")
        f.write(struct.pack("<II", len(cluster_ids), K))
        f.write(b"\x00\x00\x00\x00")
        f.write(cluster_ids.tobytes(order="C"))
    print(f"  wrote {path}  ({len(cluster_ids)} cluster IDs, K={K})")


def load_dataset(dataset, n, data_dir):
    """Load (base, queries) for a dataset. Returns (base_arr, query_arr, dim, metric)."""
    if dataset == "sift":
        base, dim = load_fvecs(os.path.join(data_dir, "sift_base.fvecs"), n)
        queries, _ = load_fvecs(os.path.join(data_dir, "sift_query.fvecs"))
        metric = "l2"
    elif dataset == "gist":
        base, dim = load_fvecs(os.path.join(data_dir, "gist_base.fvecs"), n)
        queries, _ = load_fvecs(os.path.join(data_dir, "gist_query.fvecs"))
        metric = "l2"
    elif dataset == "text2image":
        base, dim = load_fbin(os.path.join(data_dir, "text2image-40M.fbin"), n)
        queries, _ = load_fbin(os.path.join(data_dir, "text2image-query-100K.fbin"))
        metric = "cosine"
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return base, queries, dim, metric


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["sift", "gist", "text2image"])
    parser.add_argument("--n", type=int, required=True, help="Base vector count")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/clusters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--max-no-improvement", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    # Deferred import — see top-of-file note.
    from sklearn.cluster import MiniBatchKMeans

    base_path = os.path.join(args.out_dir, f"cluster_{args.dataset}_n{args.n}_k{args.k}.bin")
    query_path = os.path.join(args.out_dir, f"cluster_{args.dataset}_query_n{args.n}_k{args.k}.bin")
    if not args.force and os.path.exists(base_path) and os.path.exists(query_path):
        print(f"[skip] outputs already exist:\n  {base_path}\n  {query_path}")
        print("       (use --force to overwrite)")
        return 0

    print(f"=== build_clusters: dataset={args.dataset} n={args.n} k={args.k} ===")
    t0 = time.time()
    base, queries, dim, metric = load_dataset(args.dataset, args.n, args.data_dir)
    print(f"  loaded base={base.shape} queries={queries.shape} dim={dim} metric={metric}  ({time.time()-t0:.1f}s)")

    if metric == "cosine":
        t1 = time.time()
        l2_normalize(base)
        l2_normalize(queries)
        print(f"  L2-normalized for cosine metric  ({time.time()-t1:.1f}s)")

    # MiniBatchKMeans handles both 1M and 40M scales identically; numpy/BLAS
    # parallelism already saturates inner-loop work via OMP_NUM_THREADS.
    print(f"  fitting MiniBatchKMeans (k={args.k}, batch={args.batch_size}, n_init={args.n_init})...")
    t1 = time.time()
    km = MiniBatchKMeans(
        n_clusters=args.k,
        batch_size=args.batch_size,
        random_state=args.seed,
        n_init=args.n_init,
        max_iter=args.max_iter,
        max_no_improvement=args.max_no_improvement,
        verbose=args.verbose,
    )
    km.fit(base)
    print(f"  fit done  ({time.time()-t1:.1f}s, inertia={km.inertia_:.3e})")

    t1 = time.time()
    base_cids = km.predict(base)
    query_cids = km.predict(queries)
    print(f"  predicted base+query cluster IDs  ({time.time()-t1:.1f}s)")

    # cluster size distribution (for sanity check)
    sizes = np.bincount(base_cids, minlength=args.k)
    print(f"  cluster sizes: min={sizes.min()} max={sizes.max()} mean={sizes.mean():.1f} "
          f"std={sizes.std():.1f}")
    print(f"  query distribution across clusters: queries-per-cluster "
          f"min={np.bincount(query_cids, minlength=args.k).min()} "
          f"max={np.bincount(query_cids, minlength=args.k).max()}")

    save_cluster_file(base_path, base_cids, args.k)
    save_cluster_file(query_path, query_cids, args.k)
    print(f"=== done in {time.time()-t0:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
