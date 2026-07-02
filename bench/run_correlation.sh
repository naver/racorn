#!/usr/bin/env bash
# Cluster-correlation sweep for ACORN-family search modes.
#
# QPS-mode only. Each invocation runs bench_acorn in `--qps` mode for one
# (dataset, correlation, K) tuple, restricted to HNSW / ACORN-1 / RACORN-1
# via `--modes hnsw,acorn-1,racorn-1`. Per-query selectivity stats
# (mean / p50 / p90 / p95 / p99) are printed in the result log header.
#
# Prerequisites:
#   1. Cluster files for every (dataset, K) in scope, generated via
#        bench/scripts/build_clusters.py / build_all_clusters.sh
#      (defaults: ../data/clusters).
#   2. Existing index volume per dataset (reuses the same volume as
#      run_final_numa.sh: m=16, efc=100).
#   3. GT files are produced inline by the --build-gt pass below — the
#      cluster GT files are independent of the modulo GT files because
#      the filename gets a `_cluster_<pos|neg>_kK` suffix.
#
# Usage:
#   ./run_correlation.sh [OUT_DIR]
#
# Env overrides:
#   BENCH       binary (default: ./bench_acorn)
#   THREADS     parallel search threads (default: 32)
#   K_LIST      cluster K values (default: "10 100")
#   CORR_LIST   correlations (default: "positive negative")
#   NQ          query count (default: 992 — matches run_final_numa.sh)
#   RUNS        per-mode runs (overrides per-dataset RUNS column)
#   DATASETS    override row list (see DATASETS array below for format)
#   FORCE_GT    set to 1 to rebuild GT files even if they exist (default: 0)
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${1:-.}"
mkdir -p "$OUT_DIR"

NUMA_NODE="${NUMA_NODE:-0}"
NUMACTL="${NUMACTL:-numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE}}"
THREADS="${THREADS:-32}"
TS=$(date +%Y%m%d_%H%M)
DIR_OPTS="--data-dir ./data/raw --volume-dir ./data/volumes --gt-dir ./data/gt --cluster-dir ./data/clusters"
BENCH="${NUMACTL} ${BENCH:-../build/bench/bench_acorn}"
TAG="corr_numa"
K_LIST="${K_LIST:-10 100}"
CORR_LIST="${CORR_LIST:-positive negative}"
NQ_DEFAULT="${NQ:-992}"

# Dataset table — same shape as run_final_numa.sh:
#   DATASET  N  NAME  FB  FB_EFS100  FB_EFS400  NQ  RUNS
DATASETS_DEFAULT=(
    "gist        1000000   gist_1m         0.003   0.0015   0.006   $NQ_DEFAULT  10"
    "sift        1000000   sift_1m         0.003   0.0015   0.006   $NQ_DEFAULT  10"
    "text2image  1000000   text2image_1m   0.003   0.0015   0.006   $NQ_DEFAULT  10"
    "text2image  40000000  text2image_40m  0.0003  0.00015  0.0006  $NQ_DEFAULT  3"
)
if [ -n "${DATASETS:-}" ]; then
    eval "DATASETS_ARR=( $DATASETS )"
else
    DATASETS_ARR=("${DATASETS_DEFAULT[@]}")
fi

# Build (or reuse) GT for one (dataset, correlation, K) tuple.
# Skips the build if a matching GT file already exists in ./data/gt. The
# bench_acorn binary itself does not check file existence — without this
# script-level guard, every --build-gt call recomputes and overwrites the
# existing file. Set FORCE_GT=1 to rebuild unconditionally.
build_gt_one() {
    local DATASET="$1" N="$2" NAME="$3" CORR="$4" KCL="$5" NQV="$6"
    local SUFFIX
    if [ "$CORR" = "positive" ]; then SUFFIX="pos"; else SUFFIX="neg"; fi
    # GT filename pattern (dim varies per dataset, m/ef are default 16/100):
    #   data/gt/<DS>_<N>_<DIM>_m16_ef100_gt_k*_nq*_cluster_<pos|neg>_k<K>.bin
    local GT_PATTERN="./data/gt/${DATASET}_${N}_*_m16_ef100_gt_k*_nq*_cluster_${SUFFIX}_k${KCL}.bin"
    if [ "${FORCE_GT:-0}" != "1" ] && compgen -G "$GT_PATTERN" > /dev/null; then
        echo "=== [$TAG] $NAME GT cluster=$CORR K=$KCL — skip (existing file) ==="
        return
    fi
    echo "=== [$TAG] $NAME GT cluster=$CORR K=$KCL nq=$NQV ==="
    $BENCH --build-gt --dataset "$DATASET" --n "$N" --nq "$NQV" \
           --cluster-correlation "$CORR" --cluster-k "$KCL" \
           --search-threads "$THREADS" $DIR_OPTS \
           > "${OUT_DIR}"/gt_${TAG}_${NAME}_${SUFFIX}_k${KCL}_${TS}.log 2>&1
}

# QPS benchmark: HNSW / ACORN-1 / RACORN-1 only, sweeps default ef_values.
qps_one() {
    local DATASET="$1" N="$2" NAME="$3" NQV="$4" CORR="$5" KCL="$6"
    local SUFFIX
    if [ "$CORR" = "positive" ]; then SUFFIX="pos"; else SUFFIX="neg"; fi
    local COMMON="--dataset $DATASET --n $N --nq $NQV --qps --runs ${RUNS:-3} --warmup 1"
    COMMON+=" --search-threads $THREADS $DIR_OPTS"
    COMMON+=" --cluster-correlation $CORR --cluster-k $KCL"
    COMMON+=" --modes hnsw,acorn-1,racorn-1"
    echo "=== [$TAG] $NAME qps cluster=$CORR K=$KCL ==="
    $BENCH $COMMON \
        > "${OUT_DIR}"/qps_${TAG}_${NAME}_${SUFFIX}_k${KCL}_${TS}.log 2>&1
}

# 1. GT pass (parallel-friendly, but here serial — index loads share memory).
for ROW in "${DATASETS_ARR[@]}"; do
    read -r DATASET N NAME FB FB_EFS100 FB_EFS400 NQV _RUNS_V <<< "$ROW"  # _RUNS_V unused; RUNS env (default 3) wins
    for CORR in $CORR_LIST; do
        for KCL in $K_LIST; do
            build_gt_one "$DATASET" "$N" "$NAME" "$CORR" "$KCL" "$NQV"
        done
    done
done

# 2. QPS benchmark.
for ROW in "${DATASETS_ARR[@]}"; do
    read -r DATASET N NAME FB FB_EFS100 FB_EFS400 NQV _RUNS_V <<< "$ROW"  # _RUNS_V unused; RUNS env (default 3) wins
    for CORR in $CORR_LIST; do
        for KCL in $K_LIST; do
            qps_one "$DATASET" "$N" "$NAME" "$NQV" "$CORR" "$KCL"
        done
    done
done

echo "All cluster-correlation benchmarks complete. (timestamp: ${TS})"
