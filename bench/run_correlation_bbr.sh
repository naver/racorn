#!/usr/bin/env bash
# Cluster-correlation sweep with non-default bridge_budget_ratio values.
#
# QPS-mode only. Mirrors run_correlation.sh but adds a `bbr` axis
# (default: 0.25 0.5). `bbr=1.0` is the default value already exercised by
# run_correlation.sh, so this script intentionally restricts to {0.25, 0.5}
# to fill in the lower end of the trade-off curve. modes are limited to
# HNSW / RACORN-1 — HNSW serves as the baseline (bbr is a no-op for HNSW)
# and RACORN-1 actually responds to the bbr axis.
#
# Cells:
#   4 datasets × |CORR_LIST| × |K_LIST| × |BBR_LIST|
# For default values that's 4 × 2 × 2 × 2 = 32 QPS cells.
#
# GT files (suffix `_cluster_<pos|neg>_kK`) are reused when present — the
# common case is that run_correlation.sh has already produced them. The
# inline --build-gt step below auto-skips any (dataset, corr, K) tuple whose
# GT file already exists in ./data/gt. Set FORCE_GT=1 to rebuild.
#
# Usage:
#   ./run_correlation_bbr.sh [OUT_DIR]
#
# Env overrides:
#   BENCH       binary (default: ./bench_acorn)
#   THREADS     parallel search threads (default: 32)
#   K_LIST      cluster K values (default: "10 100")
#   CORR_LIST   correlations (default: "positive negative")
#   BBR_LIST    bridge budget ratios (default: "0.25 0.5")
#   NQ          query count (default: 992)
#   RUNS        per-mode runs (overrides per-dataset RUNS column)
#   DATASETS    override row list (single string, see DATASETS array below)
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
TAG="corr_numa_bbr"
K_LIST="${K_LIST:-10 100}"
CORR_LIST="${CORR_LIST:-positive negative}"
BBR_LIST="${BBR_LIST:-0.25 0.5}"
NQ_DEFAULT="${NQ:-992}"

# Dataset table — same shape as run_correlation.sh:
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
# Skips the build if a matching GT file already exists in ./data/gt — the
# common case when run_correlation.sh has been executed first. Set
# FORCE_GT=1 to rebuild unconditionally.
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

# Helper: bbr value → filename tag (1.0 → 1_0, 0.25 → 0_25).
bbr_tag() { echo "$1" | tr . _; }

# QPS benchmark per (dataset, corr, K, bbr).
qps_one() {
    local DATASET="$1" N="$2" NAME="$3" NQV="$4" CORR="$5" KCL="$6" BR="$7"
    local SUFFIX BRTAG
    if [ "$CORR" = "positive" ]; then SUFFIX="pos"; else SUFFIX="neg"; fi
    BRTAG=$(bbr_tag "$BR")
    local COMMON="--dataset $DATASET --n $N --nq $NQV --qps --runs ${RUNS:-3} --warmup 1"
    COMMON+=" --search-threads $THREADS $DIR_OPTS"
    COMMON+=" --cluster-correlation $CORR --cluster-k $KCL"
    COMMON+=" --bridge-budget-ratio $BR --modes hnsw,racorn-1"
    echo "=== [$TAG] $NAME qps cluster=$CORR K=$KCL bbr=$BR ==="
    $BENCH $COMMON \
        > "${OUT_DIR}"/qps_${TAG}_${NAME}_${SUFFIX}_k${KCL}_bbr${BRTAG}_${TS}.log 2>&1
}

# 1. GT pass (one per (dataset, corr, K), reused across all bbr cells).
for ROW in "${DATASETS_ARR[@]}"; do
    read -r DATASET N NAME FB FB_EFS100 FB_EFS400 NQV _RUNS_V <<< "$ROW"  # _RUNS_V unused; RUNS env (default 3) wins
    for CORR in $CORR_LIST; do
        for KCL in $K_LIST; do
            build_gt_one "$DATASET" "$N" "$NAME" "$CORR" "$KCL" "$NQV"
        done
    done
done

# 2. QPS benchmark — bbr is the innermost loop so each (corr, K) batch is
#    contiguous in the log timeline.
for ROW in "${DATASETS_ARR[@]}"; do
    read -r DATASET N NAME FB FB_EFS100 FB_EFS400 NQV _RUNS_V <<< "$ROW"  # _RUNS_V unused; RUNS env (default 3) wins
    for CORR in $CORR_LIST; do
        for KCL in $K_LIST; do
            for BR in $BBR_LIST; do
                qps_one "$DATASET" "$N" "$NAME" "$NQV" "$CORR" "$KCL" "$BR"
            done
        done
    done
done

echo "All cluster-correlation bbr-sweep benchmarks complete. (timestamp: ${TS})"
