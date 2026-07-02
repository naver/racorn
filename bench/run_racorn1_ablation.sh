#!/usr/bin/env bash
# RACORN-1 ablation sweep — gist_1m / sift_1m / text2image_1m / text2image_40m.
#
# Three cells per dataset, all at default parameters:
#   (1) Current      : --modes acorn-1,racorn-1
#                      RACORN-1 with stride sampling + bridge fallback (vs ACORN-1).
#   (2) Stride off   : --modes racorn-1 --racorn-stride-sampling off
#                      RACORN-1 keeps the bridge fallback path but skips
#                      stride down-sampling at both call sites
#                      (acorn_expand_two_hop_ 2-hop passed compaction +
#                       racorn_fallback_ bridge pool).
#   (3) Fallback off : --modes racorn-1 --bridge-budget-ratio 0
#                      RACORN-1 with bridge fallback disabled (degenerates
#                      to ACORN-1 behavior; stride flag has no effect once
#                      fallback is gated out).
#
# Selectivity sweep `30%~0.01%` covers the range where RACORN-1 actually
# differentiates from ACORN-1 (mid sel: stride contribution; low sel:
# bridge fallback contribution; very low sel: cross-over toward exact).
#
# NUMA pinning (node 0) + THREADS=32 + nq=992 mirror run_final_numa.sh,
# so result/qps logs land in the same shape and per-dataset RUNS column
# stays consistent (10 for 1M datasets, 3 for 40M).
#
# Usage:
#   ./run_racorn1_ablation.sh [OUT_DIR]
#     OUT_DIR : optional output directory (default: current directory).
#
# Env var overrides:
#   BENCH=./build/bench/bench_acorn ./run_racorn1_ablation.sh OUT/
#   NQ      queries per dataset row (overrides 4th column of 5-col DATASETS)
#   RUNS    measurement runs per cell (overrides 5th column)
#   DATASETS  full 5-col row override ("dataset N name nq runs")
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${1:-.}"
mkdir -p "$OUT_DIR"

NUMA_NODE=0
NUMACTL="numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE}"
THREADS=32
TS=$(date +%Y%m%d_%H%M)
DIR_OPTS="--data-dir ./data/raw --volume-dir ./data/volumes --gt-dir ./data/gt"
BENCH="${NUMACTL} ${BENCH:-../build/bench/bench_acorn}"
TAG="racorn1_ablation"
SEL="30%,20%,10%,5%,4%,3%,2%,1%,0.8%,0.6%,0.5%,0.4%,0.3%,0.2%,0.1%,0.05%,0.01%"

# Dataset table (one row per dataset):
#   DATASET  N  NAME  NQ  RUNS
NQ_DEFAULT="${NQ:-992}"

DATASETS_DEFAULT=(
    "gist        1000000   gist_1m         $NQ_DEFAULT  10"
    "sift        1000000   sift_1m         $NQ_DEFAULT  10"
    "text2image  1000000   text2image_1m   $NQ_DEFAULT  10"
    "text2image  40000000  text2image_40m  $NQ_DEFAULT  3"
)
if [ -n "${DATASETS:-}" ]; then
    eval "DATASETS_ARR=( $DATASETS )"
else
    DATASETS_ARR=("${DATASETS_DEFAULT[@]}")
fi

run_dataset() {
    local DATASET=$1 N=$2 NAME=$3 NQ=$4 RUNS=$5
    local COMMON_BENCH="--dataset $DATASET --n $N --nq $NQ --runs $RUNS --warmup 1 --skip-exact --selectivity $SEL --search-threads $THREADS $DIR_OPTS"

    echo "=== [$TAG] $NAME (1) Current ACORN-1 + RACORN-1 ==="
    $BENCH --modes acorn-1,racorn-1 $COMMON_BENCH \
        > "${OUT_DIR}/result_${TAG}_${NAME}_current_${TS}.log" 2>&1

    echo "=== [$TAG] $NAME (2) RACORN-1 stride sampling off ==="
    $BENCH --modes racorn-1 --racorn-stride-sampling off $COMMON_BENCH \
        > "${OUT_DIR}/result_${TAG}_${NAME}_stride_off_${TS}.log" 2>&1

    echo "=== [$TAG] $NAME (3) RACORN-1 bridge fallback off (bbr=0) ==="
    $BENCH --modes racorn-1 --bridge-budget-ratio 0 $COMMON_BENCH \
        > "${OUT_DIR}/result_${TAG}_${NAME}_bbr0_${TS}.log" 2>&1
}

for ROW in "${DATASETS_ARR[@]}"; do
    read -r DATASET N NAME _NQ_ROW _RUNS_ROW <<< "$ROW"
    # Lowercase locals — see run_bench_default.sh for rationale (env override
    # must not leak between iterations).
    nq_final="${NQ:-$_NQ_ROW}"
    runs_final="${RUNS:-$_RUNS_ROW}"
    run_dataset "$DATASET" "$N" "$NAME" "$nq_final" "$runs_final"
done

echo "All RACORN-1 ablation benchmarks complete. (timestamp: ${TS})"
