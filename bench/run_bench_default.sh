#!/usr/bin/env bash
# Latency benchmark with default parameters + bridge_budget_ratio sweep.
#
# Covers 4 datasets × per-dataset bbr sweep at the default
# (m=16, ef_construction=100, ef_search=200, fallback_threshold default
# per row of DATASETS table) settings. bbr=1.0 runs all 5 modes; other
# bbr values run racorn-1 / racorn-1+ only (bbr changes their behaviour).
#
# Output: result_${TAG}_<dataset>_bbr<bbr_tag>_<ts>.log under OUT_DIR.
#
# Usage:
#   ./run_bench_default.sh [OUT_DIR]
#
# Env overrides:
#   BENCH      bench_acorn binary (default: ./bench_acorn)
#   NUMA_NODE  numactl node (default: 0)
#   NUMACTL    full numactl prefix (default: --cpunodebind=N --membind=N)
#   THREADS    parallel search threads (default: 32)
#   NQ         queries per dataset row (overrides 7th column of DATASETS)
#   RUNS       measurement runs per cell (overrides 8th column of DATASETS)
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${1:-.}"
mkdir -p "$OUT_DIR"

NUMA_NODE="${NUMA_NODE:-0}"
NUMACTL="${NUMACTL:-numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE}}"
THREADS="${THREADS:-32}"
TS="${TS:-$(date +%Y%m%d_%H%M)}"
DIR_OPTS="--data-dir ./data/raw --volume-dir ./data/volumes --gt-dir ./data/gt"
BENCH="${NUMACTL} ${BENCH:-../build/bench/bench_acorn}"
TAG="acorn_numa"

# Dataset table: DATASET  N  NAME  FB  FB_EFS100  FB_EFS400  NQ  RUNS
NQ_DEFAULT="${NQ:-992}"

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

# Per-dataset bridge_budget_ratio sweep range.
bbrs_for() {
    case "$1" in
        sift_1m)        echo "1.0 1.5 2.0" ;;
        gist_1m)        echo "1.0 1.5 2.0 4.0" ;;
        text2image_1m)  echo "1.0 1.5 2.0 2.5 3.0 4.0" ;;
        text2image_40m) echo "1.0 1.5 2.0 2.5 3.0 4.0 8.0 16.0" ;;
        *)              echo "1.0 1.5 2.0" ;;
    esac
}

for ROW in "${DATASETS_ARR[@]}"; do
    read -r DATASET N NAME FB FB_EFS100 FB_EFS400 _NQ_ROW _RUNS_ROW <<< "$ROW"
    # Use lowercase locals so the env override (NQ / RUNS) stays untouched
    # across iterations — otherwise the first row's value leaks into
    # subsequent rows and shadows their per-row defaults.
    nq_final="${NQ:-$_NQ_ROW}"
    runs_final="${RUNS:-$_RUNS_ROW}"
    COMMON_BENCH="--dataset $DATASET --n $N --nq $nq_final --runs $runs_final --warmup 1 --skip-exact --search-threads $THREADS $DIR_OPTS"
    for BBR in $(bbrs_for "$NAME"); do
        BBR_TAG=$(echo "$BBR" | tr . _)
        echo "=== [$TAG] $NAME bench bbr=$BBR ==="
        if [ "$BBR" = "1.0" ]; then
            $BENCH --fallback-threshold $FB --bridge-budget-ratio $BBR $COMMON_BENCH \
                > "${OUT_DIR}"/result_${TAG}_${NAME}_bbr${BBR_TAG}_${TS}.log 2>&1
        else
            $BENCH --fallback-threshold $FB --bridge-budget-ratio $BBR --modes racorn-1,racorn-1+ $COMMON_BENCH \
                > "${OUT_DIR}"/result_${TAG}_${NAME}_bbr${BBR_TAG}_${TS}.log 2>&1
        fi
    done
done
