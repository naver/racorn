#!/usr/bin/env bash
# QPS benchmark with default parameters + a single bridge_budget_ratio per
# dataset (no bbr sweep). Executes in `--qps` mode (ef_search sweep, recall +
# QPS + per-query latency).
#
# Per-dataset bbr defaults (override via BBR env to apply one value to all):
#   sift_1m        : 1.5
#   gist_1m        : 1.0
#   text2image_1m  : 2.5
#   text2image_40m : 2.5
#
# bbr=1.0 runs all 4 modes (HNSW + ACORN-1 / RACORN-1 / RACORN-1+); other bbr
# values run racorn-1 / racorn-1+ only.
#
# Runs are fixed at 3 for every dataset (override via RUNS env).
#
# Output: qps_${TAG}_<dataset>_bbr<bbr_tag>_<ts>.log under OUT_DIR.
#
# Usage / env: see run_bench_default.sh — same conventions.
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

NQ_DEFAULT="${NQ:-992}"
RUNS_FINAL="${RUNS:-3}"

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

# Per-dataset bbr mapping. Override globally via `BBR=...` env.
bbr_for() {
    if [ -n "${BBR:-}" ]; then
        echo "$BBR"; return
    fi
    case "$1" in
        sift_1m)        echo 1.5 ;;
        gist_1m)        echo 1.0 ;;
        text2image_1m)  echo 2.5 ;;
        text2image_40m) echo 2.5 ;;
        *)              echo 1.0 ;;
    esac
}

for ROW in "${DATASETS_ARR[@]}"; do
    read -r DATASET N NAME FB FB_EFS100 FB_EFS400 _NQ_ROW _RUNS_ROW <<< "$ROW"
    NQ_FINAL="${NQ:-$_NQ_ROW}"
    BBR_VAL=$(bbr_for "$NAME")
    BBR_TAG=$(echo "$BBR_VAL" | tr . _)
    COMMON_QPS="--dataset $DATASET --n $N --nq $NQ_FINAL --qps --runs $RUNS_FINAL --warmup 1 --search-threads $THREADS $DIR_OPTS"
    echo "=== [$TAG] $NAME qps bbr=$BBR_VAL ==="
    if [ "$BBR_VAL" = "1.0" ]; then
        $BENCH --bridge-budget-ratio $BBR_VAL $COMMON_QPS \
            > "${OUT_DIR}"/qps_${TAG}_${NAME}_bbr${BBR_TAG}_${TS}.log 2>&1
    else
        $BENCH --bridge-budget-ratio $BBR_VAL --modes racorn-1,racorn-1+ $COMMON_QPS \
            > "${OUT_DIR}"/qps_${TAG}_${NAME}_bbr${BBR_TAG}_${TS}.log 2>&1
    fi
done
