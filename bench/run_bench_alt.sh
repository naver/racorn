#!/usr/bin/env bash
# Latency benchmark with alternate parameters.
#
# Per dataset, runs five parameter variants:
#   - m=32                 (heavier graph)
#   - ef_construction=500  (higher-quality build)
#   - ef_search=100        (smaller search expansion + scaled fb threshold)
#   - ef_search=400 × bbr sweep (1.0 / 1.5 / 2.0 × scaled fb threshold)
#   - 2hop_threshold sweep (0.1 / 0.5 / 1.0 — ACORN-1 / RACORN-1 / RACORN-1+)
#
# Output: result_${TAG}_<dataset>_<variant>_<ts>.log under OUT_DIR.
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

for ROW in "${DATASETS_ARR[@]}"; do
    read -r DATASET N NAME FB FB_EFS100 FB_EFS400 _NQ_ROW _RUNS_ROW <<< "$ROW"
    # Lowercase locals — see run_bench_default.sh for rationale (env override
    # must not leak between iterations).
    nq_final="${NQ:-$_NQ_ROW}"
    runs_final="${RUNS:-$_RUNS_ROW}"
    COMMON_BENCH="--dataset $DATASET --n $N --nq $nq_final --runs $runs_final --warmup 1 --skip-exact --search-threads $THREADS $DIR_OPTS"

    echo "=== [$TAG] $NAME bench m=32 ==="
    $BENCH --m 32 --fallback-threshold $FB $COMMON_BENCH \
        > "${OUT_DIR}"/result_${TAG}_${NAME}_m32_${TS}.log 2>&1

    echo "=== [$TAG] $NAME bench ef_construction=500 ==="
    $BENCH --ef-construction 500 --fallback-threshold $FB $COMMON_BENCH \
        > "${OUT_DIR}"/result_${TAG}_${NAME}_efc500_${TS}.log 2>&1

    echo "=== [$TAG] $NAME bench efs=100 ==="
    $BENCH --ef-search 100 --fallback-threshold $FB_EFS100 $COMMON_BENCH \
        > "${OUT_DIR}"/result_${TAG}_${NAME}_efs100_${TS}.log 2>&1

    for BBR in 1.0 1.5 2.0; do
        BBR_TAG=$(echo "$BBR" | tr . _)
        echo "=== [$TAG] $NAME bench efs=400 bbr=$BBR ==="
        if [ "$BBR" = "1.0" ]; then
            $BENCH --ef-search 400 --fallback-threshold $FB_EFS400 --bridge-budget-ratio $BBR $COMMON_BENCH \
                > "${OUT_DIR}"/result_${TAG}_${NAME}_efs400_bbr${BBR_TAG}_${TS}.log 2>&1
        else
            $BENCH --ef-search 400 --fallback-threshold $FB_EFS400 --bridge-budget-ratio $BBR --modes racorn-1,racorn-1+ $COMMON_BENCH \
                > "${OUT_DIR}"/result_${TAG}_${NAME}_efs400_bbr${BBR_TAG}_${TS}.log 2>&1
        fi
    done

    for HOP in 0.1 0.5 1.0; do
        HOP_TAG=$(echo "$HOP" | tr . _)
        echo "=== [$TAG] $NAME bench 2hop=$HOP ==="
        $BENCH --2hop-threshold $HOP --fallback-threshold $FB --modes acorn-1,racorn-1,racorn-1+ $COMMON_BENCH \
            > "${OUT_DIR}"/result_${TAG}_${NAME}_2hop${HOP_TAG}_${TS}.log 2>&1
    done
done
