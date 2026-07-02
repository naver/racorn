#!/usr/bin/env bash
# 3-hop expansion sweep (bench mode, requires bench_acorn_3hop binary).
#
# Requires the bench_acorn_3hop binary built with USEARCH_ENABLE_3HOP=1
# so that `--3hop-threshold` takes effect. Modes are limited to acorn-1
# and racorn-1 — 3-hop expansion is mutually exclusive with RACORN-1's
# bridge fallback, so racorn-1 is the baseline reference in this sweep.
# Single fallback threshold per dataset (ef_search=200 baseline, NOT scaled
# per ef_search) — keeps the 3-hop sweep comparable to the bench-default baseline.
#
# Sweeps ef_search ∈ {128, 256, 512} with fixed HOP2=0.3 / HOP3=1.0.
#
# Output: result_${TAG}_<dataset>_2hop<...>_3hop<...>_efs<EFS>_<ts>.log
#         under OUT_DIR.
#
# Usage:
#   ./run_bench_3hop.sh [OUT_DIR]
#
# Env overrides:
#   BENCH_3HOP  binary path (default: ./bench_acorn_3hop). Must be built
#               with USEARCH_ENABLE_3HOP=1.
#   NUMA_NODE / NUMACTL / THREADS — same as run_bench_default.sh.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${1:-.}"
mkdir -p "$OUT_DIR"

NUMA_NODE="${NUMA_NODE:-0}"
NUMACTL="${NUMACTL:-numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE}}"
THREADS="${THREADS:-32}"
TS="${TS:-$(date +%Y%m%d_%H%M)}"
DIR_OPTS="--data-dir ./data/raw --volume-dir ./data/volumes --gt-dir ./data/gt"
BENCH_3HOP="${NUMACTL} ${BENCH_3HOP:-../build/bench/bench_acorn_3hop}"
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

HOP2=0.3
HOP3=1.0
HOP_TAG="2hop$(echo "$HOP2" | tr . _)_3hop$(echo "$HOP3" | tr . _)"
MODES="acorn-1,racorn-1"

for ROW in "${DATASETS_ARR[@]}"; do
    read -r DATASET N NAME FB FB_EFS100 FB_EFS400 _NQ_ROW _RUNS_ROW <<< "$ROW"
    # Lowercase locals — see run_bench_default.sh for rationale (env override
    # must not leak between iterations).
    nq_final="${NQ:-$_NQ_ROW}"
    runs_final="${RUNS:-$_RUNS_ROW}"
    COMMON_BENCH="--dataset $DATASET --n $N --nq $nq_final --runs $runs_final --warmup 1 --bridge-budget-ratio 1.0 --skip-exact --search-threads $THREADS $DIR_OPTS"
    for EFS in 128 256 512; do
        echo "=== [$TAG] $NAME bench ${HOP_TAG} efs=$EFS ==="
        $BENCH_3HOP --2hop-threshold $HOP2 --3hop-threshold $HOP3 \
                   --fallback-threshold $FB --modes $MODES $COMMON_BENCH --ef-search $EFS \
            > "${OUT_DIR}"/result_${TAG}_${NAME}_${HOP_TAG}_efs${EFS}_${TS}.log 2>&1
    done
done
