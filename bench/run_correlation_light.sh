#!/usr/bin/env bash
# Light variant of run_correlation.sh — sift_1m only, K=10 + K=100, both
# correlations. Used to sanity-check the cluster experiment locally before
# running the full sweep (run_correlation.sh) on the benchmark machine.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${1:-.}"
mkdir -p "$OUT_DIR"

# DATASETS env is eval'd as a bash-array literal by run_correlation.sh
# (each element is one quoted 7-col row), so the row must be quoted to
# survive eval as a single entry.
DATASETS='"sift 1000000 sift_1m 0.003 0.0015 0.006 992 5"' \
NQ=992 \
RUNS=5 \
K_LIST="${K_LIST:-10 100}" \
CORR_LIST="${CORR_LIST:-positive negative}" \
    ./run_correlation.sh "$OUT_DIR"
