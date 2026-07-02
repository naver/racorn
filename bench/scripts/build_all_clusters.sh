#!/usr/bin/env bash
# Build all cluster files for ACORN correlation experiments.
#
# Iterates over the dataset table and K values, calling build_clusters.py
# for each combination.
#
# - Optional dataset filter via positional args (substring match against
#   per-row aliases — sift / sift_1m / gist / gist_1m / text2image /
#   text2image_1m / text2image_40m are all valid):
#     ./build_all_clusters.sh sift            # only sift
#     ./build_all_clusters.sh sift gist       # sift + gist
#     ./build_all_clusters.sh text2image_1m   # only the 1M T2I row
# - Combos whose raw dataset file is missing are skipped (so a partial
#   download / fresh checkout still progresses for what's available).
# - Per-K output pairs that already exist are skipped (idempotent;
#   pass --force to build_clusters.py to rebuild a specific combo).
#
# Env overrides:
#   DATA_DIR : raw dataset dir (default: ../data/raw)
#   OUT_DIR  : cluster output dir (default: ../data/clusters)
#   PYTHON   : python interpreter (default: python3)
#   K_LIST   : space-separated K values (default: "10 100")
set -euo pipefail
cd "$(dirname "$0")"

# Cap BLAS / OpenMP threads to avoid OpenBLAS memory-region exhaustion
# on large NUMA hosts. See build_clusters.py top-of-file for context.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

DATA_DIR="${DATA_DIR:-../data/raw}"
OUT_DIR="${OUT_DIR:-../data/clusters}"
PYTHON="${PYTHON:-python3}"
K_LIST="${K_LIST:-10 100}"

# (DATASET, N) — same N column as run_final_numa.sh
DATASETS_DEFAULT=(
    "sift        1000000"
    "gist        1000000"
    "text2image  1000000"
    "text2image  40000000"
)
if [ -n "${DATASETS:-}" ]; then
    eval "DATASETS_ARR=( $DATASETS )"
else
    DATASETS_ARR=("${DATASETS_DEFAULT[@]}")
fi

mkdir -p "$OUT_DIR"

# Build a per-combo alias list so the filter can target size-qualified
# names (sift_1m, gist_1m, text2image_1m, text2image_40m) in addition to
# the bare dataset family. Mirrors build_volumes.sh / build_all_gt.sh.
combo_alias() {
    local ds=$1 n=$2
    case "$ds:$n" in
        sift:1000000)        echo "sift sift_1m" ;;
        gist:1000000)        echo "gist gist_1m" ;;
        text2image:1000000)  echo "text2image text2image_1m" ;;
        text2image:40000000) echo "text2image text2image_40m" ;;
        *)                   echo "$ds" ;;
    esac
}

filter_match_aliases() {
    local aliases=$1
    [ $# -lt 2 ] && return 0  # no filter args → match all
    shift
    for f in "$@"; do
        for a in $aliases; do
            [[ "$a" == *"$f"* ]] && return 0
        done
    done
    return 1
}

raw_file_of() {
    case "$1" in
        sift)       echo "$DATA_DIR/sift_base.fvecs" ;;
        gist)       echo "$DATA_DIR/gist_base.fvecs" ;;
        text2image) echo "$DATA_DIR/text2image-40M.fbin" ;;
    esac
}

for ROW in "${DATASETS_ARR[@]}"; do
    read -r DS N <<< "$ROW"
    ALIASES=$(combo_alias "$DS" "$N")
    filter_match_aliases "$ALIASES" "$@" || continue

    RAW=$(raw_file_of "$DS")
    if [ -z "$RAW" ] || [ ! -f "$RAW" ]; then
        echo "[skip] $DS N=$N — raw file ${RAW:-<unknown>} missing"
        continue
    fi

    for K in $K_LIST; do
        BASE="$OUT_DIR/cluster_${DS}_n${N}_k${K}.bin"
        QUERY="$OUT_DIR/cluster_${DS}_query_n${N}_k${K}.bin"
        if [ -f "$BASE" ] && [ -f "$QUERY" ]; then
            echo "[skip] $DS N=$N K=$K (both files exist)"
            continue
        fi
        echo "=== build $DS N=$N K=$K ==="
        $PYTHON build_clusters.py --dataset "$DS" --n "$N" --k "$K" \
            --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
    done
done
echo "All requested cluster combos processed. ($OUT_DIR)"
