#!/usr/bin/env bash
# Build HNSW index volumes for every (dataset, N, M, ef_construction)
# combo consumed by the bench scripts.
#
# - Skips combos whose raw dataset file is missing (idempotent partial setup)
# - Skips combos whose volume already exists (idempotent)
# - Optional dataset filter via positional args (substring match on the
#   dataset name): `./build_volumes.sh sift` only builds sift volumes.
#
# Env overrides:
#   BENCH      bench_acorn binary (default: ../build/bench/bench_acorn)
#   DATA_DIR   raw dataset dir   (default: ./data/raw)
#   VOLUME_DIR index volume dir  (default: ./data/volumes)
set -euo pipefail
cd "$(dirname "$0")"

BENCH="${BENCH:-../build/bench/bench_acorn}"
DATA_DIR="${DATA_DIR:-./data/raw}"
VOLUME_DIR="${VOLUME_DIR:-./data/volumes}"
mkdir -p "$VOLUME_DIR"

# (dataset, N, M, ef_construction) combos. Must cover every (M, efc) pair
# consumed by the bench scripts. Currently:
#   m=16  efc=100  — default index (run_bench_default.sh, run_qps_default.sh, run_correlation*)
#   m=32  efc=100  — `--m 32` cell in run_bench_alt.sh
#   m=16  efc=200  — kept for ef_search=200 baseline / sanity comparisons
#   m=16  efc=500  — `--ef-construction 500` cell in run_bench_alt.sh
COMBOS=(
    "sift        1000000   16  100"
    "sift        1000000   32  100"
    "sift        1000000   16  200"
    "sift        1000000   16  500"
    "gist        1000000   16  100"
    "gist        1000000   32  100"
    "gist        1000000   16  200"
    "gist        1000000   16  500"
    "text2image  1000000   16  100"
    "text2image  1000000   32  100"
    "text2image  1000000   16  200"
    "text2image  1000000   16  500"
    "text2image  40000000  16  100"
    "text2image  40000000  32  100"
    "text2image  40000000  16  200"
    "text2image  40000000  16  500"
)

filter_match() {
    local dataset=$1
    [ $# -lt 2 ] && return 0  # no filter args → match all
    shift
    for f in "$@"; do
        [[ "$dataset" == *"$f"* ]] && return 0
    done
    return 1
}

dims_of() {
    case "$1" in
        sift)       echo 128 ;;
        gist)       echo 960 ;;
        text2image) echo 200 ;;
        *) echo "Unknown dataset: $1" >&2; exit 1 ;;
    esac
}

raw_file_of() {
    local dataset=$1
    case "$dataset" in
        sift)       echo "$DATA_DIR/sift_base.fvecs" ;;
        gist)       echo "$DATA_DIR/gist_base.fvecs" ;;
        text2image) echo "$DATA_DIR/text2image-40M.fbin" ;;
    esac
}

vol_path_of() {
    local dataset=$1 N=$2 M=$3 EFC=$4
    local D=$(dims_of "$dataset")
    echo "$VOLUME_DIR/${dataset}_${N}_${D}_m${M}_ef${EFC}.usearch"
}

# Build a per-combo alias so the filter can target text2image_1m or
# text2image_40m specifically (in addition to the bare dataset name).
combo_alias() {
    local ds=$1 n=$2
    case "$ds:$n" in
        text2image:1000000)  echo "text2image text2image_1m" ;;
        text2image:40000000) echo "text2image text2image_40m" ;;
        *)                   echo "$ds" ;;
    esac
}

filter_match_aliases() {
    local aliases=$1
    [ $# -lt 2 ] && return 0
    shift
    for f in "$@"; do
        for a in $aliases; do
            [[ "$a" == *"$f"* ]] && return 0
        done
    done
    return 1
}

for COMBO in "${COMBOS[@]}"; do
    read -r DS N M EFC <<< "$COMBO"
    ALIASES=$(combo_alias "$DS" "$N")
    filter_match_aliases "$ALIASES" "$@" || continue

    RAW=$(raw_file_of "$DS")
    if [ ! -f "$RAW" ]; then
        echo "--- $DS ${N} (m=$M, ef=$EFC): SKIP — raw file $RAW missing ---"
        continue
    fi

    VOL=$(vol_path_of "$DS" "$N" "$M" "$EFC")
    if [ -f "$VOL" ]; then
        echo "--- $DS ${N} (m=$M, ef=$EFC): SKIP — $VOL already present ---"
        continue
    fi

    echo "--- $DS ${N} (m=$M, ef=$EFC): BUILDING ---"
    $BENCH --dataset "$DS" --n "$N" --m "$M" --ef-construction "$EFC" --build-index \
           --data-dir "$DATA_DIR" --volume-dir "$VOLUME_DIR"
done

echo "=== build_volumes.sh done ==="
