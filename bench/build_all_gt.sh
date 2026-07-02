#!/usr/bin/env bash
# Build ground-truth (.bin) files for every (dataset, N, M, ef_construction)
# combination used by run_final_numa.sh.
#
# - Skips combos whose GT already exists (idempotent)
# - Auto-builds the index volume first if it's missing
# - Logs go to log/build_gt_<TS>.log
#
# Usage:
#   ./build_all_gt.sh              # all combos
#   ./build_all_gt.sh sift gist    # filter by dataset name (substring match)
set -euo pipefail
cd "$(dirname "$0")"

BENCH="${BENCH:-../build/bench/bench_acorn}"
DIR_OPTS="--data-dir ./data/raw --volume-dir ./data/volumes --gt-dir ./data/gt"
THREADS="${THREADS:-32}"   # override via env; 32 matches the run_*.sh convention
GT_K=100
GT_NQ=${GT_NQ:-1000}    # GT query count (override via env: GT_NQ=10000 ./build_all_gt.sh)
TS=$(date +%Y%m%d_%H%M)
LOG_DIR=./log
mkdir -p "$LOG_DIR" ./data/volumes ./data/gt
LOG="$LOG_DIR/build_gt_${TS}.log"

# (dataset, N, M, ef_construction) combos. Must cover every (M, efc) pair
# whose volume the bench scripts will load — kept aligned with build_volumes.sh.
COMBOS=(
    "gist        1000000   16  100"
    "gist        1000000   32  100"
    "gist        1000000   16  200"
    "gist        1000000   16  500"
    "sift        1000000   16  100"
    "sift        1000000   32  100"
    "sift        1000000   16  200"
    "sift        1000000   16  500"
    "text2image  1000000   16  100"
    "text2image  1000000   32  100"
    "text2image  1000000   16  200"
    "text2image  1000000   16  500"
    "text2image  40000000  16  100"
    "text2image  40000000  32  100"
    "text2image  40000000  16  200"
    "text2image  40000000  16  500"
)

# Optional filter: only run combos whose dataset name matches one of the args
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
        gist)       echo 960 ;;
        text2image) echo 200 ;;
        sift)       echo 128 ;;
        *) echo "Unknown dataset: $1" >&2; exit 1 ;;
    esac
}

vol_path_of() {
    local dataset=$1 N=$2 M=$3 EFC=$4
    local D=$(dims_of "$dataset")
    echo "./data/volumes/${dataset}_${N}_${D}_m${M}_ef${EFC}.usearch"
}

gt_path_of() {
    local dataset=$1 N=$2 M=$3 EFC=$4
    local D=$(dims_of "$dataset")
    echo "./data/gt/${dataset}_${N}_${D}_m${M}_ef${EFC}_gt_k${GT_K}_nq${GT_NQ}.bin"
}

ensure_volume() {
    local dataset=$1 N=$2 M=$3 EFC=$4
    local vol=$(vol_path_of "$dataset" "$N" "$M" "$EFC")
    if [ -f "$vol" ]; then
        echo "  [vol] exists: $vol" | tee -a "$LOG"
        return 0
    fi
    echo "  [vol] BUILDING: $vol" | tee -a "$LOG"
    $BENCH --dataset "$dataset" --n "$N" --m "$M" --ef-construction "$EFC" \
        --build-index --search-threads "$THREADS" $DIR_OPTS 2>&1 | tee -a "$LOG"
}

ensure_gt() {
    local dataset=$1 N=$2 M=$3 EFC=$4
    local gt=$(gt_path_of "$dataset" "$N" "$M" "$EFC")
    if [ -f "$gt" ]; then
        echo "  [gt]  exists: $gt" | tee -a "$LOG"
        return 0
    fi
    echo "  [gt]  BUILDING: $gt" | tee -a "$LOG"
    $BENCH --dataset "$dataset" --n "$N" --m "$M" --ef-construction "$EFC" \
        --build-gt --build-gt-k "$GT_K" --nq "$GT_NQ" --search-threads "$THREADS" \
        $DIR_OPTS 2>&1 | tee -a "$LOG"
}

echo "==== Build all GT files ===="
echo "  Threads : $THREADS"
echo "  GT k    : $GT_K"
echo "  GT nq   : $GT_NQ"
echo "  Log     : $LOG"
[ $# -gt 0 ] && echo "  Filter  : $*"
echo "  Combos  : ${#COMBOS[@]}"
echo

# Combo-level alias so the filter can target text2image_1m / text2image_40m
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

for ROW in "${COMBOS[@]}"; do
    read -r DATASET N M EFC <<< "$ROW"
    ALIASES=$(combo_alias "$DATASET" "$N")
    if ! filter_match_aliases "$ALIASES" "$@"; then
        continue
    fi
    echo "=== [$DATASET N=$N M=$M EFC=$EFC] ===" | tee -a "$LOG"
    ensure_volume "$DATASET" "$N" "$M" "$EFC"
    ensure_gt "$DATASET" "$N" "$M" "$EFC"
    echo | tee -a "$LOG"
done

echo "==== All GT builds complete (timestamp: $TS) ===="
echo "Log: $LOG"
