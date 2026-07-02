#!/usr/bin/env bash
# run_light.sh — single-command light-mode reproduction for one dataset.
#
# Prerequisite: the chosen dataset must already be downloaded under
# bench/data/raw/. Run the download step separately:
#   ./download_datasets.sh sift          # or gist / text2image
#
# Then this script runs:
#   1. cmake build                     (skipped if already built)
#   2. Build HNSW index volumes        (skipped if already built)
#   3. Build ground-truth files        (skipped if already built)
#   4. Build cluster files             (skipped if already built)
#   5. run_final_numa_light.sh         (nq=100, runs=1 by default)
#
# Usage (run from repository root):
#   ./download_datasets.sh <ds>                                 # step 1
#   ./run_light.sh [OUT_DIR] [--no-numa]                        # step 2
#
# Choose dataset via DATASET env (only one at a time):
#   DATASET=sift            (default; 1M, 128d, L2)
#   DATASET=gist            (1M, 960d, L2)
#   DATASET=text2image_1m   (1M subset of T2I, 200d, cosine)
#   DATASET=text2image_40m  (40M T2I, 200d, cosine; needs ~32 GB disk + RAM)
#
# Examples:
#   ./run_light.sh                                # sift, NUMA off (default; run_all.sh forces NUMACTL="")
#   DATASET=gist ./run_light.sh results/          # gist
#   DATASET=text2image_1m ./run_light.sh --no-numa
#   THREADS=8 DATASET=gist ./run_light.sh --no-numa
#
# Env overrides (all optional):
#   DATASET    sift|gist|text2image_1m|text2image_40m (default: sift)
#   NUMACTL    full numactl prefix (overrides --no-numa if set explicitly)
#   THREADS    search threads (default: 32 via run_all.sh)
#   DATA_DIR   raw dataset directory (default: bench/data/raw)
#   JOBS       cmake build parallelism (default: nproc)

set -euo pipefail
cd "$(dirname "$0")"

TS=$(date +%Y%m%d_%H%M)

# ── Argument parsing ─────────────────────────────────────────────────────────
NUMA_FLAG=""
OUT_DIR="results_light_${TS}"
for arg in "$@"; do
    case "$arg" in
        --no-numa) NUMA_FLAG="--no-numa" ;;
        -*) echo "Unknown option: $arg" >&2; exit 1 ;;
        *)  OUT_DIR="$arg" ;;
    esac
done

# ── Dataset selection (DATASET env; default sift) ────────────────────────────
DATASET="${DATASET:-sift}"
case "$DATASET" in
    sift)
        RUN_DS=sift
        DS_ROW='"sift 1000000 sift_1m 0.003 0.0015 0.006 100 1"'
        REQUIRED_FILES=(sift_base.fvecs sift_query.fvecs)
        ;;
    gist)
        RUN_DS=gist
        DS_ROW='"gist 1000000 gist_1m 0.003 0.0015 0.006 100 1"'
        REQUIRED_FILES=(gist_base.fvecs gist_query.fvecs)
        ;;
    text2image_1m|t2i_1m)
        RUN_DS=text2image
        DS_ROW='"text2image 1000000 text2image_1m 0.003 0.0015 0.006 100 1"'
        REQUIRED_FILES=(text2image-40M.fbin text2image-query-100K.fbin)
        ;;
    text2image_40m|t2i_40m)
        RUN_DS=text2image_40m
        DS_ROW='"text2image 40000000 text2image_40m 0.0003 0.00015 0.0006 100 1"'
        REQUIRED_FILES=(text2image-40M.fbin text2image-query-100K.fbin)
        ;;
    *)
        echo "ERROR: unknown DATASET=$DATASET" >&2
        echo "  valid: sift | gist | text2image_1m | text2image_40m" >&2
        exit 1
        ;;
esac

echo "======================================================"
echo " RACORN-1 / ACORN-1 — Light Mode ($DATASET smoke run)"
echo " $(date)"
echo " OUT_DIR : $OUT_DIR"
echo "======================================================"
echo ""

# ── Prerequisite: dataset files must already be present ──────────────────────
DATA_DIR_CHECK="${DATA_DIR:-bench/data/raw}"
for FILE in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$DATA_DIR_CHECK/$FILE" ]; then
        echo "ERROR: $FILE (required by DATASET=$DATASET) not found under $DATA_DIR_CHECK/."
        echo ""
        echo "Run the dataset download step first:"
        echo "  ./download_datasets.sh ${DATASET%_*}"
        echo ""
        echo "If your environment blocks FTP, see bench/datasets.md for the"
        echo "manual download recipe."
        exit 1
    fi
done

# Build + index + GT + clusters + bench (single dataset)
# Pass --no-numa to run_all.sh via NUMACTL env if requested
if [ -n "$NUMA_FLAG" ] && [ -z "${NUMACTL+set}" ]; then
    export NUMACTL=""
fi
# LIGHT_DS_ROW: dataset row forwarded by run_all.sh to run_final_numa_light.sh
# so the bench step actually uses the selected dataset (not the default sift).
LIGHT_DS_ROW="$DS_ROW" DATASETS="$RUN_DS" ./run_all.sh "$OUT_DIR"
